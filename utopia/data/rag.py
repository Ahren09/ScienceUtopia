"""Batched retrieval from the public SciEvo scientific-paper corpus."""

import asyncio
import numpy as np
import os
import re
import datetime
import pickle
import tempfile
from pathlib import Path
from utopia.utils.data_utils import json_sha256
from collections import Counter
import torch
import torch.nn.functional as F
from dataclasses import dataclass
# LangChain imports
from langchain_core.documents import Document
# Scientific computing
from time import time
from tqdm import trange, tqdm
from typing import List, Dict, Optional

from datasets import load_dataset


from utopia.agents.conference import ALL_CS_TOPICS, ARXIV_CATEGORY_NAMES

import pandas as pd


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    batch_size: int = 64
    chunk_size: int = 512
    embedding_model: str = "thenlper/gte-small" # "Qwen/Qwen3-Embedding-0.6B" 
    embedding_revision: str = "17e1f347d17fe144873b1201da91788898c639cd"
    dataset_revision: str = "f80fb38a034b2f763dfcbdd2cd187f5b42ddcce6"
    cache_dir: str = "data/cache/rag"
    num_retrieved_docs: int = 1000
    num_docs_final: int = 5
    normalize_embeddings: bool = True
    use_langchain: bool = False
    device: str = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    n_workers: int = 32  # Number of workers for parallel dataset processing (1 = sequential)


class RAG:
    """Year-aware retrieval with caches bound to the actual input documents."""

    def __init__(self, config: RAGConfig = None, use_langchain: bool = False, debug: bool = False, **kwargs):
        self.config = config
        self.embeddings = None
        self.knowledge_index = None
        self.debug = debug
        self.device = config.device
        self.use_langchain: bool = use_langchain

        # Year-based data partitioning
        self.start_year: int = kwargs.get('start_year', 2016)
        self.current_sim_year: int = 0

        # All documents and their mappings
        self.documents: List[Document] = []  # All documents
        self.document_embeddings: Optional[torch.Tensor] = None
        self.id2docs: Dict[str, Document] = {}


        self.document_status = pd.DataFrame(columns=['id', 'status', 'publication_year'])
        
        
        
        # Track original document count (for separating external vs simulation papers)
        self.original_doc_count: int = 0

        self.sentence_transformer_model = kwargs.get('model')
        
    def load_documents(self, data_path: str = None, num_years: int = 10) -> List[Document]:
        """Load documents from JSON file

        Args:
            data_path: Path to local JSON data file (used in debug mode)
            num_years: Number of simulation years
        """

        def _normalize_whitespace(text: str) -> str:
            return re.sub(r'\s+', ' ', (text or '').replace('\n', ' ')).strip()

        def _as_datetime(ts):
            """Ensure 'published' is a datetime-like object (prefer pd.Timestamp)."""
            if ts is None or ts == '':
                return pd.NaT
            if isinstance(ts, (pd.Timestamp, datetime.datetime)):
                return pd.Timestamp(ts)

            return pd.to_datetime(ts)

        self.num_years = num_years
        end_real_year = self.start_year + self.num_years - 1
        valid_years = set(range(self.start_year, end_real_year + 1))

        self.id2docs = {}

        revision = self.config.dataset_revision
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info('Ahren09/SciEvo', revision=revision).sha
        self.dataset_identity = {
            "dataset": "Ahren09/SciEvo", "configuration": "arxiv",
            "revision": revision,
            "start_year": self.start_year, "num_years": num_years,
            "categories": sorted(ALL_CS_TOPICS),
            "normalization": "parallel" if self.config.n_workers > 1 else "sequential",
        }
        cache_root = Path(self.config.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        expected = getattr(self, '_checkpoint_dataset_identity', None)
        if expected and expected != self.dataset_identity:
            raise ValueError('Checkpoint dataset identity differs from the requested dataset/year selection')
        cache_path = cache_root / f"documents-{json_sha256(self.dataset_identity, sort_keys=True)}.pkl"
        cache_valid = False
        if os.path.exists(cache_path):
            print(f"[Cache HIT] Loading documents from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            self.documents = cache_data['documents']
            # Validate cache has 'tags' metadata and id2year_pos mapping
            if (cache_data.get('identity') == self.dataset_identity and self.documents
                    and 'tags' in self.documents[0].metadata and 'id2year_pos' in cache_data):
                self.id2year_pos = cache_data['id2year_pos']
                cache_valid = True
            else:
                print(f"[Cache STALE] Missing 'tags' or 'id2year_pos' — regenerating cache")
                self.documents = []

        if not cache_valid:
            dataset = load_dataset("Ahren09/SciEvo", "arxiv", split="train",
                                   revision=revision)
            n_workers = self.config.n_workers
            num_proc = n_workers if n_workers > 1 else None

            dataset = dataset.filter(lambda x: len(set(x['tags']) & set(ALL_CS_TOPICS)) > 0, num_proc=num_proc)

            self.id2docs = {}


            if num_proc:
                # Parallel path: use HF datasets built-in multiprocessing
                print(f"[RAG] Processing dataset with {n_workers} workers...")

                def _parse_pub_year(item):
                    ts = item.get('published', '')
                    if ts is None or ts == '':
                        return {'_pub_year': -1}
                    try:
                        dt = pd.to_datetime(ts)
                        return {'_pub_year': dt.year if not pd.isna(dt) else -1}
                    except Exception:
                        return {'_pub_year': -1}

                dataset = dataset.map(_parse_pub_year, num_proc=num_proc, desc="Parsing dates")

                nat_count = sum(1 for y in dataset['_pub_year'] if y == -1)

                valid_years_list = sorted(valid_years)
                valid_years_set_for_filter = set(valid_years_list)
                dataset = dataset.filter(
                    lambda x: x['_pub_year'] in valid_years_set_for_filter,
                    num_proc=num_proc,
                    desc="Filtering by year",
                )

                df = dataset.to_pandas()

                # Vectorized operations — no per-item Python calls
                df['_published'] = pd.to_datetime(df['published'])
                df['_id'] = df['id'].str.replace('http://', 'https://', regex=False)

                # Build Document list via zip over numpy arrays (no for-loop)
                self.documents = [
                    Document(
                        page_content=abstract,
                        metadata={'id': id_, 'title': title, 'published': pub, 'publication_year': int(year), 'tags': list(tags), 'topics': list([ARXIV_CATEGORY_NAMES[tag] for tag in tags if tag in ARXIV_CATEGORY_NAMES])}
                    )
                    for abstract, id_, title, pub, year, tags in zip(
                        df['summary'].values, df['_id'].values, df['title'].values,
                        df['_published'].values, df['_pub_year'].values, df['tags'].values,
                    )
                ]
                self.id2docs = dict(zip(df['_id'].values, self.documents))

            else:
                # Sequential path (original)
                nat_count = 0
                for item in tqdm(dataset, desc="Processing dataset"):
                    title = _normalize_whitespace(item.get('title', ''))
                    abstract = _normalize_whitespace(item.get('summary', ''))

                    published = _as_datetime(item.get('published', ''))
                    if pd.isna(published):
                        nat_count += 1
                        continue

                    pub_year = published.year
                    if pub_year not in valid_years:
                        continue

                    metadata = {
                        'id': item.get('id', '').replace('http://', 'https://'),
                        'title': title,
                        'published': published,
                        'publication_year': int(pub_year),
                        'tags': list(item['tags']),
                        'topics': list([ARXIV_CATEGORY_NAMES[tag] for tag in item['tags'] if tag in ARXIV_CATEGORY_NAMES]),
                    }
                    doc = Document(page_content=abstract, metadata=metadata)
                    self.id2docs[metadata['id']] = doc

            if nat_count > 0:
                print(f"[RAG] Skipped {nat_count} papers with missing publication dates")

            docs = list(self.id2docs.values())
            self.documents = docs
            self.documents.sort(key=lambda doc: doc.metadata['publication_year'])

            # Build id -> (year, position_within_year) mapping
            year_counts = {}
            id2year_pos = {}
            for doc in self.documents:
                yr = doc.metadata['publication_year']
                pos = year_counts.get(yr, 0)
                id2year_pos[doc.metadata['id']] = (yr, pos)
                year_counts[yr] = pos + 1
            self.id2year_pos = id2year_pos

            cache_data = {
                'identity': self.dataset_identity,
                'documents': self.documents,
                'id2year_pos': id2year_pos,
            }
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_root, delete=False) as f:
                pickle.dump(cache_data, f)
                temporary = f.name
            os.replace(temporary, cache_path)

        self.id2docs = {doc.metadata['id']: doc for doc in self.documents}
        if not self.documents:
            raise ValueError(f"No CS papers in SciEvo for {self.start_year}-{end_real_year}")
        self.document_identity = json_sha256(
            [(doc.metadata, doc.page_content) for doc in self.documents],
            sort_keys=True, default=str)

        expected = getattr(self, '_checkpoint_document_identity', None)
        if expected and expected != self.document_identity:
            raise ValueError('Checkpoint corpus content or document order changed')

        # Log year distribution
        year_counts = Counter(doc.metadata['publication_year'] for doc in self.documents)
        print(f"[RAG] Loaded {len(self.documents)} papers for years {self.start_year}-{end_real_year}:")
        for yr in sorted(year_counts):
            print(f"  {yr}: {year_counts[yr]} papers")

        self.document_status = pd.DataFrame(
            data=[{'id': doc.metadata['id'], 'status': 'unsubmitted', 'publication_year': doc.metadata['publication_year']} for doc in self.documents],
            columns=['id', 'status', 'publication_year']
        )
            
        
    def set_current_year(self, sim_year: int):
        """Set the current simulation year for year-based data partitioning"""
        self.current_sim_year = sim_year
        real_year = self.start_year + sim_year - 1
        available = (self.document_status['publication_year'] == real_year).sum()
        unsubmitted = ((self.document_status['publication_year'] == real_year) & (self.document_status['status'] == 'unsubmitted')).sum()
        print(f"[RAG] Sim year {sim_year} (real: {real_year}): {available} total papers, {unsubmitted} available for submission")


    def build_knowledge_index(self):
        """Build normalized document embeddings for PyTorch retrieval

        Builds TWO separate indices:
        1. Submission Index: For finding papers to submit
        2. Citation Index: For finding papers to cite (superset of submission index)

        Initially both indices contain the same documents since no papers are submitted yet.
        """
        print(f"Building dual knowledge indices with {self.config.embedding_model}...")
        print(f"Using device: {self.device}")

        # Store original document count for tracking
        self.original_doc_count = len(self.documents)
        t0 = time()

        print("[SentenceTransformer] Encoding documents using SentenceTransformer")

        # Lazy-load model on first use (fp16 only on CUDA; CPU runs fp32)
        if getattr(self, 'sentence_transformer_model', None) is None:
            from sentence_transformers import SentenceTransformer
            model_kwargs = {"torch_dtype": torch.float16} if 'cuda' in str(self.device) else {}
            self.sentence_transformer_model = SentenceTransformer(
                self.config.embedding_model, revision=self.config.embedding_revision,
                device=self.device, model_kwargs=model_kwargs
            )

        doc_texts = [doc.page_content for doc in self.documents]
        print(f"Encoding {len(doc_texts)} documents...")

        # https://arxiv.org/pdf/1234.12345.pdf -> 0
        self.id2document_indices = {doc.metadata['id']: i for i, doc in enumerate(self.documents)}

        # The ordered metadata/text binding rejects equal-length but different corpora.
        identity = {
            "documents": json_sha256(
                [(doc.metadata, doc.page_content) for doc in self.documents],
                sort_keys=True, default=str),
            "model": self.config.embedding_model,
            "revision": self.config.embedding_revision,
            "normalized": self.config.normalize_embeddings,
            "dtype": "float16" if "cuda" in str(self.device) else "float32",
        }
        cache_root = Path(self.config.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = cache_root / f"embeddings-{json_sha256(identity, sort_keys=True)}.pt"

        if os.path.exists(cache_file):
            print(f"[Cache HIT] Loading embeddings from {cache_file}")
            self.document_embeddings = torch.load(cache_file, map_location=self.device, weights_only=True)
            # Cache was written as fp16 on GPU; CPU queries encode as fp32 —
            # unify dtype so torch.mm never sees mixed precision.
            if 'cuda' not in str(self.device) and self.document_embeddings.dtype != torch.float32:
                self.document_embeddings = self.document_embeddings.float()
            if self.document_embeddings.shape[0] != len(self.documents):
                print(f"[Cache STALE] Embedding count {self.document_embeddings.shape[0]} != document count {len(self.documents)} — re-encoding")
                self.document_embeddings = None
        else:
            self.document_embeddings = None

        if self.document_embeddings is None:
            print(f"Encoding {len(doc_texts)} documents...")
            self.document_embeddings = self.sentence_transformer_model.encode(
                doc_texts,
                convert_to_tensor=True,
                normalize_embeddings=self.config.normalize_embeddings,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                device=self.device,
            )
            with tempfile.NamedTemporaryFile(dir=cache_root, delete=False) as stream:
                temporary = stream.name
                torch.save(self.document_embeddings, stream)
            os.replace(temporary, cache_file)
            print(f"[Cache SAVE] Saved {self.document_embeddings.shape[0]} embeddings to {cache_file}")

        print(f"[SentenceTransformer] Encode documents: {time() - t0:.2f} seconds")

        return self.knowledge_index

    def batch_retrieve(
        self,
        queries: List[str],
        retrieval_type: str = "submission",
        cite_only_accepted_papers: bool = True
    ) -> Dict:
        """
        Batch retrieve documents for multiple queries using PyTorch tensor operations
        Optimized for large document sets with GPU acceleration

        Args:
            queries: List of query strings
            retrieval_type: Which index to use - "submission" for paper submission, "citation" for citing papers

        Returns:
            Dictionary with 'topk_indices' and 'similarity_scores'
        """
        
        assert retrieval_type in ["submission", "accepted_papers", "accepted_or_rejected_papers"], "Invalid retrieval_type"

        # Select the appropriate index
        if retrieval_type == "submission":
            # Year-specific: only unsubmitted papers from current real-world year
            real_year = self.start_year + self.current_sim_year - 1
            indices = self.document_status[
                (self.document_status['status'] == "unsubmitted") &
                (self.document_status['publication_year'] == real_year)
            ].index.values
            assert len(indices) > 0, f"No candidate papers found for year {real_year}!"

            print(f"[SUBMISSION INDEX] Retrieving papers for {len(queries)} queries from {len(indices)} papers (year={real_year})")

        elif retrieval_type in ["accepted_papers", "accepted_or_rejected_papers"]:
            # Cumulative: all accepted/rejected papers from any prior year
            if retrieval_type == "accepted_papers":
                indices = (self.document_status[self.document_status['status'] == "accept"]).index.values
            else:
                indices = (self.document_status[(self.document_status['status'].isin({"accept", "reject"}))]).index.values
            assert len(indices) > 0, "No candidate papers found!"
            print(f"\n[CITATION INDEX] Batch retrieving for {len(queries)} queries...")
            
        else:
            raise ValueError(f"Invalid retrieval_type: {retrieval_type}. Must be 'submission' or 'citation'")
        
        indices = np.sort(indices)
        mapping = {i: original_idx for i, original_idx in enumerate(indices)}
        indices = torch.tensor(indices, device=self.device)
        
        
        
        document_embeddings = self.document_embeddings[indices]

        t0 = time()

        if self.use_langchain:
            # Branch 1: LangChain
            print("[LangChain Branch] Embedding queries...")

            # The document and query paths must use the same immutable encoder.
            query_embeddings_np = self.sentence_transformer_model.encode(
                queries, convert_to_numpy=True, normalize_embeddings=self.config.normalize_embeddings,
                device=self.device)
            query_embeddings = torch.from_numpy(query_embeddings_np).to(device=self.device, dtype=document_embeddings.dtype)

            if self.config.normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        else:
            # Branch 2: SentenceTransformer - Simple Q x D similarity matrix
            print("[SentenceTransformer Branch] Computing Q x D similarity matrix...")
            if getattr(self, 'sentence_transformer_model', None) is None:
                from sentence_transformers import SentenceTransformer
                self.sentence_transformer_model = SentenceTransformer(
                    self.config.embedding_model, device=self.device
                )

            # Encode queries: shape (Q, embedding_dim)
            query_embeddings = self.sentence_transformer_model.encode(
                queries,
                convert_to_tensor=True,
                normalize_embeddings=self.config.normalize_embeddings,
                device=self.device
            )  # (Q, dim)

        print(f"Query embeddings shape: {query_embeddings.shape}")
        print(f"Document embeddings shape: {document_embeddings.shape}")

        # Compute similarity matrix: (Q, embedding_dim) @ (embedding_dim, D) = (Q, D)
        print("Computing similarity matrix...")
        t1 = time()

        all_similarity_scores = []

        for i in trange(0, len(document_embeddings), self.config.batch_size):
            batch_embeddings = document_embeddings[i:i + self.config.batch_size]
            similarity_scores = torch.mm(query_embeddings, batch_embeddings.T)
            all_similarity_scores.append(similarity_scores)

        similarity_scores = torch.cat(all_similarity_scores, dim=1)  # (Q, D)
        assert similarity_scores.shape == (len(queries), document_embeddings.shape[0])
        del all_similarity_scores
        print(f"Similarity computation time: {time() - t1:.2f} seconds")
        # Get top-k indices for each query (vectorized)
        print("Finding top-k documents...")
        t2 = time()
        _, topk_indices = torch.topk(similarity_scores, k=min(self.config.num_retrieved_docs, len(indices)), dim=1)
        topk_indices = topk_indices.cpu().numpy()
        
        topk_indices = np.vectorize(mapping.get)(topk_indices)

        print(f"Top-k selection time: {time() - t2:.2f} seconds")

        print(f"Total batch retrieval time: {time() - t0:.2f} seconds")
        return {
            'topk_indices': topk_indices,
            'similarity_scores': similarity_scores,
        }

    async def batch_retrieve_async(
            self,
            queries: List[str],
            retrieval_type: str = "submission"
    ) -> Dict:
        """
        Async batch retrieval using vectorized operations

        Note: The actual computation is still synchronous (vectorized),
        but wrapped in async for API compatibility
        """
        print(f"\nAsync batch retrieving for {len(queries)} queries...")

        # Wrap synchronous batch operation in async
        # The vectorized operations are already efficient
        results = await asyncio.to_thread(
            self.batch_retrieve, queries, retrieval_type
        )

        return results


    def print_results(
            self,
            queries: List[str],
            results: List[List[Document]],
            max_docs_display: int = 3
    ):
        """Pretty print retrieval results"""
        print("\n" + "=" * 80)
        print("RETRIEVAL RESULTS")
        print("=" * 80)

        for q_idx, (query, docs) in enumerate(zip(queries, results), 1):
            print(f"\nQuery {q_idx}: {query}")
            print("-" * 80)

            for d_idx, doc in enumerate(docs[:max_docs_display], 1):
                print(f"\n  [{d_idx}] {doc.metadata.get('title', 'No title')}")
                print(f"      Authors: {doc.metadata.get('authors', 'Unknown')}")
                print(f"      Tags: {doc.metadata['tags']}")
                print(f"      Summary: {doc.page_content[:150]}...")

            if len(docs) > max_docs_display:
                print(f"\n  ... and {len(docs) - max_docs_display} more documents")
                
    def mark_papers_as_submitted(self, accepted_paper_ids: List[str], rejected_paper_ids: List[str]):
        """Mark papers as submitted"""
        self.document_status.loc[self.document_status['id'].isin(accepted_paper_ids), 'status'] = 'accept'
        self.document_status.loc[self.document_status['id'].isin(rejected_paper_ids), 'status'] = 'reject'
        assert len(self.document_status.loc[self.document_status['status'] == 'submitted']) == 0, "Some papers still require a decision!"

    def to_dict(self) -> Dict:
        """Serialize RAG state (document status only, embeddings rebuilt on load)"""
        return {
            'document_status': self.document_status.to_dict('records'),
            'start_year': self.start_year,
            'current_sim_year': self.current_sim_year,
            'dataset_identity': getattr(self, 'dataset_identity', None),
            'ordered_documents_sha256': getattr(self, 'document_identity', None),
            'embedding_model': self.config.embedding_model,
            'embedding_revision': self.config.embedding_revision,
        }

    @classmethod
    def from_dict(cls, data: Dict, config: RAGConfig, model=None, debug: bool = False, **kwargs) -> 'RAG':
        """Reconstruct RAG from saved state"""
        start_year = data.get('start_year', kwargs.get('start_year', 2016))
        rag = cls(config=config, model=model, debug=debug, start_year=start_year)
        rag.document_status = pd.DataFrame(data['document_status'])
        # Backward compatibility: add publication_year column if missing
        if 'publication_year' not in rag.document_status.columns:
            rag.document_status['publication_year'] = 0
        rag.current_sim_year = data.get('current_sim_year', 0)
        rag._checkpoint_dataset_identity = data.get('dataset_identity')
        rag._checkpoint_document_identity = data.get('ordered_documents_sha256')
        for key in ('embedding_model', 'embedding_revision'):
            if data.get(key) and data[key] != getattr(config, key):
                raise ValueError(f'Checkpoint {key} differs from current retrieval configuration')
        return rag

    def add_documents_to_corpus(self, new_documents: List[Document]):
        """
        Add new documents (e.g., accepted papers from simulation) to the corpus.
        These papers are added ONLY to the citation index (not submission index)
        since they represent already-submitted papers.

        Args:
            new_documents: List of LangChain Document objects to add

        Note:
            This incrementally updates the knowledge base without rebuilding from scratch.
        """
        if not new_documents:
            print("No new documents to add.")
            return

        print(f"Adding {len(new_documents)} new documents to citation index (already-submitted papers)...")
        t0 = time()

        # Add to document lists
        docs_to_add = []
        for doc in new_documents:
            doc_id = doc.metadata.get('id', f"sim_{len(self.documents)}")
            if doc_id not in self.id2docs:
                self.documents.append(doc)
                self.id2docs[doc_id] = doc
                docs_to_add.append(doc)

                # Add to citation index only
                self.citation_documents.append(doc)
                self.citation_doc_ids.add(doc_id)

        if not docs_to_add:
            print("All documents already in corpus.")
            return

        # Encode new documents
        new_doc_texts = [doc.page_content for doc in docs_to_add]

        if self.use_langchain:
            # LangChain branch
            print("[LangChain] Encoding new documents...")
            new_embeddings_np = np.array(self.embeddings.embed_documents(new_doc_texts))
            new_embeddings = torch.from_numpy(new_embeddings_np).float().to(self.device)

            if self.config.normalize_embeddings:
                new_embeddings = F.normalize(new_embeddings, p=2, dim=1)

            # Update citation index only
            self.citation_embeddings = torch.cat([self.citation_embeddings, new_embeddings], dim=0)

            # Update FAISS index (legacy)
            print("[LangChain] Updating FAISS index...")
            self.knowledge_index.add_documents(docs_to_add)

        else:
            # SentenceTransformer branch
            print("[SentenceTransformer] Encoding new documents...")
            self._ensure_sentence_transformer()
            new_embeddings = self.sentence_transformer_model.encode(
                new_doc_texts,
                convert_to_tensor=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=True,
                batch_size=64,
                device=self.device
            )

            # Update citation index only
            self.citation_embeddings = torch.cat([self.citation_embeddings, new_embeddings], dim=0)

        print(f"Added {len(docs_to_add)} documents in {time() - t0:.2f} seconds")
        print(f"Total documents in corpus: {len(self.documents)}")
        print(f"Submission index: {len(self.submission_documents)} documents")
        print(f"Citation index: {len(self.citation_documents)} documents")

    def get_corpus_statistics(self) -> Dict:
        """Get statistics about the corpus"""
        simulation_papers = sum(1 for doc in self.documents
                               if doc.metadata.get('source') == 'simulation')
        external_papers = len(self.documents) - simulation_papers

        return {
            'total_documents': len(self.documents),
            'external_papers': external_papers,
            'simulation_papers': simulation_papers,
            'original_doc_count': self.original_doc_count
        }
