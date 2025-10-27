"""
Advanced RAG Demo with Batch Retrieval and Visualization

This demo implements advanced RAG techniques based on HuggingFace cookbook:
- Batch retrieval from scientific paper summaries
- Visualization with PaCMAP dimensionality reduction
- Interactive scatter plots showing document clusters

References:
- https://huggingface.co/learn/cookbook/en/advanced_rag
"""

import asyncio
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import pacmap
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# LangChain imports
from langchain_core.documents import Document
from pathlib import Path
from sentence_transformers import SentenceTransformer
# Scientific computing
from sklearn.preprocessing import normalize
from time import time
from tqdm import trange
from typing import List, Dict, Tuple, Optional

from utopia.utils.data_utils import read_json

import pandas as pd


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    batch_size: int = 64
    chunk_size: int = 512
    embedding_model: str = "thenlper/gte-small"  # 384-dimensional embeddings
    num_retrieved_docs: int = 50
    num_docs_final: int = 5
    normalize_embeddings: bool = True
    use_langchain: bool = False
    device: str = "mps:0" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class RAG:
    """Advanced RAG system with batch retrieval, and visualization"""

    def __init__(self, config: RAGConfig = None, use_langchain: bool = False, debug: bool = False, **kwargs):
        self.config = config
        self.embeddings = None
        self.knowledge_index: Optional[FAISS] = None
        self.debug = debug
        self.device = config.device
        self.use_langchain: bool = use_langchain

        # All documents and their mappings
        self.documents: List[Document] = []  # All documents
        self.document_embeddings: Optional[torch.Tensor] = None
        self.id2docs: Dict[str, Document] = {}

        
        self.document_status = pd.DataFrame(columns=['id', 'status'])
        
        
        
        # Track original document count (for separating external vs simulation papers)
        self.original_doc_count: int = 0

        if self.use_langchain:
            pass

        else:
            assert kwargs.get('model') is not None, "Model must be provided if use_langchain is False"
            self.sentence_transformer_model = kwargs.get('model')
        
    def load_documents(self, data_path: str = None) -> List[Document]:
        """Load documents from JSON file"""
        if self.debug:
            dataset = read_json("data/additional_data_2024_7.json")
            for item in dataset[:128]:

                
                title = re.sub(r'\s+', ' ', item.get('title', '').replace('\n', ' ')).strip()
                abstract = re.sub(r'\s+', ' ', item.get('summary', '').replace('\n', ' ')).strip()
                
                metadata = {
                    'id': item.get('id', ''),
                    'title': title,
                    'tags': ', '.join(item.get('tags', [])),
                    'published': item.get('published', ''),
                }

                # Create document with summary as content
                doc = Document(
                    page_content=abstract,
                    metadata=metadata
                )
                self.id2docs[item['id']] = doc

            docs = list(self.id2docs.values())
            self.documents = docs
            self.document_status = pd.DataFrame(columns=['id', 'status'], data=[{'id': doc.metadata['id'], 'status': 'unsubmitted'} for doc in docs])

        else:
            dataset = load_dataset("Ahren09/SciEvo", "arxiv", split="train")
            self.documents = dataset
            
        
    def mark_papers_as_submitted(self, paper_ids: List[str]):
        """Mark papers as submitted"""
        self.document_status.loc[self.document_status['id'].isin(paper_ids), 'status'] = 'submitted'
        

    def initialize_sentence_transformer(self, model):
        """Initialize SentenceTransformer embeddings"""
        assert isinstance(model, SentenceTransformer), "Model must be a SentenceTransformer instance"
        self.sentence_transformer_model = model

    def build_knowledge_index(self) -> FAISS:
        """Build FAISS index with normalized embeddings using PyTorch tensors

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

        if self.use_langchain:
            raise NotImplementedError("LangChain is not supported yet")
            # Branch 1: LangChain + FAISS
            print("[LangChain] Using HuggingFaceEmbeddings + FAISS")
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': self.config.normalize_embeddings},
                show_progress=True
            )

            # Build FAISS index (legacy - kept for compatibility)
            self.knowledge_index = FAISS.from_documents(
                self.documents,
                self.embeddings
            )

            # Store document embeddings for batch retrieval
            doc_texts = [doc.page_content for doc in self.documents]
            print(f"Embedding {len(doc_texts)} documents for both indices...")

            batch_size = 1024
            all_embeddings = []

            for i in range(0, len(doc_texts), batch_size):
                batch_texts = doc_texts[i:i + batch_size]
                print(
                    f"  Batch {i // batch_size + 1}/{(len(doc_texts) + batch_size - 1) // batch_size} ({len(batch_texts)} docs)")
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

            doc_embeddings_np = np.array(all_embeddings)
            embeddings_tensor = torch.from_numpy(doc_embeddings_np).float().to(self.device)

            if self.config.normalize_embeddings:
                embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)

            # Both indices start with same embeddings
            self.submission_embeddings = embeddings_tensor.clone()
            self.citation_embeddings = embeddings_tensor.clone()

            print(f"[LangChain] Build indices: {time() - t0:.2f} seconds")

        else:
            # Branch 2: Pure SentenceTransformer
            print("[SentenceTransformer] Encoding documents using SentenceTransformer")

            # Initialize SentenceTransformer if not already done
            if not hasattr(self, 'sentence_transformer_model') or self.sentence_transformer_model is None:
                self.sentence_transformer_model = SentenceTransformer(
                    self.config.embedding_model,
                    device=self.device
                )

            # Encode all documents at once
            doc_texts = [doc.page_content for doc in self.documents]
            print(f"Encoding {len(doc_texts)} documents...")
            
            # https://arxiv.org/pdf/1234.12345.pdf -> 0
            self.id2document_indices = {doc.metadata['id']: i for i, doc in enumerate(self.documents)}

            self.document_embeddings = self.sentence_transformer_model.encode(
                doc_texts,
                convert_to_tensor=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=True,
                batch_size=64,
                device=self.device
            )  # (D, dim)
            

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
        # Select the appropriate index
        if retrieval_type == "submission":
            indices = (self.document_status[self.document_status['status'] == "unsubmitted"]).index.values
            assert len(indices) > 0, "No candidate papers found!"
            
            print(f"[SUBMISSION INDEX] Retrieving papers for {len(queries)} queries from {len(indices)} papers ...")
        elif retrieval_type == "citation":
            if cite_only_accepted_papers:
                indices = (self.document_status[self.document_status['status'] == "accepted"]).index.values
            else:
                indices = (self.document_status[(self.document_status['status'] == "accepted") | (self.document_status['status'] == "rejected")]).index.values
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

            query_embeddings_np = np.array(self.embeddings.embed_documents(queries))
            query_embeddings = torch.from_numpy(query_embeddings_np).float().to(self.device)

            if self.config.normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        else:
            # Branch 2: SentenceTransformer - Simple Q x D similarity matrix
            print("[SentenceTransformer Branch] Computing Q x D similarity matrix...")

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

    def visualize_embeddings(
            self,
            queries: List[str] = None,
            save_path: str = None,
            show_plot: bool = True
    ):
        """
        Visualize document embeddings using PaCMAP dimensionality reduction

        Args:
            queries: Optional list of queries to visualize alongside documents
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if not PACMAP_AVAILABLE:
            print("PaCMAP not available. Skipping visualization.")
            return

        if self.document_embeddings is None:
            raise ValueError("Document embeddings not available. Build knowledge index first.")

        print("\nGenerating embedding visualization...")

        # Convert PyTorch tensors to numpy for visualization
        embeddings_to_plot = self.document_embeddings.cpu().numpy()
        labels = ['document'] * len(self.documents)
        titles = [doc.metadata.get('title', '')[:50] for doc in self.documents]

        # Add query embeddings if provided
        if queries:
            query_embeddings = np.array([
                self.embeddings.embed_query(q) for q in queries
            ])
            if self.config.normalize_embeddings:
                query_embeddings = normalize(query_embeddings)

            embeddings_to_plot = np.vstack([embeddings_to_plot, query_embeddings])
            labels.extend(['query'] * len(queries))
            titles.extend(queries)

        # Apply PaCMAP for dimensionality reduction
        print("Applying PaCMAP dimensionality reduction...")
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42
        )
        embeddings_2d = reducer.fit_transform(embeddings_to_plot)

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot documents
        doc_mask = np.array(labels) == 'document'
        scatter_docs = ax.scatter(
            embeddings_2d[doc_mask, 0],
            embeddings_2d[doc_mask, 1],
            c='lightblue',
            alpha=0.6,
            s=50,
            label='Documents',
            edgecolors='navy',
            linewidth=0.5
        )

        # Plot queries if available
        if queries:
            query_mask = np.array(labels) == 'query'
            scatter_queries = ax.scatter(
                embeddings_2d[query_mask, 0],
                embeddings_2d[query_mask, 1],
                c='red',
                alpha=0.8,
                s=200,
                marker='*',
                label='Queries',
                edgecolors='darkred',
                linewidth=1.5
            )

            # Annotate queries
            for i, (x, y) in enumerate(embeddings_2d[query_mask]):
                ax.annotate(
                    f"Q{i + 1}: {queries[i][:30]}...",
                    (x, y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        ax.set_xlabel('PaCMAP Dimension 1', fontsize=12)
        ax.set_ylabel('PaCMAP Dimension 2', fontsize=12)
        ax.set_title(
            'Document Embeddings Visualization (PaCMAP 2D Projection)',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

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
                print(f"      Tags: {doc.metadata.get('tags', 'None')}")
                print(f"      Summary: {doc.page_content[:150]}...")

            if len(docs) > max_docs_display:
                print(f"\n  ... and {len(docs) - max_docs_display} more documents")
                
    def mark_papers_as_submitted(self, accepted_paper_ids: List[str], rejected_paper_ids: List[str]):
        """Mark papers as submitted"""
        self.document_status.loc[self.document_status['id'].isin(accepted_paper_ids), 'status'] = 'accepted'
        self.document_status.loc[self.document_status['id'].isin(rejected_paper_ids), 'status'] = 'rejected'
        

    def to_dict(self) -> Dict:
        """Serialize RAG state (document status only, embeddings rebuilt on load)"""
        return {
            'document_status': self.document_status.to_dict('records')
        }

    @classmethod
    def from_dict(cls, data: Dict, config: RAGConfig, model=None, debug: bool = False) -> 'RAG':
        """Reconstruct RAG from saved state"""
        rag = cls(config=config, model=model, debug=debug)
        rag.document_status = pd.DataFrame(data['document_status'])
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
