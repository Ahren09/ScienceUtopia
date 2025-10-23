"""
Advanced RAG Demo with Batch Retrieval and Visualization

This demo implements advanced RAG techniques based on HuggingFace cookbook:
- Batch retrieval from scientific paper summaries
- Reranking with cross-encoder
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
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

# Set device for PyTorch operations
DEVICE = "mps:0" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    batch_size: int = 64
    chunk_size: int = 512
    embedding_model: str = "thenlper/gte-small"  # 384-dimensional embeddings
    num_retrieved_docs: int = 50
    num_docs_final: int = 5
    normalize_embeddings: bool = True
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"


class RAG:
    """Advanced RAG system with batch retrieval, reranking, and visualization"""

    def __init__(self, config: RAGConfig = None, use_langchain: bool = False, debug: bool = False):
        self.config = config or RAGConfig()
        self.embeddings = None
        self.knowledge_index: Optional[FAISS] = None
        self.document_embeddings: Optional[torch.Tensor] = None
        self.debug = debug
        self.device = DEVICE
        self.use_langchain: bool = use_langchain
        
        # Each document is stored twice.
        self.documents: List[Document] = []
        self.id2docs: Dict[str, Document] = {}
        
    def load_documents(self, data_path: str = None) -> List[Document]:
        """Load documents from JSON file"""
        if self.debug:
            dataset = read_json("data/additional_data_2024_7.json")
            for item in dataset:

                
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

        else:
            dataset = load_dataset("Ahren09/SciEvo", "arxiv", split="train")
            self.documents = dataset

    def initialize_sentence_transformer(self):
        """Initialize SentenceTransformer embeddings"""
        self.sentence_transformer_model = SentenceTransformer(self.config.embedding_model, device=self.device)

    def build_knowledge_index(self) -> FAISS:
        """Build FAISS index with normalized embeddings using PyTorch tensors"""
        print(f"Building knowledge index with {self.config.embedding_model}...")
        print(f"Using device: {self.device}")

        t0 = time()

        if self.use_langchain:
            # Branch 1: LangChain + FAISS
            print("[LangChain] Using HuggingFaceEmbeddings + FAISS")

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': self.config.normalize_embeddings},
                show_progress=True
            )

            # Build FAISS index
            self.knowledge_index = FAISS.from_documents(
                self.documents,
                self.embeddings
            )

            # Store document embeddings for batch retrieval
            doc_texts = [doc.page_content for doc in self.documents]
            print(f"Embedding {len(doc_texts)} documents...")

            batch_size = 1024
            all_embeddings = []

            for i in range(0, len(doc_texts), batch_size):
                batch_texts = doc_texts[i:i + batch_size]
                print(
                    f"  Batch {i // batch_size + 1}/{(len(doc_texts) + batch_size - 1) // batch_size} ({len(batch_texts)} docs)")
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

            doc_embeddings_np = np.array(all_embeddings)
            self.document_embeddings = torch.from_numpy(doc_embeddings_np).float().to(self.device)

            if self.config.normalize_embeddings:
                self.document_embeddings = F.normalize(self.document_embeddings, p=2, dim=1)

            print(f"[LangChain] Build index: {time() - t0:.2f} seconds")

        else:
            # Branch 2: Pure SentenceTransformer
            print("[SentenceTransformer] Using pure SentenceTransformer")

            # Initialize SentenceTransformer if not already done
            self.sentence_transformer_model = SentenceTransformer(
                self.config.embedding_model,
                device=self.device
            )

            # Encode all documents at once
            doc_texts = [doc.page_content for doc in self.documents]
            print(f"Encoding {len(doc_texts)} documents...")

            self.document_embeddings = self.sentence_transformer_model.encode(
                doc_texts,
                convert_to_tensor=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=True,
                batch_size=64,
                device=self.device
            )  # (D, dim)
            print(f"[SentenceTransformer] Encode documents: {time() - t0:.2f} seconds")

        print(f"Document embeddings shape: {self.document_embeddings.shape}")
        print(f"Knowledge index built with {len(self.documents)} documents")

        return self.knowledge_index

    def batch_retrieve(
            self,
            queries: List[str],
            use_reranker: bool = None
    ) -> List[List[Document]]:
        """
        Batch retrieve documents for multiple queries using PyTorch tensor operations
        Optimized for large document sets with GPU acceleration

        Args:
            queries: List of query strings
            use_reranker: Whether to use reranker (default from config)

        Returns:
            List of document lists, one per query
        """
        if self.document_embeddings is None:
            raise ValueError("Knowledge index not built. Call build_knowledge_index() first.")

        use_reranker = use_reranker if use_reranker is not None else self.config.use_reranker

        print(f"\nBatch retrieving for {len(queries)} queries...")
        print(f"Initial retrieval: {self.config.num_retrieved_docs} docs per query")
        if use_reranker:
            print(f"Reranking to: {self.config.num_docs_final} docs per query")

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
        print(f"Document embeddings shape: {self.document_embeddings.shape}")

        # Compute similarity matrix: (Q, embedding_dim) @ (embedding_dim, D) = (Q, D)
        print("Computing similarity matrix...")
        t1 = time()

        all_similarity_scores = []

        for i in trange(0, len(self.document_embeddings), self.config.batch_size):
            batch_embeddings = self.document_embeddings[i:i + self.config.batch_size]
            similarity_scores = torch.mm(query_embeddings, batch_embeddings.T)
            all_similarity_scores.append(similarity_scores)

        similarity_scores = torch.cat(all_similarity_scores, dim=1)  # (Q, D)
        assert similarity_scores.shape == (len(queries), len(self.documents))
        del all_similarity_scores
        print(f"Similarity computation time: {time() - t1:.2f} seconds")
        # Get top-k indices for each query (vectorized)
        print("Finding top-k documents...")
        t2 = time()
        _, topk_indices = torch.topk(similarity_scores, k=self.config.num_retrieved_docs, dim=1)
        topk_indices = topk_indices.cpu().numpy()
        print(f"Top-k selection time: {time() - t2:.2f} seconds")

        # Build results list from indices
        topk_docs = [[self.documents[idx] for idx in query_indices] for query_indices in topk_indices]

        # Optional reranking
        if use_reranker:
            print("Applying reranking...")
            if self.use_langchain:
                reranked_indices = self._batch_rerank_documents_torch(queries, topk_indices, query_embeddings)
            else:
                reranked_indices = self._batch_rerank_documents_sentencetransformer(queries, topk_indices)

            print(f"Total batch retrieval time: {time() - t0:.2f} seconds")
            return {
                'topk_indices': reranked_indices,
                'similarity_scores': similarity_scores,
            }
        else:
            print(f"Total batch retrieval time: {time() - t0:.2f} seconds")
            return {
                'topk_indices': topk_indices,
                'similarity_scores': similarity_scores,
            }

    async def batch_retrieve_async(
            self,
            queries: List[str],
            use_reranker: bool = None
    ) -> List[List[Document]]:
        """
        Async batch retrieval using vectorized operations

        Note: The actual computation is still synchronous (vectorized),
        but wrapped in async for API compatibility
        """
        if self.knowledge_index is None:
            raise ValueError("Knowledge index not built. Call build_knowledge_index() first.")

        use_reranker = use_reranker if use_reranker is not None else self.config.use_reranker

        print(f"\nAsync batch retrieving for {len(queries)} queries...")

        # Wrap synchronous batch operation in async
        # The vectorized operations are already efficient
        results = await asyncio.to_thread(
            self.batch_retrieve, queries, use_reranker
        )

        return results

    def _batch_rerank_documents_sentencetransformer(
            self,
            queries: List[str],
            topk_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Batch rerank using SentenceTransformer - returns reranked indices
        No for-loops: processes all queries at once using vectorized operations

        Args:
            queries: List of query strings
            topk_indices: Initial top-k indices (Q, num_retrieved_docs)
            topk_docs: List of document lists (one per query)

        Returns:
            Reranked indices (Q, num_docs_final)
        """
        print("[SentenceTransformer Branch] Reranking with vectorized operations...")
        t0 = time()

        raise NotImplementedError("Reranking has bugs now")

        # Encode all queries at once: (Q, dim)
        query_embeddings = self.sentence_transformer_model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=self.config.normalize_embeddings,
            device=self.device
        )

        # Flatten all candidate documents across all queries
        all_doc_texts = [doc.page_content for doc in self.documents]

        # Encode all documents at once: (total_docs, dim)
        all_doc_embeddings = self.sentence_transformer_model.encode(
            all_doc_texts,
            convert_to_tensor=True,
            normalize_embeddings=self.config.normalize_embeddings,
            device=self.device,
            show_progress_bar=False,
        )

        # Build block diagonal structure for batch processing
        # Each query only computes similarity with its candidate documents
        reranked_indices = []
        doc_offset = 0

        num_docs_per_query = topk_indices.shape[1]

        for query_idx, num_docs in enumerate(num_docs_per_query):
            if num_docs == 0:
                reranked_indices.append([])
                continue

            # Get embeddings for this query's documents
            doc_embs = all_doc_embeddings[topk_indices[query_idx]]  # (num_docs, dim)
            query_emb = query_embeddings[query_idx]  # (dim,)

            # Compute similarities: (num_docs,) = (num_docs, dim) @ (dim,)
            similarities = torch.mv(doc_embs, query_emb)

            # Get top-k local indices
            _, local_top_indices = torch.topk(similarities, k=min(self.config.num_docs_final, num_docs))
            local_top_indices = local_top_indices.cpu().numpy()

            # Map to global indices
            global_indices = topk_indices[query_idx][local_top_indices]
            reranked_indices.append(global_indices)

            doc_offset += num_docs

        print(f"SentenceTransformer reranking time: {time() - t0:.2f} seconds")
        return np.array(reranked_indices)

    def _batch_rerank_documents_torch(
            self,
            queries: List[str],
            topk_indices: np.ndarray,
            query_embeddings: torch.Tensor
    ) -> np.ndarray:
        """
        PyTorch-optimized batch rerank - returns reranked indices
        No for-loops: processes all queries at once using vectorized operations

        Args:
            queries: List of query strings
            topk_indices: Initial top-k indices (Q, num_retrieved_docs)
            topk_docs: List of document lists (one per query)
            query_embeddings: Pre-computed query embeddings as PyTorch tensor

        Returns:
            Reranked indices (Q, num_docs_final)
        """
        print("Reranking with PyTorch tensors (vectorized)...")
        t0 = time()

        # Flatten all candidate documents across all queries
        all_doc_texts = [doc.page_content for docs in topk_docs for doc in docs]

        # Encode all documents at once: (total_docs, dim)
        print(f"    Encoding {len(all_doc_texts)} total candidate documents...")
        all_doc_embeddings_np = np.array(self.embeddings.embed_documents(all_doc_texts))
        all_doc_embeddings = torch.from_numpy(all_doc_embeddings_np).float().to(self.device)

        # Normalize if required
        if self.config.normalize_embeddings:
            all_doc_embeddings = F.normalize(all_doc_embeddings, p=2, dim=1)

        # Build block diagonal structure for batch processing
        reranked_indices = []
        doc_offset = 0

        num_docs_per_query = [len(docs) for docs in topk_docs]

        for query_idx, num_docs in enumerate(num_docs_per_query):
            if num_docs == 0:
                reranked_indices.append([])
                continue

            # Get embeddings for this query's documents
            doc_embs = all_doc_embeddings[doc_offset:doc_offset + num_docs]  # (num_docs, dim)
            query_emb = query_embeddings[query_idx]  # (dim,)

            # Compute similarities: (num_docs,) = (num_docs, dim) @ (dim,)
            similarities = torch.mv(doc_embs, query_emb)

            # Get top-k local indices
            _, local_top_indices = torch.topk(similarities, k=min(self.config.num_docs_final, num_docs))
            local_top_indices = local_top_indices.cpu().numpy()

            # Map to global indices
            global_indices = topk_indices[query_idx][local_top_indices]
            reranked_indices.append(global_indices)

            doc_offset += num_docs

        print(f"PyTorch reranking time: {time() - t0:.2f} seconds")
        return np.array(reranked_indices)

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
