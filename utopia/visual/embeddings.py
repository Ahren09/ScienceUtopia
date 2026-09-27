"""PaCMAP corpus and PCA paper visualizations with separate projection policies."""

from typing import List
import numpy as np


def plot_rag_embeddings(
    rag, queries: List[str] = None, save_path: str = None, show_plot: bool = True
):
    """
    Visualize document embeddings using PaCMAP dimensionality reduction

    Args:
        queries: Optional list of queries to visualize alongside documents
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    try:
        import pacmap
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import normalize
    except ImportError:
        print("PaCMAP/matplotlib not available. Skipping visualization.")
        return

    if rag.document_embeddings is None:
        raise ValueError(
            "Document embeddings not available. Build knowledge index first."
        )

    print("\nGenerating embedding visualization...")

    # Convert PyTorch tensors to numpy for visualization
    embeddings_to_plot = rag.document_embeddings.cpu().numpy()
    labels = ["document"] * len(rag.documents)
    titles = [doc.metadata.get("title", "")[:50] for doc in rag.documents]

    # Add query embeddings if provided
    if queries:
        query_embeddings = np.array([rag.embeddings.embed_query(q) for q in queries])
        if rag.config.normalize_embeddings:
            query_embeddings = normalize(query_embeddings)

        embeddings_to_plot = np.vstack([embeddings_to_plot, query_embeddings])
        labels.extend(["query"] * len(queries))
        titles.extend(queries)

    # Apply PaCMAP for dimensionality reduction
    print("Applying PaCMAP dimensionality reduction...")
    reducer = pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42
    )
    embeddings_2d = reducer.fit_transform(embeddings_to_plot)

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot documents
    doc_mask = np.array(labels) == "document"
    scatter_docs = ax.scatter(
        embeddings_2d[doc_mask, 0],
        embeddings_2d[doc_mask, 1],
        c="lightblue",
        alpha=0.6,
        s=50,
        label="Documents",
        edgecolors="navy",
        linewidth=0.5,
    )

    # Plot queries if available
    if queries:
        query_mask = np.array(labels) == "query"
        scatter_queries = ax.scatter(
            embeddings_2d[query_mask, 0],
            embeddings_2d[query_mask, 1],
            c="red",
            alpha=0.8,
            s=200,
            marker="*",
            label="Queries",
            edgecolors="darkred",
            linewidth=1.5,
        )

        # Annotate queries
        for i, (x, y) in enumerate(embeddings_2d[query_mask]):
            ax.annotate(
                f"Q{i + 1}: {queries[i][:30]}...",
                (x, y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    ax.set_xlabel("PaCMAP Dimension 1", fontsize=12)
    ax.set_ylabel("PaCMAP Dimension 2", fontsize=12)
    ax.set_title(
        "Document Embeddings Visualization (PaCMAP 2D Projection)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_paper_embeddings(simulation):
    """Create PCA scatter of paper embeddings colored by strategy."""
    import matplotlib.pyplot as plt

    tracker = simulation.embedding_tracker
    if not tracker or not tracker.embeddings:
        return None

    from sklearn.decomposition import PCA

    paper_ids = list(tracker.embeddings.keys())
    embeddings = [tracker.embeddings[pid] for pid in paper_ids]
    if len(embeddings) < 3:
        return None

    embeddings_array = np.array(embeddings)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_array)

    strategies = []
    for pid in paper_ids:
        meta = tracker.paper_metadata[pid]
        author_id = meta["author_id"]
        agent = simulation.ecosystem.agent_population[author_id]
        strategies.append(agent.exploration_strategy)

    unique_strategies = sorted(set(strategies))
    strategy_colors = {s: plt.cm.tab10(i / 10) for i, s in enumerate(unique_strategies)}
    point_colors = [strategy_colors[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, alpha=0.6, s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Paper Embedding Space (PCA)")

    if len(unique_strategies) <= 10:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=strategy_colors[s],
                markersize=8,
                label=s,
            )
            for s in unique_strategies
        ]
        ax.legend(handles=handles, loc="best", fontsize=8)

    return fig
