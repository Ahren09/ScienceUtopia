"""Visualize paper abstract embeddings colored by simulation year or topic."""

import argparse
import json
import logging
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from utopia.utils.generail_utils import project_setup
logger = logging.getLogger(__name__)
matplotlib.use('Agg')

# (marker, size, alpha) for each review status
MARKER_MAP = {'accept': ('o', 30, 0.7), 'reject': ('x', 20, 0.4)}


def load_checkpoint_papers(checkpoint_path: str) -> list[dict]:
    """Load submitted papers and author-direction mapping from a checkpoint JSON.

    Returns:
        papers: list of dicts with keys: id, sim_year, status, author_id,
                project_start_year
        author_directions: dict mapping author_id -> {year -> direction}
    """
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)

    papers = []
    for p in data['paper_tracker']['papers']:
        papers.append({
            'id': p['id'],
            'sim_year': p['year'],
            'status': p['status'],
            'author_id': p['author_id'],
            'project_start_year': p['project_start_year'],
        })

    # Build author -> {year -> direction} from agent memory banks
    author_directions = {}
    for agent in data['ecosystem_data']['agents']:
        aid = agent['id']
        author_directions[aid] = {}
        for m in agent.get('memory_bank', []):
            if m.get('type') == 'select_research_direction':
                author_directions[aid][m['year']] = m['direction']

    return papers, author_directions


def load_scievo_cache(cache_path: str):
    """Load SciEvo pickle cache with id->global_index mapping.

    Returns:
        id2global_idx: dict mapping paper ID -> global document index
        documents: list of Document objects
    """
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    documents = cache_data['documents']
    id2global_idx = {doc.metadata['id']: i for i, doc in enumerate(documents)}

    return id2global_idx, documents


def resolve_topic(author_id: str, project_start_year: int, author_directions: dict) -> str:
    """Resolve the research direction for a paper from its author's history."""
    dirs = author_directions.get(author_id, {})
    direction = dirs.get(project_start_year)
    if not direction:
        earlier = [y for y in dirs if y <= project_start_year]
        if earlier:
            direction = dirs[max(earlier)]
    return direction or 'unknown'


def build_embedding_dataframe(
    papers: list[dict],
    author_directions: dict,
    id2global_idx: dict,
    embedding_path: str,
    start_year: int,
    method: str = 'tsne',
) -> pd.DataFrame:
    """Build a DataFrame with paper metadata, 2D coords, and topic labels.

    Columns: id, sim_year, real_year, status, author_id, topic, x, y
    """
    all_embeddings = torch.load(embedding_path, map_location='cpu', weights_only=True)

    rows = []
    valid_indices = []
    skipped = 0

    for p in papers:
        idx = id2global_idx.get(p['id'])
        if idx is None or idx >= all_embeddings.shape[0]:
            skipped += 1
            continue
        valid_indices.append(idx)
        lead_author = p['author_id'][0] if isinstance(p['author_id'], list) else p['author_id']
        topic = resolve_topic(lead_author, p['project_start_year'], author_directions)
        rows.append({
            'id': p['id'],
            'sim_year': p['sim_year'],
            'real_year': start_year + p['sim_year'] - 1,
            'status': p['status'],
            'author_id': p['author_id'],
            'topic': topic,
        })

    if skipped:
        logger.warning(f"Skipped {skipped} papers (missing in cache or embeddings)")

    # Batch index with torch, convert to numpy only for dimensionality reduction
    selected = all_embeddings[torch.tensor(valid_indices)]
    embeddings = selected.numpy()

    logger.info(f"Reducing {embeddings.shape[0]} embeddings with {method}...")
    if method == 'tsne':
        perplexity = min(30, max(5, len(embeddings) // 5))
        coords = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000).fit_transform(embeddings)
    elif method == 'umap':
        from umap import UMAP
        coords = UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    df = pd.DataFrame(rows)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    return df


# ---------------------------------------------------------------------------
# Plotting functions — all read from the DataFrame
# ---------------------------------------------------------------------------

def _scatter_by_group(df, group_col, colors, label_fn, title, save_path, figsize=(12, 9)):
    """Shared scatter: color by *group_col*, marker by accept/reject status."""
    fig, ax = plt.subplots(figsize=figsize)

    for group_val in colors:
        for status, (marker, size, alpha) in MARKER_MAP.items():
            sub = df[(df[group_col] == group_val) & (df['status'] == status)]
            if sub.empty:
                continue
            ax.scatter(
                sub['x'], sub['y'],
                c=[colors[group_val]], marker=marker, s=size, alpha=alpha,
                edgecolors='none' if status == 'accept' else None,
                linewidths=0.8,
                label=label_fn(group_val) if status == 'accept' else None,
            )

    group_legend = ax.legend(
        loc='upper left', fontsize=7, title_fontsize=9,
        framealpha=0.9, markerscale=1.2,
    )
    ax.add_artist(group_legend)

    status_handles = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=6, label='Accepted'),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=6, label='Rejected'),
    ]
    ax.legend(handles=status_handles, loc='upper right', title='Status',
              fontsize=8, title_fontsize=9, framealpha=0.9)

    ax.set(title=title, xlabel='Dimension 1', ylabel='Dimension 2')
    ax.grid(True, linestyle='--', alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def plot_embedding_scatter(df: pd.DataFrame, start_year: int, save_path: str):
    """Scatter plot colored by simulation year with marker shape for accept/reject."""
    years = sorted(df['sim_year'].unique())
    cmap = matplotlib.colormaps.get_cmap('tab10').resampled(len(years))
    colors = {yr: cmap(i) for i, yr in enumerate(years)}
    n_acc = (df['status'] == 'accept').sum()
    n_rej = (df['status'] == 'reject').sum()
    _scatter_by_group(
        df, 'sim_year', colors,
        label_fn=lambda yr: f"Year {yr} ({start_year + yr - 1})",
        title=f"Paper Abstract Embeddings by Simulation Year\n({len(df)} papers: {n_acc} accepted, {n_rej} rejected)",
        save_path=save_path,
    )


def plot_embedding_by_topic(df: pd.DataFrame, save_path: str, top_n: int = 15):
    """Scatter plot colored by research topic (author direction) with accept/reject markers."""
    df = df.copy()
    top_topics = list(df['topic'].value_counts().index[:top_n])
    df['topic_display'] = df['topic'].where(df['topic'].isin(top_topics), other='other')
    ordered = top_topics + (['other'] if (df['topic_display'] == 'other').any() else [])
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(ordered))
    colors = {t: cmap(i) for i, t in enumerate(ordered)}
    n_topics = df['topic'].nunique()
    _scatter_by_group(
        df, 'topic_display', colors,
        label_fn=lambda t: t.replace('_', ' '),
        title=f"Paper Abstract Embeddings by Research Topic\n({len(df)} papers, {n_topics} topics — showing top {min(top_n, n_topics)})",
        save_path=save_path, figsize=(14, 10),
    )


def main():
    project_setup()

    parser = argparse.ArgumentParser(description="Plot paper abstract embeddings by simulation year or topic")
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Experiment output directory (e.g., outputs/default_Qwen3-32B)')
    parser.add_argument('--num_years', type=int, default=10)
    parser.add_argument('--start_year', type=int, default=2016)
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'])
    parser.add_argument('--embedding_model', type=str, default='thenlper/gte-small',
                        help='Embedding model tag used for cache files')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    output_dir = args.output_dir

    # 1. Load papers + author directions from final checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_year_{args.num_years}.json")
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    papers, author_directions = load_checkpoint_papers(checkpoint_path)
    logger.info(f"Loaded {len(papers)} submitted papers")

    # 2. Load SciEvo document cache
    cache_path = "outputs/cache_scievo.pkl"
    logger.info(f"Loading SciEvo cache: {cache_path}")
    id2global_idx, documents = load_scievo_cache(cache_path)
    logger.info(f"SciEvo cache: {len(id2global_idx)} documents")

    # 3. Build DataFrame with embeddings, 2D coords, and topics
    model_tag = args.embedding_model.replace("/", "_")
    embedding_path = f"outputs/{model_tag}/cache_embeddings.pt"
    df = build_embedding_dataframe(
        papers, author_directions, id2global_idx,
        embedding_path, args.start_year, method=args.method,
    )
    logger.info(f"DataFrame: {len(df)} papers, {df['topic'].nunique()} topics")

    # 4. Generate both plots from the same DataFrame
    visual_dir = os.path.join(output_dir, 'visual')
    plot_embedding_by_topic(df, os.path.join(visual_dir, f"embedding_{args.method}_by_topic.png"))
    plot_embedding_scatter(df, args.start_year, os.path.join(visual_dir, f"embedding_{args.method}.png"))


if __name__ == '__main__':
    main()
