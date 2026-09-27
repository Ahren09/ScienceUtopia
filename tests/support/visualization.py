"""Small, deterministic inputs and figure inspection for renderer contracts."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


def diagnostic_inputs():
    strategies = ["explorer", "exploiter", "cautious_explorer"]
    agent_rows = [
        dict(
            seed=seed,
            year=year,
            strategy=strategy,
            avg_career_distance=0.1 * (i + year),
            topic_switched=(i + year) % 2,
            acceptance_rate=0.2 * (i + 1),
            funding=20 + seed + i + year,
        )
        for seed in (1, 2)
        for year in (1, 2)
        for i, strategy in enumerate(strategies)
    ]
    papers = [
        dict(
            age=age,
            strategy=strategy,
            citations=1 + i + j + age,
            novelty_score=0.2 * (i + 1) + j / 20,
            cd=0.1 * (j - i),
        )
        for age in (1, 3)
        for i, strategy in enumerate(strategies)
        for j in range(3)
    ]
    ecosystem = pd.DataFrame(
        [
            dict(
                seed=seed,
                year=year,
                topic_entropy_normalized=0.5 + year / 10,
                funding_gini=0.3,
                citation_gini=0.4,
                num_active_researchers=12 + year,
            )
            for seed in (1, 2)
            for year in (1, 2)
        ]
    )
    researchers = pd.DataFrame(
        [
            dict(seed=seed, strategy=strategy, hit_papers=seed + i)
            for seed in (1, 2)
            for i, strategy in enumerate(strategies)
        ]
    )
    x = np.tile(np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]), 3)
    rd = pd.DataFrame(
        {
            "x": x,
            "funded": (x > 0).astype(int),
            "cumulative_subsequent_funding_3y": 20 + x + (x > 0) * 3,
            "pubs_written_3y": 2 + x / 3,
            "any_application_3y": (x > -1).astype(int),
            "time_to_attrition": np.tile([1, 2, 3, 1, 2, 3], 3),
            "attrition_event": np.tile([1, 1, 0, 1, 0, 0], 3),
            "applicant_id": [f"a{i % 6}" for i in range(18)],
            "competition": [f"c{i // 6}" for i in range(18)],
        }
    )
    rd_result = {
        "primary": {"tau": 3.0, "ci_lo": 2.0, "ci_hi": 4.0},
        "robustness": {
            "bandwidth_sensitivity": {"2": {"tau": 3.0, "se": 0.5}},
            "placebo_cutoffs": {"-1": {"tau": 0.1, "se": 0.2}},
        },
    }
    pairs = pd.DataFrame(
        [
            dict(
                paper_id=f"p{i}",
                stratum=s,
                delta=0.1 * (i - j),
                focal_betweenness=0.1 * i,
                similarity=0.6 + 0.02 * (i + j),
                identity_effect=0.05 * (i + j),
            )
            for i in range(4)
            for j, s in enumerate(("d2", "d3", "far"))
        ]
    )
    paper_effects = pd.DataFrame(
        {"paper_id": [f"p{i}" for i in range(4)], "paper_effect": [-0.2, 0.1, 0.2, 0.4]}
    )
    secondaries = {
        "S2_distance_trend": {"estimate": 0.1, "p_boot": 0.3},
        "S5_author_betweenness_moderation": {"p_boot": 0.4},
    }
    primary = {"primary_d2_minus_far": {"ci_lo": -0.1, "ci_hi": 0.3}}
    stability = pd.DataFrame(
        {
            "paper_effect_main": [-0.2, 0.1, 0.2, 0.4],
            "paper_effect_stab": [-0.1, 0.15, 0.22, 0.35],
        }
    )
    reviews = pd.DataFrame(
        [
            dict(
                success=True,
                arm="blind",
                panel=1,
                regime=regime,
                score_raw=2 + i / 2,
                mean_just_len=40 + i * 10,
            )
            for regime in ("A", "B", "C")
            for i in range(4)
        ]
    )
    per_paper = {
        "P1": pd.DataFrame({"A": [0.2, 0.4, 0.6], "B": [0.3, 0.5, 0.4]}),
        "P2": pd.DataFrame({"A": [0.1, 0.3, 0.5], "B": [0.2, 0.2, 0.4]}),
        "P3": pd.DataFrame(
            {
                "delta_score": [0.1, 0.2, 0.4],
                "novelty_score": [0.2, 0.4, 0.7],
                "ipw_weight": [1.0, 2.0, 1.5],
            }
        ),
        "P4": pd.DataFrame(
            {"d_A": [0.2, 0.3, 0.4], "d_B": [0.1, 0.1, 0.2], "diff": [0.1, 0.2, 0.2]}
        ),
    }
    return {
        "exploration": (
            researchers,
            pd.DataFrame(agent_rows),
            ecosystem,
            pd.DataFrame(papers),
        ),
        "funding": (rd, rd_result, 2.0),
        "network": (pairs, paper_effects, secondaries, primary, stability),
        "review": (reviews, reviews, per_paper),
    }


def figure_record(fig):
    fig.canvas.draw()
    axes = []
    for ax in fig.axes:
        axes.append(
            {
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
                "xticks": [text.get_text() for text in ax.get_xticklabels()],
                "lines": [
                    {
                        "x": np.asarray(line.get_xdata(orig=False)).tolist(),
                        "y": np.asarray(line.get_ydata(orig=False)).tolist(),
                        "color": line.get_color(),
                        "label": line.get_label(),
                    }
                    for line in ax.lines
                ],
                "collections": [
                    {
                        "offsets": np.asarray(collection.get_offsets()).tolist(),
                        "array": None
                        if collection.get_array() is None
                        else np.asarray(collection.get_array()).tolist(),
                    }
                    for collection in ax.collections
                ],
                "patch_count": len(ax.patches),
            }
        )
    return {
        "size": fig.get_size_inches().tolist(),
        "axes": axes,
        "suptitle": fig._suptitle.get_text() if fig._suptitle else None,
        "rgba_sha256": hashlib.sha256(bytes(fig.canvas.buffer_rgba())).hexdigest(),
    }


def capture_figures(function, arguments, directory):
    import matplotlib
    from matplotlib.figure import Figure

    records = []

    def capture(fig, path, **options):
        records.append(
            dict(name=Path(path).name, options=options, **figure_record(fig))
        )

    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        with patch.object(Figure, "savefig", capture):
            function(*arguments, str(directory))
    return records


def embedding_simulation():
    tracker = SimpleNamespace(
        embeddings={f"p{i}": np.array([i, i * i, i % 2, 1.0]) for i in range(4)},
        paper_metadata={f"p{i}": {"author_id": f"a{i}"} for i in range(4)},
    )
    population = {
        f"a{i}": SimpleNamespace(
            exploration_strategy=[
                "explorer",
                "exploiter",
                "cautious_explorer",
                "explorer",
            ][i],
            resources=20 + i,
            is_active=True,
        )
        for i in range(4)
    }
    return SimpleNamespace(
        is_exploration_experiment=True,
        embedding_tracker=tracker,
        ecosystem=SimpleNamespace(agent_population=population),
    )


def embedding_contract(rag_renderer, paper_renderer, charts):
    """Exercise projection parameters, query order and W&B's RNG use without services."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    projection = {}

    class PaCMAP:
        def __init__(self, **options):
            projection["options"] = options

        def fit_transform(self, values):
            projection["input"] = values.tolist()
            return values[:, [0, 2]]

    class Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([[1, 0, 2, 1], [0, 1, 3, 1], [2, 1, 1, 1]], dtype=float)

    queries = []

    class Embedder:
        def embed_query(self, text):
            queries.append(text)
            return [len(text), 1.0, 0.5, 1.0]

    rag = SimpleNamespace(
        document_embeddings=Tensor(),
        documents=[
            SimpleNamespace(metadata={"title": f"Document {i}"}) for i in range(3)
        ],
        embeddings=Embedder(),
        config=SimpleNamespace(normalize_embeddings=True),
    )
    pacmap = ModuleType("pacmap")
    pacmap.PaCMAP = PaCMAP
    wandb = ModuleType("wandb")
    wandb.Histogram = lambda values: {"histogram": list(values)}
    wandb.plot = SimpleNamespace(line_series=lambda **options: {"line_series": options})
    wandb.Image = lambda fig: {"figure": figure_record(fig)}
    corpus_figures = []
    state = np.random.get_state()
    try:
        with matplotlib.rc_context(matplotlib.rcParamsDefault):
            with patch.dict(sys.modules, {"pacmap": pacmap, "wandb": wandb}):
                with patch.object(
                    Figure,
                    "savefig",
                    lambda fig, *args, **kwargs: corpus_figures.append(
                        dict(options=kwargs, **figure_record(fig))
                    ),
                ):
                    rag_renderer(
                        rag,
                        queries=["graph", "model"],
                        save_path="corpus.png",
                        show_plot=False,
                    )
                simulation = embedding_simulation()
                fig = paper_renderer(simulation)
                paper = figure_record(fig)
                plt.close(fig)
                np.random.seed(103)
                history = {
                    "acceptance_rate": [0.2, 0.3],
                    "gini": [0.4, 0.5],
                    "mean_score": [3.0, 4.0],
                }
                results = {
                    "peer_review": {
                        "score_metrics": {
                            "overall_mean": 3.0,
                            "overall_std": 0.7,
                            "total_reviews": 5,
                        }
                    }
                }
                chart_values = charts(2, results, simulation, history)
                rng = np.random.get_state()
                rng_digest = hashlib.sha256(
                    rng[1].tobytes() + str(rng[2:]).encode()
                ).hexdigest()
    finally:
        np.random.set_state(state)
        plt.close("all")
    return json.loads(
        json.dumps(
            {
                "corpus": corpus_figures,
                "projection": projection,
                "queries": queries,
                "paper": paper,
                "charts": chart_values,
                "rng_sha256": rng_digest,
            }
        )
    )
