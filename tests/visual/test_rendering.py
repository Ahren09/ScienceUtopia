import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from matplotlib.colors import to_rgba

from tests.support.visualization import (
    capture_figures,
    diagnostic_inputs,
    embedding_contract,
)
from utopia.utils.paths import project_root
from utopia.visual import (
    plot_exploration_strategies,
    plot_funding_cutoff,
    plot_network_bias_forest,
    review_regime,
)
from utopia.visual.embeddings import plot_paper_embeddings, plot_rag_embeddings
from utopia.visual.wandb import build_charts

GOLDEN = json.loads(
    (project_root() / "tests/support/fixtures/ownership_golden.json").read_text()
)
MODULES = {
    "exploration": plot_exploration_strategies,
    "funding": plot_funding_cutoff,
    "network": plot_network_bias_forest,
    "review": review_regime,
}


@pytest.mark.parametrize("family", list(MODULES))
def test_diagnostic_figures_preserve_data_labels_and_export_options(family, tmp_path):
    module = MODULES[family]
    with patch.object(
        plot_funding_cutoff, "guard_write_path", side_effect=lambda path: path
    ):
        records = capture_figures(
            module.make_figures, diagnostic_inputs()[family], tmp_path
        )
    expected = deepcopy(GOLDEN["figures"][family])
    # Pixel hashes are checked separately against the baseline in the same environment.
    # The durable contract checks artist data and options across rendering environments.
    for record in records + expected:
        record.pop("rgba_sha256")
        for axis in record["axes"]:
            for line in axis["lines"]:
                line["color"] = to_rgba(line["color"])
    assert records == expected


def test_embedding_projections_and_wandb_charts_preserve_parameters_and_rng():
    actual = embedding_contract(
        plot_rag_embeddings, plot_paper_embeddings, build_charts
    )

    def structural(value):
        if isinstance(value, dict):
            return {
                key: structural(item)
                for key, item in value.items()
                if key != "rgba_sha256"
            }
        if isinstance(value, list):
            return [structural(item) for item in value]
        return value

    assert structural(actual) == structural(GOLDEN["embeddings"])
