import json

import numpy as np
import pandas as pd
import pytest

from utopia.analysis import statistics
from utopia.utils.paths import project_root


GOLDEN = json.loads(
    (project_root() / "tests/support/fixtures/ownership_golden.json").read_text()
)


def test_statistical_helpers_preserve_sampling_order_and_results():
    values = np.array([0.2, np.nan, -0.7, 1.1, 0.4, -0.2])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    reviewers = np.array(["r1", "r2", "r1", "r2", "r1", "r2"])
    frame = pd.DataFrame(
        {
            "y": [1.0, 1.8, np.nan, 3.9, 5.2, 4.6],
            "x": [0, 1, 2, 3, 4, 5],
            "w": [1, 3, 2, 4, 2, 1],
        }
    )
    results = {
        "paired": statistics.paired_boot(values, n_boot=257, seed=71),
        "paired_median": statistics.paired_boot(
            values, n_boot=257, seed=71, stat=np.median
        ),
        "weighted": statistics.boot_weighted_slope(
            frame, "y", "x", "w", n_boot=257, seed=91
        ),
        "cluster": statistics.cluster_boot_mean(values, groups, 257, 19),
        "crossed": statistics.twoway_boot_mean(values, groups, reviewers, 257, 23),
    }
    assert results == GOLDEN["statistics"]


def test_wilson_interval_preserves_scalar_clipping_and_vector_behavior():
    for row in GOLDEN["wilson"]:
        assert (
            list(statistics.wilson_ci(row["k"], row["n"], clip_lower=True))
            == row["clipped"]
        )
        assert list(statistics.wilson_ci(row["k"], row["n"])) == row["unclipped"]
    lower, upper = statistics.wilson_ci(np.array([0, 2, 10]), np.array([10, 10, 10]))
    np.testing.assert_array_equal(
        lower, [row["unclipped"][0] for row in GOLDEN["wilson"]]
    )
    np.testing.assert_array_equal(
        upper, [row["unclipped"][1] for row in GOLDEN["wilson"]]
    )
    with pytest.raises(ZeroDivisionError):
        statistics.wilson_ci(0, 0)
