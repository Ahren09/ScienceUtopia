"""Optional standalone export of the unified figure's acceptance-price panel.

The manuscript now uses the combined three-panel figure rendered by
plot_resubmission_cascades. Reuse its panel function so both exports retain the
same values, uncertainty calculation, and visual encoding.

Run:
    python -m scripts.figures.plot_resubmission_difficulty
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from scripts.figures.visual_utils import nature_style
from scripts.figures.plot_resubmission_cascades import (
    DEFAULT_DECLINING, DEFAULT_FIXED, DEFAULT_OUT_DIR, acceptance_panel)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixed_stats', default=DEFAULT_FIXED)
    ap.add_argument('--declining_stats', default=DEFAULT_DECLINING)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    stats = {'Fixed': json.load(open(args.fixed_stats)),
             'Declining': json.load(open(args.declining_stats))}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    fig, ax = plt.subplots(figsize=(2.6, 2.1))
    acceptance_panel(ax, stats)
    sns.despine(fig)
    fig.tight_layout()

    for ext in {'pdf'}:
        path = out_dir / f'fig_resubmission_difficulty.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
