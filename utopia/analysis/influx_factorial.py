"""Cross-cell factorial decomposition for the influx2x2 experiment.

Experiment: 2x2 factorial {researcher influx OFF/ON} x {resubmission OFF/ON},
single seed 42, full paper scale (302 agents, 10 years, 10 conferences, fixed
30% acceptance), model Qwen/Qwen3-32B served locally (bf16 TP=4, port 8036).
Cells:
    A = no influx, resubmission disabled
    B = influx (+30 juniors/year, staggered 10%%), resubmission disabled
    C = no influx, resubmission enabled (config twin of the published paper
        baseline outputs/default_Qwen3-32B_fixed30_paired)
    D = influx + resubmission (flagship world)

Decomposition identities, per endpoint:
    entry effect      = B - A   (submission growth from population influx alone)
    recycling effect  = C - A   (growth from manuscript recycling alone)
    interaction       = (D - B) - (C - A)
                      (does influx amplify the rejection-resubmission feedback?)
    total effect      = D - A;  additivity gap = total - entry - recycling
                        (identical to the interaction term; reported for
                        readability)

Endpoints (frozen in outputs/docs/influx2x2_qwen3_32b_seed42/design.md):
    primary   - year-10 submission events; cumulative submission events (y1-10);
                year-10 reviews per active reviewer
    secondary - year-10 recycling multiplier M_t; year-10 active researchers

Per-cell metrics come from utopia.analysis.resubmission_stats.compute_stats
(single source of truth for metric definitions). Also emits a replication
table comparing cell C against the published fixed30 baseline numbers — a
same-config replication under vLLM serving nondeterminism, not a bitwise check.

Run (after the four cells complete):
    python -m utopia.analysis.influx_factorial
"""

from utopia.utils.data_utils import write_json as write_json_file
import argparse
import json
from pathlib import Path

from utopia.analysis.resubmission_stats import compute_stats

DEFAULT_CELLS = {
    'A': 'outputs/influx2x2_A_noinflux_noresub_seed42',
    'B': 'outputs/influx2x2_B_influx_noresub_seed42',
    'C': 'outputs/influx2x2_C_noinflux_resub_seed42',
    'D': 'outputs/influx2x2_D_influx_resub_seed42',
}
DEFAULT_PAPER_BASELINE = 'outputs/default_Qwen3-32B_fixed30_paired'
DEFAULT_OUT_DIR = 'outputs/docs/influx2x2_qwen3_32b_seed42'

CELL_LABELS = {
    'A': 'stable population, no recycling',
    'B': 'growing population, no recycling',
    'C': 'stable population, recycling',
    'D': 'growing population, recycling',
}


def _year(stats: dict, field: str, year: int):
    """Per-year dicts round-trip through JSON with string keys; accept both."""
    d = stats[field]
    return d.get(year, d.get(str(year)))


def cell_endpoints(stats: dict, num_years: int) -> dict:
    subs = stats['submissions_per_year']
    cumulative = sum(subs.get(y, subs.get(str(y), 0)) for y in range(1, num_years + 1))
    return {
        'submissions_y1': _year(stats, 'submissions_per_year', 1),
        'submissions_yT': _year(stats, 'submissions_per_year', num_years),
        'submissions_cumulative': cumulative,
        'reviews_per_active_reviewer_yT': _year(
            stats, 'reviews_per_active_reviewer_by_year', num_years),
        'reviews_per_pool_member_yT': _year(
            stats, 'reviews_per_pool_member_by_year', num_years),
        'recycling_multiplier_yT': _year(stats, 'recycling_multiplier_by_year', num_years),
        'active_researchers_y1': _year(stats, 'active_researchers_by_year', 1),
        'active_researchers_yT': _year(stats, 'active_researchers_by_year', num_years),
        'population_yT': _year(stats, 'population_by_year', num_years),
        'cascade_depth_mean': stats['cascade_depth_mean'],
        'recycle_rate_pct': stats['recycle_rate_pct'],
    }


def decompose(endpoints_by_cell: dict) -> dict:
    """Entry/recycling/interaction decomposition on every shared numeric endpoint."""
    a, b, c, d = (endpoints_by_cell[k] for k in 'ABCD')
    out = {}
    for key in a:
        va, vb, vc, vd = a[key], b[key], c[key], d[key]
        if not all(isinstance(v, (int, float)) for v in (va, vb, vc, vd)):
            continue
        entry = vb - va
        recycling = vc - va
        total = vd - va
        interaction = (vd - vb) - (vc - va)
        out[key] = {
            'A': va, 'B': vb, 'C': vc, 'D': vd,
            'entry_effect_B_minus_A': entry,
            'recycling_effect_C_minus_A': recycling,
            'total_effect_D_minus_A': total,
            'interaction': interaction,
            'additivity_gap': total - entry - recycling,  # == interaction
        }
    return out


def replication_check(c_endpoints: dict, baseline_stats: dict, num_years: int) -> dict:
    """Cell C vs the published fixed30 baseline (same config, fresh serving)."""
    base = cell_endpoints(baseline_stats, num_years)
    rows = {}
    for key in ('submissions_y1', 'submissions_yT', 'cascade_depth_mean',
                'recycle_rate_pct', 'reviews_per_active_reviewer_yT',
                'active_researchers_yT'):
        b, c = base[key], c_endpoints[key]
        rows[key] = {
            'paper_baseline': b, 'cell_C': c, 'delta': c - b,
            'rel_delta_pct': (c - b) / b * 100 if b else float('nan'),
        }
    return rows


def render_markdown(summary: dict) -> str:
    ny = summary['num_years']
    lines = [
        '# Influx x resubmission 2x2 factorial - summary',
        '',
        f"Seed {summary['seed']}, {ny} years, fixed 30% acceptance. "
        'Cells: ' + '; '.join(f"{k} = {v}" for k, v in CELL_LABELS.items()) + '.',
        '',
        '## Endpoints by cell',
        '',
    ]
    keys = list(next(iter(summary['decomposition'].values())).keys())[:4]  # A B C D
    endpoint_names = list(summary['decomposition'].keys())
    header = '| endpoint | ' + ' | '.join(keys) + ' | entry (B-A) | recycling (C-A) | interaction |'
    lines += [header, '|' + '---|' * (len(keys) + 4)]
    for name in endpoint_names:
        row = summary['decomposition'][name]
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
        lines.append('| ' + name + ' | '
                     + ' | '.join(fmt(row[k]) for k in keys) + ' | '
                     + fmt(row['entry_effect_B_minus_A']) + ' | '
                     + fmt(row['recycling_effect_C_minus_A']) + ' | '
                     + fmt(row['interaction']) + ' |')
    lines += [
        '',
        '## Replication check: cell C vs published fixed30 baseline',
        '',
        'Same config, fresh run; differences reflect vLLM serving nondeterminism, '
        'not code changes.',
        '',
        '| metric | paper baseline | cell C | delta | rel % |',
        '|---|---|---|---|---|',
    ]
    for name, row in summary['replication_check_C_vs_paper'].items():
        lines.append(f"| {name} | {row['paper_baseline']:.2f} | {row['cell_C']:.2f} "
                     f"| {row['delta']:+.2f} | {row['rel_delta_pct']:+.1f}% |")
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    import sys
    from utopia.analysis.release import main as report_main
    return report_main([*(sys.argv[1:] if argv is None else argv), '--family-analysis'])


if __name__ == '__main__':
    main()
