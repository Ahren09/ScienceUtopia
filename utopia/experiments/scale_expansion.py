"""Scale expansion experiment and reusable scientific operations."""
#!/usr/bin/env python3

from utopia.utils.data_utils import write_json as write_json_file

from utopia.runtime.commands import module_command

from utopia.utils.paths import project_root
import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO_ROOT = str(project_root(__file__))
MODEL = "Qwen/Qwen3-32B"
DEFAULT_VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8036/v1")
LAUNCHER_DIR = os.path.join(REPO_ROOT, 'outputs', 'logs', 'scale_expansion_launcher')

# cell name -> factor levels (S=k, R=budget, K=capacity on, E=standardized, slots=fixed slots)
CELLS = {
    'S0R0':      dict(k=1, budget='track', cap=False, std=False, slots=False),
    'S1R0':      dict(k=2, budget='track', cap=False, std=False, slots=False),
    'S0R1':      dict(k=1, budget='fixed', cap=False, std=False, slots=False),
    'S1R1':      dict(k=2, budget='fixed', cap=False, std=False, slots=False),
    'S1R1_K1':   dict(k=2, budget='fixed', cap=True,  std=False, slots=False),
    'S1R1_E1':   dict(k=2, budget='fixed', cap=False, std=True,  slots=False),
    'S1R1_K1E1': dict(k=2, budget='fixed', cap=True,  std=True,  slots=False),
    'S0R0_E1':   dict(k=1, budget='track', cap=False, std=True,  slots=False),
    'S1R0_slots': dict(k=2, budget='track', cap=False, std=False, slots=True),
    # counterfactual: cut the resubmission channel in the pressure world (mechanism sufficiency)
    'S1R1_noresub': dict(k=2, budget='fixed', cap=False, std=False, slots=False, noresub=True),
}
# Cost control keeps the actual legacy production debit (zero) and waives the
# per-paper resubmission fee. Resubmission decisions and review workload remain.
# The configured funding_cost_per_paper=15 is unused in the legacy code, so
# activating it here would add a new expense rather than remove a confound.
for _base in ('S0R0', 'S1R0', 'S0R1', 'S1R1'):
    CELLS[_base + '_costcontrol'] = {
        **CELLS[_base], 'production_cost_mode': 'per_paper',
        'resubmission_cost': 0, 'log_resource_ledger': True,
    }
STAGES = {
    'smoke': ['S0R0'],
    'a': ['S0R0', 'S1R0', 'S0R1', 'S1R1'],
    'b': ['S1R1', 'S1R1_K1', 'S1R1_E1', 'S1R1_K1E1', 'S0R0_E1'],
    'sens': ['S1R0_slots'],
    'counterfactual': ['S1R1_noresub'],
    'costcontrol': [cell + '_costcontrol' for cell in ('S0R0', 'S1R0', 'S0R1', 'S1R1')],
    'costpilot': ['S0R0_costcontrol', 'S1R0_costcontrol', 'S0R1_costcontrol', 'S1R1_costcontrol'],
}


def build_command(cell: str, seed: int, args, vllm_url: str = None) -> list:
    spec = CELLS[cell]
    vllm_url = vllm_url or args.vllm_url[0]
    cmd = module_command(sys.executable, 'utopia', '--experiment_name', 'scale_expansion', '--experiment_stage', 'scale', '--seed', str(seed), '--population_mode', args.population)
    if args.population == 'university_only':
        # synthetic balanced institutions, every researcher without an exploration prior
        cmd += ['--num_institutions', str(args.num_institutions),
                '--researchers_per_institution', '5',
                '--strategy_mix', 'balanced']
    # default population: the paper-baseline roster (10 rich + 20 normal universities,
    # 10 rich + 20 normal companies, 5 each = 300); all researchers are 'balanced' there
    cmd += [
        '--num_years', str(args.num_years),
        '--num_conferences', str(args.num_conferences),
        '--start_year', str(args.start_year),
        '--model', MODEL,
        '--vllm_url', vllm_url,
        '--batch_size', str(args.batch_size),
        '--industry_funding_mode', 'performance',
        '--funding_allocation_mode', 'fixed',
        '--funding_panel_max_apps', str(args.funding_panel_max_apps),  # 0 = global ranking
        '--log_funding_applications',         # full pre-decision rankings for P3
        '--papers_per_project', str(spec['k']),
        '--funding_budget_mode', spec['budget'],
        '--review_policy', 'standardized' if spec['std'] else 'persona',
    ]
    if spec['budget'] == 'fixed':
        if args.slots_frac is None:
            sys.exit(f"cell {cell} needs --slots_frac (preregistered f)")
        cmd += ['--funding_budget_slots_frac', str(args.slots_frac)]
    if spec['cap']:
        if args.reviewer_capacity is None:
            sys.exit(f"cell {cell} needs --reviewer_capacity (preregistered c)")
        cmd += ['--reviewer_capacity', str(args.reviewer_capacity)]
        if args.reviewer_matching != 'random':
            cmd += ['--reviewer_matching', args.reviewer_matching]
    if spec['slots']:
        cmd += ['--acceptance_mode', 'fixed_slots']
    if spec.get('noresub'):
        cmd.append('--disable_resubmission')
    if 'production_cost_mode' in spec:
        cmd += ['--production_cost_mode', spec['production_cost_mode'],
                '--resubmission_cost', str(spec['resubmission_cost'])]
    if spec.get('log_resource_ledger'):
        cmd.append('--log_resource_ledger')
    if not args.legacy_scoring:
        cmd += ['--review_score_mode', 'float', '--acceptance_tiebreak', 'seeded']
    if args.always_rerun:
        cmd.append('--always_rerun')
    return cmd


def experiment_id_for(cmd: list) -> str:
    """Resolve the experiment_id exactly as simulation will. parse_arguments creates the
    canonical output dirs as a side effect; remove them again if they are still empty so a
    dry run leaves no trace (the real run recreates them)."""
    from utopia.arguments import parse_arguments
    a = parse_arguments(cmd[2:])
    for d in (a.log_dir, a.checkpoint_dir, a.docs_dir):
        try:
            os.rmdir(d)  # only succeeds when empty
        except OSError:
            pass
    return a.experiment_id












def main(argv=None):
    from utopia.experiments.common import main_for
    return main_for("scale_expansion", argv)


if __name__ == "__main__":
    main()
