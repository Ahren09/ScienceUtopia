# ScienceUtopia

ScienceUtopia simulates researchers, institutions, funding agencies, peer review,
publication, and citation networks. Experiment drivers reuse the same simulator
and record the configuration, request usage, checkpoints, and results.

## Code layout

All production Python lives in `utopia/`. Tests and their fixtures live in
`tests/`; there is no `src/` directory.

| Package | Responsibility |
| --- | --- |
| `utopia.simulation`, `utopia.constants` | Simulation phases and shared constants |
| `utopia.experiments` | Experiment configuration, launch plans, and orchestration |
| `utopia.analysis` | Data admission, estimates, and statistics |
| `utopia.funding` | Accounting, ranking validation, sequential selection, feedback |
| `utopia.runtime` | Server commands, application setup, and run provenance |
| `utopia.visual` | Figure construction |
| `utopia.utils` | File IO, seeding, console logging, and project-root discovery |
| `tests/support` | Shared simulation, funding, and transport fixtures |

Import shared operations from their owning package. Library code does not import
experiment runners. Tests do not import other `test_*` modules. Use functional
names for new modules, fields, environment variables, and output directories.

Generic utilities do not import domain packages. Statistical estimators live in
`utopia.analysis.statistics`, keyword extraction in `utopia.data.keyword_extractor`,
and funding rules in `utopia.funding`. The keyword module owns both the cached
`KeywordExtractor` batch API and the distinct `CorpusKeywordExtractor` protocol.

All plotting, themes, figure export, and W&B chart construction live in
`utopia.visual`. `visual.common` shares the existing theme and export operation.
`visual.embeddings` preserves the separate PaCMAP corpus and PCA paper views.
`visual.wandb` builds charts while the metrics logger owns logging and history.
Analysis entrypoints may invoke renderers, but do not construct charts themselves.

Use `utils.seeding` for RNG configuration, `utils.logging` for console logging,
`runtime.setup` for application initialization, and `runtime.provenance` for
run manifests. IO writers explicitly preserve streaming, final-newline, file
creation, and atomic-publication policies.

## Entry points

Run commands from the repository root in the existing project environment:

```bash
python -m utopia --help
python -m utopia.experiments.exploration --help
python -m utopia.experiments.scale_expansion --help
python -m utopia.experiments.influx_factorial --help
python -m utopia.experiments.funding_feedback --help
python -m utopia.experiments.switching_propensity --help
python -m utopia.experiments.resource_size --help
python -m utopia.experiments.funding_cutoff --help
python -m utopia.experiments.review_replay --help
python -m utopia.experiments.project_cost --help
python -m utopia.data.keyword_extractor --help
```

Local server launchers are `utopia.runtime.local_server` (named profiles) and
`utopia.runtime.fp8_server`. The Modal declarations are
`utopia.runtime.modal_vllm` and `utopia.runtime.modal_resource_size`; they share
image and command construction in `utopia.runtime.modal_server`. Inspect a local
profile without launching it with `python -m utopia.runtime.local_server PROFILE
--dry-run`.

The manual structured-output model probe is
`python -m utopia.runtime.structured_output_probe`. Its `--help` and module
imports do not load a model; executing the probe requires the local vLLM setup.

Generated data stays in `data/` and `outputs/`, with logs, checkpoints, figures,
and reports in `outputs/logs`, `outputs/checkpoints`, `outputs/visual`, and
`outputs/docs` respectively. Importing a utility or analysis module must not
instantiate a model or launch a process.

The historical machine-specific queue and checkpoint-capacity probe have been
retired. Start servers explicitly and use the existing experiment entrypoints.
The cost-control planner preserves its four pilot cells and original full-scale
design, but no longer allocates GPUs or manages servers. Print native commands,
run them in the shell, and validate their checkpoints:

```bash
python -m utopia.experiments.project_cost plan \
  --directory outputs/docs/project_cost_control --vllm-url http://localhost:8032/v1
python -m utopia.experiments.project_cost commands \
  --plan outputs/docs/project_cost_control/plan.json --stage costpilot
python -m utopia.experiments.project_cost validate \
  --plan outputs/docs/project_cost_control/plan.json --stage costpilot
python -m utopia.experiments.project_cost commands \
  --plan outputs/docs/project_cost_control/plan.json --stage costcontrol
python -m utopia.analysis.project_cost \
  --plan outputs/docs/project_cost_control/plan.json --out outputs/docs/project_cost_control/results
```

`commands --stage costcontrol` validates all pilot results before printing the
full-scale commands. Native simulation resumes saved worlds and missing requests.

## Validation

Collect both pytest functions and unittest cases, including all former `src/`
tests, through the single test root:

```bash
python -B -m pytest tests --collect-only -q
python -B -m unittest discover -s tests -t . -v
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
python -B -m pytest tests -q \
  -k 'not synthetic_near_mid_far_ordering and not batched_encode_matches_single and not empirical_thresholds_from_direction_distribution'
```

The three excluded embedding tests require a real embedding model. Other
optional integration tests retain their explicit environment gates. Review
replay unit tests use small synthetic policy fixtures and need no archived outputs.
Package-boundary tests reject domain imports from generic utilities and plotting
code outside `utopia.visual`. Ownership regression fixtures capture original
keyword request hashes, cache bytes, statistical estimates, and figure artists.
CPU fixtures exercise the actual simulation/accounting/request code with mocked
transport, and compare prompts, RNG states, ranking failures, and checkpoints.

## Historical records

Existing result, checkpoint, manifest, and audit files remain intact. Resume by
stable world, task, request, and candidate IDs, reusing complete records and
processing missing IDs. Source, configuration, request, and cache hashes are
historical metadata, never conditions for execution, acceptance, or forced reruns.
Validate scientific settings, IDs, duplicates, coverage, scores, and full rankings
directly. Preserve the original seed derivation and checkpoint RNG state.

`utopia.runtime.historical` resolves old source paths and experiment names.
Ordinary functions use descriptive names without a leading underscore. Python
special methods and externally required API names retain their prescribed names.

See `CLAUDE.md` for the repository engineering rules.
