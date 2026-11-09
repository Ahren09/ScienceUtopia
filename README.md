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

## Population growth and portable continuation

The native `--annual-population-influx` defaults to zero. It adds a proportion
of **all researchers, including inactive researchers**, at the start of each year,
rounded half up. Funding agencies are excluded. An initial population of 5000
therefore becomes 6000, 7200, and 8640 in the first three years at 20%.
Entrants use the existing researcher constructors and join existing institutions
in deterministic round-robin order, with symmetric institutional conflicts.
Old checkpoints with `annual_influx_rate` retain their historical fixed-cohort
meaning; those results are not compound-growth controls.

Start the approved Qwen service separately:

```bash
VLLM_USE_DEEP_GEMM=0 vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.93 \
  --max-model-len 262144 --max-num-seqs 256 \
  --max-num-batched-tokens 32768 --enable-chunked-prefill \
  --reasoning-parser qwen3 --language-model-only \
  --seed 42 --port 8100
```

Set `CUDA_HOME` to the installed CUDA toolkit and include its `bin` directory in
`PATH` so the serving environment can compile the model's kernels.

Use `python -m utopia` for the simulation:

```bash
python -m utopia --experiment_name default_population_influx --experiment_stage scale \
  --model Qwen/Qwen3.5-397B-A17B-FP8 --vllm_url http://localhost:8100/v1 \
  --population_mode university_only --num_institutions 1000 \
  --researchers_per_institution 5 --initial_funding 100 --strategy_mix balanced \
  --num_years 10 --num_conferences 6 --reviews_per_paper 3 --seed 42 \
  --annual-population-influx 0.2 --funding_panel_max_apps 25 \
  --funding_allocation_mode fixed --funding_budget_mode track \
  --review_score_mode float --acceptance_tiebreak seeded \
  --tensor_parallel_size 8 --gpu_memory_utilization 0.93 --max_model_len 262144 \
  --batch_size 256 --completion_token_floor 16384 --thinking_token_budget 8192 \
  --fulltext_index /nvme/fulltext/index.json --review-main-text-pages 9 \
  --save_model_replies --log_resource_ledger \
  --checkpoint-s3-prefix s3://aaron0929/population_influx/EXPERIMENT_ID \
  --stop_at_utc 2026-11-15T08:30:00Z
```

Give each condition its own S3 experiment prefix. The 10% comparison uses the
same initial population and seed with `--annual-population-influx 0.1`.
The additional Luna 20% condition uses `--model openai.gpt-5.6-luna --backend bedrock`
with the same scientific arguments and page selection. Supply its Bedrock bearer
token through the environment and set an explicit `--api_budget_usd`; API rate and
spending limits can be changed on resume while accumulated usage remains counted.
Both conditions use the first nine physical PDF pages and stop at the appendix
heading if it occurs earlier. References within those pages remain in the input.
Original PDFs are cached; HTML-only sources use an A4 print layout recorded in the
text index. `--review-main-text-pages 0` retains the older complete-text behavior.
Unavailable text, oversized tokenized prompts plus output budgets, and incomplete
review sets stop the phase; no review falls back to an abstract. Page-limited
reviews have distinct semantic request IDs, preserving earlier complete-text
replies as historical records while keeping directions and submissions reusable.

Every phase saves its entry and completed state. Replies are appended and synced
to disk individually. An interrupted phase replays complete task IDs and requests
only missing replies; result application follows input order. S3 snapshots
include code, scientific configuration, state, reply segments, corpus caches,
and full text. Objects are read back before advancing `latest.json`; boundaries
force synchronization, and phase progress syncs every 128 replies or 60 seconds.
Upload failure stops dispatch. Replies lost before upload are recomputed by ID.

On another machine, install the saved code and its dependencies, provide the
same model, and restore directly from the confirmed S3 position:

```bash
python -m utopia --resume-from s3://aaron0929/population_influx/EXPERIMENT_ID \
  --output-dir /nvme/restored --vllm_url http://localhost:8100/v1 \
  --batch_size 512 --max_model_len 262144
```

`--resume-from` also accepts a local checkpoint or checkpoint directory.
Scientific settings are restored from the checkpoint; paths, service address,
and concurrency can change. Model weights and the Python environment must be
provisioned on the destination machine. Wall-clock timestamps may differ after
replaying a phase; scientific state and applied ledger events retain their IDs.

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
