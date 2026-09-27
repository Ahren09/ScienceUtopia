# ScienceUtopia

ScienceUtopia simulates researchers, institutions, funding agencies, peer review,
publication, and citation networks. Experiment drivers reuse the same simulator
and record the configuration, request audits, source fingerprints, and results.

## Code layout

All production Python lives in `utopia/`. Tests and their fixtures live in
`tests/`; there is no `src/` directory.

| Package | Responsibility |
| --- | --- |
| `utopia.simulation`, `utopia.constants` | Simulation phases and shared constants |
| `utopia.experiments` | Experiment configuration, launch plans, and orchestration |
| `utopia.analysis` | Data admission, estimates, and statistics |
| `utopia.funding` | Accounting, ranking validation, sequential selection, feedback |
| `utopia.runtime` | Queue control, server profiles, launchers, and provenance |
| `utopia.visual` | Figure construction |
| `utopia.utils` | Generic IO, hashing, seeding, console logging, and project-root discovery |
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
python -m utopia.runtime.experiment_queue --help
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
replay tests read the existing frozen preregistrations under `outputs/docs`.
Package-boundary tests reject domain imports from generic utilities and plotting
code outside `utopia.visual`. Ownership regression fixtures capture original
keyword request hashes, cache bytes, statistical estimates, and figure artists.
CPU fixtures exercise the actual simulation/accounting/request code with mocked
transport, and compare prompts, RNG states, ranking failures, and checkpoints.

## Historical records

The migration changes source locations and their fingerprints. Frozen plans
continue to bind their original worktree; a new checkout must never impersonate
that source identity or resume a historical plan by substituting new hashes.

Existing result, checkpoint, manifest, and audit files are not rewritten.
`utopia.runtime.historical` contains read-time aliases for old names and source
paths, plus the original strings used in seed derivation. Historical audit
validation requires an independently trusted source manifest and checks its
exact hashes and required source set. Aliases only resolve names.

See `CLAUDE.md` for the repository engineering rules.
