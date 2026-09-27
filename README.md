<div align="center">

# ScienceUtopia

### *A configurable simulation of research, peer review, and scientific funding*

[**Project website**](https://ahren09.github.io/ScienceUtopia/)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57.3-FFD21E.svg)](https://huggingface.co/docs/transformers)
[![vLLM](https://img.shields.io/badge/vLLM-0.12.0-30A14E.svg)](https://github.com/vllm-project/vllm)

</div>

---

## Overview

ScienceUtopia models researchers choosing projects, submitting papers, reviewing one another's work, receiving citations, and competing for funding. Language models make structured decisions within an explicit yearly simulation. Researchers retrieve candidate papers from the public [SciEvo dataset](https://huggingface.co/datasets/Ahren09/SciEvo).

This release supports **new experiments**. It contains the simulator, scientific prompts, experiment definitions, numerical estimators, and regression tests. Historical campaign artifacts and their original results are not distributed. Simulation outputs are checkpoints, request audits, manifests, and numerical tables; experiment plotting and rendering code are excluded.

The [project website](https://ahren09.github.io/ScienceUtopia/) is a static page in `index.html`, with styles in `assets/css/style.css` and the copy-button script in `assets/js/main.js`. Open `index.html` directly in a browser, or view the GitHub Pages site. No backend, package installation, or frontend build is required. GitHub Pages publishes the root of `main`; `.nojekyll` keeps the files static.

## Highlights

- **Nine experiment families** with configurable models, seeds, endpoints, and output locations.
- **Scientific mechanisms:** researcher populations, project duration and output, retrieval, conflicts of interest, peer review, citations, funding, and attrition.
- **Auditable runs:** yearly checkpoints, structured request records, funding evidence, source hashes, and input identities.
- **Portable data:** public dataset/model revisions and caches tied to their actual inputs.
- **Numerical analysis:** within-institution contrasts, factorial decompositions, bootstrap intervals, randomization tests, and funding-cutoff estimators.

## Capabilities

| Family | Experiment module | Conditions / purpose |
| :--- | :--- | :--- |
| Exploration | `utopia.experiments.exploration` | Research strategies and funding/citation novelty incentives |
| Scale expansion | `utopia.experiments.scale_expansion` | Project output, budget, reviewer capacity, review policy, and resubmission |
| Influx | `utopia.experiments.influx_factorial` | Researcher entry × resubmission |
| Funding feedback | `utopia.experiments.funding_feedback` | Publication and resource feedback mechanisms |
| Switching propensity | `utopia.experiments.switching_propensity` | Switching probability × distance from research history |
| Resource size | `utopia.experiments.resource_size` | Resource randomization and institution size |
| Funding cutoff | `utopia.experiments.funding_cutoff` | Funding-rank cutoffs and application-cost sensitivity |
| Review replay | `utopia.experiments.review_replay` | Controlled reviewer policies and network information |
| Project cost | `utopia.experiments.project_cost` | Output × budget with direct output/resubmission fees set to zero |

## Installation

The local inference setup targets **Linux, Python 3.11, and an NVIDIA GPU** compatible with PyTorch's CUDA 12.8 wheels. The example uses an 80 GB GPU, reserves 45% of its memory for vLLM, and runs retrieval on the same GPU. Select a GPU with sufficient free memory. On another machine, adjust the reservation or use `--rag_device cpu`; CPU retrieval takes considerably longer.

Allow approximately **70 GB of free disk space** for the environment, installation cache, model weights, public data, and initial outputs. Put the checkout and caches on a large local volume.

```bash
# 1) Clone the public repository.
git clone https://github.com/Ahren09/ScienceUtopia.git
cd ScienceUtopia

# 2) Place caches and the environment on the chosen storage volume.
export SCIENCEUTOPIA_STORAGE="$PWD/.cache"
export HF_HOME="$SCIENCEUTOPIA_STORAGE/huggingface"
export PIP_CACHE_DIR="$SCIENCEUTOPIA_STORAGE/pip"
export TMPDIR="$SCIENCEUTOPIA_STORAGE/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# 3) Create an isolated Python 3.11 environment.
python3.11 -m venv "$SCIENCEUTOPIA_STORAGE/env"
source "$SCIENCEUTOPIA_STORAGE/env/bin/activate"

# 4) Install the simulator, local server, and tests.
python -m pip install -r requirements-vllm.txt -r requirements-test.txt
python -m pip check
```

If Python 3.11 is unavailable but Conda is installed, replace step 3 with:

```bash
# Standard Anaconda installations live here; use the installed Conda executable.
export PATH="$HOME/anaconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda create --override-channels -c conda-forge --prefix "$SCIENCEUTOPIA_STORAGE/env" python=3.11 -y
conda activate "$SCIENCEUTOPIA_STORAGE/env"
```

| File | Purpose |
| :--- | :--- |
| `requirements.txt` | Simulator, retrieval, numerical analysis, and endpoint client |
| `requirements-vllm.txt` | Core plus the local vLLM server |
| `requirements-optional.txt` | Optional providers and scalar W&B logging |
| `requirements-test.txt` | Core plus pytest |

`constraints-linux-py311.txt` pins the resolved transitive versions for these groups. An existing compatible model endpoint needs only the core requirements. Optional providers require their own credentials; the quick start uses local inference and makes no paid API calls. W&B is disabled by default. If installed and explicitly enabled with `--wandb`, it logs scalars.

### Verify the installation

```bash
python -c "import torch, vllm, transformers, sentence_transformers; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available()); print('vllm', vllm.__version__); print('transformers', transformers.__version__); print('sentence-transformers', sentence_transformers.__version__)"
python -m utopia --help
```

Expected versions are PyTorch `2.9.0` (possibly with a CUDA suffix), vLLM `0.12.0`, Transformers `4.57.3`, and Sentence Transformers `5.1.2`. `pip check` must report no broken requirements.

## Quick start

Run these commands from the repository root with the environment activated. This is a genuine **Qwen3-8B simulation with ten researchers, three years, and seed 42**.

```bash
# 1) Select an available GPU and unused local port.
export SCIENCEUTOPIA_GPU="${SCIENCEUTOPIA_GPU:-0}"
export SCIENCEUTOPIA_PORT="${SCIENCEUTOPIA_PORT:-8000}"
export CUDA_VISIBLE_DEVICES="$SCIENCEUTOPIA_GPU"
export OMP_NUM_THREADS=4
# Runtime IPC needs short paths for both vLLM and the dataset loader.
# Large downloads and caches remain in HF_HOME and PIP_CACHE_DIR.
export TMPDIR="$(mktemp -d /tmp/scienceutopia-runtime-XXXXXX)"
export VLLM_RPC_BASE_PATH="$TMPDIR"
mkdir -p outputs/logs

# 2) Start the model server. The first launch downloads Qwen3-8B.
vllm serve Qwen/Qwen3-8B \
    --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --host 127.0.0.1 --port "$SCIENCEUTOPIA_PORT" \
    --dtype bfloat16 --max-model-len 32768 --max-num-seqs 8 \
    --gpu-memory-utilization 0.45 --enforce-eager \
    --reasoning-parser qwen3 --seed 42 \
    > outputs/logs/model-server.log 2>&1 &
echo $! > outputs/logs/model-server.pid

# 3) Wait for the server (at most 15 minutes).
python - <<'PY'
import os, time, urllib.request
url = f"http://127.0.0.1:{os.environ['SCIENCEUTOPIA_PORT']}/health"
for _ in range(450):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                print("Model server ready")
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise SystemExit("Server did not become ready; read outputs/logs/model-server.log")
PY

# 4) Run the simulation.
python -B -m utopia \
    --model Qwen/Qwen3-8B \
    --vllm_url "http://127.0.0.1:$SCIENCEUTOPIA_PORT/v1" \
    --experiment_name exploration_vs_exploitation --experiment_stage smoke \
    --population_mode university_only \
    --num_institutions 2 --researchers_per_institution 5 \
    --num_years 3 --num_conferences 2 --seed 42 --batch_size 8 \
    --rag_device cuda:0 --log_funding_applications --log_resource_ledger \
    > outputs/logs/quickstart.log 2>&1

# 5) Validate all three years and produce a numerical report.
python -m utopia.analysis.release \
    --run explore_smoke_qwen3_8b_neutral_i2_n10_y3_seed42_costv1_per_paper_resub5_ledger1_acdad001284e \
    --require-activity --out-dir outputs/docs/quickstart-report
```

During installation, `TMPDIR` points to the chosen storage volume for package downloads. At runtime, the short `TMPDIR` above prevents Unix-socket path limits in both vLLM and the dataset loader; `VLLM_RPC_BASE_PATH` uses the same directory. `HF_HOME`, `PIP_CACHE_DIR`, generated data caches, and simulation outputs remain on the chosen storage volume. Use these runtime exports for experiment commands too.

The first simulation downloads SciEvo's `arxiv` configuration and builds the retrieval index. This example selects CS papers from 2016–2018 and downloads the small embedding models. Subsequent runs reuse valid caches. Runtime depends on download speed, GPU load, and model response lengths; allow tens of minutes for the full first run.

Inspect `outputs/logs/quickstart.log` for progress. Failed attempts preserve their evidence. Three years and one seed demonstrate execution; they are not a research sweep or an uncertainty estimate across worlds.

### Expected outputs

The experiment ID is `explore_smoke_qwen3_8b_neutral_i2_n10_y3_seed42_costv1_per_paper_resub5_ledger1_acdad001284e`.

- `outputs/checkpoints/<experiment_id>/`: complete yearly JSON checkpoints, population blueprint, funding applications, resource ledgers, and final simulation report.
- `outputs/docs/<experiment_id>/`: run manifest, JSONL request audit and its closed summary, and numerical exploration tables.
- `outputs/docs/quickstart-report/`: `report.json`, `yearly.csv`, and `report.md`.

The validation command requires completed years, valid checkpoints, a closed request audit, and nonzero paper/review activity. It rejects visualization files.

The direct simulator resumes from its latest complete yearly checkpoint. Repeating a completed command validates it and exits without starting inference. Experiment drivers require fresh output directories to keep conditions and attempts separate. Select a fresh output location for a new run.

When finished, stop the server started above:

```bash
kill "$(cat outputs/logs/model-server.pid)"
```

## Data and models

| Input | Public identity |
| :--- | :--- |
| Candidate papers | `Ahren09/SciEvo`, `arxiv` configuration, `train` split |
| Quick-start model | `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218` |
| Retrieval encoder | `thenlper/gte-small` at `17e1f347d17fe144873b1201da91788898c639cd` |
| Distance encoder | `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

The SciEvo revision is pinned in `RAGConfig` and exposed as `--dataset-revision`. Use `--start_year` and `--num_years` to select the corpus window. Document caches bind the dataset revision, year selection, filters, and preprocessing. Embedding caches additionally bind document order/content and the embedding model revision.

Set `HF_HOME` for downloads and `--data-cache-dir` (direct simulator) or `--cache-dir` (experiment drivers) for generated caches. These locations are independent of outputs. Git contains no dataset, model weights, private caches, or historical results.

Experiment configs default to Qwen3-32B. Override with `--model Qwen/Qwen3-8B` to use the quick-start server. For another compatible model, provide an immutable `--model-revision` in drivers (`--model_revision` in the direct simulator), and serve that same revision with a 32,768-token context. Changing the model creates a new experiment; results are not expected to match another model's outputs. The request audit uses the selected model's tokenizer.

## Experiments

Study definitions live in `configs/`. All eight simulation drivers accept `--config`, `--cell`, `--all-cells`, `--model`, `--model-revision`, `--seed`, `--num-years`, `--num-institutions`, `--num-conferences`, `--batch-size`, `--vllm-url`, `--rag-device`, `--cache-dir`, `--output-root`, and `--dry-run`. Config files expose additional study settings.

```bash
# Setup-only previews: no inference or output directories.
python -m utopia.experiments.exploration --cell neutral --dry-run
python -m utopia.experiments.scale_expansion --cell S1R1 --dry-run
python -m utopia.experiments.influx_factorial --cell D --dry-run
python -m utopia.experiments.funding_feedback --cell P0F0 --dry-run
python -m utopia.experiments.switching_propensity --initialize --dry-run
python -m utopia.experiments.resource_size --dry-run
python -m utopia.experiments.funding_cutoff --dry-run
python -m utopia.experiments.project_cost --cell S1R1_costcontrol --dry-run

# Launch a fresh exploration world against the running local server.
python -m utopia.experiments.exploration \
    --cell neutral --model Qwen/Qwen3-8B --seed 43 \
    --vllm-url "http://127.0.0.1:$SCIENCEUTOPIA_PORT/v1" \
    --rag-device cuda:0 --output-root outputs/exploration-seed43
```

Increase population, years, and independent seeds for larger studies. Keep the same model, source, configuration, data revision, and initial state across paired conditions. Funding feedback and influx use audited, panelized sequential selection; this differs scientifically from a single global funding ranking.

Switching first runs `--initialize` to record treatment-independent first choices. Pass the resulting `initial_choices.json` via `--initial-choices <path>` to every LN/LF/HN/HF condition with the same config and seed. Resource-size populations require a multiple of three institutions. Influx requires a multiple of six institutions and five researchers per institution.

Review replay takes completed simulation outputs as inputs:

```bash
# Replace SOURCE_RUN_ID with completed experiment IDs under outputs/.
python -m utopia.experiments.review_replay --run-name study1 \
    extract --source-run SOURCE_RUN_ID --source_outputs_root outputs
python -m utopia.experiments.review_replay --run-name study1 --model Qwen/Qwen3-8B \
    generate --stage smoke --vllm_url "http://127.0.0.1:$SCIENCEUTOPIA_PORT/v1"
python -m utopia.experiments.review_replay --run-name study1 analyze --stage smoke
```

Replay needs enough eligible papers per source to satisfy its quotas. Defaults target larger source worlds, not the ten-researcher quick start. `extract --help` exposes quotas and network-study inputs; use `--study network_proximity` for the alternative study.

## Numerical analysis

```bash
# Validate and summarize any completed new simulation.
python -m utopia.analysis.release --run RUN_ID \
    --outputs-root outputs --out-dir outputs/docs/my-report

# Supply every paired condition for the selected family and seed.
python -m utopia.analysis.release --run RUN_A RUN_B RUN_C RUN_D \
    --family-analysis --out-dir outputs/docs/paired-report

# Dedicated exploration, resource-size, and funding-cutoff estimators.
python -m utopia.analysis.exploration_cross_seed \
    --multi_run_prefix exploration_Qwen3-8B_neutral_ \
    --outputs_dir outputs --out_dir outputs/docs/exploration-analysis
python -m utopia.analysis.resource_size \
    --docs-dir outputs/docs/RESOURCE_RUN_ID \
    --checkpoints-dir outputs/checkpoints/RESOURCE_RUN_ID
python -m utopia.analysis.funding_cutoff \
    --run CUTOFF_RUN_ID --outputs-root outputs \
    --out-dir outputs/docs/funding_cutoff_analysis
```

The family report supports scale expansion, project cost, influx, funding feedback, and switching comparisons. Resource-size analysis writes numerical results alongside the run's tables. Funding-cutoff analysis needs at least six complete years for its follow-up windows and reports exclusions and sample-size gates. Undefined estimates remain explicitly undefined. Independent simulation seeds are the replication units; researchers and years within one world are not independent worlds.

## Testing

The offline suite uses synthetic data, scripted model responses, and small regression fixtures:

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    UTOPIA_ACTUAL_MODULE_TESTS=1 \
    python -B -m pytest tests -q -m "not integration"
```

It covers native simulation phases, funding conservation, checkpoint resume, scientific prompts and sampling, structured responses, experiment configuration, cache invalidation, and numerical estimators. It makes no model API calls.

Additional CPU integration tests use the public, pinned MiniLM encoder. After the quick start has downloaded it:

```bash
CUDA_VISIBLE_DEVICES="" python -B -m pytest tests -q -m integration
```

The quick-start simulation and `--require-activity` report provide the separate real-model end-to-end check.

## Acknowledgements

ScienceUtopia builds on [SciEvo](https://huggingface.co/datasets/Ahren09/SciEvo), [Qwen](https://github.com/QwenLM/Qwen3), [vLLM](https://github.com/vllm-project/vllm), [Transformers](https://github.com/huggingface/transformers), [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), [PyTorch](https://pytorch.org/), and the scientific Python ecosystem. Model and dataset licenses apply to those separately downloaded inputs.

## License

Released under the [Apache License 2.0](LICENSE).
