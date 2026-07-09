# Repository Engineering Rules

These rules apply to all code changes, experiments, training jobs, evaluation jobs, and generated artifacts in this repository.

## 1. Rule Priority

When making changes, follow this priority order:

1. Correctness and reproducibility.
2. Reuse of existing implementations.
3. LoRA-first model storage and loading.
4. Code readability and simplicity.
5. Computational efficiency.
6. Minimal changes to the repository structure.

Do not sacrifice correctness or reproducibility merely to reduce the number of changed lines.

---

## 2. Repository Layout

All source code must live under `src/`.

All generated artifacts must use the following repository-relative locations:

```text
src/                     # All library code, training code, evaluation code, and utilities
data/                    # Raw, processed, cached, and generated datasets
outputs/
├── logs/                # Training, evaluation, profiling, and job logs
├── checkpoints/         # LoRA adapters and resumable training states
├── visual/              # Figures, plots, qualitative examples, and videos
└── docs/                # Experiment reports, summaries, tables, and notes
```

Required placement:

* Source code: `src/`
* Logs: `outputs/logs/`
* LoRA checkpoints: `outputs/checkpoints/`
* Visualizations: `outputs/visual/`
* Data and caches: `data/`
* Generated reports and documentation: `outputs/docs/`

Do not save generated files in the repository root or scatter output directories across source-code folders.

Before adding a new output path, verify that one of the directories above cannot be used.

---

## 3. Modify Existing Code Before Adding New Code

Before implementing a feature:

1. Search the repository for related implementations, utilities, configuration fields, and command-line arguments.
2. Extend or generalize the existing implementation whenever practical.
3. Update all relevant callers instead of introducing a parallel implementation.
4. Create a new file only when the functionality is genuinely distinct and cannot be placed naturally in an existing module under `src/`.

Never create files such as:

```text
train_new.py
train_v2.py
train_fixed.py
utils_new.py
evaluation_updated.py
```

unless there is a clear architectural reason.

Do not copy and slightly modify an existing function. If two code paths share substantial logic, extract the shared logic into one reusable implementation.

Before creating a new function, class, CLI option, configuration field, or file, search for an existing equivalent.

---

## 4. Avoid Duplicate Logic

There must be a single source of truth for:

* Dataset loading and preprocessing
* Prompt construction
* Diffusion schedules
* Noise or masking policies
* Model loading
* LoRA configuration
* Generation and sampling
* Metric computation
* Checkpoint saving
* Logging
* Random seed handling
* Device and dtype selection

Do not independently reimplement these behaviors in multiple training or evaluation scripts.

When the same nontrivial logic appears more than once, consolidate it into a shared function or module and update all call sites.

Do not duplicate constants or default hyperparameters across files. Store shared defaults in an existing configuration object, argument parser, or centralized configuration module.

---

## 5. Code Style and Readability

Prefer direct, readable, and explicit code.

* Keep control flow easy to follow.
* Use descriptive variable and function names.
* Avoid unnecessary classes, wrappers, factories, and abstraction layers.
* Avoid deeply nested functions and excessive indirection.
* Do not split a simple operation across many tiny helper functions.
* Extract a helper when it removes meaningful duplication, isolates a coherent operation, or makes the main pipeline substantially easier to read.
* Add comments for non-obvious reasoning, tensor shapes, invariants, and algorithmic decisions—not for code that is already self-explanatory.
* Include tensor shapes in comments or assertions at important boundaries.
* Remove dead code, obsolete branches, stale arguments, and unused imports introduced or exposed by the change.

Prefer extending an existing function over creating a nearly identical alternative. However, do not turn one function into a large collection of unrelated conditional branches merely to avoid creating a new function.

---

## 6. Minimal and Focused Changes

Keep each implementation focused on the requested task.

Do not perform unrelated refactors, rename unrelated files, or reformat the entire repository while implementing a feature.

When changing an interface:

* Update all call sites.
* Update relevant tests and documentation.
* Remove the obsolete interface when backward compatibility is not required.
* Do not silently maintain two equivalent interfaces.

Do not leave temporary debugging scripts, copied code, scratch files, or one-off experiment files in the repository.

All reusable code should be placed under `src/`.

---

## 7. Model Storage and LoRA

All trainable model updates must use LoRA or another explicitly approved parameter-efficient adapter method.

LoRA is the default and strongly preferred mechanism for training, saving, loading, evaluation, and deployment workflows.

Always try to load LoRA adapters instead of full model checkpoints whenever possible.

Always try to save LoRA adapters instead of full model checkpoints whenever possible.

Never save:

* A duplicated copy of the base model
* A fully fine-tuned model checkpoint
* A merged LoRA-plus-base-model checkpoint
* Large model weights inside an experiment directory
* Full model checkpoints when a LoRA adapter is sufficient

Prefer workflows of the form:

```text
Base Model + LoRA Adapter
```

instead of:

```text
Fully Saved Fine-Tuned Model
```

Save only the artifacts needed to reproduce or resume the run, such as:

```text
outputs/checkpoints/<experiment_name>/
├── adapter_config.json
├── adapter_model.safetensors
├── trainer_state.json
├── optimizer.pt              # Only when required for resuming
├── scheduler.pt              # Only when required for resuming
├── training_args.json
└── tokenizer files           # Only when modified or required
```

The base model must be referenced by its model identifier or external path, not copied into this repository.

When loading a trained model, prefer loading:

1. The base model from its original source.
2. The LoRA adapter from `outputs/checkpoints/`.
3. The adapter onto the base model at runtime.

Do not create workflows that require storing or distributing merged model weights unless explicitly required.

Checkpoint directories must use stable, descriptive experiment names. Do not create ambiguous directories such as `test`, `new`, `final2`, or `tmp_model`.

---

## 8. Batched Training and Evaluation

All model-facing training, inference, generation, and evaluation operations must be batched.

Forbidden pattern:

```python
for example in dataset:
    output = model(example)
```

Required pattern:

```python
for batch in dataloader:
    outputs = model(**batch)
```

In particular:

* Do not perform one model forward pass per example.
* Do not call `generate`, the diffusion sampler, the teacher model, or the reward model separately for every sample.
* Tokenization should use batched tokenizer calls.
* Metric computation should be vectorized or accumulated by batch whenever practical.
* Teacher-label or self-distillation target generation must be batched.
* Dataset preprocessing should use batched dataset transforms when supported.

Small Python loops are acceptable for lightweight bookkeeping, formatting, variable-length postprocessing, or operations that cannot reasonably be vectorized. They must not cause repeated single-example GPU model calls.

The final partial batch must be handled correctly. Do not drop examples during evaluation unless explicitly requested.

---

## 9. GPU Utilization

Training and evaluation jobs should maximize GPU utilization without causing instability or unnecessary contention.

GPU memory is a critical resource and should not sit mostly idle. Any GPU assigned to a job should generally maintain at least **50% memory utilization**, with **80%+ utilization preferred whenever safely achievable**.

Before launching a job:

1. Inspect available GPUs and their current memory usage.
2. Estimate or measure the memory required by one job.
3. Select the largest safe batch size.
4. Use gradient accumulation only when the desired effective batch size cannot fit directly.
5. Enable appropriate efficiency features, such as BF16, gradient checkpointing, Flash Attention, or efficient attention kernels, when supported and numerically appropriate.

When several jobs use significantly less than one GPU's memory, they should generally be consolidated onto fewer GPUs rather than spread across many underutilized devices.

Only place multiple jobs on the same GPU or GPU set when:

* Their combined peak memory safely fits.
* There is sufficient memory headroom for temporary allocations.
* Concurrent execution does not cause repeated OOMs or severe throughput degradation.
* The GPU assignment is explicitly recorded in the launch command and logs.

Mandatory utilization rules:

* Any GPU actively assigned to training, evaluation, generation, or preprocessing should target at least **50% memory utilization**.
* Utilization of **80%+ memory usage** is preferred whenever it can be achieved safely.
* Do not reserve GPUs that remain mostly empty.
* Do not spread lightweight jobs across multiple GPUs while each GPU remains substantially underutilized.
* There must never be two concurrently running jobs where both jobs individually use **less than 45% GPU memory utilization**. Such jobs must be packed together, assigned fewer GPUs, or launched with larger batch sizes if feasible.
* If two jobs would each occupy less than 45% of a GPU, they should generally share hardware resources rather than consume separate underutilized GPUs.
* Before launching additional jobs, verify that existing GPU allocations cannot be consolidated to improve utilization.

Conversely, do not blindly launch concurrent jobs merely to fill memory. Stability, throughput, and reproducibility take priority over achieving a specific memory-utilization percentage.

For distributed jobs:

* Use all assigned GPUs effectively.
* Verify that each process receives data rather than redundantly processing the same examples.
* Avoid assigning GPUs that remain below the minimum utilization targets without a documented justification.

Any exception to the utilization requirements must be explicitly justified in the logs.

---

## 10. Job Launching and Logging

All job-launching scripts must be written in Python.

Prefer Python launchers, orchestration scripts, and experiment runners over Bash scripts.

Do not create Bash wrappers when the same functionality can be implemented in Python.

Training, evaluation, preprocessing, and generation workflows should be reproducible through Python entry points under `src/`.

Every training, evaluation, preprocessing, or large-scale generation job must:

* Write its logs to `outputs/logs/<experiment_name>/`.
* Record the complete command used to launch the job.
* Record resolved hyperparameters and configuration.
* Record the base model identifier and LoRA configuration.
* Record random seeds.
* Record dataset names, splits, and sample counts.
* Record GPU assignments.
* Record checkpoint and output paths.
* Preserve errors and stack traces.

Do not redirect different concurrent jobs into the same log file.

Use descriptive experiment names that encode the method and important settings, for example:

```text
dg_gsm8k_onpolicy_lora_r8_seed42
dg_gsm8k_mixedhard_steps200_seed42
```

Avoid names such as:

```text
run1
test
latest
new_exp
final
```

Training scripts should periodically log:

* Step and epoch
* Loss components
* Learning rate
* Gradient norm when available
* Throughput
* GPU memory usage
* Evaluation metrics at configured intervals

Do not print large tensors, complete datasets, or per-example outputs into the main log.

---

## 11. Experiment Outputs

Each experiment should use one consistent experiment identifier across:

```text
outputs/logs/<experiment_id>/
outputs/checkpoints/<experiment_id>/
outputs/visual/<experiment_id>/
outputs/docs/<experiment_id>/
```

Do not create multiple differently named directories for the same run.

Machine-readable results should be stored in structured formats such as JSON, JSONL, CSV, or Parquet under the appropriate output directory. Large generated datasets belong under `data/`, not under `outputs/docs/`.

Human-readable summaries, tables, and experiment reports belong under:

```text
outputs/docs/<experiment_id>/
```

Visual outputs belong under:

```text
outputs/visual/<experiment_id>/
```

---

## 12. Reproducibility

Every experiment must support deterministic seeding where practical.

Set and record seeds for:

* Python
* NumPy
* PyTorch
* CUDA
* Dataset shuffling
* Diffusion noise or masking
* Sampling and generation

Do not hide important experiment settings as hard-coded values inside implementation files. Expose settings through the existing configuration or CLI system.

A run must not depend on an untracked manual edit.

---

## 13. Validation Before Completion

Before reporting a task as complete:

1. Confirm that the changed code imports successfully.
2. Run the narrowest relevant test or smoke test.
3. Verify tensor shapes and dtypes at modified interfaces.
4. Verify that training and evaluation are batched.
5. Verify that no full model checkpoint is saved.
6. Verify that LoRA adapters are used whenever possible instead of full model checkpoints.
7. Verify that all artifacts use the required directories.
8. Search for duplicated implementations introduced by the change.
9. Check that no temporary files or redundant scripts were added.
10. Confirm that existing callers still work or were updated.
11. Report the exact commands executed and the locations of generated outputs.

Do not claim success solely because the code was written. Distinguish clearly among:

* Implemented but not executed
* Smoke-tested
* Fully executed
* Evaluated with final results

---

## 14. Change Summary

After completing a task, provide a concise summary containing:

* Files modified
* Files created, with justification for every new file
* Existing code reused or generalized
* Commands executed
* Tests or experiments completed
* Output locations
* Known limitations or unverified behavior

If no new file was necessary, explicitly state that the existing implementation was extended.

If a model-related change was made, explicitly state:

* Whether LoRA adapters were used
* Whether any full model checkpoints were created
* Where LoRA adapters were stored or loaded from
