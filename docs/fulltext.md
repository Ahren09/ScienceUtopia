# Frozen full text and portable experiments

Retrieval and paper selection use abstracts. Actual peer reviews receive the
complete extracted text from the corresponding frozen corpus entry. The text
comes from structured source extraction, not visual inspection of PDF pages.
Corpus eligibility, source hashes, year coverage, and exclusions are recorded
in `index.json`.

## Data location

Use a corpus already staged on the server when available. A repository's `data`
and `outputs` directories may link to a larger storage volume. Resolving these
links is sufficient to locate the data; cloning Git does not download it.

For a private archived corpus, transfer directly from object storage to the
server and validate it against its accompanying manifest. Keep response
databases, checkpoints, and full-text packs off laptops and out of Git.

To prepare a new corpus from an existing source archive:

```bash
python -m utopia.data.fulltext --help
```

This command requires the source archive, its metadata, an output directory,
and an immutable tokenizer revision. It saves the text pack, abstracts,
eligibility records, and source identity before any world is created.
`rapidgzip` is optional for parallel decompression; the default reader uses
Python's standard library.

## Funding feedback

The full-text entrypoint requires explicit model, corpus, server, run, and budget
identities. For example, after filling in the server manifest and absolute
deadline for a **new** run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=42 python \
  -m utopia.experiments.funding_feedback run-fulltext \
  --cell P1F1 --run-id example --founders 10000 --years 6 \
  --vllm-url http://localhost:8000/v1 \
  --server-manifest data/server.json \
  --fulltext-index data/fulltext2024/index.json \
  --deadline-epoch "$DEADLINE_EPOCH"
```

The server manifest contains `model`, `model_revision`, `endpoint`, and the
server's `context_tokens` capacity. The requested context budget must fit
within that capacity. Defaults are the pinned Qwen3.5-397B-A17B-FP8 revision,
seed 42, an 8192-token answer budget, a 131072-token request context, eight
concurrent requests, and zero SDK retries. `--thinking` / `--no-thinking`
explicitly bind a new world's generation mode; the default follows the
simulation configuration.

Use another P/F cell to change the funding treatment. The reviewer-matching
entrypoint fixes its own RANDOM/TOPIC design. The switching entrypoint provides
`prospective-initialize` and `prospective`, sharing a validated initial-choice
file between conditions. Their `--help` output lists the required arguments.

## Recovery and preservation

`--resume` requires an existing portable checkpoint with identical scientific
parameters, corpus identity, and source binding. It restores the last completed
year and replays already committed responses before requesting missing ones.
Replayed responses have `cache_replay=true`; only new non-cache responses are
new inference. Changing the machine or cache location does not authorize
changing the scientific inputs.

At an annual boundary, failure boundary, or finalization, the runner first
preserves the checkpoint, a consistent complete response database, raw audits,
native outputs, logs, and source on the server. Optional `--s3-prefix` uploads
that snapshot using the caller's AWS environment, such as `AWS_PROFILE` and
`AWS_DEFAULT_REGION`. A failed upload leaves the server snapshot available and
marks it pending. Later boundaries in that process continue saving locally.

`server_preserved` and `server_preserved_s3_pending` are not off-host backups.
`s3_uploaded` records a successful upload; `independent_readback_verified=false`
remains explicit until a separate readback check has been performed.
Finalization and storage status are reported separately.

Historical experiments retain their original execution source, binding,
deadlines, amendments, and preservation receipts. This public source does not
silently migrate those worlds or extend their budgets.
