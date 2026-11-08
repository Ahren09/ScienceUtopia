# ScienceUtopia engineering rules

ScienceUtopia models researchers, publication, peer review, and funding. Keep the
implementation close to the scientific method, with direct functions and a small
number of shared operations. Extend existing modules and entrypoints.

## Ownership

- `utopia.agents`: agent decisions, personas, and scientific state.
- `utopia.models`: model transport, structured responses, quota pacing, and reply reuse.
- `utopia.data`: corpora, retrieval, full text, and data preparation.
- `utopia.funding`: ranking, allocation, and resource accounting.
- `utopia.simulation`: native annual phases, checkpoints, and continuation.
- `utopia.experiments`: existing experimental protocols and conditions.
- `utopia.analysis`: numerical analysis of saved results.
- `utopia.visual`: figures and shared plotting operations.
- `utopia.runtime`: existing execution and environment support.
- `utopia.utils`: domain-independent operations only.

Keep chart construction inside `utopia.visual`. Importing a module must not load
a model, change the working directory, start a process, or require run artifacts.
Use descriptive ordinary function names without a leading underscore. Preserve
Python special methods and names prescribed by external APIs.

## Experiments and continuation

Use the authorized model, conditions, seed, horizon, population, review protocol,
and funding rules. Do not add seeds, change scientific settings, replace full
papers with abstracts, or expand compute without authorization.

Resume by stable world, task, request, and candidate IDs. Reuse complete saved
records and process only missing IDs. Source, configuration, request, model, and
cache hashes may remain descriptive historical metadata. They must never block
execution, require fresh outputs, or force reruns. Check actual IDs, fields,
duplicates, coverage, finite scores, complete rankings, and scientific conditions.
Preserve deterministic seed derivation and Python, NumPy, and Torch RNG state.

Keep all paid replies, including incomplete attempts. Drain issued model requests
before terminating a client. Track actual token usage separately from scientific
completion. Budget and deadline options apply only when explicitly authorized.

Inspect current process identity and output paths before changing a running job.
Never launch a second writer or interrupt unrelated work. Preserve existing
checkpoints and peer changes. Distinguish new inference from loading, CPU
preprocessing, cache replay, and waiting. After a failure, repair the concrete
cause and verify fresh missing outputs before reporting recovery.

Use the existing status records and entrypoints. Do not add a launcher,
controller, registry, lock, or retry framework for a one-off experiment. Simple
command sequences belong in the shell. Credentials belong in child environments,
not source, logs, command arguments, or committed artifacts.

## Validation and publication

Use `tests/` for both pytest and unittest cases. Prefer existing relevant tests.
Exercise real implementation paths with controlled fixtures and mocked transport.
Tests should run from a clean checkout without paid inference or private outputs.
Optional model integration tests must remain explicitly enabled.

Check full inputs, response reuse, RNG continuation, ranking completeness, and
scientific denominators when those paths change. Run the relevant tests and
`git diff --check`. Keep reports concise and distinguish completed results from
partial evidence. Leave unmeasured result cells blank.

Keep generated data and outputs out of source commits. Preserve published Git
history and make coherent commits. Follow the user's requested publication path,
including direct pushes when authorized. Never commit credentials.
