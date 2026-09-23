> **Staging scope, 19 September 2026:** Sections 1–9 and their preceding status text are preserved from the reviewed v0.3 document as historical context. The dated Section 10 governs the new `james-propensity-v3-single-world-sequential` replacement. Earlier compact-funding, initialization, and review-status statements do not describe the replacement's current execution status. This staging amendment does not modify either frozen predecessor worktree.

# James response: switching propensity × conditional distance

**Status:** reviewed prospective protocol v0.3, 19 September 2026; not externally registered. Herschel and the main agent have reviewed the document amendment. Implementation remains under CPU validation; no outcomes from this family have been observed. This is an authorized third experiment family, not optional future work. Protocol and implementation freeze remain prospective and pending completion of the required validation; document review does not certify runtime completion or scientific execution.

**Preserved predecessors:** the v0.3 scratch draft at `/tmp/scienceutopia-llm-review/propensity_distance_protocol_v0.3.md`, SHA256 `bc7b660ca9a20271fa2ee5ab527b23fcdbf188fd877d26fc17d571eff869a826`; v0.2 at `/tmp/scienceutopia-llm-review/propensity_distance_protocol_v0.2.md`, SHA256 `d18942e838ff3699916663dcff148898ee5bac3072bf9fc5cbdf38edeb95a935`; and v0.1 at `/tmp/scienceutopia-llm-review/propensity_distance_protocol_draft.md`, SHA256 `dcee284e34e4becb887e4e2276e781daad76bd8c42b9a025078b786aa222d006`, remain unchanged. This amendment documents the shared v4 compact funding baseline and the exact topic-history embedding implementation; it preserves the v0.2 scientific gate, menus, project cycle, validity rules, and estimands.

**Scope:** four matched conditions of one initialized population, one seed (42), 1,200 university-researcher founders per condition, ten years. The first eight runs belong to different experiment families; their source and outputs must remain frozen. Implement this family in a separate worktree with its own source commit, protocol version, and fresh output roots. EC2 only, within the shared maximum of eight GPUs; no additional GPU entitlement and no Cheesecake inference.

## 1. Question and limits of identification

James asks: “Maybe parameterize and report exploration propensity separately from distance, and show that the strategies differ on both before attributing impact and diversity differences to cautious exploration” (`paper/Comments_James.md:8`).

Question: holding the project cycle, initial population, funding rules, and all other institutional rules fixed, does increasing the probability of changing topic have a more positive effect on sustained topic diversity when new topics are selected from a near-history menu than from a far-history menu? A relative difference need not be a benefit in either menu condition.

The intervention controls a switch gate separately from a candidate-menu rule. “Distance” means **direction-to-historical-paper-centroid cosine distance**, using the existing embedding implementation. It does not mean distance between two consecutive papers, nor necessarily distance from the immediately preceding direction. Use “near-history” and “far-history” in analysis labels; do not rename these conditions “cautious explorers” or claim they reproduce the original strategy bundles.

These are effects of two specified dynamic policies in one stochastic ecosystem realization. Menu membership changes topic identities as well as their distance. The experiment does not identify a universal effect of a scalar cosine distance, a human behavioral effect, or the original strategy effect after removing every mediator. Policies are independently set; their realized distance distributions need not remain statistically independent of propensity after histories diverge.

## 2. Four conditions and common baseline

| Condition | Switch probability at an eligible repeat choice | Menu if switching |
|---|---:|---|
| LN | 0.25 | 17 nearest eligible directions |
| LF | 0.25 | 17 farthest eligible directions |
| HN | 0.75 | 17 nearest eligible directions |
| HF | 0.75 | 17 farthest eligible directions |

All four conditions use the following explicit, resolved settings:

- Seed and `PYTHONHASHSEED`: 42. Exactly 240 institutions × five university researchers = 1,200 researcher founders, excluding funding agencies from that count. No entrants or replacements.
- The existing deterministic university-only population blueprint; 100 initial resource units; native equal initial reputation; identical researcher IDs, institutions, expertise, conflicts, program objects, and initial memory. Every researcher has the existing `balanced` persona, with no explorer/exploiter/cautious label or strategy-dependent score, citation, or funding bonus.
- **`funding_panel_max_apps=25`**: native deterministic **balanced panels of at most 25 applications per program**. For n>0 applications let k=ceil(n/25); after the native seeded shuffle, panel j is `shuffled[j::k]`, j=0,...,k−1. Sizes differ by at most one: 26 applications produce 13/13; 53 produce 18/18/17. For n≤25 retain the native single panel; no applications produce no ranking prompt. Do not implement chunks of 25 plus a residual: native per-panel quotas make that a scientifically different funding rule. Explicitly set the override and validate actual memberships and sizes.
- **`funding_budget_mode=track`**: award slots follow applications under the original panel/program allocation rule; no fixed system slot cap or initial-population budget. Retain deterministic native panelization and tie handling.
- **`funding_allocation_mode=fixed`**: 20 resource units per awarded grant. This controls award size and is distinct from the `track` slot-budget setting. All award credits are paid; structured publication records and proposal text are shown normally. Preserve native program rates and program eligibility, recording their resolved values rather than replacing them with the generic configuration defaults. The currently inspected agency constructor specifies NSF 0.23 and DARPA 0.10 as agency defaults.
- Install the reviewed shared `src/james_funding_validation.py` gate in every cell using `install_funding_validation(funding_class, audit_path, compact=True)`, adopting the common v4 compact funding baseline. Before native award processing, require complete application-level identifier coverage and a valid full ordering; duplicate program/panel identities must be rejected, including copied metadata. No imputed tails or final fallback rankings may generate scientific awards. Persist the audit and `handle.summary()`; require `summary()["completion_allowed"] == True` at each cell's completion boundary and when admitting all four cells to analysis. Panel-size checks alone are insufficient. Valid zero-application years are retained, not imputed or mistaken for missing audit records.
- **Compact funding output:** the representation label is `ordered_application_ids_v1`; the model returns a JSON object with the single field `ranked_application_ids`. For a panel of n applications, its value must be a complete unique permutation of integer indices 0,...,n−1, ordered from most to least preferred. Validate the entire permutation before deriving rank 1,...,n and each exact `applicant_id` from the frozen panel metadata. Expand into fresh native `ranked_applications` records with `reason=""`; funding explanations are deliberately omitted and unused by the simulation. Do not fill missing IDs, remove duplicates, coerce invalid values, or change the model-selected order. Retain original responses and message histories. The substantive application text and evaluation criteria remain unchanged; the response-format instructions and schema change. This elicitation change may alter sampled rankings, so no equivalence to legacy elicitation is claimed. Native award-processing parity means only the same downstream result for the same validated order and metadata. Funding evaluation requests retain `max_tokens=8192`.
- Six conferences, native 30% acceptance regime, integer score handling and stable acceptance ties, original reviewer assignment and review count, and no hard reviewer-capacity intervention.
- One new-paper opportunity per completed project (`papers_per_project=1`); original resubmission policy enabled, original resubmission cost 5; original citation and publication mechanisms; no new collaboration intervention.
- Common one-year project duration for every direction and condition, explained below. Annual active-project cost remains 10. Do not activate the configured but currently uncharged 15-unit new-paper cost.
- The same ordered corpus, RAG retrieval cache, temporal restrictions, and validated production Qwen3-32B model/revision/runtime as the other new families. Apart from the explicit compact funding response-format amendment above, preserve reasoning, token limits, prompt defaults, response schemas, retry behavior, and the reviewed constrained-output transport. No silent thinking-mode or precision change.
- **Topic-history embeddings:** use `sentence-transformers/all-MiniLM-L6-v2` pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, resolving that exact local snapshot before construction and loading it with `device="cpu", local_files_only=True`. This MiniLM model embeds initial expertise, unique first-submission abstracts, and candidate directions for the historical centroid/menu rule. It is distinct from the `thenlper/gte-small` RAG retrieval embeddings and their separately verified corpus cache; do not substitute GTE vectors for the menu/history vectors. Freeze the MiniLM snapshot file hashes; the current driver also requires `model.safetensors` SHA256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`. Missing or changed pinned artifacts fail validation; no download, unpinned model, alternate encoder, or GPU fallback is permitted during execution.

The final freeze must store resolved configuration, program rates, all source/runtime/model/corpus/cache hashes, the population blueprint and initial-state hashes, exact prompt templates, output namespaces, and the analysis specification. A settings mismatch is an integration failure, not another experimental factor.

## 3. Initial choice, project cycle, and risk set

### Initial choice

The first direction choice is common initialization, not a treatment event. It has no preceding direction, so exclude it from both switching propensity and conditional-switch-distance summaries.

Generate one common cache of validated initial direction responses for all founders, using the native balanced direction prompt, the founder's original expertise candidates, strict candidate validation, and seed 42. Apply the common one-year wording before generation. Cache the whole validated response (`topic`, `detailed_focus`, `reason`), fallback flag, candidate list, request seed, and input hash; then replay it identically into all four cells through the normal direction-memory and project-state update. Preserve memory content and canonicalize or exclude wall-clock timestamps only for hash comparison.

This cache is treatment-blind initialization, prepared after protocol freeze and before any condition's outcomes. It is not a fifth scientific world. Do not select among several caches. All four cells must have the same initial choice assignments and descriptions. Do not reuse condition-specific year-one outcomes or claim later trajectories are bitwise coupled. Empty founder expertise is an initialization failure; do not substitute a favorable initial topic.

### Common project cycle

Use copies of the existing 53 direction objects with `years=1`, preserving topic, keywords, focus areas, related topics, and embeddings. Every phase-1 candidate and returned direction must resolve to this common duration, including the lookup through `DIRECTIONS_DICT`. Do not change the frozen source used by the first eight runs.

Keep the native timeline formula: a project selected in year \(t\) has `project_start_year=t`, `project_end_year=(t-1)+1=t`. It can submit in phase 2 of that year; the next project choice becomes due in year \(t+1\). Make the prompt's general duration statement, candidate duration lines, and duration-related advice consistent with one-year projects. The prompt must never advertise 1–3-year or 3–4-year alternatives that are not implemented.

At each repeat opportunity, **both switching and staying start a new one-year project**, incur the same annual charge, and permit the same maximum new output. Staying is a singleton menu containing the previous topic, with a fresh native LLM `detailed_focus` and `reason`; it does not skip the direction call, reuse a completed project, prolong its duration, avoid its annual cost, or suppress its output opportunity.

### Eligibility and resources

Use the native phase-1 risk set, after phase-0 resubmission: `get_available_authors()` filtered by `is_active` and `project_end_year < t`. Repeat opportunities require a preceding valid direction. Do not draw a new behavior for inactive researchers or force a departed researcher back into the risk set.

Preserve the native order: resubmission and its charges; direction choice; new submission/citation; annual project charge; review; acceptance/memory; funding/attrition. Negative resource updates, clamping, participation thresholds, and the known exact-zero year-end boundary behavior remain unchanged and are logged. There is no compensating endowment, forced survival, or cost reset.

One-year cycles make nominal project opportunities equal conditional on remaining eligible. Realized opportunities can still differ because resource outcomes and attrition are downstream of treatment, including deactivation during phase 0. Log year-start activity, phase-1 eligibility, and exclusions separately. Do not claim that equal duration holds survival or exposure fixed.

## 4. Exact gate, candidate construction, and validation

### Seeded gate

For every founder and calendar year 2–10, define one common uniform number:

`U(i,t) = Random(derive_seed(42, "james_propensity_distance_v1_switch", founder_id, t)).random()`.

Use the existing stable `derive_seed`; specify and freeze the Python/RNG version. Cell, menu setting, processing order, worker, batch position, and retry count must not enter this gate. The common table may be materialized before simulation. At an eligible repeat opportunity, request a switch iff `U(i,t) < p`; otherwise request stay. Calendar-year keys preserve alignment when risk sets diverge.

This isolates the gate from mutable RNG consumption. Preserve the existing production LLM request-seeding policy elsewhere; do not claim that this small gate table creates per-agent common random numbers for all later model calls.

### Historical reference and candidate universe

**Required neutral tracking hooks.** The stock `_initialize_experiment_trackers`, `_record_paper_metadata`, and `_record_paper_metadata_batch` return early outside `exploration_vs_exploitation`. Calling the centroid helper alone does not populate history. In the separate third-family wrapper, initialize the existing `EmbeddingTracker` with the frozen embedding model/cache and register all founders' expertise before initial direction choices. Add a neutral phase-2 observer that ingests each unique new submission's actual abstract, stable manuscript ID, lead-author ID, and **first simulated submission year**, irrespective of eventual acceptance. Ingest after new submissions are materialized and before the next year's direction decisions; the centroid lookup still restricts history to t−1 and earlier. Resubmissions must not duplicate an embedding, overwrite its first year, or reassign its author. Conflicting IDs/provenance, missing abstracts, or invalid embeddings are coverage/integrity failures.

Reuse the existing expertise- and abstract-embedding routines. These hooks are observational bookkeeping used by the assigned menu rule; do not turn on the full exploration-experiment flag, strategy bonuses, keyword-generation LLM calls, or unrelated scoring/metadata interventions merely to initialize history. Freeze the minimal hooks and their order across all four cells.

At each year end reconcile the unique first-submission registry with the embedding metadata: every required manuscript has one matching author/year/abstract-hash record. At each centroid query log the eligible historical manuscript IDs and expected count. Expertise fallback is legitimate only when that historical window is empty under complete ingestion. Missing ingestion must fail coverage rather than masquerade as an empty history or a cold start. Check paper and direction embeddings for finite coordinates and strictly positive norms, not only finite cosine outputs; the stock epsilon normalization can produce finite distances for a zero vector.

At each eligible repeat state, call the existing:

`compute_career_centroid(agent.id, max_years=3, current_year=t-1)`.

Its inclusive implementation uses paper metadata years **t−4 through t−1**, with the registered initial-expertise centroid if no historical paper embeddings exist. Preserve that behavior and log whether paper history or expertise supplied the reference. Require a finite, nonzero reference vector. Do not use future-year papers or redefine the centroid as an average of direction labels.

Start with all 53 unique canonical direction topics. Remove the preceding topic before ranking, leaving 52 eligible directions. Use the existing `compute_direction_distances` for every eligible direction; require complete finite values. No default distance of 0.5 and no truncation at distance 1. The mathematical range is [0,2], allowing only machine-precision boundary tolerance in validation.

Sort by `(computed_distance, canonical_topic_name)` ascending. Define:

- Near-history menu: first 17 entries.
- Far-history menu: last 17 entries.
- The middle 18 entries are unused at that state.

This deliberately equalizes menu sizes. The original 53-direction helper splits at 17 and 34, producing sizes 17/17/19; calling that helper unchanged would not implement this protocol. Using the nearest and farthest 17 **after removing the preceding topic** is the exact new selection rule.

Compute and log both menus at every eligible repeat state, even when the gate says stay, so validity is not selected by p. Require disjoint topic sets and log the full-precision separation gap `g = min(distance in far) - max(distance in near)`. Rank ties use canonical topic names.

Retain the requested strict statewise condition **g>0**, with an explicit domain restriction: the comparison is evaluable only if every encountered eligible repeat state supports strictly separated menus. A legitimate geometric tie (g=0) is an **out-of-domain/absent-separation state**, not evidence of a coding defect or scientific falsification. Stop that cell under the predeclared guard, retain its diagnostics, and withhold the complete factorial contrast; do not silently relabel ties or change the rule after outcomes. A negative gap with correctly sorted finite distances would instead indicate an implementation/integrity failure. Fifty-three unique topic labels establish cardinality only; they cannot guarantee a positive gap for every future centroid, and static preflight must not claim that guarantee.

Present each selected menu in canonical topic-name order, the same rule in both factors. This prevents a “first candidate” parser fallback from always meaning closest in one condition and least distant of the far candidates in another.

### Same topic, empty menus, and failures

- Gate = stay: pass `[previous_direction]` explicitly to the native batch selector. A returned different topic is invalid.
- Gate = switch: pass the selected 17-topic menu explicitly. The previous topic is absent; returning it is invalid.
- Require the explicit `candidate_map` strict-validation path. Do not rely on `balanced` strategy filtering, which currently returns expertise and would ignore the distance intervention.
- Use one common balanced prompt template. Preserve its substantive research context and expertise information, but state clearly that only the displayed candidates are valid. Do not mention p, the cell, a desired diversity result, or a strategy-specific justification. Apply the same duration corrections everywhere.
- Preserve the native parse/semantic-validation fallback to the first displayed valid candidate and log it. Such a fallback must still satisfy stay/switch and menu membership. Do not discard its downstream outcomes or silently resample the gate. Report fallback counts and rates by cell, year, and stay/switch mode. Stock fallback also substitutes generic `detailed_focus` and `reason`; differential fallback can therefore affect later submissions through proposal content as well as topic choice. This pathway is included in the policy contrast, not evidence for a scalar-distance effect.
- An empty menu, missing preceding topic, duplicate canonical topics, missing/nonfinite distances, zero/invalid direction embeddings, unusable required reference, or identical menus is a **data/implementation validity failure before the LLM call**, not a stay event. Stop the affected cell, retain diagnostics, and withhold a complete factorial analysis. Legitimate g=0 is separately recorded as the policy-domain failure described above, not as a malformed-data error. Do not silently widen menus, shrink their sizes, borrow expertise topics, or use the hard-coded `artificial_intelligence` empty-list fallback. Preflight checks the initial inputs and exercised states; it cannot certify separation in all future evolved states.
- A year with no eligible active researchers is a legitimate empty year, not an invalid empty menu. Continue the institutional clock using the validated shared empty-year adapters; do not add researchers or choices.

## 5. Manipulation checks and estimands

For cell \(c\), year \(t\), let \(E_{ct}\) be eligible repeat-choice events, \(S_{ct}\) realized topic changes, \(A^{start}_{ct}\) year-start active founders, and \(N_{ct}\) all new-project direction choices, including the initial choices and repeat stays.

Report:

1. Realized switch propensity: \(\hat p_c=\sum_t S_{ct}/\sum_t E_{ct}\), with raw counts. It need not equal p exactly. If the eligible-repeat denominator is zero, report undefined/NA, not zero. Eventwise `realized_switch == (U < p)` must hold for every eligible repeat event in a valid cell.
2. Mean, median, and full empirical distribution of stored direction-to-history distance **among realized switches**. Zero switches produces undefined conditional distance, not zero. Report the history/expertise reference split.
3. Eligible repeat decisions per year-start active person-year, and switches per year-start active person-year; also give raw phase-1 risk sets and phase-0 losses. Report undefined/NA for any zero active-person denominator. For ten-year rates, sum numerator and exposure over years 1–10, with zero repeat events in the common initialization year; for annual rates use that year's own denominator. No rate inherits the special zero convention defined below for empty-year entropy.
4. For every logged state, both menus' sizes, topic sets, distance extrema, and separation. This is the check of the controlled menu factor. Do not require an aggregate far-versus-near distance difference between different evolved populations as an implementation invariant.
5. Fallback counts; annual charges; project start/end dates; initial-state equality; history-ingestion coverage and reference provenance; direction-vector norms; and funding-panel memberships, budget mode, and terminal shared-validator summary.

### Descriptive companion from the original strategy worlds

Reuse the already completed audited manipulation reanalysis for frozen original worlds 1001–1003, or reconstruct it from the same frozen event logs if an output is missing. Report by original strategy and world: eligible repeat choices, switch propensity excluding initial choices, conditional direction-to-history distance, choices/switches per year-start active person-year, and initial-choice counts. Preserve raw denominators, source hashes, and variable-duration context; do not pool researchers as world replicates. This supplies James's descriptive check of the **original** strategy labels. Assigned balanced-agent policies in the new factorial cannot establish how those old labels behaved. Keep this existing-data companion separate from new-world outcomes and do not describe it as a new simulation result.

### Primary endpoint

For all direction-choice events in year \(t\), including repeat stays, let \(q_{ctd}=n_{ctd}/N_{ct}\) for topic \(d\) in the frozen 53-topic universe.

\[
H_{ct}=-\frac{\sum_{d=1}^{53}q_{ctd}\log q_{ctd}}{\log 53},\qquad
Y_c=\sum_{t=1}^{9}\frac{H_{ct}+H_{c,t+1}}{2}.
\]

Use \(0\log0=0\). If \(N_{ct}=0\), assign \(H_{ct}=0\) by a predeclared **no-activity convention**, and flag that year; do not call it measured concentration among an existing choice population. \(Y_c\) lies in [0,9]. Do not omit empty years, average only surviving years, or change the denominator to occupied topics.

This endpoint is diversity of new project choices, not active-project stock diversity or diversity only among switches. Reporting only switches would change its sampling population with p. The common one-year cycle is a deliberate simplification; do not numerically pool these endpoints with the original variable-duration strategy experiments.

### Primary contrast and conditional falsification

\[
\Delta_{\mathrm{interaction}}
=(Y_{HN}-Y_{LN})-(Y_{HF}-Y_{LF}).
\]

Pre-specified hypothesis: the propensity effect on entropy AUC is **more positive under the near-history rule**, i.e. \(\Delta_{\mathrm{interaction}}>0\). This is a scale-specific interaction on the bounded entropy-AUC scale, not by itself a diversity benefit. For example, near/far simple effects of −1 and −3 yield an interaction of +2 although both reduce diversity.

In a completed, valid four-cell comparison:

- \(\Delta_{\mathrm{interaction}}\leq0\): this realization does not support the proposed positive relative interaction; report the nonpositive finding.
- \(\Delta_{\mathrm{interaction}}>0\): report a more positive relative propensity effect under near-history **in this seed-42 realization only**. It does not establish a benefit, robustness, practical importance, or that HN has the largest absolute diversity. A near-history propensity benefit separately requires \(Y_{HN}-Y_{LN}>0\).
- Separately report all four \(Y_c\), the two propensity simple effects, the two menu simple effects, and averaged main effects. A claim that HN exceeds both one-factor alternatives additionally requires \(Y_{HN}>Y_{LN}\) and \(Y_{HN}>Y_{HF}\); these are secondary descriptive comparisons, not alternative primaries.
- An invalid, incomplete, or out-of-domain cell leaves the primary contrast unevaluated. A failed gate/menu check or legitimate zero separation gap is not evidence against the scientific hypothesis; report its distinct reason.

No p-values, inferential confidence intervals, equivalence tests, standard errors across cells, or bootstrapping agents/papers/years as world replicates. Report exact descriptive magnitudes and trajectories. One seed remains one matched realization.

### Fixed secondary endpoints

- Year-ten active founders / 1,200, accepted unique papers / 1,200, and cumulative earned funding / 1,200.
- Participation-weighted entropy AUC, using \((N_{ct}/1200)H_{ct}\) with the same trapezoidal rule. Report alongside raw choice counts so high entropy among few survivors is visible; never replace the primary after seeing results.
- Realized citation yield for every unique manuscript first submitted in years 1–7, measured at the end of first-submission year + 3. Include unpublished manuscripts as zero realized publication-mediated citations; published manuscripts require the observed count at that same deadline. Report total yield / 1,200 founders and mean yield / eligible manuscript, with cohort counts. The latter is undefined if its cohort is empty. Missing required observations are data-integrity failures, not zeros.
- First-attempt acceptance among all unique manuscripts first submitted in years 1–10, with an undefined rate if there are no first submissions.

All effects are assignment-policy contrasts, retaining technical parse fallbacks and natural attrition. Menu topic identities alter venue/funding eligibility and other institutional opportunities; generic-text fallbacks may change later submissions; `track` permits treatment-dependent application counts, awards, and funding volume. These are included pathways, not nuisance differences that matching or fixed duration eliminates. Conditional-on-switch distance and per-manuscript outcomes are manipulation/secondary summaries with changing risk sets, not alternative causal primary populations.

## 6. Freeze, validation, execution, and reporting

Before any scientific outcome is inspected, freeze this reviewed protocol and a separate implementation/analysis commit. A draft is not a completed preregistration. Preserve earlier versions and document every amendment with its timing.

Required CPU checks in the isolated third-family worktree:

- Actual 53-direction helper output and canonical lookup: previous-topic exclusion, exactly 17/17 distinct candidates, deterministic tie ordering, complete distance range, and nonzero finite direction embeddings. Exercise a positive-gap state, a legitimate zero-gap state with its distinct domain-failure outcome, and malformed/empty inputs. These fixtures do not establish future separation.
- Neutral tracking initialization outside the exploration experiment: expertise registered for every founder; unique accepted and rejected new manuscripts ingested with first simulated submission years; resubmissions leave metadata unchanged; empty/expired history legitimately falls back to expertise; deliberately missing ingestion fails coverage. Verify the exact local MiniLM revision and weight hash, CPU-only construction, and separation from the GTE RAG cache. No extra strategy adjustments or keyword LLM calls are introduced.
- Initial cached responses are identical across four cells; first choices never enter repeat denominators.
- Synthetic U values on both sides of 0.25 and 0.75 verify the exact gates and common random numbers. Reordering agents/retries does not change the gate table.
- Use the actual phase-1/prompt/parser path with scripted model responses to verify singleton stays and menu-constrained switches; all returns, prompt durations, and project timelines remain one year. Bad JSON and out-of-menu topics exercise the native valid-candidate fallback.
- Actual project-cycle checks verify equal timing and 10-unit annual charges for stay versus switch, including natural phase-0 deactivation and the existing exact-zero boundary.
- Actual effective funding prompts use the native balanced partition and contain at most 25 applications; verify 26→13/13 and 53→18/18/17, native quotas and downstream outcome parity for an identical validated order, plus zero-application years. Every cell resolves to `track` slots and fixed 20-unit award size. Verify `compact=True`, the exact `ranked_application_ids` schema and response-format instructions, complete unique-permutation validation before ID/rank expansion, empty unused reasons, and `max_tokens=8192`. Exercise malformed/incomplete/duplicate ID arrays, strict final ranking validation, duplicate program/panel rejection, and refusal to complete/analyze when the shared validator's `completion_allowed` is false. These tests do not establish equivalence of sampled rankings under legacy and compact elicitation.
- Analysis fixtures check initial-choice inclusion in entropy but exclusion from repeat propensity, repeat stays, no-active-choice years, NA for zero event/exposure/conditional-distance denominators, unique manuscripts/deadlines, the interaction sign including two negative simple effects, and refusal of incomplete, out-of-domain, or funding-invalid four-cell input.

A bounded live transport preflight may verify the inherited constrained-output contract. It is not a pilot for choosing p, menu size, duration, endpoints, or signs. Any pilot outcomes must be labeled and excluded from confirmatory execution; do not tune this design on them.

Run exactly the four frozen conditions to year ten with the same source and initial cache, unless a predeclared integrity or policy-domain guard stops a cell. Admit scientific completion only when the per-cell shared funding validator also permits completion. Do not stop on an attractive sign, choose a favorable subset of years, add seeds, silently resume with changed source, or combine failed attempts with replacement cells. Infrastructure failures retain their outputs and reasons. Any scientific-source correction requires a new version and a fresh matched set, unless an outcome-blind, independently reviewed argument establishes an identical scientific execution; never delete failed-attempt evidence.

The gate table, initial choices, prompts, candidate-distance records, actual selected topics, fallback flags, risk-set counts, project state, per-year resources/funding panels, source hashes, request metadata, and completion records must permit reconstruction. Preserve the current production transport/semantic-validation audit. Protect the existing eight-run freeze by using a different worktree, launch plan, data-cache namespace, and output namespace.

Final figures and manuscript integration occur only after outcome validation and independent reviewer critique. New figures use green text/marks on white; no inferential error bars. Manuscript additions remain green and replaced text struck through, on the paper branch only.

## 7. Source basis inspected for this draft

The inherited v0.2 references below are under `/tmp/scienceutopia-llm-review/code/`; they describe the original implementation basis. Current third-family sources for this amendment are listed separately below. Source inspection does not certify completed CPU validation or scientific execution.

- `utopia/agents/research_direction.py:477–550`: original 17/17/19 strategy menus, balanced expertise shortcut, and heuristic fallbacks.
- `utopia/agents/research_direction.py:294–400`: explicit `candidate_map`, strict topic validation, deterministic first-candidate parse fallback, memory update, and `DIRECTIONS_DICT` lookup.
- `utopia/agents/research_direction.py:189–245`: current duration text and balanced/strategy prompt branches requiring a small common-template adaptation.
- `utopia/metrics/embedding_tracker.py:61–82,172–232`: unique-paper embedding ingestion, registered expertise, inclusive historical-paper centroid, fallback, and cosine-distance computation.
- `run_simulation.py:692–719,750–785`: exploration-only early returns and the required neutral-hook coverage gap.
- `utopia/agents/funding_agents.py:328–341`: native seeded balanced `shuffled[j::ceil(n/25)]` panel partitioning.
- `src/james_funding_validation.py:321,409–426,460`: duplicate-panel rejection, terminal `completion_allowed` summary, and installation API. The reviewed helper must be version-pinned in the later implementation freeze; this draft does not modify it.
- `run_simulation.py:1971–2082`: phase-1 risk set, timeline update, recorded switch indicator, and submission eligibility.
- `run_simulation.py:2992–3005,3998–4032`: annual project charges and actual phase ordering.
- `run_simulation.py:616–675,3681–3695,3762–3827`: founder initialization, agency default rates, panelization, track/fixed slot distinction, and grant credits.
- `utopia/arguments.py:101–107,216–230,291–307`: fixed award-size default, balanced population override, explicit funding-panel override, and track-budget option.
- `utopia/config.py:19–41,68–124`: project/resubmission costs, 20-unit award, initial resources, neutral strategy toggles, history-window argument, and panel default.

## 8. Independent-review disposition for v0.2

Herschel and Godel reviewed v0.1 independently. Both accepted the common initialization, singleton stays, common one-year cycle, downstream attrition interpretation, and bounded single-seed policy question. Their request/funding-helper implementation reviews were separate from this protocol review.

- Herschel's tracking prerequisite is incorporated as explicit neutral expertise/history hooks with first-submission-year coverage reconciliation.
- Both reviewers' panel correction is incorporated with the exact native balanced partition. Godel's shared-validator/terminal-completion requirement is explicit.
- Both rejected any guarantee of strict future distance separation from 53 distinct labels. Godel preferred continuing through zero gaps with a weak-manipulation label. V0.2 retains the strict guard as a prospective design decision. V0.2 therefore states the restricted policy domain, logs legitimate geometric ties separately from implementation defects, stops that cell, and withholds the factorial contrast. This choice is fixed before outcomes.
- Godel's additional corrections are incorporated: nonzero direction embeddings; a separate existing-world descriptive strategy companion; relative-interaction wording instead of an automatic benefit claim; explicit fallback-text and track-funding pathways; and undefined/NA for zero rate denominators.

At v0.2, no experimental factor, seed, population, horizon, primary endpoint, or interaction contrast changed. That protocol revision made no implementation, simulation, or manuscript change and required separate-worktree implementation and actual-path validity tests before freeze and scientific execution.


## 9. Prospective v0.3 amendment and review disposition

This amendment records the already reviewed common v4 compact funding baseline and the current third-family MiniLM pin before any observed third-family outcomes. It changes funding response elicitation explicitly, rather than asserting that response schemas or sampled rankings remain identical to legacy elicitation. The compact representation, 8,192-token funding limit, and exact CPU MiniLM snapshot are shared across all four cells. No switch probability, seed, population, horizon, initial-choice rule, project cycle, menu construction or separation guard, endpoint, or interaction contrast changes from v0.2.

Current amendment sources, all under `/tmp/scienceutopia-llm-review/code-propensity/`:

- `src/james_propensity.py:35–42,153–169,480–489,817–818`: four-cell constants, representation label, exact MiniLM snapshot/weight validation, local CPU construction, and `compact=True` installation.
- `src/james_funding_compact.py:17–30,44–88`: representation versus JSON-field names, prompt-tail replacement, full-permutation validation, metadata-derived IDs/ranks, and empty reasons.
- `src/james_funding_validation.py:517–525,549–621`: opt-in API, batch preflight before expansion and stock processing, retained original responses, and compact prompt/schema installation.
- `run_simulation.py:3714–3718`: funding evaluation `max_tokens=8192` and existing request-seed context.
- `src/james_mechanisms.py:228–242`: separately pinned `thenlper/gte-small` CPU retrieval-cache verification. This does not define the MiniLM topic-history geometry.

**Review state:** Herschel returned GO for the v0.3 draft with no correction on the reviewed compact-funding, MiniLM, or prospective-status points. The main agent approved that scope and requested removing the unnecessary attribution of the strict-guard decision; this reviewed copy makes that wording correction without changing the domain rule. Document review is complete. Implementation CPU validation and the prospective freeze remain pending. Record this reviewed version and the completed validation evidence at the later freeze; neither document review nor earlier helper reviews constitute completed runtime validation or observed scientific results.

## 10. Failure-informed amendment: third-family v3, 19 September 2026

### Status, scope, and retained history

This is a prospective replacement amendment after an observed runtime failure, **not externally registered**. The recovery design has received the requested independent methodological review; replacement implementation validation, final independent review, and the new source/protocol freeze remain prerequisites for scientific launch. No replacement-family outcome has been observed or used to choose the amendment. Partial predecessor trajectories exist and are preserved; the historical status text above is not a current claim that no study has ever run.

The predecessor third queue, `james_propensity_seed42_n1200_20260919_v2`, used source `9e01705b69a55748a3f0c0c6ddaf1a3407735c02` and scientific output IDs containing `james_propensity_v1`. Its protocol before this addition had SHA256 `322658683f37e58874d842034d23d2a710e684018284e6dffebb673c8aea50c1`.

The predecessor queue emitted `queue_failed` at 11:40:31.767863 UTC after LN exited 1 during year-one compact funding. Its final status was updated at 11:40:44.596820 UTC after cleanup. For `NSF_SYSTEM`, panel 6, the final audit recorded three semantic validation attempts, `n_expected=25`, `n_returned=25`, `n_valid=5`, and `n_rejected=20`, with duplicate-ID and compact-validation errors. Its `missing_application_ids` values identify failures of exactly-once coverage, not necessarily literal omissions. The terminal funding summary recorded 134 audited final panels, one invalid final panel, one failed batch, **zero successful batches**, `completion_allowed=false`, and zero imputed/fallback rankings. The whole-batch guard prevented native awards from that year-one batch.

LF was interrupted during year-two resubmission preparation after saving its year-one checkpoint. Its observed funding audit contained 135 valid final panels and no invalid final panel. Do not describe this collateral interruption as an independently demonstrated ranking failure. HN and HF never started. None of the four conditions completed the required ten years; no predecessor primary contrast or scientific null result is adopted.

The verified snapshot is `outputs/logs/ec2_preparation/heartbeat_20260919T115313Z_verified.json`, SHA256 `9d3e5a15d8489514e7d49d812ed62a1e00388e0fd636404a087fe647bbec46c3`, under `/opt/dlami/nvme/ubuntu/scienceutopia_llm_james_20260918`. The 37-file failure evidence inventory is `propensity_v2_failure_evidence_20260919.json`, SHA256 `51f041cde4a969901c3099fd38299df8de88a5cdb86032efc9168482b4dde5eb`. The original uncorrected snapshot is preserved separately. The corresponding LN and LF `funding_validation.jsonl` hashes are `dc24945cb2c4e246ceafcedc5f528ed26179ca71cf5d3766cb8644b488b8d7be` and `8f021a894cb38fe14f68017e1cdf8b5c96a30f22920112b4c26aa930c82cdaa8`. Preserve raw requests, validation attempts, partial outputs, checkpoints, initializer artifacts, source freezes, and termination receipts. A stale `running` field in an interrupted manifest does not supersede queue termination evidence.

### One fresh initializer and one whole four-cell replacement

The new protocol identifier is **`james-propensity-v3-single-world-sequential`**. Use a new branch/worktree and fresh output roots with these exact IDs:

- `james_propensity_v3_n1200_y10_initial_seed42`
- `james_propensity_v3_n1200_y10_LN_seed42`
- `james_propensity_v3_n1200_y10_LF_seed42`
- `james_propensity_v3_n1200_y10_HN_seed42`
- `james_propensity_v3_n1200_y10_HF_seed42`

After the new freeze, generate **one fresh treatment-blind technical initializer** under that source. Replay its single immutable cache in all four new conditions through the native parser, memory update, and project initialization. Retain seed 42, the same 1,200 university-founder blueprint rules, the native initial expertise candidates, common one-year duration wording, and the original initial-choice request policy.

The predecessor initializer completed before the failed conditions and was independently reverified: 1,200 requests/responses, no pending requests or reported transport/guard errors, and zero funding evaluations. Its cache SHA256 is `efb85f89fd8be4d3318dd4607c80857f24236c62d95ea02089e81448b40c1b21`; its six-artifact receipt SHA256 is `040c4f7c4da516462de0ade19631e55cab71e63552020cd2ded8b15d323b8743`. **Preserve that complete cache but do not adopt it in v3.** No cross-version cache import, metadata rewriting, or relaxation of strict source/cache matching is part of this decision.

The new cache is not promised to reproduce the old choices or initial-state hash merely because the world seed is unchanged. This is a new matched realization under the amended funding method, not continuation of the old world. Do not generate multiple initialization candidates, select among caches, or switch back to the predecessor cache after seeing the new one. Initialization is not a fifth scientific world.

No predecessor LN/LF checkpoint, year-one outcome, submission history, funding allocation, or partially accepted ranking prefix enters v3. Execute exactly one fresh four-cell replacement set; no selective-cell retry, trajectory mixing, or automatic further replacement after failure is allowed.

### Unchanged scientific intervention and amended funding

Retain p=0.25/0.75, 1,200 university founders, no entrants/replacements, ten scientific years, the balanced persona, and the exact native-history 17-nearest/17-farthest menus excluding the current topic. Preserve the strict positive separation guard, common one-year cycles, singleton stays, project charge 10, resubmission charge 5, native attrition/zero-balance behavior, and complete neutral first-submission history ingestion. Retain the exact local CPU MiniLM revision and weight pin specified above, distinct from the RAG GTE cache.

All four new conditions adopt the reviewed **`sequential_remaining_ids_v1`** funding method from the frozen V6 source `34112b33eb6ba37ff2a8bf6eff7cceddc704d508f`. Its algorithm, prompt preservation, per-request schema, exact seed context, singleton request, three-SDK-invocations-per-step bound, SDK internal-retry distinction, final permutation guard, and whole-batch award barrier are specified in `docs/james_sequential_funding_protocol.md`, including its appended third-family scope. Preserve balanced panels of at most 25, `track` slots, NSF/DARPA rates 0.23/0.10, native quota rounding, and fixed 20-unit earned awards with normal credits in all propensity cells.

The propensity study does not acquire the first-eight population/influx settings or the P0/F0 interventions. Its funding prompt remains the original native prompt appropriate to each evolving propensity condition; the sequential helper replaces only the response-format tail. Structured publication records remain normally available.

This changes funding elicitation and the number of deliberative model calls. Sequential history conditioning can change rankings, resources, attrition, and subsequent topic histories. No ranking-distribution equivalence to compact or legacy output, or pure syntax-repair claim, is made. The repeated-call compute limits and SDK-free probe limitations remain as disclosed in the sequential protocol.

Keep the primary normalized 53-topic choice-entropy AUC over years 1–10, including initial choices and repeat stays, with the original empty-year convention. Keep `(HN−LN)−(HF−LF)`, all simple effects, and every secondary endpoint and denominator from Section 5. A positive relative interaction alone is not a diversity benefit. Retain the conditional single-realization interpretation and no world-level SE, CI, p-values, or agent/paper/year pseudo-replication.

### Exact raw-audit admission and initialization boundary

The initializer performs only native initial-choice generation and state construction: no submissions, reviews, awards, or scientific-year outcomes. Its sequential funding-audit admission uses **`num_years=0`**, meaning zero expected native funding years, zero agency-year registrations, zero panels, and zero sequential funding SDK calls. This is the audit's expected horizon: native `Simulation` retains `YEARS=10`, and initialization exits at the native phase-one boundary after producing the cache. The request audit must still be present, finalized, and joined exactly; an empty funding trace cannot conceal an unregistered legacy or sequential funding request.

Each scientific condition instead requires **ten** started/completed native funding-year boundaries covering years 1–10 and **twenty agency-year registrations**: one NSF and one DARPA registration in every year, including years with zero applications. Twenty is the number of agency-year registrations, not twenty agencies or twenty panels. Native program/panel and application counts remain data-dependent. Record legitimate empty agency registrations explicitly rather than inventing panels or treating absent coverage as empty.

Require the finalized sequential trace to join exactly to the independent raw request audit by context, item index, and attempt, with matching payload hashes, effective seeds, token reservations, response/usage evidence, and terminal states. Reject duplicate, missing, unexpected, late, or unmatched funding requests and unknown legacy funding contexts. The initializer must pass this same join with zero expected funding calls; worlds must pass it for every observed sequential call. SDK audit counts do not claim visibility into each internal HTTP retry.

Use the shared read-only admission API `validate_funding_evidence(directory, manifest, *, expected_source)` for the initializer and each condition, including downstream file analysis. It validates the role-specific zero-/ten-year funding coverage and raw evidence against the explicit expected source identity. A stored success flag alone does not substitute for this validation.

Bind **both** `docs/james_propensity_protocol.md` and `docs/james_sequential_funding_protocol.md` in driver, initializer, controller, and analysis provenance. The new cache and all four consumers must match the same complete v3 source/configuration/runtime identity; all four must reproduce that cache's pre/post initialization-state hashes under the existing state canonicalization. Do not edit old cache fields or loosen comparison to make a new consumer pass.

Every world manifest must stamp the fresh producer's `initialization_manifest_path` and `initialization_manifest_sha256`, as well as `initial_choices_path` and `initial_choices_sha256`. Validate those exact files and hashes, the producer's complete initialization-only status, its source/configuration identity and zero-funding audit, and its agreement with all four consumers. Matching cache bytes without the validated producer evidence is insufficient.

Require all native ten-year files, agency/panel coverage, award ledgers, founder/risk-set accounting, direction events and embedding norms, unique first-submission registry, acceptance and fixed-deadline citation observations, request/funding audits, and source-bound completion receipts before analysis. Missing data are integrity failures, not zeros. All four new conditions must complete ten years and pass these gates before the primary contrast is evaluated.

### Validation, failure policy, and comparison across studies

The reviewed V6 helper and its earlier tests do not certify this new third-driver integration. Before launch, require targeted actual-path CPU tests and independent final review of fresh-cache identity, initializer `num_years=0` admission, exact request-audit join, ten-year/twenty-registration coverage including empties, funding atomicity, unchanged propensity/menu behavior, and analyzer/receipt compatibility. The prescribed technical probe must use a legitimately available server allocation and its disclosed one-HTTP-attempt route; no diagnostic may repurpose a running V6 scientific server. Neither frozen predecessor source is edited by this staging work.

A failed initialization, exhausted selection budget, malformed or incomplete audit, false completion gate, changed source/model/cache, broken initial-state equality, or invalid funding batch prevents scientific admission. Preserve the originating failure and collateral interruptions. A legitimate zero near/far gap remains a policy-domain failure, not a scientific null. Natural attrition and genuine empty years retain the original conventions. No failed state is repaired by changing p, menus, duration, horizon, or ranking rules.

Using the same verified sequential funding helper/settings as V6 permits a narrow shared-funding-method description. It does not create one twelve-cell factorial or justify controlled cross-family effect-size comparisons across different populations, horizons, and interventions. Keep the failed compact third family, the new sequential third family, and archived multi-world studies separately identified. Report all completion/failure statuses; James's full requested evidence remains incomplete until the required complete analyses and reviewer-checked findings exist.

## Prospective single-GPU runtime amendment, 19 September 2026

The user explicitly requested use of idle EC2 GPU 7 instead of waiting for a
second card. The original v3 queue had no server, initializer, or scientific
client started when it was retired. Its source, plan, status, and retirement
receipt remain preserved. This amendment changes the execution configuration
before any v3 initialization or outcome; it does not rerun a completed or
partially observed scientific condition.

Use fresh IDs `james_propensity_v3tp1_n1200_y10_{initial,LN,LF,HN,HF}_seed42`
in a separate worktree. The method identifier remains
`james-propensity-v3-single-world-sequential`. One fresh initializer and all four
conditions must use the same fixed server profile `propensity_tp1_v1`, whose
runtime identifier is `qwen3_32b_bf16_tp1_v1`. The pinned Qwen3-32B weights,
BF16 precision, 32,768-token context, thinking mode, request generation
parameters, seed derivation, 1,200 founders, ten years, direction policies,
funding algorithm, and estimands remain unchanged.

The explicit serving changes are tensor parallel size 1, maximum 16 active
sequences, GPU memory utilization 0.94, eager execution, chunked prefill, and
maximum 2,048 batched tokens. These are resource and execution settings, not a
reduction of the request context or output-token limits. The shared profile
module is bound in source provenance. Each manifest and the initialization
cache record the runtime profile and its hash; a TP2 or differently configured
server is rejected. No numerical, sampled-trajectory, or distributional
equivalence to TP2 is claimed. All within-family comparisons use this one
profile; the ongoing TP2 V6 families remain separately scoped.

Every new server must acquire its physical GPU's shared UUID lock, verify
exclusive ownership and actual startup capacity, and pass its own nonce and
all 25 sequential selections before starting any client. Capacity estimates
alone are not a readiness result. Preserve the existing full context/token
checks, zero-year initializer audit, ten-year world audits, native funding
ledgers, shared-cache and state checks, complete receipts, and all-four-cell
analysis gate. No scientific settings or guards may be relaxed to make the
single-GPU launch succeed. The prior LLaDA tasks continue naturally.

## Initializer admission path correction, 19 September 2026

The v3tp1 initializer finished its 1,200 initial-direction requests, but its
queue rejected admission before issuing a completion receipt or starting any
scientific condition. The queue supplied the canonical private-storage
directory to a validator that requires the producer's recorded
`worktree/outputs` spelling. Those paths pointed to the same files. Independent
CPU replays of the unchanged validator passed the complete raw request,
zero-year funding, runtime, source, and native-ledger checks when called through
the recorded producer path. This was an integration error in admission; it
does not establish a scientific result.

Preserve the failed queue, its initializer, source, and original audit files.
The corrected queue uses a new worktree and fresh IDs
`james_propensity_v3tp1p1_n1200_y10_{initial,LN,LF,HN,HF}_seed42`. It constructs a
new treatment-blind initializer and shares that one cache across all four
conditions. No old initializer or scientific output is adopted. The method
identifier, seed42, population, horizons, TP1 runtime profile, model,
generation settings, funding procedure, direction policies, and estimands
remain unchanged. A fresh initializer is not claimed to reproduce the previous
sampled realization.

Declare output artifacts, the initializer cache, and every consumer's cache
argument using the producer's worktree spelling. The controller must derive
that path from its trusted plan, verify that it resolves to the same owned
private-storage job directory, and preserve all artifact, source, byte-hash,
and before/after checks. Do not rewrite stored audit paths, relax literal
raw-audit joins, trust an arbitrary manifest-supplied alias, or bypass any
zero-year or ten-year validation. Downstream file analysis must likewise use
the recorded producer directories and its shared producer cache, while
retaining the source, runtime, receipt, and all-four-condition requirements.
