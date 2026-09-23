> **Staging scope, 19 September 2026:** The V6-origin document below is preserved verbatim as history. Its new Section 8 adopts the same reviewed algorithm for the separate third-family v3 replacement. The earlier statement that the third family continues with compact funding describes the prior plan and is superseded for this new replacement only. The frozen V6 document and running V6 worktree remain unchanged.

# Sequential funding ranking: prospective v1 protocol for the v6 replacement set

**Status, 19 September 2026:** prospective protocol amendment, not externally registered. Implementation, technical validation, and independent review must be completed before launching the v6 scientific jobs. No v6 outcomes have been observed or used to select this protocol. This document does not declare implementation validation, a source freeze, or scientific completion.

This amendment applies only to one fresh replacement set of the first eight conditions. Its source basis is the first-eight worktree at commit `adddc6ec611af7cb19e03f244579e8dbe3b452de`. The parent task owns integration, the final source/protocol freeze, and any later launch. Authoring this document does not modify or authorize modification of running worktrees.

## 1. Reason for amendment and retained failure history

The v5 influx condition B failed during scientific year 2 after three compact-ranking attempts. The final ranking for `NSF_AI_FOUNDATIONS`, panel 4, violated exactly-once coverage for IDs 3 and 24: the audit recorded 25 entries and 23 uniquely valid entries. Its `missing_application_ids` field identifies failed exactly-once coverage, not necessarily literal omissions. The strict final guard rejected the batch before award processing. The controller also stopped A while it was processing year 3; this collateral interruption is not a demonstrated invalid-ranking event in A. Neither condition completed its specified horizon.

The failure receipt is `outputs/logs/ec2_preparation/heartbeat_20260919T100159Z.json` under `/opt/dlami/nvme/ubuntu/scienceutopia_llm_james_20260918`, SHA256 `f4887122920433f61ebde6541e5bdfcae68fbe6ac13ae3839f35603ad51bfbd8`. Preserve the receipt, v5 source and launch records, raw requests/responses, validation attempts, partial checkpoints, and termination evidence. Preserve earlier failed versions as well.

V5 was a failure during scientific execution, not solely a startup or transport probe. No v5 trajectory, partial outcome, award history, or checkpoint is adopted into v6. This is a disclosed failure-informed method amendment. It is not recovery of the original realization or retrospective registration of the earlier runs.

## 2. Fixed replacement set and unchanged scientific questions

Exactly one fresh v6 set comprises the following eight conditions:

| Family | Conditions | Researcher founders per condition | Horizon |
|---|---|---:|---:|
| Influx × resubmission | A: neither; B: influx only; C: resubmission only; D: both | 1,202 | 10 years |
| Structured publication-record display × award credit | P1F1, P0F1, P1F0, P0F0 | 1,200 | 6 years |

Every world uses seed 42. The influx family retains 240 universities with five researchers each and two industry founders; funding agencies are not counted as researcher founders. B and D retain the existing entrant schedule: 120 university researchers each year and one industry researcher in years 5 and 10. The other family retains 1,200 university founders and its existing population settings.

Preserve the existing founder/cohort generation and its `james1202_v2` population RNG namespace. Renaming output paths to v6 must not regenerate different founder expertise or entrant cohorts through a new population namespace. Retain the existing initialization rules for the P/F family.

The intervention definitions, resource and project rules, resubmission settings, venue rules, outcome definitions, denominators, family-specific contrasts, and horizons remain those of the frozen first-eight design. Bind the predecessor analysis and plan definitions in the v6 freeze; this amendment introduces no alternative endpoint or outcome-based selection rule.

P0 continues to withhold structured publication counts, acceptance statuses, and titles from funding prompts and retains its condition-specific instruction about withheld records. Proposals may still reveal earlier work. F0 continues to withhold spendable award credits while preserving earned awards and award histories; it is not an equal-resource redistribution. These interpretations are unchanged.

## 3. Common funding and inference settings

All eight v6 conditions use the same sequential ranking method below. Preserve:

- Native deterministic balanced funding panels of at most 25 applications. For a program with N>0 applications, use the existing seeded shuffle and k=ceil(N/25), with panel j equal to `shuffled[j::k]`. Thus 26 applications yield 13/13 and 53 yield 18/18/17. Do not replace this with chunks of 25 plus a residual.
- `funding_budget_mode=track`, existing program eligibility, NSF rate 0.23 and DARPA rate 0.10, and the native per-panel quota `max(1, int(n * program.funding_rate))`.
- Fixed earned award size of 20 resource units, with the original condition-specific F1/F0 credit treatment. Panelization remains distinct from uncapped global ranking because competitors and quota rounding differ.
- `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`, BF16, TP2, vLLM 0.12, the existing 32,768-token context, `enable_thinking=True`, the Qwen3 reasoning parser, and temperature 0.7.
- `max_tokens=8192` for each funding rank-selection request. This is a per-request cap, not a panel-wide budget. Non-funding request settings, the ordered corpus, retrieval cache, and temporal restrictions remain unchanged.
- The existing OpenAI SDK client setting `max_retries=5` for internal transport retries. It is distinct from the three rank-selection SDK invocations specified below.

Record the exact effective settings and runtime versions at freeze. No silent reduction in reasoning, precision, context checks, population, horizon, or panel size is permitted to obtain completion.

## 4. Sequential next-choice ranking

### Immutable panel and condition-specific prompt

For each nonempty panel, freeze its original application order and the exact mapping from integer application indices 0,...,n−1 to applicant metadata. Require 1≤n≤25. An empty program produces no ranking requests and remains a valid empty case.

Start from the actual **original condition-specific funding prompt** and its original application order and evaluation criteria. Replace the original `## Response Format` tail with the sequential instruction and state; do not append competing instructions after the old tail. Preserve the exact substantive prefix before that boundary. Preserve the complete original panel at every position, including applications already selected. Do not rebuild a generic P1 prompt for P0, restore withheld records, remove chosen applications from the substantive panel text, reorder applications, or change program criteria. Preserve the inherited system message and common client instructions.

The new tail shows the ordered selected-ID prefix and the remaining IDs in original index order, and asks for the next most preferred remaining application. The selected prefix is an ordered list, not a set. Each request uses a schema whose allowed integer enum is exactly that request's remaining IDs; a shared static enum is insufficient. Freeze the exact response wrapper/field, schema, prompt-tail template, and representation identifier with the implementation.

Construct a fresh request from this immutable prompt and the current prefix/remaining state. Previous reasoning, explanations, invalid answers, and conversation transcripts are retained as evidence, not appended to later prompts. Failed attempts do not modify the displayed state.

### Positions and validation

Initialize an empty accepted prefix. For each zero-based step s=0,...,n−1, corresponding to rank s+1:

1. Compute the remaining set as the original panel indices minus the accepted prefix.
2. Request exactly one model-selected integer from that remaining enum. Reject booleans, coerced strings/floats, out-of-enum values, missing selections, and malformed responses.
3. Accept the first valid selection within the fixed attempt bound. Append it once to the prefix and advance to the next position. Do not reconsider an accepted selection or reset an accepted prefix.
4. If no valid selection is obtained within the bound, mark the panel and its funding batch failed. Propagate a fatal failure rather than returning a fallback ranking.

Request all n positions, including the final singleton. Do not fill the last remaining ID without a validated model response. A singleton failure is fatal under this protocol, despite its lack of substantive choice.

After n validated selections, apply the unchanged full complete-permutation guard. Only then derive rank numbers and exact applicant IDs from the frozen metadata for native processing; retain an empty, unused `reason` field rather than inventing explanations. No deduplication, missing-ID completion, sorting of model choices, or repaired ranking is allowed.

Validate all panels in the funding batch before processing **any** awards in that batch. A failure in its last panel must leave earlier panels without applied awards, award-history entries, tracker allocations, or resource credits from that batch. Valid prefixes and panel records may be persisted for audit, but cannot serve as partial scientific awards.

### Attempt bounds and seed mapping

Each position permits at most **three actual SDK completion invocations**, counting failed transport/parse/semantic invocations within that limit. The implementation must not multiply this limit through nested batch, panel, or semantic-retry wrappers. There is no outer compact-ranking retry or accepted-prefix restart.

Consequently a panel permits at most 3n≤75 SDK completion invocations. A first-attempt-success 25-application panel uses 25 invocations. At 8,192 requested completion tokens per invocation, the production upper bound is 75×8,192=614,400 requested completion tokens per full panel, before internal SDK transport retries. This is a cap calculation, not measured usage or a runtime prediction.

The inherited SDK permits up to five internal transport retries within each invocation: an initial HTTP attempt plus up to five retries. Therefore 75 SDK invocations is **not** a 75-HTTP-request bound; under those configured bounds a full panel could issue up to 450 HTTP attempts. Distinguish configured bounds from observed counts. Internal transport retries do not become additional semantic attempts or change the selection seed/state. These production attempt bounds do not establish a 600-second total wall-clock limit for an SDK invocation including its internal retries.

Use the exact production seed contract:

```python
ctx = (
    "phase5_funding_eval",
    year,
    "sequential_remaining_ids_v1",
    "production",
    program_id,
    panel_index,
    step,
)
request_seed = derive_seed(42, *ctx, 0, attempt)
```

Here step is 0,...,n−1, attempt is 0,1,2, and the item-index argument is always 0. Freeze the derivation helper, encoding, and effective wire seeds with the source. Cell label, worker identity, wall time, queue version, mutable batch position, and retry-list position must not alter this mapping. Internal SDK retries retain the same request seed and scientific payload. The new funding request namespace does not replace the preserved population/cohort namespace.

The same identifiers do not guarantee identical random draws once panels or histories differ across conditions. Preserve request-level provenance rather than claiming that seed 42 couples every later decision or reproduces v5 randomness.

## 5. Interpretation and bounded execution

Sequential conditional choice changes funding elicitation. It introduces repeated deliberation and explicit conditioning on earlier selections. Preserving the model, temperature, panel contents, and quotas does not preserve sampled rankings or total inference expenditure. No equivalence to legacy tuple/reason output, compact joint ranking, or uncapped global competition is claimed.

This is exactly one fresh v6 replacement set, with fresh output roots and a common frozen implementation. No v5 resume, prefix reuse, trajectory stitching, selective world restart, or automatic replacement after another failure is allowed. A fatal failure stops the replacement queue under the declared failure policy; preserve every condition's status and the distinction between originating failures and collateral interruptions. A subsequent redesign would require a separately documented decision, not an implicit retry of this set.

Admit a family's contrast only when all four designated conditions complete their original horizons and pass all audit gates. Report every family's completion/failure status; do not silently present only successful families as the entire requested evidence. Missing or failed outcomes are not zero outcomes or evidence for a scientific null.

The four propensity conditions continue under their separate, unchanged frozen compact-ranking protocol. If all four finish validly, they may support their own within-family contrast with the different funding elicitation disclosed. Do not pool the three families into a common twelve-condition funding baseline or compare effect sizes across the differing funding methods as controlled cross-family effects.

All new analyses remain conditional single-seed case studies. Do not calculate world-level standard errors, confidence intervals, or p-values from researchers, papers, years, cells, request attempts, or replacement versions as if they were independent worlds. Preserve the archived multi-world evidence separately.

## 6. Required audit coverage and provenance

The final freeze and execution records must support reconstruction without trusting a success flag alone:

- **Freeze identity:** this protocol's SHA256; exact source/analysis commits and file hashes; predecessor source/plan references; v6 commands and output roots; model/tokenizer/runtime identities; corpus/cache hashes; effective condition settings; founder/cohort hashes; and the request-seed specification. Record the technical reason and timing of the amendment.
- **Panel identity:** world, condition, year, program, panel index, original ordered application indices and applicant mapping, n, quota, and original condition-specific prompt. Preserve the exact substantive prefix and its hash so P0 withholding and unchanged application order are verifiable.
- **Every SDK invocation:** a unique identifier; panel/step/attempt identity; prefix and remaining IDs; exact prompt/messages and per-request schema, or immutable references to their complete bytes; effective seed, model, temperature, token cap, and reasoning setting; start/terminal events; raw returned content and returned reasoning when present; parsed selection; validation decision/error; finish reason; usage and timing.
- **Retry accounting:** reconcile all observed SDK invocations with the three-per-step/3n-per-panel limits. Preserve internal HTTP-retry evidence where observed and the SDK version/configuration in all cases. Do not label SDK-level counts as complete HTTP-attempt counts if internal retries are not directly observed. Unknown transport counts remain explicitly unobserved, not zero.
- **Completion coverage:** reconcile each nonempty panel to exactly n accepted model selections, including its singleton request, then to one final permutation validation. Reconcile every expected panel with its batch record and successful batches with actual award ledgers. Record valid empty cases explicitly. Missing audit records cannot masquerade as no applications or no failures.
- **Raw outcome coverage:** preserve all required years, fixed-cohort/entrant identities, application/award records, resource histories, publication/review/submission records, native analysis inputs, and request/funding summaries. Final failure, incomplete raw coverage, or a false validator completion gate bars admission of the affected family.
- **Failure preservation:** keep partial prefixes, invalid responses, exceptions, outstanding request identities, incomplete batch records, checkpoints, and controller receipts. Keep v5 evidence immutable and separate from v6 outputs.

Large repeated prompt bodies may be stored once with immutable content-addressed references, provided each request's complete bytes remain reconstructible. Hash-only logs without the referenced raw content do not establish raw coverage.

## 7. Technical validation and independent review before launch

Validate the actual implementation path, not just a prompt example or a generic JSON smoke:

1. Exercise n=1 and n=25 panels, all positions including singleton, changing remaining enums, exact integer validation, malformed/duplicate/previously selected IDs, partial-prefix failures, and exhaustion at the third SDK invocation. Prove that no outer wrapper increases the bound or resets a prefix.
2. Verify condition-specific prefix preservation across positions and attempts, including real P0/P1 prompt differences. The sequential tail must neither expose withheld structured records nor replace the original criteria, identities, or application order.
3. Check exact per-request schemas and stable request seeds through the real client/transport adapter. Preserve the inherited SDK internal-retry setting while testing and auditing the distinct SDK-invocation bound. Validate input tokens plus the 8,192-token output reservation against the fixed context at every step.
4. Exercise multiple panels with a late failure and show that no batch award/history/resource mutation occurs. Exercise valid complete rankings through the existing final guard and native quota/award/F0 processing. An identical supplied validated order may support downstream processing parity; this does not test equivalence of sampled rankings.
5. Verify fresh v6 namespaces, preserved population/cohort generation, all eight planned cells/horizons, complete raw-audit binding, and refusal of incomplete or mixed-version family inputs.
6. Independently review the protocol, actual call-count/seed contract, P0 preservation, batch atomicity, and analysis admission rules. Retain review disposition and test evidence with the final freeze.

### Bounded technical probe

After CPU validation and independent review, the technical preflight against the production server must exercise the same sequential algorithm and actual pinned inference settings before scientific clients start. Use the immutable 25-application fixture, its actual program ID and panel metadata, year 0, and the exact seed contract above with `"technical_probe"` replacing `"production"`. Preserve the fixture's original compact-prompt substantive prefix and replace its response-format tail using the shared sequential helper. Steps are 0,...,24 and attempt is 0.

A successful probe makes exactly one explicit HTTP POST at each of the 25 steps, including singleton: 25 successful HTTP requests in total. It uses standard-library `urllib`, with no OpenAI SDK, automatic HTTP retries, proxy, or redirect operation. There is no second semantic attempt, accepted-prefix reset, skipped singleton, or forced tail. Any failed selection aborts the probe and blocks scientific launch. Production's limit of 75 SDK invocations per full panel and its five internal SDK retries remain separate; neither retry layer is exercised by the probe.

Shared native production message, per-request schema, and context builders establish the corresponding payload-construction contract, including the declared technical namespace. This is payload equivalence, not SDK-stack or internal-retry parity.

The probe limits are fixed: at most 600 seconds per HTTP request and at most 15,000 seconds aggregate, **including tokenizer startup and token counting**. Tokenizer startup is bounded by 120 seconds and each token count by 30 seconds, each further capped by the remaining aggregate budget. `SIGALRM` enforces wall-clock deadlines; each HTTP operation also uses a socket timeout no greater than 600 seconds or the remaining aggregate budget. Each selection requests at most 8,192 output tokens, and its exact input plus output reservation must fit the fixed context. These are enforced timeout ceilings and token limits, not elapsed-time or usage predictions. Audit the limits actually enforced, all step/schema/seed bindings, raw responses, and terminal probe status; do not describe a partial prefix or a generic JSON success as a complete sequential probe.

Transport/preflight outputs are technical evidence, not additional worlds or candidates from which to choose favorable outcomes. Passing fixtures does not guarantee every future panel or world will complete.

Do not select probabilities, interventions, horizons, rankings, or retry policies from scientific outcome signs. The parent records completed validation and the reviewed freeze before launch; no GPU launch, commit, remote synchronization, or manuscript integration is authorized by this document alone.

## 8. Third-family v3 adoption after compact-ranking failure, 19 September 2026

This is a prospective, failure-informed adoption of the V6 sequential algorithm for **one new whole four-cell propensity replacement**, not externally registered. The inherited text remains a record of the first-eight V6 amendment; this section supersedes its third-family continuation statement for the new v3 jobs. It does not alter the frozen V6 source, protocol, queue, or results.

The copied V6-origin document had SHA256 `ed278eabd477542a55393a649678932c23b62ce03cca6cd16a39c9e930cdeb4f`; the reviewed algorithm's frozen source is `34112b33eb6ba37ff2a8bf6eff7cceddc704d508f`. V6 review/validation supports the algorithm, not completed validation of its integration into the third driver.

### Failure evidence and fresh realization

The old third V2 queue emitted `queue_failed` at 11:40:31.767863 UTC after LN's year-one `NSF_SYSTEM` panel 6 failed compact ranking following three semantic validation attempts. Its final status was updated at 11:40:44.596820 UTC after cleanup. The final audit has 25 expected/returned entries, five valid entries, twenty rejected entries, and duplicate-ID/compact-validation errors. `missing_application_ids` is an exactly-once-coverage diagnostic; do not infer literal omissions. The terminal summary has 134 audited final panels, one invalid final panel, one failed batch, zero successful batches, and `completion_allowed=false`; no imputed/fallback ranking was admitted. LF was interrupted during year-two preparation with a year-one checkpoint, and HN/HF had not started. None completed ten years.

Use verified snapshot `heartbeat_20260919T115313Z_verified.json` (SHA256 `9d3e5a15d8489514e7d49d812ed62a1e00388e0fd636404a087fe647bbec46c3`) and the 37-file failure evidence inventory `propensity_v2_failure_evidence_20260919.json` (SHA256 `51f041cde4a969901c3099fd38299df8de88a5cdb86032efc9168482b4dde5eb`). The original uncorrected snapshot is preserved separately. Preserve all raw attempts and partial outcomes, and the complete old initializer. The old cache is retained but **not adopted**. No partial trajectory, checkpoint, or previously chosen funding prefix enters the new set.

Use protocol `james-propensity-v3-single-world-sequential` and fresh IDs `james_propensity_v3_n1200_y10_{initial,LN,LF,HN,HF}_seed42`. Generate one new treatment-blind technical initializer after the new freeze, then replay that same source-bound immutable cache into all four new worlds. Each world has seed 42, 1,200 university founders, ten years, and the original p×near/far intervention and estimands from `docs/james_propensity_protocol.md`. This is a new matched realization; seed 42 does not promise the old initializer's LLM choices. Initialization is not a fifth scientific world.

### Exact algorithm adoption and admission

All four new conditions use `sequential_remaining_ids_v1` exactly as specified above: the original condition-specific substantive prompt is retained, its response-format tail is replaced, every remaining-ID selection including singleton is model-generated and validated, and the original full-permutation/batch-atomicity guards remain. Preserve the production request context and zero-based step, SDK call and internal-retry bounds, and the separate SDK-free technical-probe route/deadlines. The propensity cells have normal publication-record display and award credits; adopting this helper does not introduce P0/F0, influx, different project durations, or other first-eight driver settings.

Require the initializer's sequential-audit expected horizon **`num_years=0`**: no native funding years, agency registrations, panels, or sequential funding calls. Still run the exact join against its finalized independent request audit so a zero-count summary cannot hide any funding request. Native `Simulation` retains `YEARS=10` and exits at the initial phase-one boundary; the zero audit horizon does not eliminate initial-choice generation.

Each scientific world requires ten completed native funding years and twenty agency-year registrations—NSF and DARPA each year, including explicit zero-application registrations. Require exact raw sequential/request-audit correspondence, no unknown legacy funding requests, all expected native-year/ledger/submission/citation artifacts, and a true final completion gate. Do not replace missing agency coverage with empty records or interpret twenty registrations as twenty agencies/panels.

Use the shared read-only API `validate_funding_evidence(directory, manifest, *, expected_source)` for the initializer and each world, including analysis admission, rather than accepting summary flags without raw validation.

Both protocol Markdown files are fingerprinted in the new driver, initialization, controller, and analysis identities. Preserve strict complete source/cache matching and common pre/post initialization-state hashes for all four consumers. Stamp every world with `initialization_manifest_path` and `initialization_manifest_sha256` for the fresh producer and with `initial_choices_path` and `initial_choices_sha256` for its cache. Validate the producer's complete status, source/configuration identity, zero-funding evidence, and exact correspondence to every consumer. New manifests and admission checks must reflect the sequential method honestly; no predecessor metadata is rewritten to satisfy them.

### Interpretation and release boundary

The more numerous, history-conditioned ranking calls can change funding and later scientific trajectories. There is no equivalence claim to compact joint ranking, legacy ranking, or the old partial worlds. Identical funding settings across the new families support only a shared-method description, not a pooled twelve-cell factorial, controlled cross-family effect comparisons, seed robustness, or world-level SE/CI/p-values.

The full four-cell replacement and initializer policy are fixed before replacement outcomes. No selective-cell restart, old-cache switch, trajectory adoption, or automatic further replacement is allowed. Preserve and report another failure; withhold the factorial contrast unless all four new conditions complete their original horizons and pass every raw-data/provenance gate.

Targeted actual-path integration tests, exact zero-/ten-year audit coverage, independent final review, and the technical preflight are required before launch. Do not change the frozen V6 worktree or use its active scientific servers for diagnostics. This amendment records the new study contract; it does not assert that the replacement integration, final freeze, launch, or outcomes are complete.

## 9. Single-GPU execution of the unstarted third family

The subsequent runtime amendment in `docs/james_propensity_protocol.md` replaces
the never-started v3 queue with fresh `james_propensity_v3tp1_...` operations.
Its common initializer and all four conditions use the explicit
`qwen3_32b_bf16_tp1_v1` profile: the same pinned Qwen3-32B BF16 model and 32,768
context with TP1, 16 active sequences, 0.94 memory utilization, eager execution,
chunked prefill, and 2,048 maximum batched tokens. TP2 equivalence is not claimed.

This changes no funding prompt, schema, selection algorithm, seed context,
output-token reservation, retry bound, probe deadline, or raw-audit admission
rule. Each newly owned single-GPU server still requires the complete nonce and
25-selection probe, including singleton, before any client. The shared runtime
profile and both protocol documents enter the new source freeze. Existing V6
sources and servers, the retired unstarted v3 queue, and all older failure
evidence remain preserved.
