# Sequential funding selection

For each funding panel, ask the model for the next best remaining application ID.
The candidate text and evaluation criteria come from the existing funding prompt.
Each selected ID must belong to the remaining set. Previously selected IDs are
never reintroduced. At most three SDK attempts are allowed per selection.
A complete permutation must pass validation before native award processing.
Record requests, selections, ranking validation, and the actual source hashes.
Failure aborts the run instead of fabricating or repairing an award order.

This is sequential elicitation. It does not claim distributional equivalence to
joint ranking. Per-panel competition and quota rounding differ from global ranking.
