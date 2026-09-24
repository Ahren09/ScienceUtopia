The original `test_scale_expansion.py` referenced an ignored, untracked JSON
fixture. These prompt hashes were recovered on 2026-09-24 by executing that
test's `golden_inputs()` and five prompt calls in a clean detached checkout of
`2bb7e6917418f2722b2217cc5453188f51734662`, before the cost-control changes.
They were not generated from the implementation being tested.
