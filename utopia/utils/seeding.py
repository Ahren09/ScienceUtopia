"""Deterministic seed derivation and process RNG configuration."""

import os
import random


def derive_seed(*components) -> int:
    """Deterministically derive a 31-bit seed from arbitrary components.

    Uses SHA-256 (never Python's randomized hash()) so derivation is stable
    across processes and runs. Scheme: derive_seed(run_seed, phase, year,
    batch_index, item_index, ...).
    """
    import hashlib

    key = "|".join(str(c) for c in components)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def set_seed(seed: int = 42, use_torch: bool = True):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid parallel tokenization races
    # single-threaded math libs to avoid nondeterministic reductions (optional but helpful)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # stricter CUDA determinism (PyTorch recommendation)

    if use_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False
                # Needed for deterministic cublas GEMMs (pick one)
                os.environ.setdefault(
                    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
                )  # or ":16:8"
                # helps with debugging op order; not required for determinism but useful
                os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

        except ImportError:
            print("Fail to import torch. Skipping torch seed setting")
