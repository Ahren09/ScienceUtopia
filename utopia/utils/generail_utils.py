
import os
import random
from datetime import datetime

from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm
import json
import argparse
import torch

import logging
import warnings
import pandas as pd


def check_cwd():
    """
    Checks if the current working directory is the repository root directory (ScienceUtopia/)
    """
    basename = os.path.basename(os.path.normpath(os.getcwd()))
    assert basename.lower() in [
        "scienceutopia"], "Please run this file from the repository root directory (ScienceUtopia/)"


def derive_seed(*components) -> int:
    """Deterministically derive a 31-bit seed from arbitrary components.

    Uses SHA-256 (never Python's randomized hash()) so derivation is stable
    across processes and runs. Scheme: derive_seed(run_seed, phase, year,
    batch_index, item_index, ...).
    """
    import hashlib
    key = "|".join(str(c) for c in components)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def config_hash(config: dict) -> str:
    """SHA-256 hex digest of a config dict (sorted-key JSON, tuples->lists)."""
    import hashlib
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def write_run_manifest(docs_dir: str, status: str, args=None, resolved_config: dict = None,
                       extra: dict = None) -> str:
    """Write/update outputs/docs/<experiment_id>/run_manifest.json.

    Called at run start (status='running') and end (status='complete'/'failed').
    Merges into an existing manifest so start-time fields survive the final write.
    """
    import subprocess
    import sys

    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "run_manifest.json")
    manifest = {}
    if os.path.exists(path):
        with open(path) as f:
            manifest = json.load(f)

    def _git(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    manifest.update({
        "status": status,
        f"timestamp_{status}": datetime.now().isoformat(),
    })
    if "command" not in manifest:
        manifest.update({
            "command": " ".join(sys.argv),
            "git_commit": _git(["git", "rev-parse", "HEAD"]),
            "git_dirty_files": _git(["git", "status", "--short"]),
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        try:
            import torch, vllm, transformers, sentence_transformers
            manifest["package_versions"] = {
                "torch": torch.__version__, "vllm": vllm.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
            }
        except ImportError:
            pass
    if args is not None:
        manifest["args"] = {k: v for k, v in sorted(vars(args).items())}
    if resolved_config is not None:
        manifest["resolved_config"] = resolved_config
        manifest["scientific_config_hash"] = config_hash(resolved_config)
    if extra:
        # Resume-safe: llm_call_stats accumulate across restarts instead of
        # being overwritten by a short resume run's counters.
        new_stats = extra.get("llm_call_stats")
        old_stats = manifest.get("llm_call_stats")
        if isinstance(new_stats, dict) and isinstance(old_stats, dict):
            extra = dict(extra)
            extra["llm_call_stats"] = {
                k: (old_stats.get(k, 0) or 0) + (new_stats.get(k, 0) or 0)
                for k in set(old_stats) | set(new_stats)}
        manifest.update(extra)

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def set_seed(seed: int = 42, use_torch: bool = True):
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
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # or ":16:8"
                # helps with debugging op order; not required for determinism but useful
                os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

        except ImportError:
            print("Fail to import torch. Skipping torch seed setting")



def print_colored(text, color):
    # Define ANSI color codes
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m"  # Reset to default color
    }

    # Get the color code from the dictionary (default to reset if not found)
    color_code = colors.get(color.lower(), colors["reset"])

    # Print the colored text
    print(f"{color_code}{text}{colors['reset']}")
    
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        log_fmt = '%(levelname)s | %(asctime)s | %(name)s | %(message)s'
        formatter = logging.Formatter(log_fmt, datefmt='%m-%d %H:%M:%S')
        log_msg = formatter.format(record)

        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{log_msg}{self.RESET}"


def project_setup():
    check_cwd()
    
    
    # Create handler and set formatter
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s | %(asctime)s | %(filename)s:%(lineno)d - %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=[handler]
    )

    for package_name in ["urllib3", "httpx", "httpcore", "openai", "langchain", "matplotlib", "PIL", "h5py"]:
        logging.getLogger(package_name).setLevel(logging.ERROR)
        
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 20)
    set_seed(42)
    # Set logging level for the specific logger 'bm25s' to WARNING
    logging.getLogger('bm25s').setLevel(logging.WARNING)

