
import os
import random
from datetime import datetime

from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm
import json
import argparse
import torch

def check_cwd():
    """
    Checks if the current working directory is the repository root directory (ScienceUtopia/)
    """
    basename = os.path.basename(os.path.normpath(os.getcwd()))
    assert basename.lower() in [
        "scienceutopia"], "Please run this file from the repository root directory (ScienceUtopia/)"


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


def project_setup():
    check_cwd()
    import logging
    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 20)
    set_seed(42)
    # Set logging level for the specific logger 'bm25s' to WARNING
    logging.getLogger('bm25s').setLevel(logging.WARNING)

