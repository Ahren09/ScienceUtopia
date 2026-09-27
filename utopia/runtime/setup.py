"""Initialize the simulation command-line environment."""

import logging
import warnings

from utopia.utils.paths import check_cwd
from utopia.utils.logging import configure_logging
from utopia.utils.seeding import set_seed


def project_setup():
    import pandas as pd

    check_cwd()

    configure_logging()

    # Configure application-wide warnings and display
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.max_columns", 20)
    set_seed(42)
    # Set logging level for the specific logger 'bm25s' to WARNING
    logging.getLogger("bm25s").setLevel(logging.WARNING)
