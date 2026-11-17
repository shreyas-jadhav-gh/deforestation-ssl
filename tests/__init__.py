"""
Unit tests. From the repository root, either of:

    python -m pytest              (nicer output; pip install pytest)
    python -m unittest -v         (needs nothing beyond the project's own requirements)

Most tests use small synthetic arrays. Some also use the files kept in the repository: the 16
sample patches in sample_data/ and the saved results in results/kaggle_*/. Tests that need
PyTorch are skipped when it is not installed.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SAMPLE_DATA = REPO / "sample_data"
RESULTS = {"full": REPO / "results" / "kaggle_full" / "results",
           "cd": REPO / "results" / "kaggle_cd" / "results",
           "lrsearch": REPO / "results" / "kaggle_lrsearch" / "results"}

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
