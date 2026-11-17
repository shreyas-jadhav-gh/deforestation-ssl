"""
forestwatch: a small Python toolkit around the deforestation-mapping pipeline.

The training scripts in src/ do the heavy work (Earth Engine downloads, PyTorch training on
Kaggle). This package holds the parts that are plain Python and can be tested on any computer:

    config    bands, label values, the study regions and the settings of one training run
    data      one image patch, a catalogue of patches, two patch loaders, nested label budgets
    metrics   pixel counts and probability histograms, and the scores computed from them
    results   the table of training runs and the comparisons made in the report
    jobs      jobs made of steps, and the runner that starts one stream of steps per GPU
    cli       the command line:  python -m forestwatch --help
"""

from .config import BANDS, CLEARED, FOREST, IGNORE, INITS, REGIONS, SCALE, Region, RunConfig
from .data import InMemoryLoader, LabelBudget, Patch, PatchCatalog, PatchLoader, SharedCacheLoader
from .jobs import Job, JobError, JobRunner, PartResult, Step
from .metrics import ConfusionCounts, ProbabilityHistogram
from .results import RunResults

__version__ = "1.1.0"

__all__ = [
    "BANDS", "CLEARED", "FOREST", "IGNORE", "INITS", "REGIONS", "SCALE", "Region", "RunConfig",
    "InMemoryLoader", "LabelBudget", "Patch", "PatchCatalog", "PatchLoader", "SharedCacheLoader",
    "Job", "JobError", "JobRunner", "PartResult", "Step",
    "ConfusionCounts", "ProbabilityHistogram", "RunResults",
]
