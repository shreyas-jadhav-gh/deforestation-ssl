"""
The table of training runs (one row per run, as written by the training script) and the
comparisons the report makes from it: means over seeds, label-efficiency tables, the label
fraction needed to reach a score, and the learning rate picked by a search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RunConfig

KEYS = ("tag", "init", "input", "lr", "frac", "seed")


class RunResults:
    """All runs of one or more results CSVs. Filtering returns a new, smaller RunResults."""

    def __init__(self, table: pd.DataFrame):
        table = table.copy()
        if "input" not in table:
            table["input"] = "post"
        table["input"] = table["input"].fillna("post")
        missing = [k for k in KEYS if k not in table.columns]
        if missing:
            raise ValueError(f"results table is missing column(s) {missing}")
        self.table = table.reset_index(drop=True)

    @classmethod
    def from_csv(cls, *paths) -> "RunResults":
        if not paths:
            raise ValueError("no results files given")
        return cls(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))

    @classmethod
    def from_folders(cls, *folders) -> "RunResults":
        """Every <job>_all.csv in the given results folders."""
        files = sorted(f for folder in folders for f in Path(folder).glob("*_all.csv"))
        if not files:
            raise FileNotFoundError(f"no *_all.csv files in {[str(f) for f in folders]}")
        return cls.from_csv(*files)

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"RunResults({len(self)} runs)"

    def filter(self, **conditions) -> "RunResults":
        """Keep runs matching every condition, e.g. filter(tag="full", frac=[0.05, 1.0]).
        A list or tuple means 'any of these'; floats are compared with a small tolerance."""
        rows = self.table
        for column, wanted in conditions.items():
            if column not in rows:
                raise KeyError(f"no column {column!r} in the results")
            values = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
            col = rows[column]
            if pd.api.types.is_float_dtype(col):
                mask = np.zeros(len(rows), dtype=bool)
                for v in values:
                    mask |= np.isclose(col.to_numpy(), float(v), rtol=1e-9, atol=1e-12)
            else:
                mask = col.isin(values).to_numpy()
            rows = rows[mask]
        return RunResults(rows)

    @property
    def inits(self) -> list[str]:
        return sorted(self.table.init.unique())

    @property
    def fractions(self) -> list[float]:
        return sorted(self.table.frac.unique())

    def _metric(self, metric: str) -> str:
        if metric not in self.table:
            raise KeyError(f"no metric {metric!r} in the results")
        return metric

    def summary(self, metric: str, by=("init", "frac")) -> pd.DataFrame:
        """Mean, standard deviation and number of runs of `metric` per group."""
        return (self.table.groupby(list(by))[self._metric(metric)]
                .agg(mean="mean", std="std", runs="size"))

    def label_efficiency(self, metric: str) -> pd.DataFrame:
        """Mean of `metric` for every starting point (rows) and label fraction (columns)."""
        return self.table.pivot_table(index="init", columns="frac",
                                      values=self._metric(metric), aggfunc="mean")

    def labels_needed(self, init: str, target: float, metric: str) -> Optional[float]:
        """Smallest label fraction at which `init` reaches a mean `metric` of at least
        `target`, or None if it never does."""
        means = self.filter(init=init).summary(metric, by=("frac",))["mean"]
        reached = means[means >= target]
        return float(reached.index.min()) if len(reached) else None

    def best_learning_rates(self, metric: str = "amazon_val_iou_t") -> dict[str, float]:
        """For a learning-rate search: per starting point, the rate with the highest mean
        `metric` over the searched fractions (the rule fixed before the search ran)."""
        means = self.table.groupby(["init", "lr"])[self._metric(metric)].mean()
        return {init: float(s.droplevel("init").idxmax()) for init, s in means.groupby(level="init")}

    def run(self, config: RunConfig) -> pd.Series:
        """The one row of the run described by `config`."""
        rows = self.filter(tag=config.tag, init=config.init, input=config.input,
                           lr=config.lr, frac=config.frac, seed=config.seed).table
        if len(rows) != 1:
            raise KeyError(f"expected one run called {config.name}, found {len(rows)}")
        return rows.iloc[0]
