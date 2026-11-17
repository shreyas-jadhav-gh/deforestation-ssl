"""
Scores for the 'cleared' class.

ConfusionCounts holds the four pixel counts at one threshold. ProbabilityHistogram holds, for
truly forest and truly cleared pixels, how many got each predicted probability; from it the
counts, and so every score, follow at any threshold without keeping the predictions. This is
how the training script stores each evaluation (results/.../hists/<run>.npz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CLEARED, FOREST, IGNORE


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


@dataclass(frozen=True)
class ConfusionCounts:
    """Pixel counts for the 'cleared' class: true and false positives and negatives.
    Pixels labelled IGNORE are never counted. Counts from several patches can be added."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __post_init__(self):
        if min(self.tp, self.fp, self.fn, self.tn) < 0:
            raise ValueError("pixel counts cannot be negative")

    @classmethod
    def from_masks(cls, predicted, label, ignore: int = IGNORE) -> "ConfusionCounts":
        """From a boolean map of pixels predicted 'cleared' and the label map."""
        predicted = np.asarray(predicted, dtype=bool)
        label = np.asarray(label)
        if predicted.shape != label.shape:
            raise ValueError(f"prediction {predicted.shape} and label {label.shape} differ in shape")
        scored = label != ignore
        truth = (label == CLEARED) & scored
        pred = predicted & scored
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        fn = int(np.sum(~pred & truth))
        return cls(tp, fp, fn, int(np.sum(scored)) - tp - fp - fn)

    @classmethod
    def from_probabilities(cls, prob, label, threshold: float = 0.5) -> "ConfusionCounts":
        return cls.from_masks(np.asarray(prob) >= threshold, label)

    def __add__(self, other):
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        return ConfusionCounts(self.tp + other.tp, self.fp + other.fp,
                               self.fn + other.fn, self.tn + other.tn)

    def __radd__(self, other):      # lets sum() start from 0
        return self if other == 0 else self.__add__(other)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def iou(self) -> float:
        """Intersection over union of predicted and true cleared areas."""
        return _ratio(self.tp, self.tp + self.fp + self.fn)

    @property
    def iou_forest(self) -> float:
        return _ratio(self.tn, self.tn + self.fp + self.fn)

    @property
    def miou(self) -> float:
        return (self.iou + self.iou_forest) / 2

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.total)

    @property
    def predicted_share(self) -> float:
        """Share of the scored area that the model calls cleared."""
        return _ratio(self.tp + self.fp, self.total)

    @property
    def true_share(self) -> float:
        """Share of the scored area that really was cleared."""
        return _ratio(self.tp + self.fn, self.total)

    def as_dict(self) -> dict[str, float]:
        return {"iou": self.iou, "f1": self.f1, "precision": self.precision,
                "recall": self.recall, "miou": self.miou, "accuracy": self.accuracy}


class ProbabilityHistogram:
    """counts[0, k] = forest pixels and counts[1, k] = cleared pixels whose predicted probability
    of 'cleared' fell in bin k, i.e. in [k / bins, (k + 1) / bins). Predicting 'cleared' when the
    probability is at least k / bins therefore marks exactly the pixels in bins k and above."""

    def __init__(self, counts=None, bins: int = 1000):
        if counts is None:
            counts = np.zeros((2, bins), dtype=np.int64)
        counts = np.asarray(counts)
        if counts.ndim != 2 or counts.shape[0] != 2 or counts.shape[1] < 1:
            raise ValueError(f"histogram must have shape (2, bins), got {counts.shape}")
        if np.any(counts < 0):
            raise ValueError("histogram counts cannot be negative")
        self.counts = counts.astype(np.int64)

    @classmethod
    def from_npz(cls, path, eval_set: str) -> "ProbabilityHistogram":
        """One evaluation set (e.g. 'amazon_test') of a saved run histogram file."""
        with np.load(path) as d:
            if eval_set not in d.files:
                raise KeyError(f"{path} has no histogram {eval_set!r} (has {d.files})")
            return cls(d[eval_set])

    @property
    def bins(self) -> int:
        return self.counts.shape[1]

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def __repr__(self) -> str:
        return f"ProbabilityHistogram(bins={self.bins}, pixels={self.total})"

    def __eq__(self, other) -> bool:
        return isinstance(other, ProbabilityHistogram) and np.array_equal(self.counts, other.counts)

    def __add__(self, other):
        if not isinstance(other, ProbabilityHistogram):
            return NotImplemented
        if other.bins != self.bins:
            raise ValueError("cannot add histograms with different numbers of bins")
        return ProbabilityHistogram(self.counts + other.counts)

    def add(self, prob, label, ignore: int = IGNORE) -> "ProbabilityHistogram":
        """Count one batch of predictions (any shape) against its labels; returns self."""
        prob = np.asarray(prob, dtype=np.float64)
        label = np.asarray(label)
        if prob.shape != label.shape:
            raise ValueError(f"probabilities {prob.shape} and labels {label.shape} differ in shape")
        k = np.clip((prob * self.bins).astype(np.int64), 0, self.bins - 1)
        scored = label != ignore
        for row, value in ((0, FOREST), (1, CLEARED)):
            self.counts[row] += np.bincount(k[scored & (label == value)], minlength=self.bins)
        return self

    def _bin(self, threshold: float) -> int:
        k = int(round(threshold * self.bins))
        if not 0 <= k <= self.bins or abs(k - threshold * self.bins) > 1e-6:
            raise ValueError(f"threshold {threshold} is not a bin edge of this histogram")
        return k

    def _cumulative(self):
        """tp, fp, fn, tn for the thresholds k / bins, k = 0 ... bins (one array each)."""
        neg, pos = self.counts[0].astype(np.float64), self.counts[1].astype(np.float64)
        tp = np.append(pos[::-1].cumsum()[::-1], 0.0)
        fp = np.append(neg[::-1].cumsum()[::-1], 0.0)
        return tp, fp, pos.sum() - tp, neg.sum() - fp

    def counts_at(self, threshold: float) -> ConfusionCounts:
        tp, fp, fn, tn = (int(round(v[self._bin(threshold)])) for v in self._cumulative())
        return ConfusionCounts(tp, fp, fn, tn)

    def iou_curve(self) -> np.ndarray:
        tp, fp, fn, _ = self._cumulative()
        return np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)

    def best_threshold(self) -> float:
        """The threshold (a bin edge) with the highest IoU; the lowest one if several tie."""
        return int(np.argmax(self.iou_curve()[: self.bins])) / self.bins

    def average_precision(self) -> float:
        """Area under the precision-recall curve, summed step-wise from the strictest threshold."""
        tp, fp, fn, _ = self._cumulative()
        n_pos = tp[0] + fn[0]
        if n_pos == 0:
            return float("nan")
        tp, fp = tp[: self.bins][::-1], fp[: self.bins][::-1]
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = tp / n_pos
        return float(np.sum(np.diff(recall, prepend=0.0) * precision))

    def metrics_at(self, threshold: float) -> dict[str, float]:
        c = self.counts_at(threshold)
        return {**c.as_dict(), "pred_cleared_share": (c.tp + c.fp) / max(c.total, 1)}
