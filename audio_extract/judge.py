"""Learned-judge calibration + pairwise ranking (docs/v2 §4.1–4.6).

A learned judge (SAM Audio Judge primary; Music Flamingo / AF3 as *experimental*
annotators) may influence selection **only after** passing a local calibration
gate: A/B order stability, repeatability, injected-defect monotonicity, human
agreement, and no loudness-only bias. Until then it is a diagnostic annotation
source, never a selector.

The judge *inference* (SAJ / Music Flamingo) runs on the GPU box and is wired
through the ``Judge`` protocol; the ranker and the calibration gate here are pure
deterministic logic, tested against a metric-based mock judge and an adversarial
one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from . import metrics, panel_runner

# Promotion thresholds (§4.6). Explicitly provisional; a judge that fails stays
# an annotation source, not a selector.
PROMOTION = {
    "ab_order_consistency": 0.85,
    "repeated_pair_agreement": 0.80,
    "defect_monotonicity": 0.90,
    "human_agreement": 0.70,
    "max_loudness_bias": 0.10,  # |P(prefer louder) - 0.5|
}


@dataclass
class JudgeScore:
    recall: float = 0.0
    precision: float = 0.0
    faithfulness: float = 0.0
    overall: float = 0.0


@runtime_checkable
class Judge(Protocol):
    name: str

    def compare(self, a: np.ndarray, b: np.ndarray, sr: int, **ctx) -> str:
        """Return ``"a"`` or ``"b"`` — which candidate is the better accompaniment."""
        ...


# --------------------------------------------------------------------------
# reference judges
# --------------------------------------------------------------------------
class MetricJudge:
    """Deterministic judge over the objective metric bank. Gain-matches each
    candidate to the reference first (the two-copies rule), so it is
    loudness-robust — the behaviour a real judge must also show."""

    name = "metric-judge/v1"

    def __init__(self, reference: np.ndarray, sr: int, vocal_ref: np.ndarray | None = None,
                 axes=("leakage", "holes_db", "pump_depth_db", "brightness_deviation")):
        self.reference = reference
        self.sr = sr
        self.vocal_ref = vocal_ref
        self.axes = axes

    def cost(self, x: np.ndarray) -> float:
        xm = metrics.gain_match(x, self.reference)
        c = panel_runner.score_candidate(xm, self.sr, reference=self.reference, vocal_ref=self.vocal_ref)
        return float(sum(c[a] for a in self.axes))

    def compare(self, a: np.ndarray, b: np.ndarray, sr: int, **ctx) -> str:
        return "a" if self.cost(a) <= self.cost(b) else "b"


class NoisyJudge:
    """Adversarial control: agrees with truth only ``accuracy`` of the time and is
    order-dependent — should FAIL the calibration gate."""

    name = "noisy-judge"

    def __init__(self, truth: "MetricJudge", accuracy: float = 0.5, seed: int = 0):
        self.truth = truth
        self.accuracy = accuracy
        self._rng = np.random.default_rng(seed)

    def compare(self, a: np.ndarray, b: np.ndarray, sr: int, **ctx) -> str:
        honest = self.truth.compare(a, b, sr)
        # order-dependent + noisy: flip on a coin biased by (1 - accuracy)
        if self._rng.random() > self.accuracy:
            return "b" if honest == "a" else "a"
        return honest


# --------------------------------------------------------------------------
# pairwise ranking
# --------------------------------------------------------------------------
def bradley_terry(items: list[str], pairs: list[tuple[str, str]], iters: int = 200) -> dict[str, float]:
    """Bradley–Terry strengths via the MM algorithm. ``pairs`` are ``(winner, loser)``.
    Strengths are normalized to geometric mean 1."""
    idx = {c: i for i, c in enumerate(items)}
    n = len(items)
    wins = np.zeros(n)
    counts = np.zeros((n, n))
    for win, lose in pairs:
        i, j = idx[win], idx[lose]
        wins[i] += 1
        counts[i, j] += 1
        counts[j, i] += 1
    p = np.ones(n)
    for _ in range(iters):
        newp = p.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if counts[i, j] > 0:
                    denom += counts[i, j] / (p[i] + p[j])
            if denom > 0 and wins[i] > 0:
                newp[i] = wins[i] / denom
        newp = np.maximum(newp, 1e-12)
        newp /= np.exp(np.mean(np.log(newp)))  # normalize geometric mean to 1
        if np.allclose(newp, p, rtol=1e-9):
            p = newp
            break
        p = newp
    return {c: float(p[idx[c]]) for c in items}


def fit_logistic_ranker(feature_map: dict[str, np.ndarray], pairs: list[tuple[str, str]],
                        iters: int = 800, lr: float = 0.2, l2: float = 1e-3) -> np.ndarray:
    """Fit ``w`` in ``P(A>B)=σ(wᵀ(f_A−f_B))`` from ``(winner, loser)`` pairs.
    Model names are never features (docs/v2 §4.5)."""
    diffs = [feature_map[win] - feature_map[lose] for win, lose in pairs]
    X = np.array(diffs, dtype=np.float64)
    X = np.vstack([X, -X])  # symmetric negatives
    y = np.concatenate([np.ones(len(diffs)), np.zeros(len(diffs))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) / len(X) + l2 * w
        w -= lr * grad
    return w


def prob_a_beats_b(w: np.ndarray, fa: np.ndarray, fb: np.ndarray) -> float:
    return float(1.0 / (1.0 + np.exp(-(w @ (fa - fb)))))


# --------------------------------------------------------------------------
# calibration gate (§4.6)
# --------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    metrics: dict = field(default_factory=dict)
    passed: bool = False
    verdict: str = "annotation_only"


def _consistent(w1: str, w2: str) -> bool:
    # w1 from compare(a,b); w2 from compare(b,a). Same real winner => opposite labels.
    return (w1, w2) in (("a", "b"), ("b", "a"))


def ab_order_consistency(judge: Judge, pairs, sr: int) -> float:
    ok = 0
    for a, b in pairs:
        if _consistent(judge.compare(a, b, sr), judge.compare(b, a, sr)):
            ok += 1
    return ok / len(pairs) if pairs else 0.0


def repeated_pair_agreement(judge: Judge, pairs, sr: int) -> float:
    ok = sum(1 for a, b in pairs if judge.compare(a, b, sr) == judge.compare(a, b, sr))
    return ok / len(pairs) if pairs else 0.0


def defect_monotonicity(judge: Judge, ladders, sr: int) -> float:
    """Each ladder is ordered best→worst; the judge should prefer the earlier
    (less-damaged) item in every adjacent pair."""
    ok = tot = 0
    for ladder in ladders:
        for better, worse in zip(ladder, ladder[1:]):
            tot += 1
            if judge.compare(better, worse, sr) == "a":
                ok += 1
    return ok / tot if tot else 0.0


def human_agreement(judge: Judge, labeled, sr: int) -> float:
    """``labeled`` = list of ``(a, b, human_winner)`` with human_winner in {"a","b"}."""
    ok = sum(1 for a, b, hw in labeled if judge.compare(a, b, sr) == hw)
    return ok / len(labeled) if labeled else 0.0


def loudness_bias(judge: Judge, clips, sr: int, gain_db: float = 6.0) -> float:
    """Present each clip against a louder copy of itself, in *both* orders, so a
    deterministic tie-break toward the first argument does not masquerade as
    loudness preference. Returns ``|P(prefer louder) - 0.5|``; a good judge ~0."""
    g = 10 ** (gain_db / 20.0)
    louder, tot = 0, 0
    for x in clips:
        loud = x * g
        if judge.compare(x, loud, sr) == "b":  # louder is in slot b
            louder += 1
        if judge.compare(loud, x, sr) == "a":  # louder is in slot a
            louder += 1
        tot += 2
    return abs(louder / tot - 0.5) if tot else 0.0


def calibrate(judge: Judge, sr: int, *, ab_pairs, repeat_pairs, ladders, human_labeled, clips) -> CalibrationReport:
    m = {
        "ab_order_consistency": ab_order_consistency(judge, ab_pairs, sr),
        "repeated_pair_agreement": repeated_pair_agreement(judge, repeat_pairs, sr),
        "defect_monotonicity": defect_monotonicity(judge, ladders, sr),
        "human_agreement": human_agreement(judge, human_labeled, sr),
        "loudness_bias": loudness_bias(judge, clips, sr),
    }
    passed = (
        m["ab_order_consistency"] >= PROMOTION["ab_order_consistency"]
        and m["repeated_pair_agreement"] >= PROMOTION["repeated_pair_agreement"]
        and m["defect_monotonicity"] >= PROMOTION["defect_monotonicity"]
        and m["human_agreement"] >= PROMOTION["human_agreement"]
        and m["loudness_bias"] <= PROMOTION["max_loudness_bias"]
    )
    return CalibrationReport(metrics=m, passed=passed,
                             verdict="selector" if passed else "annotation_only")
