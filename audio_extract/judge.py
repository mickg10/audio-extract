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
        return float(sum(c[a] for a in self.axes if a in c))

    def compare(self, a: np.ndarray, b: np.ndarray, sr: int, **ctx) -> str:
        ca, cb = self.cost(a), self.cost(b)
        # §12.4: sub-epsilon differences are a TIE, not a forced choice — otherwise
        # float noise masquerades as (e.g.) loudness preference.
        if abs(ca - cb) <= 1e-6 * max(1.0, abs(ca), abs(cb)):
            return "tie"
        return "a" if ca < cb else "b"


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
    # w1 from compare(a,b); w2 from compare(b,a). Same real winner => opposite
    # labels; a stable tie/uncertain in both orders is also consistent.
    return (w1, w2) in (("a", "b"), ("b", "a"), ("tie", "tie"), ("uncertain", "uncertain"))


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
    louder = decided = 0
    for x in clips:
        loud = x * g
        for verdict, louder_slot in ((judge.compare(x, loud, sr), "b"),
                                     (judge.compare(loud, x, sr), "a")):
            if verdict in ("tie", "uncertain"):
                continue  # an honest tie is NOT a loudness preference
            decided += 1
            if verdict == louder_slot:
                louder += 1
    return abs(louder / decided - 0.5) if decided else 0.0


# ---------------------------------------------------------------------------
# WP7 (v2.1 §12): autonomous calibration — no required human agreement.
# Labels come from exact-target challenge errors, not people. Judges may answer
# "a" | "b" | "tie" | "uncertain"; forced choices hide weakness.
# ---------------------------------------------------------------------------
PROMOTION_V2 = {
    "min_total_pairs": 200,
    "min_pairs_per_defect": 25,
    "order_consistency_min": 0.98,
    "repeatability_min": 0.98,
    "synthetic_pair_accuracy_min": 0.90,
    "track_remix_pair_accuracy_min": 0.85,
    "monotonicity_each_defect_min": 0.95,
    "loudness_bias_max": 0.05,
}


@dataclass
class AutonomousCalibration:
    judge_id: str
    calibration_id: str
    order_consistency: float
    repeatability: float
    gain_invariance: float
    synthetic_pair_accuracy: float
    track_remix_pair_accuracy: float
    monotonicity_by_defect: dict
    loudness_bias: float
    ood_coverage: float
    passed_axes: set
    n_pairs: int
    passed: bool
    verdict: str


def _pair_accuracy(judge: Judge, labeled_pairs, sr: int) -> tuple[float, int]:
    """``labeled_pairs`` = (a, b, true_winner) with truth from EXACT-target error
    (lower error wins). 'tie'/'uncertain' answers count as misses (conservative)."""
    if not labeled_pairs:
        return 0.0, 0
    ok = sum(1 for a, b, w in labeled_pairs if judge.compare(a, b, sr) == w)
    return ok / len(labeled_pairs), len(labeled_pairs)


def calibrate_autonomous(judge: Judge, sr: int, *, judge_id: str, calibration_id: str,
                         exact_pairs, remix_pairs, ladders_by_defect: dict,
                         clips, thresholds: dict = PROMOTION_V2) -> AutonomousCalibration:
    """Promotion gate with zero human labels (§12). ``exact_pairs``/``remix_pairs``
    carry truth from exact-target errors; ``ladders_by_defect`` maps a defect class
    to a best→worst ladder; promotion requires EVERY defect class to pass its
    monotonicity floor (excellence in one class cannot hide failure in another)."""
    syn_acc, n_syn = _pair_accuracy(judge, exact_pairs, sr)
    remix_acc, n_remix = _pair_accuracy(judge, remix_pairs, sr)
    mono = {defect: defect_monotonicity(judge, [ladder], sr)
            for defect, ladder in ladders_by_defect.items()}
    order = ab_order_consistency(judge, [(a, b) for a, b, _ in exact_pairs], sr)
    repeat = repeated_pair_agreement(judge, [(a, b) for a, b, _ in exact_pairs], sr)
    bias = loudness_bias(judge, clips, sr)
    # gain invariance: the verdict must survive an identical gain on both sides
    g = 10 ** (3.0 / 20.0)
    gain_ok = sum(1 for a, b, _ in exact_pairs
                  if judge.compare(a, b, sr) == judge.compare(a * g, b * g, sr))
    gain_inv = gain_ok / len(exact_pairs) if exact_pairs else 0.0
    # OOD coverage: fraction of noise-vs-noise comparisons answered tie/uncertain
    rng_flip = 0
    for x in clips:
        noise = (x[::-1] if hasattr(x, "__getitem__") else x)
        if judge.compare(noise, noise, sr) in ("tie", "uncertain"):
            rng_flip += 1
    ood = rng_flip / len(clips) if clips else 0.0

    n_pairs = n_syn + n_remix
    passed_axes = {d for d, v in mono.items()
                   if v >= thresholds["monotonicity_each_defect_min"]}
    passed = (
        n_pairs >= thresholds["min_total_pairs"]
        and order >= thresholds["order_consistency_min"]
        and repeat >= thresholds["repeatability_min"]
        and syn_acc >= thresholds["synthetic_pair_accuracy_min"]
        and remix_acc >= thresholds["track_remix_pair_accuracy_min"]
        and len(passed_axes) == len(ladders_by_defect)
        and bias <= thresholds["loudness_bias_max"]
    )
    return AutonomousCalibration(
        judge_id=judge_id, calibration_id=calibration_id,
        order_consistency=order, repeatability=repeat, gain_invariance=gain_inv,
        synthetic_pair_accuracy=syn_acc, track_remix_pair_accuracy=remix_acc,
        monotonicity_by_defect=mono, loudness_bias=bias, ood_coverage=ood,
        passed_axes=passed_axes, n_pairs=n_pairs, passed=passed,
        verdict="selector" if passed else "annotation_only",
    )


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
