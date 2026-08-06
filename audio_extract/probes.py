"""Active automated probes (v2.1 WP9 / §14) — the autonomous replacement for
"ask a human": when two candidates are close, run the experiment most likely to
separate them.

``recommend_probe`` scores every registered probe by

    probe_value = rank_flip_probability × expected_uncertainty_reduction / cost

and the **deterministic fallback planner** guarantees a next step even with no
DeepSeek/Pi available — the system must never stall on LLM availability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROBE_VOCABULARY = (
    "run_track_remix_challenge",
    "run_no_vocal_theft_assay",
    "run_vocal_only_bleed_assay",
    "run_vocal_intervention",
    "run_orchestra_intervention",
    "change_overlap",
    "run_alternate_construction",
    "run_alternate_model_family",
    "build_weighted_ensemble",
    "run_additional_judge",
)

# Which probes discriminate which defect axis, with a normalized compute cost and
# an expected uncertainty reduction on that axis (seed values; recalibrated later).
PROBE_PROFILES: dict[str, dict] = {
    "run_track_remix_challenge": {"axes": {"event_hole": 0.6, "fullness": 0.5, "vocal_leakage": 0.4}, "cost": 1.0},
    "run_no_vocal_theft_assay":  {"axes": {"orchestral_theft": 0.7}, "cost": 0.6},
    "run_vocal_only_bleed_assay": {"axes": {"vocal_leakage": 0.6}, "cost": 0.5},
    "run_vocal_intervention":    {"axes": {"vocal_leakage": 0.5, "event_hole": 0.4}, "cost": 0.8},
    "run_orchestra_intervention": {"axes": {"orchestral_theft": 0.6, "brightness": 0.4,
                                            "transients": 0.4, "stereo": 0.3}, "cost": 0.8},
    "change_overlap":            {"axes": {"event_hole": 0.2, "fullness": 0.2}, "cost": 0.4},
    "run_alternate_construction": {"axes": {"event_hole": 0.3, "vocal_leakage": 0.3}, "cost": 0.7},
    "run_alternate_model_family": {"axes": {"event_hole": 0.4, "fullness": 0.4, "brightness": 0.3}, "cost": 1.5},
    "build_weighted_ensemble":   {"axes": {"event_hole": 0.3, "fullness": 0.3}, "cost": 1.2},
    "run_additional_judge":      {"axes": {"generic_quality": 0.6}, "cost": 0.5},
}

# §14.5 deterministic fallback: uncertain axis -> probe, no LLM required.
FALLBACK_RULES: dict[str, str] = {
    "event_hole": "run_track_remix_challenge",       # uncertain pumping -> forte-vocal remix
    "brightness": "run_track_remix_challenge",        # high-string/brass control remix
    "hall": "run_track_remix_challenge",              # long-tail control challenge
    "vocal_leakage": "run_vocal_only_bleed_assay",    # sparse-overlay leakage check
    "orchestral_theft": "run_no_vocal_theft_assay",
    "fullness": "run_track_remix_challenge",
    "generic_quality": "run_additional_judge",        # SAJ + instruction judge repeat
}


@dataclass
class ProbeRecommendation:
    probe: str
    axis: str
    value: float
    rank_flip_probability: float
    uncertainty_reduction: float
    cost: float
    reason: str


def _overlap_and_flip(a: dict, b: dict, axis: str) -> tuple[float, float]:
    """Overlapping-uncertainty width and a crude rank-flip probability for one
    axis, from per-axis (mean, lower, upper) bounds dicts."""
    am, al, au = a["mean"].get(axis, 0.0), a["lower"].get(axis, 0.0), a["upper"].get(axis, 0.0)
    bm, bl, bu = b["mean"].get(axis, 0.0), b["lower"].get(axis, 0.0), b["upper"].get(axis, 0.0)
    overlap = max(0.0, min(au, bu) - max(al, bl))
    spread = max(au - al, bu - bl, 1e-9)
    gap = abs(am - bm)
    flip = float(np.clip(0.5 * (1.0 - gap / (spread + gap + 1e-9)) + 0.5 * (overlap / spread), 0.0, 1.0))
    return overlap, flip


def recommend_probe(cand_a: dict, cand_b: dict, *, executed: set[str] | None = None,
                    profiles: dict = PROBE_PROFILES) -> ProbeRecommendation:
    """Pick the highest-value probe to separate the two top candidates.

    ``cand_a``/``cand_b``: {"mean": {axis: v}, "lower": {...}, "upper": {...}}.
    ``executed``: probe names already run for this pair (deduplicated).
    Always returns a recommendation (deterministic fallback included).
    """
    executed = executed or set()
    axes = sorted(set(cand_a["mean"]) | set(cand_b["mean"]))
    best: ProbeRecommendation | None = None
    for axis in axes:
        overlap, flip = _overlap_and_flip(cand_a, cand_b, axis)
        if overlap <= 0 and flip < 0.05:
            continue
        for probe, prof in profiles.items():
            if probe in executed:
                continue
            red = prof["axes"].get(axis)
            if red is None:
                continue
            value = flip * red / prof["cost"]
            if best is None or value > best.value:
                best = ProbeRecommendation(
                    probe=probe, axis=axis, value=round(value, 4),
                    rank_flip_probability=round(flip, 4),
                    uncertainty_reduction=red, cost=prof["cost"],
                    reason=f"largest overlapping uncertainty on {axis!r}; "
                           f"{probe} best reduces it per unit cost")
    if best is not None:
        return best
    # deterministic fallback (§14.5): most-uncertain axis -> fixed rule
    widths = {ax: max(cand_a["upper"].get(ax, 0) - cand_a["lower"].get(ax, 0),
                      cand_b["upper"].get(ax, 0) - cand_b["lower"].get(ax, 0))
              for ax in axes} or {"event_hole": 1.0}
    axis = max(widths, key=lambda k: widths[k])
    probe = FALLBACK_RULES.get(axis, "run_track_remix_challenge")
    if probe in executed:  # never repeat; walk the vocabulary deterministically
        probe = next((p for p in PROBE_VOCABULARY if p not in executed),
                     "run_track_remix_challenge")
    return ProbeRecommendation(probe=probe, axis=axis, value=0.0,
                               rank_flip_probability=0.0, uncertainty_reduction=0.0,
                               cost=profiles.get(probe, {"cost": 1.0})["cost"],
                               reason=f"deterministic fallback for uncertain {axis!r}")
