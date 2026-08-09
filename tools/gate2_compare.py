#!/usr/bin/env python3
"""Evaluate the preregistered 100-step HTDemucs opera continuation gate.

Input files are the immutable per-step ``report.json`` files emitted by
``audio-extract train classical``. The script refuses missing, non-finite,
incoherent, or differently-scoped reports instead of treating absent evidence
as a pass.

Example:
    python tools/gate2_compare.py \
      --fold V \
      --baseline /runs/fold-v/evaluation/step-000000/report.json \
      --candidate /runs/fold-v/evaluation/step-000100/report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

VOICE = "retained_voice_db_p90"          # more negative is better
HOLE = "event_hole_db_p90"               # lower is better
HOLE_MAX = "event_hole_db_max"           # lower is better
ALPHA = "alpha_error_p90"                 # lower is better
ARTIFACT = "artifact_ratio_p90"           # lower is better
WIDTH = "stereo_width_dev_db/v2"          # lower is better
COHERENCE = "interchannel_coherence_dev/v2"  # lower is better

REQUIRED_METRICS = (
    VOICE, HOLE, HOLE_MAX, ALPHA, ARTIFACT,
    WIDTH, COHERENCE, "_available_tiles", "_total_tiles",
)


class GateError(RuntimeError):
    """Evidence is missing, malformed, or outside the preregistered gate."""


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("works"), dict):
        raise GateError(f"invalid evaluation document: {path}")
    return doc


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GateError(f"missing/non-finite {label}: {value!r}")
    return float(value)


def _work(doc: dict[str, Any], work: str) -> dict[str, Any]:
    if work not in doc["works"]:
        raise GateError(f"missing work {work}")
    facts = doc["works"][work]
    metrics = facts.get("metrics")
    if not isinstance(metrics, dict):
        raise GateError(f"missing metrics for {work}")
    for metric in REQUIRED_METRICS:
        _finite(metrics.get(metric), f"{work}.{metric}")
    _finite(
        facts.get("no_vocal_false_positive_energy_ratio"),
        f"{work}.no_vocal_false_positive_energy_ratio",
    )
    if metrics[HOLE_MAX] + 1e-9 < metrics[HOLE]:
        raise GateError(f"incoherent hole metrics for {work}: max < p90")
    if metrics["_available_tiles"] <= 0:
        raise GateError(f"no available source-coordinate tiles for {work}")
    if metrics["_available_tiles"] > metrics["_total_tiles"]:
        raise GateError(f"invalid tile coverage for {work}")
    return facts


def _delta(base: float, candidate: float) -> float:
    """Positive means improvement for a lower-is-better metric."""
    return base - candidate


def _ratio(candidate: float, base: float) -> float:
    if base <= 1e-15:
        return 1.0 if candidate <= 1e-15 else math.inf
    return candidate / base


def compare_work(
    base_doc: dict[str, Any], candidate_doc: dict[str, Any], work: str
) -> dict[str, Any]:
    baseline = _work(base_doc, work)
    candidate = _work(candidate_doc, work)
    bm, cm = baseline["metrics"], candidate["metrics"]

    return {
        "work": work,
        "baseline": {key: bm[key] for key in REQUIRED_METRICS},
        "candidate": {key: cm[key] for key in REQUIRED_METRICS},
        "deltas": {
            "retained_voice_improvement_db": _delta(bm[VOICE], cm[VOICE]),
            "event_hole_p90_improvement_db": _delta(bm[HOLE], cm[HOLE]),
            "event_hole_max_improvement_db": _delta(bm[HOLE_MAX], cm[HOLE_MAX]),
            "alpha_error_improvement": _delta(bm[ALPHA], cm[ALPHA]),
            "artifact_improvement": _delta(bm[ARTIFACT], cm[ARTIFACT]),
            "stereo_width_improvement_db": _delta(bm[WIDTH], cm[WIDTH]),
            "coherence_improvement": _delta(bm[COHERENCE], cm[COHERENCE]),
            "available_tile_fraction_change": (
                cm["_available_tiles"] / max(1.0, cm["_total_tiles"])
                - bm["_available_tiles"] / max(1.0, bm["_total_tiles"])
            ),
        },
        "ratios": {
            "artifact_candidate_over_baseline": _ratio(cm[ARTIFACT], bm[ARTIFACT]),
            "no_vocal_fp_candidate_over_baseline": _ratio(
                candidate["no_vocal_false_positive_energy_ratio"],
                baseline["no_vocal_false_positive_energy_ratio"],
            ),
        },
    }


def _noncritical_stability(row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    deltas = row["deltas"]
    ratios = row["ratios"]

    if ratios["no_vocal_fp_candidate_over_baseline"] > 1.05:
        failures.append("no-vocal false-positive energy regressed >5%")
    if ratios["artifact_candidate_over_baseline"] > 1.10:
        failures.append("orthogonal artifact ratio regressed >10%")
    if deltas["stereo_width_improvement_db"] < -0.5:
        failures.append("stereo-width deviation regressed >0.5 dB")
    if deltas["coherence_improvement"] < -0.05:
        failures.append("interchannel-coherence deviation regressed >0.05")
    if deltas["available_tile_fraction_change"] < -0.10:
        failures.append("identifiable-tile coverage dropped >10 percentage points")
    return failures


def evaluate(
    fold: str, base_doc: dict[str, Any], candidate_doc: dict[str, Any]
) -> dict[str, Any]:
    if base_doc.get("step") != 0:
        raise GateError("baseline report is not step 0")
    if candidate_doc.get("step") != 100:
        raise GateError("candidate report is not step 100")

    if fold == "V":
        target = "bologna_verdi"
        stability = "aalto_mozart_dry"
    elif fold == "D":
        target = "bologna_donizetti"
        stability = "aalto_mozart_dry"
    else:
        raise GateError(f"unknown fold {fold}")

    if set(base_doc["works"]) != set(candidate_doc["works"]):
        raise GateError("baseline and candidate reports cover different work sets")
    expected = {target, stability}
    if not expected.issubset(base_doc["works"]):
        raise GateError(f"fold {fold} lacks required works {sorted(expected)}")

    rows = {
        work: compare_work(base_doc, candidate_doc, work)
        for work in sorted(base_doc["works"])
    }
    target_delta = rows[target]["deltas"]
    failures: list[str] = []

    if fold == "V":
        if target_delta["retained_voice_improvement_db"] < 0.5:
            failures.append("Verdi retained-voice p90 improvement <0.5 dB")
        if target_delta["event_hole_p90_improvement_db"] < -0.5:
            failures.append("Verdi event-hole p90 worsened >0.5 dB")
    else:
        if target_delta["event_hole_p90_improvement_db"] < 0.5:
            failures.append("Donizetti event-hole p90 improvement <0.5 dB")
        if target_delta["retained_voice_improvement_db"] < -0.5:
            failures.append("Donizetti retained-voice p90 worsened >0.5 dB")

    stability_delta = rows[stability]["deltas"]
    if stability_delta["retained_voice_improvement_db"] < -0.5:
        failures.append("Aalto retained-voice p90 worsened >0.5 dB")
    if stability_delta["event_hole_p90_improvement_db"] < -0.5:
        failures.append("Aalto event-hole p90 worsened >0.5 dB")

    for work, row in rows.items():
        failures.extend(
            f"{work}: {failure}" for failure in _noncritical_stability(row)
        )

    return {
        "schema": "audio-extract/gate2-exact-cv/v1",
        "fold": fold,
        "baseline_step": 0,
        "candidate_step": 100,
        "decision": "GO_TO_500" if not failures else "STOP_AT_100",
        "failures": failures,
        "works": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, choices=["V", "D"])
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        result = evaluate(args.fold, _load(args.baseline), _load(args.candidate))
    except GateError as exc:
        result = {
            "schema": "audio-extract/gate2-exact-cv/v1",
            "fold": args.fold,
            "decision": "INVALID_EVIDENCE",
            "failures": [str(exc)],
        }

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if result["decision"] == "GO_TO_500" else 1


if __name__ == "__main__":
    raise SystemExit(main())
