"""Freeze audited Cantolopera pairs for the first all-voices training pilot.

This module does not repair or rewrite audio.  It converts the immutable pair
audit into an explicit selection manifest, records every exclusion reason, and
keeps connected take/title/opera variants in one deterministic split component.
The selected source pairs are materialized by :mod:`materialize_classical_train`
as separate recipe nodes.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata
import re
from typing import Any, Iterable

from . import canon, identity
from .cantolopera_pairs import PAIR_SCHEMA


SELECTION_SCHEMA = "audio-extract/cantolopera-training-selection/v1"
SELECTION_SUMMARY_SCHEMA = "audio-extract/cantolopera-training-selection-summary/v1"
TRAINING_MANIFEST_SCHEMA = "audio-extract/cantolopera-training-manifest/v1"
SPLIT_MANIFEST_SCHEMA = "audio-extract/cantolopera-training-splits/v1"
TASK = "all_voices_vs_nonvocal"

# Frozen before the first training run.  Exact decimal strings are also used in
# recipe identity; observed floating-point evidence remains present in reports.
TIER_A_THRESHOLDS = {
    "minimum_reliable_probes": 3,
    "reliable_probe_confidence_min": "0.5",
    "maximum_reliable_delay_span_samples": 2,
    "maximum_absolute_delay_samples_exclusive": 8192,
    "null_db_p10_max": "-20",
    "identical_pair_null_db_p50_max": "-100",
    "fixed_gain_min": "0.25",
    "fixed_gain_max": "4",
    "fixed_gain_relative_p10_p90_spread_max": "0.25",
    "validation_component_fraction": "0.2",
}
TIER_B_NULL_DB_P10_MAX = -10.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _ascii_words(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _title_family(catalog: dict[str, Any]) -> tuple[str, str]:
    title = re.sub(
        r"\([^)]*\bsenza\b[^)]*\)", " ", str(catalog.get("title", "")), flags=re.I
    )
    return _ascii_words(catalog.get("composer", "")), _ascii_words(title)


def _opera_family(catalog: dict[str, Any]) -> tuple[str, str]:
    return _ascii_words(catalog.get("composer", "")), _ascii_words(catalog.get("opera", ""))


def _reliable_probes(row: dict[str, Any]) -> list[dict[str, Any]]:
    minimum = float(TIER_A_THRESHOLDS["reliable_probe_confidence_min"])
    alignment = row.get("alignment") or {}
    return [
        probe
        for probe in alignment.get("probes", [])
        if float(probe.get("confidence", -math.inf)) >= minimum
    ]


def assess_pair(row: dict[str, Any]) -> dict[str, Any]:
    """Return a complete, deterministic eligibility decision for one audit row."""

    if row.get("schema") != PAIR_SCHEMA:
        raise ValueError(f"wrong pair-audit schema for {row.get('pair_id')!r}")
    reasons: list[str] = []
    if row.get("status") != "audited":
        reasons.append("pair_audit_invalid")
        return {
            "cohort": "excluded",
            "eligible": False,
            "reasons": reasons + list(row.get("errors") or []),
            "transform": None,
        }

    probes = _reliable_probes(row)
    required = int(TIER_A_THRESHOLDS["minimum_reliable_probes"])
    if len(probes) < required:
        reasons.append("too_few_reliable_probes")
    delays = [int(probe["delay_samples"]) for probe in probes]
    maximum_span = int(TIER_A_THRESHOLDS["maximum_reliable_delay_span_samples"])
    if delays and max(delays) - min(delays) > maximum_span:
        reasons.append("unstable_delay")
    bound = int(TIER_A_THRESHOLDS["maximum_absolute_delay_samples_exclusive"])
    if any(abs(delay) >= bound for delay in delays):
        reasons.append("delay_search_bound_hit")
    if any(int(probe["polarity"]) != 1 for probe in probes):
        reasons.append("polarity_not_consistently_positive")
    if any(bool(probe["channel_swap"]) for probe in probes):
        reasons.append("channel_order_not_consistently_straight")

    evidence = row.get("null_evidence") or {}
    null_p10 = evidence.get("null_db_p10")
    null_p50 = evidence.get("null_db_p50")
    gain = evidence.get("best_quintile_gain_median")
    gain_p10 = evidence.get("best_quintile_gain_p10")
    gain_p90 = evidence.get("best_quintile_gain_p90")
    if any(value is None or not math.isfinite(float(value)) for value in (
        null_p10, null_p50, gain, gain_p10, gain_p90
    )):
        reasons.append("nonfinite_or_missing_null_evidence")
        spread = math.inf
    else:
        gain = float(gain)
        spread = (float(gain_p90) - float(gain_p10)) / (abs(gain) + 1e-12)
        if not float(TIER_A_THRESHOLDS["fixed_gain_min"]) <= gain <= float(
            TIER_A_THRESHOLDS["fixed_gain_max"]
        ):
            reasons.append("fixed_gain_out_of_range")
        if spread > float(
            TIER_A_THRESHOLDS["fixed_gain_relative_p10_p90_spread_max"]
        ):
            reasons.append("fixed_gain_unstable")
        if float(null_p50) <= float(
            TIER_A_THRESHOLDS["identical_pair_null_db_p50_max"]
        ) or row["files"]["voice"]["container_sha256"] == row["files"]["orchestra"][
            "container_sha256"
        ]:
            reasons.append("identical_voice_and_orchestra_control")

    structural_reasons = list(reasons)
    if null_p10 is None or not math.isfinite(float(null_p10)):
        cohort = "excluded"
    elif not structural_reasons and float(null_p10) <= float(
        TIER_A_THRESHOLDS["null_db_p10_max"]
    ):
        cohort = "tier_a"
    elif not structural_reasons and float(null_p10) <= TIER_B_NULL_DB_P10_MAX:
        cohort = "tier_b"
        reasons.append("null_evidence_below_tier_a")
    else:
        cohort = "excluded"
        if not structural_reasons:
            reasons.append("null_evidence_below_tier_b")

    transform = None
    if cohort in {"tier_a", "tier_b"}:
        delay = int(round(statistics.median(delays)))
        matching = [
            probe
            for probe in probes
            if abs(int(probe["delay_samples"]) - delay) <= 1
        ]
        fractional = statistics.median(
            float(probe["fractional_samples"]) for probe in matching
        )
        transform = {
            "schema": "audio-extract/cantolopera-fixed-alignment/v1",
            "mixture_to_orchestra": {
                "delay_samples": delay,
                "fractional_samples": format(float(fractional), ".17g"),
                "polarity": 1,
                "channel_swap": False,
            },
            "orchestra_gain": format(float(gain), ".17g"),
            "source_rate_hz": 48_000,
            "output_rate_hz": 44_100,
            "resampler": "soxr_vhq",
            "derive_vocal": "V=M-A",
        }
    return {
        "cohort": cohort,
        "eligible": cohort == "tier_a",
        "reasons": reasons,
        "reliable_probe_count": len(probes),
        "reliable_delay_span_samples": None if not delays else max(delays) - min(delays),
        "fixed_gain_relative_p10_p90_spread": spread,
        "transform": transform,
    }


class _UnionFind:
    def __init__(self, size: int):
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[right] = left


def _split_components(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Connect take, title, and opera relatives; return component/split per pair."""

    union = _UnionFind(len(rows))
    for key in (
        lambda row: row["take_group_id"],
        lambda row: _title_family(row["catalog"]),
        lambda row: _opera_family(row["catalog"]),
    ):
        first: dict[Any, int] = {}
        for index, row in enumerate(rows):
            value = key(row)
            if value in first:
                union.union(index, first[value])
            else:
                first[value] = index
    components: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        components.setdefault(union.find(index), []).append(row)

    component_facts = []
    for members in components.values():
        pair_ids = sorted(member["pair_id"] for member in members)
        digest = identity.blob_sha256(
            canon.canonicalize({"pair_ids": pair_ids})
        ).removeprefix("sha256:")
        component_id = f"cantolopera:component:{digest[:24]}"
        component_facts.append((digest, component_id, pair_ids))
    # Rank by the full component hash and reserve an exact fraction of independent
    # components.  A raw hash cutoff can accidentally yield only one or two
    # validation groups in a small corpus; the ranked rule is deterministic and
    # guarantees the declared group-level validation support.
    component_facts.sort()
    fraction = float(TIER_A_THRESHOLDS["validation_component_fraction"])
    validation_count = max(1, math.ceil(fraction * len(component_facts)))
    validation = {component_id for _, component_id, _ in component_facts[:validation_count]}
    result: dict[str, dict[str, str]] = {}
    for _, component_id, pair_ids in component_facts:
        split = "val" if component_id in validation else "train"
        for pair_id in pair_ids:
            result[pair_id] = {"group_id": component_id, "split": split}
    if rows and len({facts["split"] for facts in result.values()}) != 2:
        raise ValueError("deterministic component split did not produce train and val")
    return result


def build_selection(
    audit_rows: Iterable[dict[str, Any]], *, audit_sha256: str,
    selector_code_commit: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return all decisions, Tier-A training rows, summary, and split manifest."""

    rows = sorted(list(audit_rows), key=lambda row: row["pair_id"])
    decisions = []
    tier_a = []
    for row in rows:
        assessment = assess_pair(row)
        selected = {
            "schema": SELECTION_SCHEMA,
            "pair_id": row["pair_id"],
            "take_group_id": row["take_group_id"],
            "catalog": row["catalog"],
            "files": row["files"],
            "audio": row.get("audio"),
            "task": TASK,
            "audit_sha256": audit_sha256,
            "selector_code_commit": selector_code_commit,
            "audit_grade": row.get("suggested_grade"),
            "assessment": assessment,
        }
        decisions.append(selected)
        if assessment["eligible"]:
            tier_a.append(selected)

    split_facts = _split_components(tier_a)
    manifest = []
    for selected in tier_a:
        selected.update(split_facts[selected["pair_id"]])
        manifest.append(
            {
                "schema": TRAINING_MANIFEST_SCHEMA,
                "corpus_id": "cantolopera",
                "work_id": selected["pair_id"],
                "group_id": selected["group_id"],
                "split": selected["split"],
                "integrity_class": "same_take_paired_target",
                "task": TASK,
                "eligible_training_targets": ["accompaniment_A"],
                "catalog": selected["catalog"],
                "files": {
                    "M": selected["files"]["voice"],
                    "A": selected["files"]["orchestra"],
                },
                "selection_transform": selected["assessment"]["transform"],
                "audit_sha256": audit_sha256,
                "selector_code_commit": selector_code_commit,
            }
        )
    groups: dict[str, dict[str, Any]] = {}
    for item in manifest:
        group = groups.setdefault(
            item["group_id"],
            {"split": item["split"], "works": [], "corpus": "cantolopera"},
        )
        if group["split"] != item["split"]:
            raise ValueError(f"split leakage inside {item['group_id']}")
        group["works"].append(item["work_id"])
    for group in groups.values():
        group["works"].sort()
    split_manifest = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "audit_sha256": audit_sha256,
        "selector_code_commit": selector_code_commit,
        "thresholds": TIER_A_THRESHOLDS,
        "group_split": {key: value["split"] for key, value in sorted(groups.items())},
        "groups": dict(sorted(groups.items())),
    }
    cohort_counts = Counter(row["assessment"]["cohort"] for row in decisions)
    split_counts = Counter(item["split"] for item in manifest)
    split_groups = Counter(value["split"] for value in groups.values())
    summary = {
        "schema": SELECTION_SUMMARY_SCHEMA,
        "audit_sha256": audit_sha256,
        "selector_code_commit": selector_code_commit,
        "task": TASK,
        "thresholds": TIER_A_THRESHOLDS,
        "rows": len(decisions),
        "cohorts": dict(sorted(cohort_counts.items())),
        "tier_a_pairs": len(manifest),
        "tier_a_take_groups": len({row["take_group_id"] for row in tier_a}),
        "tier_a_split_components": len(groups),
        "split_pairs": dict(sorted(split_counts.items())),
        "split_components": dict(sorted(split_groups.items())),
        "tier_a_hours": sum(
            int(row["audio"]["voice"]["frames"]) for row in tier_a
        ) / 48_000 / 3600,
        "tier_a_composers": len({row["catalog"]["composer"] for row in tier_a}),
        "tier_a_opera_labels": len({row["catalog"]["opera"] for row in tier_a}),
    }
    return decisions, manifest, summary, split_manifest


def load_audit(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_outputs(
    decisions: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    summary: dict[str, Any],
    splits: dict[str, Any],
    *,
    selection_path: Path,
    manifest_path: Path,
    summary_path: Path,
    splits_path: Path,
) -> None:
    payloads = {
        selection_path: "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in decisions),
        manifest_path: "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in manifest),
        summary_path: json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        splits_path: json.dumps(splits, indent=2, sort_keys=True, allow_nan=False) + "\n",
    }
    for path, payload in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != payload:
            raise RuntimeError(f"refusing to rewrite differing selection artifact: {path}")
        if not path.exists():
            path.write_text(payload)
