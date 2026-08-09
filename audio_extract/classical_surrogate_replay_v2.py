"""Content-bound replay evaluator for `LOSS-SURROGATE-ALIGNMENT-001`.

This versioned successor fixes the v1 self-hash ambiguity and makes the
held-out decision explicit. It performs no optimization and writes no audio.
Every row binds immutable FLOAT checkpoint audio, exact A/V targets, the old
loss value, exact product metrics, and work/session/singer grouping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import os

import numpy as np
import soundfile as sf

from . import identity
from .classical_surrogate_alignment_v2 import (
    SurrogateConfigV2,
    evaluate_surrogate_v2,
    pairwise_false_safe_v2,
)

ROW_SCHEMA = "audio-extract/classical-surrogate-replay-row/v2"
REPORT_SCHEMA = "audio-extract/classical-surrogate-replay-report/v2"
REQUIRED_STEPS = (0, 25, 50, 100)
REQUIRED_EXTERNAL = (
    "retained_voice_db_p90",
    "event_hole_db_p90",
    "artifact_ratio_p90",
)
DECISIONS = (
    "SURROGATE_ALIGNED",
    "SURROGATE_REJECTED",
    "INVALID_EVIDENCE",
)


class SurrogateReplayV2Error(RuntimeError):
    """The frozen replay evidence is missing, mutated, or incoherent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if not result.startswith("sha256:") or len(result) != 71:
        raise ValueError(f"{name} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be sha256:<64 hex>") from exc
    return result.lower()


def _finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def row_sha256(value: Mapping[str, Any]) -> str:
    """Hash one row excluding only its own `row_sha256` field."""

    payload = dict(value)
    payload.pop("row_sha256", None)
    return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


def _signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class AudioRecordV2:
    path: str
    container_sha256: str
    artifact_pcm_sha256: str
    frames: int
    sample_rate_hz: int
    channels: tuple[str, ...]
    subtype: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> "AudioRecordV2":
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object")
        result = cls(
            path=str(value.get("path") or ""),
            container_sha256=_sha(
                value.get("container_sha256"), f"{name}.container_sha256"
            ),
            artifact_pcm_sha256=_sha(
                value.get("artifact_pcm_sha256"),
                f"{name}.artifact_pcm_sha256",
            ),
            frames=int(value.get("frames", 0)),
            sample_rate_hz=int(value.get("sample_rate_hz", 0)),
            channels=tuple(value.get("channels") or ()),
            subtype=str(value.get("subtype") or ""),
        )
        if not result.path:
            raise ValueError(f"{name}.path is missing")
        if result.frames <= 0 or result.sample_rate_hz <= 0:
            raise ValueError(f"{name} grid is invalid")
        if result.channels != ("FL", "FR") or result.subtype != "FLOAT":
            raise ValueError(f"{name} must be stereo FLOAT FL/FR")
        return result


@dataclass(frozen=True)
class ReplayRowV2:
    example_id: str
    split: str
    work_id: str
    session_id: str
    singer_id: str
    checkpoint_step: int
    vocal_estimate: AudioRecordV2
    accompaniment_target: AudioRecordV2
    vocal_target: AudioRecordV2
    old_declared_total: float
    external_metrics: dict[str, float]
    row_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayRowV2":
        if value.get("schema") != ROW_SCHEMA:
            raise ValueError("wrong replay-row schema")
        fields = {
            name: str(value.get(name) or "")
            for name in (
                "example_id", "split", "work_id", "session_id", "singer_id"
            )
        }
        if any(not field for field in fields.values()):
            raise ValueError("replay row has an empty grouping field")
        if fields["split"] not in {"development", "heldout"}:
            raise ValueError("split must be development or heldout")
        step = int(value.get("checkpoint_step", -1))
        if step not in REQUIRED_STEPS:
            raise ValueError(f"unsupported checkpoint step {step}")
        metrics = value.get("external_metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(
            REQUIRED_EXTERNAL
        ):
            raise ValueError("external metric set is incomplete")
        external = {
            name: _finite(metrics[name], f"external_metrics.{name}")
            for name in REQUIRED_EXTERNAL
        }
        if external["artifact_ratio_p90"] < 0:
            raise ValueError("artifact_ratio_p90 must be non-negative")
        declared_hash = _sha(value.get("row_sha256"), "row_sha256")
        if declared_hash != row_sha256(value):
            raise ValueError("replay row semantic hash mismatch")
        return cls(
            **fields,
            checkpoint_step=step,
            vocal_estimate=AudioRecordV2.from_mapping(
                value.get("vocal_estimate"), "vocal_estimate"
            ),
            accompaniment_target=AudioRecordV2.from_mapping(
                value.get("accompaniment_target"), "accompaniment_target"
            ),
            vocal_target=AudioRecordV2.from_mapping(
                value.get("vocal_target"), "vocal_target"
            ),
            old_declared_total=_finite(
                value.get("old_declared_total"), "old_declared_total"
            ),
            external_metrics=external,
            row_sha256=declared_hash,
        )


@dataclass(frozen=True)
class ReplayDecisionConfigV2:
    material_external_change_db: float = 0.5
    minimum_sign_concordance: float = 0.80
    minimum_macro_rank_correlation: float = 0.0
    catastrophic_regression_db: float = 0.5
    artifact_relative_regression: float = 0.10
    artifact_absolute_allowance: float = 1e-8
    coverage_tolerance: float = 1e-12
    required_steps: tuple[int, ...] = REQUIRED_STEPS
    heldout_split: str = "heldout"

    def validate(self) -> None:
        for name in (
            "material_external_change_db",
            "minimum_sign_concordance",
            "catastrophic_regression_db",
            "artifact_relative_regression",
            "artifact_absolute_allowance",
            "coverage_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_sign_concordance > 1:
            raise ValueError("minimum_sign_concordance must not exceed one")
        if not math.isfinite(float(self.minimum_macro_rank_correlation)):
            raise ValueError("minimum_macro_rank_correlation must be finite")
        if tuple(self.required_steps) != REQUIRED_STEPS:
            raise ValueError("required steps differ from preregistration")
        if self.heldout_split != "heldout":
            raise ValueError("heldout split identity is frozen")


def load_rows_v2(path: Path) -> tuple[ReplayRowV2, ...]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            row = ReplayRowV2.from_mapping(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SurrogateReplayV2Error(
                f"invalid row {path}:{line_number}: {exc}"
            ) from exc
        key = (row.example_id, row.checkpoint_step)
        if key in seen:
            raise SurrogateReplayV2Error(f"duplicate replay row {key}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise SurrogateReplayV2Error("replay manifest is empty")
    return tuple(rows)


def _load_audio(record: AudioRecordV2) -> np.ndarray:
    path = Path(record.path)
    try:
        if path.is_symlink():
            raise SurrogateReplayV2Error(f"refusing symlinked audio: {path}")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            handle.seek(0)
            info = sf.info(handle)
            if info.format not in {"WAV", "WAVEX"}:
                raise SurrogateReplayV2Error(
                    f"expected WAV/WAVEX container: {path}: {info.format}"
                )
            grid = (info.frames, info.samplerate, info.channels, info.subtype)
            expected = (
                record.frames, record.sample_rate_hz,
                len(record.channels), record.subtype,
            )
            if grid != expected:
                raise SurrogateReplayV2Error(
                    f"audio grid changed: {path}: {grid} != {expected}"
                )
            handle.seek(0)
            audio, sample_rate = sf.read(
                handle, dtype="float32", always_2d=True
            )
            after = os.fstat(handle.fileno())
        current = path.stat()
    except SurrogateReplayV2Error:
        raise
    except (OSError, RuntimeError) as exc:
        raise SurrogateReplayV2Error(f"cannot read audio {path}: {exc}") from exc
    if _signature(before) != _signature(after) or _signature(before) != _signature(current):
        raise SurrogateReplayV2Error(f"audio changed/path replaced: {path}")
    if "sha256:" + digest.hexdigest() != record.container_sha256:
        raise SurrogateReplayV2Error(f"audio container changed: {path}")
    if audio.shape != (record.frames, 2) or not np.all(np.isfinite(audio)):
        raise SurrogateReplayV2Error(f"audio decode is invalid: {path}")
    pcm = identity.artifact_pcm_sha256(
        audio, int(sample_rate), list(record.channels), len(audio)
    )
    if pcm != record.artifact_pcm_sha256:
        raise SurrogateReplayV2Error(f"audio PCM changed: {path}")
    return audio


def _rank_correlation(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    from scipy.stats import kendalltau, spearmanr

    if len(x) < 3 or len(x) != len(y) or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, math.nan
    return (
        float(kendalltau(x, y).statistic),
        float(spearmanr(x, y).statistic),
    )


def evaluate_replay_v2(
    rows: Sequence[ReplayRowV2],
    *,
    surrogate_config: SurrogateConfigV2 | None = None,
    decision_config: ReplayDecisionConfigV2 | None = None,
) -> dict[str, Any]:
    surrogate_cfg = surrogate_config or SurrogateConfigV2()
    decision_cfg = decision_config or ReplayDecisionConfigV2()
    surrogate_cfg.validate()
    decision_cfg.validate()
    grouped: dict[str, dict[int, ReplayRowV2]] = {}
    group_identity = {}
    for row in rows:
        grouped.setdefault(row.example_id, {})[row.checkpoint_step] = row
        current = (row.split, row.work_id, row.session_id, row.singer_id)
        if group_identity.setdefault(row.example_id, current) != current:
            raise SurrogateReplayV2Error(
                f"group identity changed for {row.example_id}"
            )
    for example_id, by_step in grouped.items():
        if tuple(sorted(by_step)) != decision_cfg.required_steps:
            raise SurrogateReplayV2Error(
                f"{example_id} lacks the exact checkpoint set"
            )
        targets = {
            (
                row.accompaniment_target.artifact_pcm_sha256,
                row.vocal_target.artifact_pcm_sha256,
            )
            for row in by_step.values()
        }
        if len(targets) != 1:
            raise SurrogateReplayV2Error(
                f"exact targets changed across checkpoints: {example_id}"
            )

    cache = {}
    evaluated = []
    for example_id in sorted(grouped):
        for step in decision_cfg.required_steps:
            row = grouped[example_id][step]

            def fetch(record: AudioRecordV2):
                key = record.artifact_pcm_sha256
                if key not in cache:
                    cache[key] = _load_audio(record)
                return cache[key]

            report = evaluate_surrogate_v2(
                fetch(row.vocal_estimate),
                fetch(row.accompaniment_target),
                fetch(row.vocal_target),
                config=surrogate_cfg,
            )
            if report.source_coordinate_tiles + report.direct_fallback_tiles != report.total_tiles:
                raise SurrogateReplayV2Error("surrogate mode coverage is incomplete")
            evaluated.append({
                "example_id": example_id,
                "split": row.split,
                "work_id": row.work_id,
                "session_id": row.session_id,
                "singer_id": row.singer_id,
                "checkpoint_step": step,
                "old_declared_total": row.old_declared_total,
                "external_metrics": dict(row.external_metrics),
                "surrogate": report.to_dict(),
                "row_sha256": row.row_sha256,
            })

    indexed = {
        (row["example_id"], row["checkpoint_step"]): row
        for row in evaluated
    }
    concordance = {"recall": [0, 0], "theft": [0, 0]}
    pairs = []
    heldout_false_safe = []
    coverage_failures = []
    for example_id in sorted(grouped):
        baseline = indexed[(example_id, 0)]
        base_fraction = baseline["surrogate"][
            "source_coordinate_measure_fraction"
        ]
        for step in decision_cfg.required_steps[1:]:
            candidate = indexed[(example_id, step)]
            failures = list(pairwise_false_safe_v2(
                baseline["surrogate"], candidate["surrogate"],
                baseline["external_metrics"], candidate["external_metrics"],
                external_regression_limit_db=(
                    decision_cfg.catastrophic_regression_db
                ),
            ))
            base_artifact = baseline["external_metrics"]["artifact_ratio_p90"]
            next_artifact = candidate["external_metrics"]["artifact_ratio_p90"]
            base_surrogate_artifact = baseline["surrogate"]["artifact_cvar"]
            next_surrogate_artifact = candidate["surrogate"]["artifact_cvar"]
            if (
                base_surrogate_artifact is not None
                and next_surrogate_artifact is not None
                and next_surrogate_artifact < base_surrogate_artifact
                and next_artifact > (
                    base_artifact
                    * (1.0 + decision_cfg.artifact_relative_regression)
                    + decision_cfg.artifact_absolute_allowance
                )
            ):
                failures.append(
                    "artifact_cvar improves while artifact_ratio_p90 regresses"
                )
            fraction = candidate["surrogate"][
                "source_coordinate_measure_fraction"
            ]
            if abs(fraction - base_fraction) > decision_cfg.coverage_tolerance:
                coverage_failures.append({
                    "example_id": example_id,
                    "checkpoint_step": step,
                    "baseline_fraction": base_fraction,
                    "candidate_fraction": fraction,
                })
            pair = {
                "example_id": example_id,
                "split": baseline["split"],
                "work_id": baseline["work_id"],
                "checkpoint_step": step,
                "false_safe": failures,
                "deltas": {},
            }
            for axis, surrogate_name, external_name in (
                ("recall", "recall_cvar", "retained_voice_db_p90"),
                ("theft", "theft_cvar", "event_hole_db_p90"),
            ):
                base_risk = baseline["surrogate"].get(surrogate_name)
                next_risk = candidate["surrogate"].get(surrogate_name)
                if base_risk is None or next_risk is None:
                    pair["deltas"][axis] = None
                    continue
                surrogate_improvement = float(base_risk) - float(next_risk)
                external_improvement = (
                    baseline["external_metrics"][external_name]
                    - candidate["external_metrics"][external_name]
                )
                pair["deltas"][axis] = {
                    "surrogate_improvement": surrogate_improvement,
                    "external_improvement_db": external_improvement,
                }
                if (
                    baseline["split"] == decision_cfg.heldout_split
                    and abs(external_improvement)
                    >= decision_cfg.material_external_change_db
                ):
                    concordance[axis][1] += 1
                    if np.sign(surrogate_improvement) == np.sign(external_improvement):
                        concordance[axis][0] += 1
            pairs.append(pair)
            if baseline["split"] == decision_cfg.heldout_split and failures:
                heldout_false_safe.append(pair)

    heldout_works = sorted({
        row["work_id"] for row in evaluated
        if row["split"] == decision_cfg.heldout_split
    })
    per_work_rank = []
    for work in heldout_works:
        work_rows = [
            row for row in evaluated
            if row["work_id"] == work and row["split"] == decision_cfg.heldout_split
        ]
        result = {"work_id": work, "axes": {}}
        for axis, surrogate_name, external_name in (
            ("recall", "recall_cvar", "retained_voice_db_p90"),
            ("theft", "theft_cvar", "event_hole_db_p90"),
        ):
            surrogate_values = []
            external_values = []
            for step in decision_cfg.required_steps:
                step_rows = [
                    row for row in work_rows
                    if row["checkpoint_step"] == step
                    and row["surrogate"].get(surrogate_name) is not None
                ]
                if not step_rows:
                    continue
                surrogate_values.append(float(np.median([
                    row["surrogate"][surrogate_name] for row in step_rows
                ])))
                external_values.append(float(np.median([
                    row["external_metrics"][external_name] for row in step_rows
                ])))
            tau, rho = _rank_correlation(surrogate_values, external_values)
            result["axes"][axis] = {
                "kendall_tau": tau,
                "spearman_rho": rho,
                "checkpoint_count": len(surrogate_values),
            }
        per_work_rank.append(result)

    concordance_report = {
        axis: {
            "matching": values[0],
            "material_pairs": values[1],
            "fraction": values[0] / values[1] if values[1] else None,
        }
        for axis, values in concordance.items()
    }
    macro_rank = {}
    for axis in ("recall", "theft"):
        taus = [
            row["axes"][axis]["kendall_tau"] for row in per_work_rank
            if math.isfinite(row["axes"][axis]["kendall_tau"])
        ]
        rhos = [
            row["axes"][axis]["spearman_rho"] for row in per_work_rank
            if math.isfinite(row["axes"][axis]["spearman_rho"])
        ]
        macro_rank[axis] = {
            "median_kendall_tau": float(np.median(taus)) if taus else None,
            "median_spearman_rho": float(np.median(rhos)) if rhos else None,
            "works": len(taus),
        }

    failures = []
    if not heldout_works:
        failures.append("no held-out work groups")
    if heldout_false_safe:
        failures.append(
            f"{len(heldout_false_safe)} catastrophic held-out false-safe pairs"
        )
    if coverage_failures:
        failures.append("source/fallback mode coverage changed across checkpoints")
    for axis in ("recall", "theft"):
        fraction = concordance_report[axis]["fraction"]
        if fraction is None or fraction < decision_cfg.minimum_sign_concordance:
            failures.append(f"{axis} sign concordance is insufficient")
        tau = macro_rank[axis]["median_kendall_tau"]
        rho = macro_rank[axis]["median_spearman_rho"]
        if (
            tau is None or rho is None
            or tau <= decision_cfg.minimum_macro_rank_correlation
            or rho <= decision_cfg.minimum_macro_rank_correlation
        ):
            failures.append(f"{axis} held-out rank association is insufficient")
    if any(
        pair["checkpoint_step"] == 100 and pair["false_safe"]
        for pair in heldout_false_safe
    ):
        failures.append("known harmful step-100 direction is falsely preferred")
    decision = (
        "INVALID_EVIDENCE" if not heldout_works
        else "SURROGATE_ALIGNED" if not failures
        else "SURROGATE_REJECTED"
    )
    if decision not in DECISIONS:
        raise AssertionError(decision)
    return {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "surrogate_config": asdict(surrogate_cfg),
        "decision_config": asdict(decision_cfg),
        "rows": evaluated,
        "pairs": pairs,
        "summary": {
            "heldout_works": heldout_works,
            "catastrophic_false_safe_pairs": heldout_false_safe,
            "coverage_failures": coverage_failures,
            "sign_concordance": concordance_report,
            "per_work_rank": per_work_rank,
            "macro_rank": macro_rank,
            "failures": failures,
        },
    }
