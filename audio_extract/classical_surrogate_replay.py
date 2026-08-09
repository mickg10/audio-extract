"""Immutable replay evaluator for `LOSS-SURROGATE-ALIGNMENT-001`.

The evaluator consumes an append-only JSONL manifest of already-rendered exact
checkpoint crops. It performs no optimization and writes no audio. Each row
binds the vocal estimate, exact accompaniment, exact vocal target, old declared
loss, product-gate metrics, and work/session grouping.

The decision is made only from rows declared `heldout`; development rows may be
used to choose one preregistered geometry before the held-out report is opened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math

import numpy as np
import soundfile as sf

from . import identity
from .classical_surrogate_alignment_v2 import (
    SurrogateConfigV2,
    evaluate_surrogate_v2,
    pairwise_false_safe_v2,
)

ROW_SCHEMA = "audio-extract/classical-surrogate-replay-row/v1"
REPORT_SCHEMA = "audio-extract/classical-surrogate-replay-report/v1"
DECISIONS = (
    "SURROGATE_ALIGNED",
    "SURROGATE_REJECTED",
    "INVALID_EVIDENCE",
)
REQUIRED_EXTERNAL = (
    "retained_voice_db_p90",
    "event_hole_db_p90",
    "artifact_ratio_p90",
)
REQUIRED_STEPS = (0, 25, 50, 100)


class SurrogateReplayError(RuntimeError):
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
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class AudioRecord:
    path: str
    container_sha256: str
    artifact_pcm_sha256: str
    frames: int
    sample_rate_hz: int
    channels: tuple[str, ...]
    subtype: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> "AudioRecord":
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object")
        record = cls(
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
        if not record.path:
            raise ValueError(f"{name}.path is missing")
        if record.frames <= 0 or record.sample_rate_hz <= 0:
            raise ValueError(f"{name} grid is invalid")
        if record.channels != ("FL", "FR") or record.subtype != "FLOAT":
            raise ValueError(f"{name} must be stereo FLOAT FL/FR")
        return record


@dataclass(frozen=True)
class ReplayRow:
    example_id: str
    split: str
    work_id: str
    session_id: str
    singer_id: str
    checkpoint_step: int
    vocal_estimate: AudioRecord
    accompaniment_target: AudioRecord
    vocal_target: AudioRecord
    old_declared_total: float
    external_metrics: dict[str, float]
    source_row_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayRow":
        if value.get("schema") != ROW_SCHEMA:
            raise ValueError("wrong replay-row schema")
        text = {
            name: str(value.get(name) or "")
            for name in (
                "example_id", "split", "work_id", "session_id", "singer_id"
            )
        }
        if any(not result for result in text.values()):
            raise ValueError("replay row has an empty grouping field")
        if text["split"] not in {"development", "heldout"}:
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
        return cls(
            **text,
            checkpoint_step=step,
            vocal_estimate=AudioRecord.from_mapping(
                value.get("vocal_estimate"), "vocal_estimate"
            ),
            accompaniment_target=AudioRecord.from_mapping(
                value.get("accompaniment_target"), "accompaniment_target"
            ),
            vocal_target=AudioRecord.from_mapping(
                value.get("vocal_target"), "vocal_target"
            ),
            old_declared_total=_finite(
                value.get("old_declared_total"), "old_declared_total"
            ),
            external_metrics=external,
            source_row_sha256=_sha(
                value.get("source_row_sha256"), "source_row_sha256"
            ),
        )


@dataclass(frozen=True)
class ReplayDecisionConfig:
    material_external_change_db: float = 0.5
    minimum_sign_concordance: float = 0.80
    minimum_macro_rank_correlation: float = 0.0
    catastrophic_regression_db: float = 0.5
    required_steps: tuple[int, ...] = REQUIRED_STEPS
    heldout_split: str = "heldout"

    def validate(self) -> None:
        for name in (
            "material_external_change_db",
            "minimum_sign_concordance",
            "catastrophic_regression_db",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_sign_concordance > 1:
            raise ValueError("minimum_sign_concordance must not exceed one")
        if not math.isfinite(float(self.minimum_macro_rank_correlation)):
            raise ValueError("minimum_macro_rank_correlation must be finite")
        if tuple(self.required_steps) != REQUIRED_STEPS:
            raise ValueError("required checkpoint steps differ from preregistration")
        if self.heldout_split != "heldout":
            raise ValueError("heldout split identity is frozen")


def load_rows(path: Path) -> tuple[ReplayRow, ...]:
    result = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            row = ReplayRow.from_mapping(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SurrogateReplayError(
                f"invalid row {path}:{line_number}: {exc}"
            ) from exc
        canonical = json.dumps(
            raw, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if "sha256:" + hashlib.sha256(canonical).hexdigest() != row.source_row_sha256:
            raise SurrogateReplayError(
                f"source row hash mismatch at {path}:{line_number}"
            )
        key = (row.example_id, row.checkpoint_step)
        if key in seen:
            raise SurrogateReplayError(f"duplicate replay row {key}")
        seen.add(key)
        result.append(row)
    if not result:
        raise SurrogateReplayError("replay manifest is empty")
    return tuple(result)


def _audio(record: AudioRecord) -> np.ndarray:
    path = Path(record.path)
    if path.is_symlink() or not path.is_file():
        raise SurrogateReplayError(f"audio is unavailable/symlinked: {path}")
    info = sf.info(path)
    grid = (info.frames, info.samplerate, info.channels, info.subtype)
    expected = (
        record.frames, record.sample_rate_hz, len(record.channels), record.subtype
    )
    if grid != expected:
        raise SurrogateReplayError(f"audio grid changed: {path}: {grid} != {expected}")
    if _sha_file(path) != record.container_sha256:
        raise SurrogateReplayError(f"audio container changed: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape != (record.frames, 2) or not np.all(np.isfinite(audio)):
        raise SurrogateReplayError(f"audio decode is invalid: {path}")
    pcm = identity.artifact_pcm_sha256(
        audio, int(sample_rate), list(record.channels), len(audio)
    )
    if pcm != record.artifact_pcm_sha256:
        raise SurrogateReplayError(f"audio PCM changed: {path}")
    return audio


def _rank_correlation(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    from scipy.stats import kendalltau, spearmanr

    if len(x) < 3 or len(x) != len(y):
        return math.nan, math.nan
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, math.nan
    tau = float(kendalltau(x, y).statistic)
    rho = float(spearmanr(x, y).statistic)
    return tau, rho


def evaluate_replay(
    rows: Sequence[ReplayRow],
    *,
    surrogate_config: SurrogateConfigV2 | None = None,
    decision_config: ReplayDecisionConfig | None = None,
) -> dict[str, Any]:
    surrogate_cfg = surrogate_config or SurrogateConfigV2()
    decision_cfg = decision_config or ReplayDecisionConfig()
    surrogate_cfg.validate()
    decision_cfg.validate()
    grouped: dict[str, dict[int, ReplayRow]] = {}
    identity: dict[str, tuple[str, str, str, str]] = {}
    for row in rows:
        grouped.setdefault(row.example_id, {})[row.checkpoint_step] = row
        current = (
            row.split, row.work_id, row.session_id, row.singer_id
        )
        previous = identity.setdefault(row.example_id, current)
        if previous != current:
            raise SurrogateReplayError(
                f"group identity changed for {row.example_id}"
            )
    for example_id, by_step in grouped.items():
        if tuple(sorted(by_step)) != decision_cfg.required_steps:
            raise SurrogateReplayError(
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
            raise SurrogateReplayError(
                f"exact targets changed across checkpoints: {example_id}"
            )

    evaluated = []
    cached: dict[str, np.ndarray] = {}
    for example_id in sorted(grouped):
        for step in decision_cfg.required_steps:
            row = grouped[example_id][step]

            def fetch(record: AudioRecord):
                key = record.artifact_pcm_sha256
                if key not in cached:
                    cached[key] = _audio(record)
                return cached[key]

            report = evaluate_surrogate_v2(
                fetch(row.vocal_estimate),
                fetch(row.accompaniment_target),
                fetch(row.vocal_target),
                config=surrogate_cfg,
            )
            if (
                report.source_coordinate_tiles + report.direct_fallback_tiles
                != report.total_tiles
            ):
                raise SurrogateReplayError("surrogate mode coverage is incomplete")
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
                "input_identities": {
                    "vocal_estimate": row.vocal_estimate.artifact_pcm_sha256,
                    "accompaniment_target": (
                        row.accompaniment_target.artifact_pcm_sha256
                    ),
                    "vocal_target": row.vocal_target.artifact_pcm_sha256,
                    "source_row_sha256": row.source_row_sha256,
                },
            })

    by_key = {
        (row["example_id"], row["checkpoint_step"]): row
        for row in evaluated
    }
    pairs = []
    false_safe = []
    concordance = {"recall": [0, 0], "theft": [0, 0]}
    for example_id in sorted(grouped):
        baseline = by_key[(example_id, 0)]
        for step in decision_cfg.required_steps[1:]:
            candidate = by_key[(example_id, step)]
            failures = pairwise_false_safe_v2(
                baseline["surrogate"],
                candidate["surrogate"],
                baseline["external_metrics"],
                candidate["external_metrics"],
                external_regression_limit_db=(
                    decision_cfg.catastrophic_regression_db
                ),
            )
            pair = {
                "example_id": example_id,
                "split": baseline["split"],
                "work_id": baseline["work_id"],
                "checkpoint_step": step,
                "false_safe": list(failures),
                "deltas": {},
            }
            for axis, surrogate_name, external_name in (
                (
                    "recall",
                    "recall_cvar",
                    "retained_voice_db_p90",
                ),
                ("theft", "theft_cvar", "event_hole_db_p90"),
            ):
                base_risk = baseline["surrogate"].get(surrogate_name)
                next_risk = candidate["surrogate"].get(surrogate_name)
                if base_risk is None or next_risk is None:
                    pair["deltas"][axis] = None
                    continue
                surrogate_improvement = float(base_risk) - float(next_risk)
                external_improvement = (
                    float(baseline["external_metrics"][external_name])
                    - float(candidate["external_metrics"][external_name])
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
                    if (
                        (surrogate_improvement > 0 and external_improvement > 0)
                        or (
                            surrogate_improvement < 0
                            and external_improvement < 0
                        )
                    ):
                        concordance[axis][0] += 1
            pairs.append(pair)
            if baseline["split"] == decision_cfg.heldout_split and failures:
                false_safe.append(pair)

    per_work_rank = []
    heldout_works = sorted({
        row["work_id"] for row in evaluated
        if row["split"] == decision_cfg.heldout_split
    })
    for work in heldout_works:
        work_rows = [
            row for row in evaluated
            if row["split"] == decision_cfg.heldout_split
            and row["work_id"] == work
        ]
        summary = {"work_id": work, "axes": {}}
        for axis, surrogate_name, external_name in (
            ("recall", "recall_cvar", "retained_voice_db_p90"),
            ("theft", "theft_cvar", "event_hole_db_p90"),
        ):
            surrogate_values = []
            external_values = []
            for step in decision_cfg.required_steps:
                rows_at_step = [
                    row for row in work_rows
                    if row["checkpoint_step"] == step
                    and row["surrogate"].get(surrogate_name) is not None
                ]
                if not rows_at_step:
                    continue
                surrogate_values.append(float(np.median([
                    row["surrogate"][surrogate_name]
                    for row in rows_at_step
                ])))
                external_values.append(float(np.median([
                    row["external_metrics"][external_name]
                    for row in rows_at_step
                ])))
            tau, rho = _rank_correlation(surrogate_values, external_values)
            summary["axes"][axis] = {
                "kendall_tau": tau,
                "spearman_rho": rho,
                "checkpoint_count": len(surrogate_values),
            }
        per_work_rank.append(summary)

    concordance_report = {}
    for axis, (matching, material) in concordance.items():
        concordance_report[axis] = {
            "matching": matching,
            "material_pairs": material,
            "fraction": (
                matching / material if material else None
            ),
        }
    macro = {}
    for axis in ("recall", "theft"):
        taus = [
            row["axes"][axis]["kendall_tau"] for row in per_work_rank
            if math.isfinite(row["axes"][axis]["kendall_tau"])
        ]
        rhos = [
            row["axes"][axis]["spearman_rho"] for row in per_work_rank
            if math.isfinite(row["axes"][axis]["spearman_rho"])
        ]
        macro[axis] = {
            "median_kendall_tau": (
                float(np.median(taus)) if taus else None
            ),
            "median_spearman_rho": (
                float(np.median(rhos)) if rhos else None
            ),
            "works": len(taus),
        }

    failures = []
    if false_safe:
        failures.append(
            f"{len(false_safe)} catastrophic held-out false-safe pairs"
        )
    for axis in ("recall", "theft"):
        fraction = concordance_report[axis]["fraction"]
        if fraction is None or fraction < decision_cfg.minimum_sign_concordance:
            failures.append(f"{axis} sign concordance is insufficient")
        tau = macro[axis]["median_kendall_tau"]
        rho = macro[axis]["median_spearman_rho"]
        if (
            tau is None
            or rho is None
            or tau <= decision_cfg.minimum_macro_rank_correlation
            or rho <= decision_cfg.minimum_macro_rank_correlation
        ):
            failures.append(f"{axis} held-out rank association is insufficient")
    known_step100_false_safe = any(
        pair["checkpoint_step"] == 100 and pair["false_safe"]
        for pair in false_safe
    )
    if known_step100_false_safe:
        failures.append("known harmful step-100 direction is falsely preferred")
    if not heldout_works:
        decision = "INVALID_EVIDENCE"
        failures.append("no held-out work groups")
    else:
        decision = "SURROGATE_ALIGNED" if not failures else "SURROGATE_REJECTED"
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
            "catastrophic_false_safe_pairs": false_safe,
            "sign_concordance": concordance_report,
            "per_work_rank": per_work_rank,
            "macro_rank": macro,
            "known_step100_false_safe": known_step100_false_safe,
            "failures": failures,
        },
    }
