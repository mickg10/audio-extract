"""Immutable no-optimizer replay for LOSS-SURROGATE-ALIGNMENT-001.

The replay consumes the already-frozen training samples and checkpoints.  It
never constructs an optimizer and never writes into the source run.  Its only
write is a new immutable report directory, published after all evidence has
been rehashed and all 400 crop/checkpoint cells have completed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from . import identity
from .classical_surrogate_alignment_v2 import SurrogateConfigV2, evaluate_surrogate_v2
from .demucs_affine import DemucsAffine, forward_demucs_production_affine
from .train_classical import SR, _read_exact, _source_metrics


SCHEMA = "audio-extract/loss-surrogate-alignment-replay/v1"
STEPS = (0, 25, 50, 100)
AXES = {
    "voice": ("recall_cvar", "retained_voice_db_p90"),
    "holes": ("theft_cvar", "event_hole_db_p90"),
    "artifacts": ("artifact_cvar", "artifact_ratio_p90"),
}
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# The frozen report did not retain vocal-estimate PCM, so replay binding uses
# the three independently recomputed external metrics.  CUDA reduction order
# may move their final decimal places across driver/runtime invocations; this
# bound is far below any gate threshold (including the 0.5 dB false-safe gate).
METRIC_PARITY_TOLERANCE = 1e-4


class ReplayEvidenceError(RuntimeError):
    """Frozen replay evidence is incomplete, mutable, or inconsistent."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ReplayEvidenceError(f"expected JSON object: {path}")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _code_commit() -> str:
    override = os.environ.get("AUDIO_EXTRACT_CODE_COMMIT", "").strip()
    if override:
        if not FULL_COMMIT.fullmatch(override):
            raise ReplayEvidenceError("AUDIO_EXTRACT_CODE_COMMIT must be 40 lowercase hex")
        return override
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    ).stdout.strip()
    if not FULL_COMMIT.fullmatch(result):
        raise ReplayEvidenceError("cannot resolve exact audio-extract code commit")
    return result


def _pcm(samples: np.ndarray) -> str:
    samples = np.asarray(samples, dtype="float32")
    return identity.artifact_pcm_sha256(samples, SR, ["FL", "FR"], len(samples))


def _load_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReplayEvidenceError(f"non-object row at {path}:{number}")
        rows.append(value)
    return rows


def _manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_lines(path)
    result = {str(row.get("work_id")): row for row in rows}
    if len(result) != len(rows):
        raise ReplayEvidenceError("manifest contains duplicate work_id values")
    return result


def _validate_run_inputs(
    run_dir: Path,
    materialized_root: Path,
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    config = yaml.safe_load((run_dir / "resolved-config.yaml").read_text())
    if not isinstance(config, dict):
        raise ReplayEvidenceError("resolved config is not a mapping")
    if int(config.get("samplerate", 0)) != SR or config.get("sources") != [
        "drums", "bass", "other", "vocals"
    ]:
        raise ReplayEvidenceError("replay refuses unexpected sample rate/source topology")
    samples = _load_lines(run_dir / "sample-log.jsonl")
    if len(samples) != 100:
        raise ReplayEvidenceError(f"expected 100 frozen samples, found {len(samples)}")
    fixed = _json(run_dir / "fixed-sampled-crop-objective.json")
    if fixed.get("mode") != "evaluation_no_grad_no_optimizer":
        raise ReplayEvidenceError("fixed objective was not a no-optimizer evaluation")
    crop_frames = int(fixed.get("crop_frames", 0))
    if crop_frames <= 0:
        raise ReplayEvidenceError("invalid fixed crop frame count")
    if _sha_file(run_dir / "resolved-config.yaml") != fixed.get("resolved_config_sha256"):
        raise ReplayEvidenceError("resolved config hash differs from fixed objective")
    fixed_rows = fixed.get("rows")
    if not isinstance(fixed_rows, list) or len(fixed_rows) != 400:
        raise ReplayEvidenceError("fixed objective must contain 400 rows")
    fixed_index: dict[tuple[int, int], dict[str, Any]] = {}
    for row in fixed_rows:
        key = (int(row["sample_index"]), int(row["step"]))
        if key in fixed_index:
            raise ReplayEvidenceError(f"duplicate fixed objective cell: {key}")
        fixed_index[key] = row
    if set(fixed_index) != {(sample, step) for sample in range(100) for step in STEPS}:
        raise ReplayEvidenceError("fixed objective lacks the exact 100 x 4 grid")

    manifest = _manifest(manifest_path)
    run_report = _json(run_dir / "run-report.json")
    if _sha_file(manifest_path) != run_report.get("manifest_sha256"):
        raise ReplayEvidenceError("manifest hash differs from frozen training run")
    facts = fixed.get("model_facts")
    if not isinstance(facts, dict) or set(facts) != {str(step) for step in STEPS}:
        raise ReplayEvidenceError("fixed objective lacks the four model facts")
    for step in STEPS:
        fact = facts[str(step)]
        path = Path(fact["path"])
        if not path.is_file() or _sha_file(path) != fact["sha256"]:
            raise ReplayEvidenceError(f"model artifact hash mismatch at step {step}")

    materials: dict[str, dict[str, Any]] = {}
    from .materialize_classical_train import _recipe_id

    for sample_index, sample in enumerate(samples):
        if set(sample) != {"recipe_id", "start_frame", "work_id"}:
            raise ReplayEvidenceError(f"sample {sample_index} has unexpected fields")
        work_id = str(sample["work_id"])
        if work_id not in manifest:
            raise ReplayEvidenceError(f"sample work absent from manifest: {work_id}")
        if work_id not in materials:
            directory = materialized_root / work_id
            recipe_path = directory / "recipe.json"
            report_path = directory / "report.json"
            recipe = _json(recipe_path)
            report = _json(report_path)
            body = dict(recipe)
            claimed = body.pop("recipe_id", None)
            if claimed != _recipe_id(body) or claimed != report.get("recipe_id"):
                raise ReplayEvidenceError(f"materialization identity mismatch: {work_id}")
            if recipe.get("sample_rate_hz") != SR or recipe.get("channel_layout") != ["FL", "FR"]:
                raise ReplayEvidenceError(f"invalid materialization grid: {work_id}")
            infos = {role: sf.info(directory / f"{role}.f32.wav") for role in "MAV"}
            grids = {(i.frames, i.samplerate, i.channels, i.subtype) for i in infos.values()}
            if len(grids) != 1 or next(iter(grids))[1:] != (SR, 2, "FLOAT"):
                raise ReplayEvidenceError(f"materialization WAV grids differ: {work_id}")
            if recipe.get("frames") != next(iter(grids))[0]:
                raise ReplayEvidenceError(f"materialization frame declaration differs: {work_id}")
            affines = recipe.get("demucs_full_track_affine")
            if not isinstance(affines, dict) or set(affines) != {"M", "A", "V"}:
                raise ReplayEvidenceError(f"materialization lacks M/A/V affines: {work_id}")
            materials[work_id] = {
                "directory": directory,
                "recipe": recipe,
                "report": report,
                "recipe_container_sha256": _sha_file(recipe_path),
                "report_container_sha256": _sha_file(report_path),
            }
        if sample["recipe_id"] != materials[work_id]["recipe"]["recipe_id"]:
            raise ReplayEvidenceError(f"sample/materialization recipe mismatch: {sample_index}")
        start = int(sample["start_frame"])
        if start < 0 or start + crop_frames > int(materials[work_id]["recipe"]["frames"]):
            raise ReplayEvidenceError(f"sample crop is off-grid: {sample_index}")
        for step in STEPS:
            old = fixed_index[(sample_index, step)]
            expected = (work_id, sample["recipe_id"], start, crop_frames)
            actual = (old["work_id"], old["recipe_id"], int(old["start_frame"]), int(old["frames"]))
            if actual != expected:
                raise ReplayEvidenceError(f"fixed cell/sample mismatch: {(sample_index, step)}")
    return config, samples, fixed_index, manifest, materials


def _load_model_for_step(run_dir: Path, config: dict[str, Any], step: int, device: str):
    import torch
    from demucs import states

    if step in {0, 100}:
        path = run_dir / f"model-step-{step:06d}.th"
        model = states.load_model(path, strict=True)
    else:
        model = states.load_model(run_dir / "model-step-000000.th", strict=True)
        checkpoint = torch.load(
            run_dir / f"checkpoint-step-{step:06d}.pt",
            map_location=device,
            weights_only=False,
        )
        if (
            checkpoint.get("step") != step
            or checkpoint.get("config") != config
            or checkpoint.get("base_checkpoint", {}).get("sha256")
            != "sha256:" + config["base_checkpoint"]["sha256"].removeprefix("sha256:")
        ):
            raise ReplayEvidenceError(f"checkpoint binding mismatch at step {step}")
        model.load_state_dict(checkpoint["model"], strict=True)
    if list(model.sources) != list(config["sources"]) or int(model.samplerate) != SR:
        raise ReplayEvidenceError(f"loaded model topology differs at step {step}")
    return model.to(device).eval()


def _surrogate_summary(report: Any) -> dict[str, Any]:
    value = asdict(report)
    value.pop("tiles")
    return value


def _metric_parity(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    maximum = 0.0
    for name in (
        "retained_voice_db_p90",
        "event_hole_db_p90",
        "artifact_ratio_p90",
    ):
        difference = abs(float(actual[name]) - float(expected[name]))
        maximum = max(maximum, difference)
    return maximum


def _sign(value: float, *, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _correlations(pairs: list[tuple[float, float]]) -> dict[str, float | None]:
    if len(pairs) < 2:
        return {"kendall_tau": None, "spearman_rho": None, "pairs": len(pairs)}
    x = np.asarray([pair[0] for pair in pairs], dtype="float64")
    y = np.asarray([pair[1] for pair in pairs], dtype="float64")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"kendall_tau": None, "spearman_rho": None, "pairs": len(pairs)}
    from scipy.stats import kendalltau, spearmanr

    tau = float(kendalltau(x, y).statistic)
    rho = float(spearmanr(x, y).statistic)
    return {
        "kendall_tau": tau if math.isfinite(tau) else None,
        "spearman_rho": rho if math.isfinite(rho) else None,
        "pairs": len(pairs),
    }


def comparison_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build paired deltas and the frozen advancement gate from replay rows."""

    index = {(int(row["sample_index"]), int(row["step"])): row for row in rows}
    expected_samples = sorted({int(row["sample_index"]) for row in rows})
    if set(index) != {(sample, step) for sample in expected_samples for step in STEPS}:
        raise ReplayEvidenceError("replay rows do not form a complete sample/checkpoint grid")
    pairs: list[dict[str, Any]] = []
    for sample in expected_samples:
        baseline = index[(sample, 0)]
        for step in STEPS[1:]:
            candidate = index[(sample, step)]
            axes = {}
            for axis, (risk_name, external_name) in AXES.items():
                base_risk = baseline["surrogate"][risk_name]
                next_risk = candidate["surrogate"][risk_name]
                risk_delta = None if base_risk is None or next_risk is None else float(next_risk) - float(base_risk)
                external_delta = float(candidate["external_metrics"][external_name]) - float(
                    baseline["external_metrics"][external_name]
                )
                axes[axis] = {
                    "surrogate_metric": risk_name,
                    "external_metric": external_name,
                    "surrogate_delta": risk_delta,
                    "external_delta": external_delta,
                    "sign_concordant": (
                        None if risk_delta is None else _sign(risk_delta) == _sign(external_delta)
                    ),
                }
            false_safes = [
                axis for axis in ("voice", "holes")
                if axes[axis]["surrogate_delta"] is not None
                and axes[axis]["surrogate_delta"] < 0
                and axes[axis]["external_delta"] > 0.5
            ]
            pairs.append({
                "sample_index": sample,
                "step": step,
                "baseline_step": 0,
                "work_id": candidate["groups"]["work_id"],
                "session_group_id": candidate["groups"]["session_group_id"],
                "axes": axes,
                "catastrophic_false_safe_axes": false_safes,
                "source_coordinate_measure_fraction_delta": float(
                    candidate["surrogate"]["source_coordinate_measure_fraction"]
                    - baseline["surrogate"]["source_coordinate_measure_fraction"]
                ),
                "direct_fallback_tiles_delta": int(
                    candidate["surrogate"]["direct_fallback_tiles"]
                    - baseline["surrogate"]["direct_fallback_tiles"]
                ),
                "old_declared_total_delta": float(candidate["old_declared_total"] - baseline["old_declared_total"]),
            })

    group_cells: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for pair in pairs:
        for axis in AXES:
            cell = pair["axes"][axis]
            if cell["surrogate_delta"] is not None:
                group_cells[(pair["work_id"], pair["step"], axis)].append(
                    (cell["surrogate_delta"], cell["external_delta"])
                )
    grouped: dict[str, list[tuple[float, float]]] = {axis: [] for axis in AXES}
    grouped_by_step: dict[str, dict[str, list[tuple[float, float]]]] = {
        str(step): {axis: [] for axis in AXES} for step in STEPS[1:]
    }
    for (_work, step, axis), values in group_cells.items():
        aggregate = (
            float(np.median([value[0] for value in values])),
            float(np.median([value[1] for value in values])),
        )
        grouped[axis].append(aggregate)
        grouped_by_step[str(step)][axis].append(aggregate)

    axis_summary = {}
    for axis in AXES:
        cells = [pair["axes"][axis] for pair in pairs if pair["axes"][axis]["sign_concordant"] is not None]
        axis_summary[axis] = {
            "crop_pair_sign_concordance": (
                None if not cells else sum(cell["sign_concordant"] for cell in cells) / len(cells)
            ),
            "crop_pairs": len(cells),
            "held_out_whole_work_rank": _correlations(grouped[axis]),
            "rank_by_checkpoint": {
                step: _correlations(values[axis]) for step, values in grouped_by_step.items()
            },
        }

    step100 = {}
    for axis in AXES:
        cells = [pair["axes"][axis] for pair in pairs if pair["step"] == 100 and pair["axes"][axis]["surrogate_delta"] is not None]
        surrogate = None if not cells else float(np.median([cell["surrogate_delta"] for cell in cells]))
        external = None if not cells else float(np.median([cell["external_delta"] for cell in cells]))
        step100[axis] = {
            "median_surrogate_delta": surrogate,
            "median_external_delta": external,
            "harmful_false_preference": bool(
                surrogate is not None and surrogate < 0 and external is not None and external > 0
            ),
        }

    false_safe_count = sum(len(pair["catastrophic_false_safe_axes"]) for pair in pairs)
    critical = ("voice", "holes")
    sign_pass = all(
        axis_summary[axis]["crop_pair_sign_concordance"] is not None
        and axis_summary[axis]["crop_pair_sign_concordance"] >= 0.80
        for axis in critical
    )
    rank_pass = all(
        axis_summary[axis]["held_out_whole_work_rank"][name] is not None
        and axis_summary[axis]["held_out_whole_work_rank"][name] > 0
        for axis in critical for name in ("kendall_tau", "spearman_rho")
    )
    step100_pass = not any(step100[axis]["harmful_false_preference"] for axis in critical)
    non_silent_coverage = all(
        int(row["surrogate"]["source_coordinate_tiles"])
        + int(row["surrogate"]["direct_fallback_tiles"])
        == int(row["surrogate"]["total_tiles"])
        for row in rows
    )
    checks = {
        "zero_catastrophic_false_safe_pairs": false_safe_count == 0,
        "critical_axis_sign_concordance_at_least_0_80": sign_pass,
        "positive_whole_work_rank_voice_and_holes": rank_pass,
        "complete_non_silent_tile_mode_coverage": non_silent_coverage,
        "no_harmful_step_100_preference": step100_pass,
    }
    summary = {
        "gate_result": "SURROGATE_ALIGNED" if all(checks.values()) else "SURROGATE_REJECTED",
        "checks": checks,
        "catastrophic_false_safe_count": false_safe_count,
        "axis_summary": axis_summary,
        "step_100_preference_audit": step100,
        "crop_checkpoint_pairs": len(pairs),
        "whole_work_groups": len({pair["work_id"] for pair in pairs}),
    }
    return pairs, summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def _verify_cached(output_dir: Path) -> dict[str, Any]:
    if not (output_dir / "COMPLETE").is_file():
        raise ReplayEvidenceError(f"refusing incomplete existing replay: {output_dir}")
    manifest = _json(output_dir / "artifact-manifest.json")
    for name, expected in manifest["files"].items():
        if _sha_file(output_dir / name) != expected:
            raise ReplayEvidenceError(f"cached replay artifact hash mismatch: {name}")
    return _json(output_dir / "summary.json")


def run_replay(
    *,
    run_dir: Path,
    materialized_root: Path,
    manifest_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    if output_dir.exists():
        return _verify_cached(output_dir)
    started = time.time()
    config, samples, fixed_index, manifest, materials = _validate_run_inputs(
        run_dir, materialized_root, manifest_path
    )
    crop_frames = int(_json(run_dir / "fixed-sampled-crop-objective.json")["crop_frames"])
    code_commit = _code_commit()
    rows: list[dict[str, Any]] = []
    max_metric_difference = 0.0
    surrogate_config = SurrogateConfigV2()
    import torch

    for step in STEPS:
        model = _load_model_for_step(run_dir, config, step, device)
        with torch.no_grad():
            for sample_index, sample in enumerate(samples):
                work_id = sample["work_id"]
                material = materials[work_id]
                directory = material["directory"]
                start = int(sample["start_frame"])
                tensors = {
                    role: _read_exact(directory / f"{role}.f32.wav", start=start, frames=crop_frames)
                    for role in "MAV"
                }
                maximum = max(1.0, float(tensors["M"].abs().max()))
                reconstruction_error = float((tensors["M"] - tensors["A"] - tensors["V"]).abs().max())
                if reconstruction_error > 8 * np.finfo(np.float32).eps * maximum:
                    raise ReplayEvidenceError(
                        f"M != A + V at sample {sample_index}: {reconstruction_error}"
                    )
                affine_record = material["recipe"]["demucs_full_track_affine"]["M"]
                mixture = tensors["M"].unsqueeze(0).to(device)
                affine = DemucsAffine(
                    mean=torch.tensor([[float(affine_record["mean"])]], device=device),
                    scale=torch.tensor([[float(affine_record["scale"])]], device=device),
                )
                estimates = forward_demucs_production_affine(model, mixture, affine)
                vocal_hat = estimates[0, int(config["vocal_source_index"])].cpu()
                accompaniment_hat = tensors["M"] - vocal_hat
                old = fixed_index[(sample_index, step)]
                actual_metrics = _source_metrics(
                    accompaniment_hat, tensors["A"], tensors["V"]
                )
                parity = _metric_parity(actual_metrics, old["metrics"])
                max_metric_difference = max(max_metric_difference, parity)
                if parity > METRIC_PARITY_TOLERANCE:
                    raise ReplayEvidenceError(
                        "checkpoint inference metric parity failed at "
                        f"{(sample_index, step)}: {parity} > {METRIC_PARITY_TOLERANCE}"
                    )
                vocal_hat_np = vocal_hat.T.contiguous().numpy().astype("float32", copy=False)
                a_np = tensors["A"].T.contiguous().numpy().astype("float32", copy=False)
                v_np = tensors["V"].T.contiguous().numpy().astype("float32", copy=False)
                report = evaluate_surrogate_v2(
                    vocal_hat_np, a_np, v_np, config=surrogate_config
                )
                manifest_row = manifest[work_id]
                rows.append({
                    "schema": SCHEMA + "/cell",
                    "sample_index": sample_index,
                    "step": step,
                    "work_id": work_id,
                    "recipe_id": sample["recipe_id"],
                    "start_frame": start,
                    "frames": crop_frames,
                    "groups": {
                        "work_id": work_id,
                        "session_group_id": manifest_row.get("group_id"),
                        "singer_group_id": None,
                        "singer_group_unavailable": True,
                        "voice_type_proxies": manifest_row.get("catalog", {}).get("voices", []),
                        "corpus_id": manifest_row.get("corpus_id"),
                    },
                    "input_hashes": {
                        "M_crop_pcm_sha256": _pcm(tensors["M"].T.numpy()),
                        "A_crop_pcm_sha256": _pcm(a_np),
                        "V_crop_pcm_sha256": _pcm(v_np),
                        "vocal_estimate_pcm_sha256": _pcm(vocal_hat_np),
                        "materialization_recipe_container_sha256": material["recipe_container_sha256"],
                        "materialization_report_container_sha256": material["report_container_sha256"],
                        "materialization_full_pcm_sha256": material["report"].get("pcm_sha256"),
                    },
                    "mixture_reconstruction_max_abs": reconstruction_error,
                    "checkpoint_metric_parity_max_abs": parity,
                    "surrogate": _surrogate_summary(report),
                    "old_declared_total": float(old["total"]),
                    "old_components": old["components"],
                    "external_metrics": {
                        name: float(old["metrics"][name])
                        for name in (
                            "retained_voice_db_p90",
                            "event_hole_db_p90",
                            "artifact_ratio_p90",
                        )
                    },
                })
                print(
                    json.dumps({"step": step, "sample": sample_index, "work_id": work_id}),
                    flush=True,
                )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    pairs, gate = comparison_summary(rows)
    input_facts = {
        "schema": SCHEMA + "/inputs",
        "source_run": str(run_dir),
        "source_run_report_sha256": _sha_file(run_dir / "run-report.json"),
        "resolved_config_sha256": _sha_file(run_dir / "resolved-config.yaml"),
        "sample_log_sha256": _sha_file(run_dir / "sample-log.jsonl"),
        "fixed_objective_sha256": _sha_file(run_dir / "fixed-sampled-crop-objective.json"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha_file(manifest_path),
        "model_artifacts": _json(run_dir / "fixed-sampled-crop-objective.json")["model_facts"],
        "replay_code_commit": code_commit,
        "optimizer_steps": 0,
        "device": device,
        "surrogate_config": asdict(surrogate_config),
        "checkpoint_metric_parity_tolerance": METRIC_PARITY_TOLERANCE,
    }
    summary = {
        "schema": SCHEMA + "/summary",
        **gate,
        "status": "needs_human_ab",
        "cells": len(rows),
        "samples": len(samples),
        "steps": list(STEPS),
        "max_checkpoint_metric_parity_abs": max_metric_difference,
        "optimizer_steps": 0,
        "elapsed_seconds": round(time.time() - started, 3),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    try:
        _write_json(temporary / "resolved-inputs.json", input_facts)
        _write_jsonl(temporary / "crop-checkpoint-report.jsonl", rows)
        _write_jsonl(temporary / "paired-deltas.jsonl", pairs)
        _write_json(temporary / "summary.json", summary)
        files = {
            name: _sha_file(temporary / name)
            for name in (
                "resolved-inputs.json",
                "crop-checkpoint-report.jsonl",
                "paired-deltas.jsonl",
                "summary.json",
            )
        }
        _write_json(temporary / "artifact-manifest.json", {"schema": SCHEMA + "/manifest", "files": files})
        (temporary / "COMPLETE").write_text("")
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary
