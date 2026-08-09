#!/usr/bin/env python3
"""Render and evaluate the preregistered five-member oracle routing envelope.

This is an exact-reference diagnostic.  It writes O1/O2/O3 as immutable FLOAT
recipe nodes and refuses implicit resampling, truncation, lossy parents, or an
unverifiable model/checkpoint identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_extract import canon, identity, recipe as recipe_mod
from audio_extract.classical_baselines import FULL_WORKS
from audio_extract.judge_labels import local_source_coordinate_labels
from audio_extract.judge_train import label_targets
from audio_extract.metrics_v2 import stereo_v2
from audio_extract.oracle_convex import (
    ConvexOracleConfig,
    local_data_objective,
    solve_true_convex_oracle,
)
from audio_extract.oracle_tail import (
    ActiveSetConfig,
    corrected_quadratic,
    solve_independent_convex,
    solve_independent_discrete,
)
from audio_extract.oracle_routing import (
    RoutingConfig,
    best_whole_track,
    one_hot_weights,
    render_spectral_route,
    seam_check,
    solve_discrete_routing,
    source_coordinate_statistics,
    stft_stack,
    validate_basis,
)
from audio_extract.separate import render_residual_candidate
from audio_extract.storage import ImmutableWriteError, TrackLayout


BASIS_ORDER = (
    "htdemucs_04573f0d",
    "htdemucs_955717e8",
    "mdx23c",
    "melband",
    "bs_roformer",
)
DEFAULT_WORKS = (
    "bologna_verdi", "bologna_donizetti", "bologna_puccini", "aalto_mozart_dry"
)

SHARED_FAILURE_THRESHOLDS = {
    "retained_voice_coef_max": 0.12,
    "retained_voice_db_max": -18.0,
    "event_hole_db_max": 18.0,
    "alpha_error_max": 0.75,
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _read_exact(path: Path, frames: int | None = None) -> np.ndarray:
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (44_100, 2, "FLOAT"):
        raise ValueError(f"not exact 44.1-kHz stereo FLOAT: {path}: {info}")
    if frames is not None and info.frames != frames:
        raise ValueError(f"frame mismatch {path}: {info.frames} != {frames}")
    value, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 44_100 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid audio values: {path}")
    return value


def _pcm(value: np.ndarray) -> str:
    return identity.artifact_pcm_sha256(value, 44_100, ["FL", "FR"], len(value))


def _input_pcm(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha256": source["input_pcm_sha256"],
        "sample_rate_hz": source["sample_rate_hz"],
        "channel_layout": source["channel_layout"],
        "frames": source["frames"],
        "sample_format": "float32-le-interleaved",
    }


def _external_demucs_member(
    *, name: str, eval_root: Path, work: str, source: dict[str, Any]
) -> dict[str, Any]:
    run_dir = eval_root.parent.parent
    run_report = json.loads((run_dir / "run-report.json").read_text())
    checkpoint = run_report["base_checkpoint"]
    expected_signature = name.rsplit("_", 1)[-1]
    if checkpoint["signature"] != expected_signature:
        raise ValueError(f"{name} resolved to wrong checkpoint: {checkpoint}")
    checkpoint_path = Path(checkpoint["path"])
    if _sha_file(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    path = eval_root / work / "accompaniment.f32.wav"
    value = _read_exact(path, source["frames"])
    recipe = {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": _input_pcm(source),
        "operation": {"type": "mixture_minus_source", "target": "instrumental",
                      "construction": "mixture_minus_source"},
        "model": {
            "model_id": f"HTDemucs-{expected_signature}",
            "weights_sha256": checkpoint["sha256"].removeprefix("sha256:"),
            "adapter": "demucs.apply_model",
            "adapter_revision": "demucs-full-track-affine/v1",
            "sources": ["drums", "bass", "other", "vocals"],
            "vocal_source_index": 3,
        },
        "effective_config": {
            "model_sample_rate_hz": 44_100,
            "normalization": "demucs-full-track-affine/v1",
            "shifts": 0,
            "split": True,
            "overlap_ppm": 250_000,
            "alignment": "source-grid-exact",
        },
        "software": {"audio_extract_commit": run_report["source_commit"]},
    }
    rid = identity.recipe_id(recipe)
    declared_raw = json.loads((eval_root / "report.json").read_text())["works"][work][
        "accompaniment_pcm_sha256"
    ]
    actual_raw = hashlib.sha256(
        np.ascontiguousarray(value, dtype="float32").tobytes()
    ).hexdigest()
    if actual_raw != declared_raw:
        raise ValueError(f"legacy evaluation PCM mismatch for {name}/{work}")
    return {
        "name": name, "recipe_id": rid, "recipe": recipe, "path": str(path),
        "artifact_pcm_sha256": _pcm(value), "container_sha256": _sha_file(path),
        "checkpoint_sha256": checkpoint["sha256"], "value": value,
    }


def _residual_members(
    *, manifest_rows: dict[tuple[str, str], dict], layout: TrackLayout,
    source: dict[str, Any], work: str, generating_commit: str,
) -> dict[str, dict[str, Any]]:
    median = manifest_rows[(work, "median_mdx_mel_bs")]
    result = {}
    for parent_id in median["parent_candidate_recipe_ids"]:
        parent = json.loads((layout.candidate_dir(parent_id) / "recipe.json").read_text())
        model = parent["model"]["model_id"]
        name = ("mdx23c" if "MDX23C" in model else
                "melband" if "mel_band" in model else "bs_roformer")
        record = render_residual_candidate(
            layout, source, vocal_recipe_id=parent_id, code_commit=generating_commit
        )
        cdir = layout.candidate_dir(record["recipe_id"])
        path = cdir / "output.f32.wav"
        value = _read_exact(path, source["frames"])
        if _pcm(value) != record["artifact_pcm_sha256"]:
            raise ValueError(f"residual PCM mismatch: {work}/{name}")
        result[name] = {
            "name": name, "recipe_id": record["recipe_id"],
            "recipe": json.loads((cdir / "recipe.json").read_text()),
            "path": str(path), "artifact_pcm_sha256": record["artifact_pcm_sha256"],
            "container_sha256": _sha_file(path), "parent_vocal_recipe_id": parent_id,
            "value": value,
        }
    if set(result) != {"mdx23c", "melband", "bs_roformer"}:
        raise ValueError(f"incomplete residual basis for {work}: {sorted(result)}")
    return result


def _route_plan_hash(
    mode: str, plan: np.ndarray, config: RoutingConfig,
    solver_config: dict[str, Any] | None = None,
) -> str:
    arr = np.ascontiguousarray(plan)
    header = canon.canonicalize({
        "mode": mode, "shape": list(arr.shape), "dtype": arr.dtype.str,
        "config": config.identity_dict(), "solver_config": solver_config,
    })
    return identity.blob_sha256(header + arr.tobytes())


def _write_route(
    *, layout: TrackLayout, source: dict[str, Any], mode: str, output: np.ndarray,
    parents: list[dict[str, Any]], plan: np.ndarray, config: RoutingConfig,
    code_commit: str, execution: dict[str, Any], truth_pcm: dict[str, str],
    solver_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan_hash = _route_plan_hash(mode, plan, config, solver_config)
    parent_ids = [member["recipe_id"] for member in parents]
    basis_hash = identity.blob_sha256(canon.canonicalize(parent_ids))
    recipe = {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": _input_pcm(source),
        "operation": {"type": "phrase_route", "target": "instrumental"},
        "model": {
            "model_id": f"oracle-routing-{mode.lower()}",
            "weights_sha256": plan_hash.removeprefix("sha256:"),
            "adapter": "audio-extract-oracle-routing",
            "adapter_revision": "source-coordinate-stft-route/v1",
            "members": parent_ids,
            "basis_sha256": basis_hash,
        },
        "effective_config": {
            **config.identity_dict(), "oracle_mode": mode,
            "routing_plan_sha256": plan_hash,
            "solver_config": solver_config,
            "truth_pcm": truth_pcm,
            "alignment": "source-grid-exact",
        },
        "software": {"audio_extract_commit": code_commit},
    }
    rid = identity.recipe_id(recipe)
    cdir = layout.candidate_dir(rid)
    artifact = _pcm(output)
    cached = (cdir / "COMPLETE").is_file()
    if cached:
        if json.loads((cdir / "recipe.json").read_text()) != recipe:
            raise ValueError(f"cached route recipe mismatch: {rid}")
        reopened = _read_exact(cdir / "output.f32.wav", source["frames"])
        if _pcm(reopened) != artifact or (cdir / "output.pcm.sha256").read_text().strip() != artifact:
            raise ValueError(f"cached route output mismatch: {rid}")
    else:
        fd, name = tempfile.mkstemp(suffix=".f32.wav")
        os.close(fd)
        tmp = Path(name)
        sf.write(tmp, output, 44_100, subtype="FLOAT")
        try:
            layout.write_candidate(
                rid, recipe,
                {**identity.execution_fingerprint(), **execution,
                 "parents": [{k: v for k, v in member.items() if k != "value"}
                             for member in parents]},
                tmp, artifact,
            )
        except ImmutableWriteError:
            cached = True
        finally:
            tmp.unlink(missing_ok=True)
    return {
        "mode": mode, "recipe_id": rid, "path": str(cdir / "output.f32.wav"),
        "artifact_pcm_sha256": artifact,
        "container_sha256": _sha_file(cdir / "output.f32.wav"), "cached": cached,
        "routing_plan_sha256": plan_hash,
    }


def _metrics(value: np.ndarray, accompaniment: np.ndarray, vocal: np.ndarray) -> tuple[dict, tuple]:
    labels = local_source_coordinate_labels(
        value, accompaniment, vocal, tile_frames=round(0.5 * 44_100),
        hop_frames=round(0.25 * 44_100),
    )
    result = label_targets(labels)
    for observation in stereo_v2(value, accompaniment):
        if observation["available"]:
            result[observation["metric"]] = observation["value"]
    return result, labels


def _worst_identifiable(labels: tuple) -> dict[str, Any]:
    available = [label for label in labels if label.available]
    if not available:
        return {"available": False}
    def score(label):
        voice = max(0.0, float(label.retained_voice_energy_ratio or 0.0))
        return (float(label.accompaniment_hole_db or 0.0) / 20.0
                + np.sqrt(voice) + float(label.orthogonal_artifact_ratio or 0.0))
    label = max(available, key=score)
    return {
        "available": True, "tile_index": label.tile_index,
        "start_frame": label.start_frame, "end_frame": label.end_frame,
        "start_seconds": label.start_frame / 44_100,
        "end_seconds": label.end_frame / 44_100,
        "accompaniment_hole_db": label.accompaniment_hole_db,
        "retained_voice_energy_db": label.retained_voice_energy_db,
        "orthogonal_artifact_ratio": label.orthogonal_artifact_ratio,
        "condition_number": label.condition_number,
        "composite_risk": score(label),
    }


def _local_objective_summary(weights: np.ndarray, quadratic) -> dict[str, Any]:
    values = local_data_objective(weights, quadratic)[quadratic.available]
    if not len(values) or not np.all(np.isfinite(values)):
        raise RuntimeError("corrected local objective is unavailable or non-finite")
    return {
        "available_cells": int(len(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _shared_failure_cells(stats, config: ConvexOracleConfig, quadratic) -> dict[str, Any]:
    """Report cells where every whole-estimator member fails a release critical axis."""
    alpha_abs = np.abs(stats.alpha)
    beta_abs = np.abs(stats.beta)
    epsilon = config.reference_floor_relative * np.maximum(
        stats.accompaniment_energy + stats.vocal_energy, np.finfo(float).tiny
    )
    retained_ratio = (
        beta_abs ** 2 * stats.vocal_energy[..., None]
        / (alpha_abs ** 2 * stats.accompaniment_energy[..., None] + epsilon[..., None])
    )
    retained_db = 10.0 * np.log10(np.maximum(retained_ratio, np.finfo(float).tiny))
    hole_db = np.maximum(
        0.0, -20.0 * np.log10(np.maximum(alpha_abs, np.finfo(float).tiny))
    )
    alpha_error = np.abs(stats.alpha - 1.0)
    failures = (
        (beta_abs > SHARED_FAILURE_THRESHOLDS["retained_voice_coef_max"])
        | (retained_db > SHARED_FAILURE_THRESHOLDS["retained_voice_db_max"])
        | (hole_db > SHARED_FAILURE_THRESHOLDS["event_hole_db_max"])
        | (alpha_error > SHARED_FAILURE_THRESHOLDS["alpha_error_max"])
    )
    shared = stats.available & np.all(failures, axis=-1)
    vertex_cost = (
        np.diagonal(quadratic.gram, axis1=-2, axis2=-1)
        - 2.0 * quadratic.linear + quadratic.constant[..., None]
    )
    coordinates = np.argwhere(shared)
    ordered = sorted(
        coordinates.tolist(),
        key=lambda item: float(np.min(vertex_cost[item[0], item[1]])),
        reverse=True,
    )
    examples = []
    for qi, bi in ordered[:20]:
        examples.append({
            "time_cell": qi,
            "frequency_band": bi,
            "stft_frame_range": list(stats.time_frame_ranges[qi]),
            "frequency_bin_range": list(stats.frequency_bin_ranges[bi]),
            "best_vertex_corrected_objective": float(np.min(vertex_cost[qi, bi])),
            "member_failure_count": int(np.count_nonzero(failures[qi, bi])),
        })
    available = int(stats.available.sum())
    return {
        "definition": "all basis members fail at least one frozen exact-release critical axis",
        "thresholds": SHARED_FAILURE_THRESHOLDS,
        "count": int(shared.sum()),
        "available_cells": available,
        "fraction": float(shared.sum() / max(available, 1)),
        "worst_examples": examples,
    }


def _load_no_vocal_manifest(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    doc = json.loads(path.read_text())
    if doc.get("schema") != "audio-extract/oracle-no-vocal-basis/v1":
        raise ValueError("wrong no-vocal manifest schema")
    members = {row["name"]: row for row in doc["members"]}
    if tuple(members) != BASIS_ORDER:
        raise ValueError(f"no-vocal basis order mismatch: {tuple(members)}")
    return members


def _no_vocal_metrics(
    members: dict[str, dict[str, Any]], basis: list[dict[str, Any]],
    stats, routes: dict[str, np.ndarray], accompaniment: np.ndarray,
    config: RoutingConfig,
) -> dict[str, Any]:
    values = []
    individual = {}
    for name in BASIS_ORDER:
        record = members[name]
        path = Path(record["path"])
        value = _read_exact(path, len(accompaniment))
        if _pcm(value) != record["artifact_pcm_sha256"] or _sha_file(path) != record["container_sha256"]:
            raise ValueError(f"no-vocal basis identity mismatch: {name}")
        values.append(value)
        removed = accompaniment.astype(np.float64) - value.astype(np.float64)
        individual[name] = float(np.sum(removed * removed) /
                                 max(np.sum(accompaniment.astype(np.float64) ** 2), 1e-12))
    spectra = stft_stack(values, config)
    output = {"basis": individual, "routes": {}}
    for mode, weights in routes.items():
        if mode == "O1":
            index = int(np.argmax(weights[0, 0]))
            routed = values[index].copy()
        else:
            routed = render_spectral_route(
                spectra, weights, stats, frames=len(accompaniment), config=config
            )
        removed = accompaniment.astype(np.float64) - routed.astype(np.float64)
        output["routes"][mode] = {
            "false_positive_energy_ratio": float(
                np.sum(removed * removed) /
                max(np.sum(accompaniment.astype(np.float64) ** 2), 1e-12)
            ),
            "artifact_pcm_sha256": _pcm(routed),
            "seam_check": seam_check(routed, stats, config),
        }
    return output


def _decision(report_works: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for mode in ("O0D", "O0C", "O2", "O3"):
        verdi_o1 = report_works["bologna_verdi"]["outputs"]["O1"]["metrics"]
        verdi = report_works["bologna_verdi"]["outputs"][mode]["metrics"]
        don_o1 = report_works["bologna_donizetti"]["outputs"]["O1"]["metrics"]
        don = report_works["bologna_donizetti"]["outputs"][mode]["metrics"]
        aalto_o1 = report_works["aalto_mozart_dry"]["outputs"]["O1"]["metrics"]
        aalto = report_works["aalto_mozart_dry"]["outputs"][mode]["metrics"]
        gains = {
            "verdi_retained_voice_db": (
                verdi_o1["retained_voice_db_p90"] - verdi["retained_voice_db_p90"]),
            "donizetti_event_hole_db": (
                don_o1["event_hole_db_p90"] - don["event_hole_db_p90"]),
        }
        regressions = {
            "verdi_event_hole_db": verdi["event_hole_db_p90"] - verdi_o1["event_hole_db_p90"],
            "donizetti_retained_voice_db": (
                don["retained_voice_db_p90"] - don_o1["retained_voice_db_p90"]),
            "aalto_retained_voice_db": (
                aalto["retained_voice_db_p90"] - aalto_o1["retained_voice_db_p90"]),
            "aalto_event_hole_db": aalto["event_hole_db_p90"] - aalto_o1["event_hole_db_p90"],
            "aalto_stereo_width_db": (
                aalto["stereo_width_dev_db/v2"] - aalto_o1["stereo_width_dev_db/v2"]),
        }
        no_vocal = report_works["aalto_mozart_dry"].get("no_vocal")
        no_vocal_ratio = None
        if no_vocal:
            base = no_vocal["routes"]["O1"]["false_positive_energy_ratio"]
            candidate = no_vocal["routes"][mode]["false_positive_energy_ratio"]
            no_vocal_ratio = 1.0 if base <= 1e-15 and candidate <= 1e-15 else (
                float("inf") if base <= 1e-15 else candidate / base
            )
        failures = []
        if max(gains.values()) < 1.5:
            failures.append("neither critical target axis improves by 1.5 dB")
        for name, value in regressions.items():
            if value > 0.5:
                failures.append(f"{name} regresses by {value:.4f} > 0.5")
        if no_vocal_ratio is None:
            failures.append("Aalto no-vocal control unavailable")
        elif no_vocal_ratio > 1.05:
            failures.append(f"Aalto no-vocal FP ratio {no_vocal_ratio:.4f} > 1.05")
        result[mode] = {
            "diagnostic_only": mode.startswith("O0"),
            "actionable_oracle_gap": not failures, "critical_gains_db": gains,
            "stability_regressions": regressions,
            "no_vocal_candidate_over_o1": no_vocal_ratio,
            "failures": failures,
        }
    o0c_gain = max(result["O0C"]["critical_gains_db"].values())
    o3_gain = max(result["O3"]["critical_gains_db"].values())
    shared = {
        work: report["shared_failure_cells"]
        for work, report in report_works.items()
    }
    shared_count = sum(item["count"] for item in shared.values())
    small_o0c = o0c_gain < 1.5
    result["interpretation"] = {
        "small_o0c_critical_axis_envelope": small_o0c,
        "o0c_max_critical_gain_db": o0c_gain,
        "o3_max_critical_gain_db": o3_gain,
        "o0c_large_o3_small": o0c_gain >= 1.5 and o3_gain < 1.5,
        "shared_failure_cell_count": shared_count,
        "shared_failure_cells_present": shared_count > 0,
        "basis_insufficiency_evidence": small_o0c and shared_count > 0,
        "guard": (
            "basis insufficiency requires both a small O0C critical-axis envelope "
            "and shared critical failures across every basis member"
        ),
        "per_work_shared_failure_cells": shared,
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = RoutingConfig()
    convex_config = ConvexOracleConfig()
    active_set_config = ActiveSetConfig()
    code_commit = _git_commit()
    rows = [json.loads(line) for line in args.candidate_manifest.read_text().splitlines()]
    manifest_rows = {(row["work_id"], row["candidate"]): row for row in rows}
    no_vocal = _load_no_vocal_manifest(args.no_vocal_manifest)
    works = tuple(args.work)
    if set(works) - set(FULL_WORKS):
        raise ValueError(f"unknown exact work(s): {sorted(set(works) - set(FULL_WORKS))}")
    report: dict[str, Any] = {
        "schema": "audio-extract/oracle-routing-envelope/v1",
        "status": "final", "code_commit": code_commit,
        "candidate_basis": list(BASIS_ORDER), "works": {},
        "config": config.to_dict(),
        "convex_o0_config": {
            "quadratic": convex_config.to_dict(),
            "active_set": active_set_config.to_dict(),
        },
        "convex_o3_config": convex_config.to_dict(),
        "shared_quadratic": "audio_extract.oracle_tail.corrected_quadratic/v1",
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": _sha_file(args.candidate_manifest),
    }
    for work in works:
        print(json.dumps({"stage": "load", "work": work}), flush=True)
        layout = TrackLayout(args.lib_root, work)
        source = json.loads((layout.source_dir / "source.json").read_text())
        truth_dir = args.truth_root / work
        mixture = _read_exact(truth_dir / "mix_with_voice.wav", source["frames"])
        accompaniment = _read_exact(truth_dir / "orchestra_only.wav", source["frames"])
        vocal = _read_exact(truth_dir / "voice_ref.wav", source["frames"])
        if _pcm(mixture) != source["input_pcm_sha256"]:
            raise ValueError(f"truth/source mixture identity mismatch: {work}")
        residuals = _residual_members(
            manifest_rows=manifest_rows, layout=layout, source=source, work=work,
            generating_commit=manifest_rows[(work, "median_mdx_mel_bs")]["code_commit"],
        )
        members = [
            _external_demucs_member(name="htdemucs_04573f0d", eval_root=args.htdemucs_045,
                                    work=work, source=source),
            _external_demucs_member(name="htdemucs_955717e8", eval_root=args.htdemucs_955,
                                    work=work, source=source),
            residuals["mdx23c"], residuals["melband"], residuals["bs_roformer"],
        ]
        if tuple(member["name"] for member in members) != BASIS_ORDER:
            raise RuntimeError("candidate basis order changed")
        values, accompaniment, vocal = validate_basis(
            [member.pop("value") for member in members], accompaniment, vocal
        )
        # Restore runtime-only values after validation; they are excluded from execution JSON.
        for member, value in zip(members, values):
            member["value"] = value
        print(json.dumps({"stage": "stft", "work": work}), flush=True)
        candidate_spectra = stft_stack(values, config)
        truth_spectra = stft_stack([accompaniment, vocal], config)
        stats = source_coordinate_statistics(
            candidate_spectra, truth_spectra[0], truth_spectra[1], config
        )
        quadratic, corrected_unary = corrected_quadratic(stats, convex_config)
        stats = replace(stats, unary_risk=corrected_unary)
        o1_index, o1_risks = best_whole_track(stats)
        print(json.dumps({"stage": "O0_independent", "work": work}), flush=True)
        o0d = solve_independent_discrete(
            quadratic, fallback_index=o1_index
        )
        print(json.dumps({"stage": "O2_milp", "work": work,
                          "cells": int(stats.available.size),
                          "available": int(stats.available.sum())}), flush=True)
        o2 = solve_discrete_routing(stats, config)
        o0c = solve_independent_convex(
            quadratic, fallback_index=o1_index, config=active_set_config
        )
        print(json.dumps({"stage": "O3_convex", "work": work}), flush=True)
        o3 = solve_true_convex_oracle(
            stats, o1_index=o1_index, o2_labels=o2.labels, config=convex_config
        )
        o1_weights = np.zeros(stats.unary_risk.shape, dtype=np.float64)
        o1_weights[..., o1_index] = 1.0
        routes = {
            "O0D": o0d.weights,
            "O0C": o0c.weights,
            "O1": o1_weights,
            "O2": one_hot_weights(o2.labels, len(members)),
            "O3": o3.weights,
        }
        outputs = {
            "O0D": render_spectral_route(
                candidate_spectra, routes["O0D"], stats,
                frames=len(mixture), config=config,
            ),
            "O0C": render_spectral_route(
                candidate_spectra, routes["O0C"], stats,
                frames=len(mixture), config=config,
            ),
            "O1": values[o1_index].copy(),
            "O2": render_spectral_route(candidate_spectra, routes["O2"], stats,
                                         frames=len(mixture), config=config),
            "O3": render_spectral_route(candidate_spectra, routes["O3"], stats,
                                         frames=len(mixture), config=config),
        }
        truth_pcm = {"accompaniment": _pcm(accompaniment), "vocal": _pcm(vocal)}
        work_report: dict[str, Any] = {
            "frames": len(mixture),
            "basis": [{k: v for k, v in member.items() if k != "value"}
                      for member in members],
            "identifiability": {
                "available_cells": int(stats.available.sum()),
                "total_cells": int(stats.available.size),
                "condition_number_p90": float(np.percentile(
                    stats.condition_number[stats.available], 90)),
                "masked_cells": int((~stats.available).sum()),
            },
            "shared_failure_cells": _shared_failure_cells(
                stats, convex_config, quadratic
            ),
            "O0D": {
                "diagnostic_only": True,
                "solver": "independent corrected-quadratic vertex minimum",
                "objective": o0d.data_objective,
                "local_objective": _local_objective_summary(o0d.weights, quadratic),
                "selection_counts": {
                    name: o0d.occupancy[i]
                    for i, name in enumerate(BASIS_ORDER)
                },
            },
            "O0C": {
                "diagnostic_only": True,
                "solver": "independent corrected-quadratic convex hull",
                "objective": o0c.data_objective,
                "local_objective": _local_objective_summary(o0c.weights, quadratic),
                "global_certificate": "all-support KKT enumeration",
                "active_set_size_histogram": o0c.active_set_size_histogram,
                "interpolated_cells": o0c.interpolated_cells,
                "interpolated_fraction": o0c.interpolated_fraction,
                "max_simplex_error": o0c.max_simplex_error,
                "max_normalized_stationarity_residual": (
                    o0c.max_normalized_stationarity_residual
                ),
                "min_normalized_inactive_reduced_gradient": (
                    o0c.min_normalized_inactive_reduced_gradient
                ),
                "cell_normalization_scale_min": o0c.cell_normalization_scale_min,
                "cell_normalization_scale_max": o0c.cell_normalization_scale_max,
                "mean_weights": {
                    name: o0c.mean_weights[i]
                    for i, name in enumerate(BASIS_ORDER)
                },
            },
            "O1": {"selected_index": o1_index, "selected_member": BASIS_ORDER[o1_index],
                   "objective_definition": "shared corrected PSD quadratic vertex unary",
                   "whole_track_risk": {name: float(value)
                                        for name, value in zip(BASIS_ORDER, o1_risks)}},
            "O2": {
                "objective_definition": "shared corrected PSD quadratic vertex unary + Potts switches",
                "objective": o2.objective, "data_objective": o2.data_objective,
                "temporal_switches": o2.temporal_switches,
                "frequency_switches": o2.frequency_switches,
                "solver_status": o2.solver_status, "mip_gap": o2.mip_gap,
                "mip_node_count": o2.mip_node_count,
                "selection_counts": {name: int(np.count_nonzero(o2.labels == i))
                                     for i, name in enumerate(BASIS_ORDER)},
            },
            "O3": {
                "objective": o3.objective, "data_objective": o3.data_objective,
                "temporal_smoothness": o3.temporal_smoothness,
                "frequency_smoothness": o3.frequency_smoothness,
                "initialization": o3.initialization, "iterations": o3.iterations,
                "converged": o3.converged,
                "projected_gradient_mapping_inf": o3.projected_gradient_mapping_inf,
                "simplex_error": o3.simplex_error,
                "min_weight": o3.min_weight,
                "solution_objective_spread": o3.solution_objective_spread,
                "best_vertex_index": o3.best_vertex_index,
                "best_vertex_objective": o3.best_vertex_objective,
                "start_objectives": o3.start_objectives,
                "restart_count": o3.restart_count,
                "accelerated_steps": o3.accelerated_steps,
                "mean_weights": {name: float(o3.weights[..., i].mean())
                                 for i, name in enumerate(BASIS_ORDER)},
            },
            "outputs": {},
        }
        for mode in ("O0D", "O0C", "O1", "O2", "O3"):
            print(json.dumps({"stage": "evaluate", "work": work, "mode": mode}), flush=True)
            plan = (
                o0d.labels.astype(np.int32) if mode == "O0D" else
                o0c.weights.astype(np.float64) if mode == "O0C" else
                np.asarray([o1_index], dtype=np.int32) if mode == "O1" else
                o2.labels.astype(np.int32) if mode == "O2" else
                o3.weights.astype(np.float64)
            )
            execution = {
                "oracle_diagnostic_only": True,
                "routing_plan": plan.tolist(),
                "source_coordinate_available_cells": int(stats.available.sum()),
                "source_coordinate_total_cells": int(stats.available.size),
                "solver": work_report[mode],
            }
            artifact = _write_route(
                layout=layout, source=source, mode=mode, output=outputs[mode],
                parents=members, plan=plan, config=config, code_commit=code_commit,
                execution=execution, truth_pcm=truth_pcm,
                solver_config=(
                    {**convex_config.identity_dict(),
                     "temporal_l2_weight": "0.0",
                     "frequency_l2_weight": "0.0",
                     "solver": "independent_vertex_minimum/v1"}
                    if mode == "O0D" else
                    {"quadratic": convex_config.identity_dict(),
                     "active_set": active_set_config.identity_dict(),
                     "solver": "all_support_kkt_convex_hull/v1"}
                    if mode == "O0C" else
                    {"quadratic": convex_config.identity_dict(),
                     "solver": "corrected_whole_candidate_mean/v1"}
                    if mode == "O1" else
                    {"quadratic": convex_config.identity_dict(),
                     "solver": "corrected_potts_milp/v1"}
                    if mode == "O2" else
                    convex_config.identity_dict()
                ),
            )
            metrics, labels = _metrics(outputs[mode], accompaniment, vocal)
            work_report["outputs"][mode] = {
                **artifact, "metrics": metrics,
                "worst_identifiable_event": _worst_identifiable(labels),
                "seam_check": seam_check(outputs[mode], stats, config),
            }
        if work == "aalto_mozart_dry" and no_vocal is not None:
            work_report["no_vocal"] = _no_vocal_metrics(
                no_vocal, members, stats, routes, accompaniment, config
            )
        report["works"][work] = work_report
        print(json.dumps({"stage": "complete", "work": work}), flush=True)
    required = {"bologna_verdi", "bologna_donizetti", "aalto_mozart_dry"}
    if required <= set(report["works"]):
        report["decision"] = _decision(report["works"])
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != text:
        raise RuntimeError(f"refusing to rewrite differing report: {args.output}")
    if not args.output.exists():
        args.output.write_text(text)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--lib-root", required=True, type=Path)
    parser.add_argument("--htdemucs-045", required=True, type=Path,
                        help="step-000000 evaluation root")
    parser.add_argument("--htdemucs-955", required=True, type=Path,
                        help="step-000000 evaluation root")
    parser.add_argument("--no-vocal-manifest", type=Path)
    parser.add_argument("--work", action="append", choices=FULL_WORKS,
                        default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.work is None:
        args.work = list(DEFAULT_WORKS)
    report = run(args)
    print(json.dumps({
        "status": report["status"], "works": list(report["works"]),
        "decision": report.get("decision"), "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
