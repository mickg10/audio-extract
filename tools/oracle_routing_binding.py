#!/usr/bin/env python3
"""Run the binding certified opera-routing experiment.

This tool is intentionally separate from ``oracle_routing_envelope.py``.  It
uses the certified convex/exact-fallback core, expands and decoded-PCM
deduplicates the frozen basis, publishes transform-identity controls, and runs
one frozen 2.0/1.0/0.5-second sensitivity across every work.  Truth is used
only by this diagnostic; every rendered route remains an immutable FLOAT DAG
node and is never a production selector by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import librosa
import numpy as np

from audio_extract import identity
from audio_extract.classical_baselines import FULL_WORKS
from audio_extract.classical_release import exact_metrics
from audio_extract.metrics_v2 import stereo_v2
from audio_extract.oracle_routing import (
    RoutingConfig,
    _frequency_ranges,
    _time_ranges,
    one_hot_weights as legacy_one_hot_weights,
    render_spectral_route,
    seam_check,
    stft_stack,
    validate_basis,
)
from audio_extract.oracle_routing_certified import (
    CertifiedRoutingError,
    CertifiedRoutingConfig,
    best_whole_track,
    build_spectral_quadratic_grid,
    one_hot_weights,
    solve_convex_certified,
    solve_discrete_global,
    unary_costs,
)
from audio_extract.storage import TrackLayout
from tools.oracle_routing_envelope import (
    _external_demucs_member,
    _metrics as _source_metrics,
    _pcm,
    _read_exact,
    _residual_members,
    _sha_file,
    _worst_identifiable,
    _write_removed_vocal,
    _write_route,
)


SCHEMA = "audio-extract/oracle-routing-binding/v2"
DEFAULT_WORKS = (
    "bologna_verdi",
    "bologna_donizetti",
    "bologna_puccini",
    "aalto_mozart_dry",
)
BASE_ORDER = (
    "htdemucs_04573f0d",
    "htdemucs_955717e8",
    "mdx23c",
    "melband",
    "bs_roformer",
    "median_mdx_mel_bs",
    "geomedian_mdx_mel_bs",
    "convex_fusion_uniform",
)
RESOLUTIONS = (2.0, 1.0, 0.5)
PRIMARY_RESOLUTION = "2"


def _local_path(value: str) -> Path:
    return Path(value.removeprefix("research6:"))


def _manifest_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["work_id"], row["candidate"])] = row
    return rows


def _manifest_member(row: dict[str, Any], layout: TrackLayout,
                     frames: int) -> dict[str, Any]:
    path = _local_path(row["path"])
    value = _read_exact(path, frames)
    if _sha_file(path) != row["container_sha256"] or _pcm(value) != row[
        "artifact_pcm_sha256"
    ]:
        raise ValueError(f"candidate-manifest identity mismatch: {row['candidate']}")
    recipe_path = layout.candidate_dir(row["recipe_id"]) / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    if identity.recipe_id(recipe) != row["recipe_id"]:
        raise ValueError(f"candidate recipe mismatch: {row['candidate']}")
    return {
        "name": row["candidate"],
        "recipe_id": row["recipe_id"],
        "recipe": recipe,
        "path": str(path),
        "artifact_pcm_sha256": row["artifact_pcm_sha256"],
        "container_sha256": row["container_sha256"],
        "executed_model_bundle_hashes": row.get(
            "executed_model_bundle_hashes", []
        ),
        "value": value,
    }


def _deduplicate_members(
    members: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    """Deduplicate by decoded PCM while retaining every alias in evidence."""

    unique: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    pcm_to_name: dict[str, str] = {}
    for member in members:
        name = str(member["name"])
        pcm = str(member["artifact_pcm_sha256"])
        if name in aliases:
            raise ValueError(f"duplicate basis name: {name}")
        if pcm in pcm_to_name:
            aliases[name] = pcm_to_name[pcm]
            continue
        aliases[name] = name
        pcm_to_name[pcm] = name
        unique.append(member)
    indices = {member["name"]: index for index, member in enumerate(unique)}
    alias_indices = {alias: indices[canonical] for alias, canonical in aliases.items()}
    required = {"median_mdx_mel_bs", "mdx23c", "melband", "bs_roformer"}
    if not required <= set(alias_indices):
        raise ValueError(f"binding basis lacks required aliases: {sorted(required-set(alias_indices))}")
    return unique, aliases, alias_indices


def _scaled_configs(tile_seconds: float) -> tuple[RoutingConfig, CertifiedRoutingConfig]:
    if tile_seconds not in RESOLUTIONS:
        raise ValueError(f"resolution is not frozen: {tile_seconds}")
    scale = 2.0 / tile_seconds
    routing = RoutingConfig(
        tile_seconds=tile_seconds,
        temporal_switch_penalty=0.05 * scale,
        frequency_switch_penalty=0.05,
    )
    certified = CertifiedRoutingConfig(
        temporal_switch_penalty=routing.temporal_switch_penalty,
        frequency_switch_penalty=routing.frequency_switch_penalty,
        temporal_weight_smoothness=0.05 * scale * scale,
        frequency_weight_smoothness=0.05,
    )
    return routing, certified


def _routing_view(grid, time_ranges, frequency_ranges):
    """Supply only the geometry fields used by the common FLOAT renderer."""

    t, b, k = grid.validate()
    return SimpleNamespace(
        unary_risk=np.zeros((t, b, k), dtype=np.float64),
        available=np.ones((t, b), dtype=bool),
        time_frame_ranges=tuple(time_ranges),
        frequency_bin_ranges=tuple(frequency_ranges),
    )


def _render(candidate_spectra: np.ndarray, weights: np.ndarray, view,
            frames: int, routing: RoutingConfig) -> np.ndarray:
    return render_spectral_route(
        candidate_spectra, weights, view, frames=frames, config=routing
    )


def _combined_spectrum(candidate_spectra: np.ndarray, weights: np.ndarray,
                       time_ranges, frequency_ranges) -> np.ndarray:
    k, _, frequencies, times = candidate_spectra.shape
    weight_map = np.empty((k, frequencies, times), dtype=np.float64)
    for ti, (t0, t1) in enumerate(time_ranges):
        for bi, (f0, f1) in enumerate(frequency_ranges):
            weight_map[:, f0:f1, t0:t1] = weights[ti, bi, :, None, None]
    return np.sum(candidate_spectra * weight_map[:, None], axis=0)


def _frequency_boundary_check(spectrum: np.ndarray, frequency_ranges) -> dict[str, Any]:
    magnitude = np.log1p(np.abs(np.asarray(spectrum, dtype=np.complex64)))
    adjacent = np.abs(np.diff(magnitude, axis=1)).reshape(-1)
    reference = max(float(np.percentile(adjacent, 99.0)), np.finfo(float).tiny)
    boundaries = [start for start, _ in frequency_ranges[1:]]
    jumps = np.concatenate([
        np.abs(magnitude[:, index] - magnitude[:, index - 1]).reshape(-1)
        for index in boundaries
    ]) if boundaries else np.zeros(0, dtype=np.float32)
    return {
        "boundary_count": len(boundaries),
        "p99_adjacent_log_magnitude_derivative": reference,
        "p99_boundary_jump": float(np.percentile(jumps, 99.0)) if len(jumps) else 0.0,
        "max_boundary_jump": float(jumps.max(initial=0.0)),
        "p99_boundary_over_p99_adjacent": (
            float(np.percentile(jumps, 99.0) / reference) if len(jumps) else 0.0
        ),
        "max_boundary_over_p99_adjacent": float(jumps.max(initial=0.0) / reference),
    }


def _mr_stft_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    error = candidate.astype(np.float64) - reference.astype(np.float64)
    ratios = []
    for n_fft in (512, 2048, 8192):
        channels = []
        for channel in range(2):
            e = librosa.stft(
                error[:, channel], n_fft=n_fft, hop_length=n_fft // 4,
                window="hann", center=True, pad_mode="constant",
            )
            r = librosa.stft(
                reference[:, channel], n_fft=n_fft, hop_length=n_fft // 4,
                window="hann", center=True, pad_mode="constant",
            )
            channels.append(float(
                np.mean(np.abs(e)) / max(np.mean(np.abs(r)), 1e-12)
            ))
        ratios.append(float(np.mean(channels)))
    return float(np.mean(ratios))


def _transform_identity_metrics(raw: np.ndarray, transformed: np.ndarray) -> dict[str, Any]:
    error = transformed.astype(np.float64) - raw.astype(np.float64)
    stereo = {
        row["metric"]: float(row["value"])
        for row in stereo_v2(transformed, raw) if row["available"]
    }
    return {
        "raw_artifact_pcm_sha256": _pcm(raw),
        "transformed_artifact_pcm_sha256": _pcm(transformed),
        "decoded_pcm_equal": bool(np.array_equal(raw, transformed)),
        "max_abs_error": float(np.max(np.abs(error))),
        "rms_error": float(np.sqrt(np.mean(np.square(error)))),
        "complex_mr_stft_error_ratio": _mr_stft_error(transformed, raw),
        "stereo_width_change_db": stereo["stereo_width_dev_db/v2"],
        "coherence_change": stereo["interchannel_coherence_dev/v2"],
    }


def _route_metrics(value: np.ndarray, accompaniment: np.ndarray,
                   vocal: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = exact_metrics(value, accompaniment, vocal, 44_100)
    _, labels = _source_metrics(value, accompaniment, vocal)
    return metrics, _worst_identifiable(labels)


def _ratio(candidate: float, baseline: float) -> float:
    return (1.0 if candidate <= 1e-15 else math.inf) if baseline <= 1e-15 else (
        candidate / baseline
    )


def _stability_failures(candidate: dict[str, Any], baseline: dict[str, Any],
                        label: str) -> list[str]:
    failures = []
    ratio_limits = {
        "artifact_ratio_p90": 1.10,
        "transient_loss/v2": 1.10,
        "transient_excess/v2": 1.10,
    }
    for metric, limit in ratio_limits.items():
        ratio = _ratio(float(candidate[metric]), float(baseline[metric]))
        if ratio > limit:
            failures.append(f"{label} {metric} ratio {ratio:.6g} > {limit}")
    additive_limits = {
        "stereo_width_dev_db/v2": 0.5,
        "interchannel_coherence_dev/v2": 0.05,
        "band_deficit_db/v2": 0.5,
        "contiguous_hole_db/v2": 0.5,
        "erb_envelope_dist_db/v2": 0.5,
    }
    for metric, limit in additive_limits.items():
        regression = float(candidate[metric]) - float(baseline[metric])
        if regression > limit:
            failures.append(f"{label} {metric} regression {regression:.6g} > {limit}")
    return failures


def _decision(works: dict[str, Any]) -> dict[str, Any]:
    decisions = {}
    for resolution in ("2", "1", "0.5"):
        for mode in ("O2", "O3"):
            key = f"{mode}@{resolution}s"
            failures: list[str] = []
            rejected = {
                work: report["resolutions"][resolution][mode].get(
                    "certificate_error"
                )
                for work, report in works.items()
                if "metrics" not in report["resolutions"][resolution][mode]
            }
            if rejected:
                decisions[key] = {
                    "binding_gate_passed": False,
                    "failures": [
                        f"{work}/{key} rejected: {reason}"
                        for work, reason in sorted(rejected.items())
                    ],
                }
                continue
            for target, metric in (
                ("bologna_verdi", "retained_voice_db_p90"),
                ("bologna_donizetti", "event_hole_db_p90"),
            ):
                route = works[target]["resolutions"][resolution][mode]["metrics"]
                report = works[target]
                bases = {
                    "O1": report["resolutions"][resolution]["O1_baseline"],
                    "median": report["baselines"]["median"],
                    "mdx": report["baselines"]["mdx"],
                }
                gains = {
                    name: float(row["metrics"][metric]) - float(route[metric])
                    for name, row in bases.items()
                }
                if min(gains.values()) < 1.5:
                    failures.append(
                        f"{target} {metric} minimum baseline gain "
                        f"{min(gains.values()):.6g} < 1.5"
                    )
                opposite = (
                    "event_hole_db_p90" if target == "bologna_verdi"
                    else "retained_voice_db_p90"
                )
                regressions = {
                    name: float(route[opposite]) - float(row["metrics"][opposite])
                    for name, row in bases.items()
                }
                if max(regressions.values()) > 0.5:
                    failures.append(
                        f"{target} {opposite} maximum regression "
                        f"{max(regressions.values()):.6g} > 0.5"
                    )
            for work, report in works.items():
                route_row = report["resolutions"][resolution][mode]
                median = report["baselines"]["median"]
                failures.extend(_stability_failures(
                    route_row["metrics"], median["metrics"], f"{work}/{key}"
                ))
                if route_row["worst_identifiable_event"]["available"]:
                    route_risk = route_row["worst_identifiable_event"]["composite_risk"]
                    median_risk = median["worst_identifiable_event"]["composite_risk"]
                    if route_risk > median_risk + 0.1:
                        failures.append(
                            f"{work}/{key} worst-event risk regression "
                            f"{route_risk-median_risk:.6g} > 0.1"
                        )
                median_frequency = report["transform_identity"][
                    "median_stft_identity"
                ]["frequency_boundary_check"]["max_boundary_over_p99_adjacent"]
                route_frequency = route_row["frequency_boundary_check"][
                    "max_boundary_over_p99_adjacent"
                ]
                if route_frequency > max(1.0, 1.5 * median_frequency):
                    failures.append(f"{work}/{key} frequency-boundary diagnostic regressed")
                median_seam = report["transform_identity"][
                    "median_stft_identity"
                ]["seam_check"]["max_boundary_jump_over_p99_derivative"]
                route_seam = route_row["seam_check"][
                    "max_boundary_jump_over_p99_derivative"
                ]
                if route_seam > max(1.0, 1.5 * median_seam):
                    failures.append(f"{work}/{key} time-boundary seam diagnostic regressed")
            aalto = works["aalto_mozart_dry"]
            control = aalto["resolutions"][resolution][mode]["no_vocal"]
            if control["candidate_over_median"] > 1.05:
                failures.append(
                    f"aalto_mozart_dry/{key} no-vocal ratio "
                    f"{control['candidate_over_median']:.6g} > 1.05"
                )
            decisions[key] = {
                "binding_gate_passed": not failures,
                "failures": failures,
            }
    primary = [decisions[f"{mode}@2s"]["binding_gate_passed"] for mode in ("O2", "O3")]
    return {
        "primary_resolution_seconds": 2.0,
        "any_primary_route_passed": any(primary),
        "routes": decisions,
        "learned_gate_authorized": any(primary),
        "authorization_scope": (
            "training-only teacher; never a deployable exact-reference selector"
        ),
    }


def _load_control(path: Path, work: str, member_names: list[str],
                  frames: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    report = json.loads(path.read_text())
    if (report.get("schema") != "audio-extract/classical-gate-controls/v1"
            or report.get("status") != "complete"
            or report.get("work_id") != work
            or report.get("control") != "no_vocal"):
        raise ValueError(f"wrong no-vocal control report: {path}")
    rows = {row["name"]: row for row in report["members"]}
    missing = sorted(set(member_names) - set(rows))
    if missing:
        raise ValueError(f"no-vocal report lacks binding members: {missing}")
    values = []
    records = []
    for name in member_names:
        record = rows[name]["accompaniment"]
        audio_path = Path(record["path"])
        value = _read_exact(audio_path, frames)
        if (_pcm(value) != record["artifact_pcm_sha256"]
                or _sha_file(audio_path) != record["container_sha256"]):
            raise ValueError(f"no-vocal control identity mismatch: {name}")
        values.append(value)
        records.append({"name": name, **record})
    return values, {
        "report": str(path.resolve()),
        "report_sha256": _sha_file(path),
        "members": records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    code_commit = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    rows = _manifest_rows(args.candidate_manifest)
    works = tuple(args.work)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "final",
        "claim": "exact-reference diagnostic only",
        "code_commit": code_commit,
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": _sha_file(args.candidate_manifest),
        "frozen_resolutions_seconds": list(RESOLUTIONS),
        "basis_requested": list(BASE_ORDER),
        "works": {},
    }
    for work in works:
        print(json.dumps({"stage": "load", "work": work}), flush=True)
        layout = TrackLayout(args.lib_root, work)
        source = json.loads((layout.source_dir / "source.json").read_text())
        truth = args.truth_root / work
        mixture = _read_exact(truth / "mix_with_voice.wav", source["frames"])
        accompaniment = _read_exact(truth / "orchestra_only.wav", source["frames"])
        vocal = _read_exact(truth / "voice_ref.wav", source["frames"])
        if _pcm(mixture) != source["input_pcm_sha256"]:
            raise ValueError(f"truth/source mixture identity mismatch: {work}")
        residuals = _residual_members(
            manifest_rows=rows, layout=layout, source=source, work=work,
            generating_commit=rows[(work, "median_mdx_mel_bs")]["code_commit"],
        )
        requested = [
            _external_demucs_member(
                name="htdemucs_04573f0d", eval_root=args.htdemucs_045,
                work=work, source=source,
            ),
            _external_demucs_member(
                name="htdemucs_955717e8", eval_root=args.htdemucs_955,
                work=work, source=source,
            ),
            residuals["mdx23c"], residuals["melband"], residuals["bs_roformer"],
        ]
        requested.extend(_manifest_member(rows[(work, name)], layout, source["frames"])
                         for name in BASE_ORDER[5:])
        if tuple(member["name"] for member in requested) != BASE_ORDER:
            raise RuntimeError("requested binding basis order changed")
        members, aliases, indices = _deduplicate_members(requested)
        values, accompaniment, vocal = validate_basis(
            [member["value"] for member in members], accompaniment, vocal
        )
        for member, value in zip(members, values):
            member["value"] = value
        member_names = [member["name"] for member in members]
        no_vocal_path = args.control_root / f"{work}.no_vocal.binding-basis.json"
        no_vocal_values, no_vocal_evidence = _load_control(
            no_vocal_path, work, member_names, len(mixture)
        )
        print(json.dumps({"stage": "stft", "work": work,
                          "members": len(members)}), flush=True)
        candidate_spectra = stft_stack(values, RoutingConfig())
        truth_spectra = stft_stack([accompaniment, vocal], RoutingConfig())
        no_vocal_spectra = stft_stack(no_vocal_values, RoutingConfig())

        baseline_indices = {
            "O1_placeholder": None,
            "median": indices["median_mdx_mel_bs"],
            "mdx": indices["mdx23c"],
        }
        work_report: dict[str, Any] = {
            "frames": len(mixture),
            "basis": [{key: value for key, value in member.items() if key != "value"}
                      for member in members],
            "basis_aliases": aliases,
            "decoded_pcm_deduplicated": len(members) != len(requested),
            "no_vocal_control": no_vocal_evidence,
            "baselines": {},
            "transform_identity": {},
            "resolutions": {},
        }
        primary_view = None
        primary_routing = None
        primary_grid = None
        for tile_seconds in RESOLUTIONS:
            resolution = format(tile_seconds, "g")
            routing, certified = _scaled_configs(tile_seconds)
            time_ranges = _time_ranges(candidate_spectra.shape[-1], routing)
            frequency_ranges = _frequency_ranges(candidate_spectra.shape[-2], routing)
            grid = build_spectral_quadratic_grid(
                candidate_spectra, truth_spectra[0], truth_spectra[1],
                time_ranges=time_ranges, frequency_ranges=frequency_ranges,
                config=certified,
            )
            view = _routing_view(grid, time_ranges, frequency_ranges)
            o1_index, o1_costs = best_whole_track(grid)
            baseline_indices["O1_placeholder"] = o1_index
            print(json.dumps({"stage": "certified_O2", "work": work,
                              "resolution": resolution}), flush=True)
            o2 = solve_discrete_global(grid, certified)
            print(json.dumps({"stage": "certified_O3", "work": work,
                              "resolution": resolution}), flush=True)
            try:
                o3 = solve_convex_certified(
                    grid, certified, o1_index=o1_index, o2_labels=o2.labels
                )
                o3_error = None
            except CertifiedRoutingError as exc:
                o3 = None
                o3_error = str(exc)
            routes = {
                "O2": one_hot_weights(o2.labels, len(members)),
            }
            if o3 is not None:
                routes["O3"] = o3.weights
            resolution_report: dict[str, Any] = {
                "routing_config": routing.to_dict(),
                "certified_config": certified.to_dict(),
                "mode_counts": dict(Counter(str(value) for value in grid.modes.flat)),
                "O1_selected_index": o1_index,
                "O1_selected_member": member_names[o1_index],
                "O1_costs": {name: float(cost) for name, cost in zip(
                    member_names, o1_costs
                )},
            }
            o1_metrics, o1_worst = _route_metrics(
                values[o1_index], accompaniment, vocal
            )
            resolution_report["O1_baseline"] = {
                "member": member_names[o1_index], "index": o1_index,
                "artifact_pcm_sha256": _pcm(values[o1_index]),
                "metrics": o1_metrics,
                "worst_identifiable_event": o1_worst,
            }
            if o3 is None:
                resolution_report["O3"] = {
                    "status": "rejected",
                    "certificate_error": o3_error,
                }
            for mode, weights in routes.items():
                output = _render(
                    candidate_spectra, weights, view, len(mixture), routing
                )
                plan = o2.labels.astype(np.int32) if mode == "O2" else (
                    o3.weights.astype(np.float64)
                )
                solver = {
                    "core": "audio_extract.oracle_routing_certified/v2",
                    "certified_config": certified.identity_dict(),
                    "resolution_scaling": {
                        "temporal_switch": "0.05*(2/tile_seconds)",
                        "temporal_weight_smoothness": "0.05*(2/tile_seconds)^2",
                        "frequency_penalties": "fixed-0.05",
                    },
                    "result": (
                        {"objective": o2.objective, "data_objective": o2.data_objective,
                         "temporal_switches": o2.temporal_switches,
                         "frequency_switches": o2.frequency_switches,
                         "solver_status": o2.solver_status, "mip_gap": o2.mip_gap}
                        if mode == "O2" else
                        {"objective": o3.objective, "data_objective": o3.data_objective,
                         "iterations": o3.iterations, "converged": o3.converged,
                         "projected_gradient_norm": o3.projected_gradient_norm,
                         "selected_start": o3.selected_start,
                         "start_objectives": o3.start_objectives}
                    ),
                }
                artifact = None
                removed = None
                if tile_seconds == 2.0:
                    artifact = _write_route(
                        layout=layout, source=source,
                        mode=f"CERTIFIED_{mode}_{resolution}S", output=output,
                        parents=members, plan=plan, config=routing,
                        code_commit=code_commit,
                        execution={
                            "oracle_diagnostic_only": True,
                            "routing_plan": plan.tolist(),
                            "cell_modes": grid.modes.tolist(),
                            "solver": solver,
                        },
                        truth_pcm={"accompaniment": _pcm(accompaniment), "vocal": _pcm(vocal)},
                        solver_config=solver,
                        adapter_revision="certified-exact-fallback-route/v2",
                    )
                    removed = _write_removed_vocal(
                        layout=layout, source=source, mixture=mixture,
                        accompaniment=output, parent=artifact, code_commit=code_commit,
                    )
                metrics, worst = _route_metrics(output, accompaniment, vocal)
                combined = _combined_spectrum(
                    candidate_spectra, weights, time_ranges, frequency_ranges
                )
                no_vocal_output = _render(
                    no_vocal_spectra, weights, view, len(mixture), routing
                )
                no_vocal_error = accompaniment.astype(np.float64) - no_vocal_output.astype(np.float64)
                median_control_error = (
                    accompaniment.astype(np.float64)
                    - no_vocal_values[indices["median_mdx_mel_bs"]].astype(np.float64)
                )
                no_vocal_energy = float(np.square(no_vocal_error).sum())
                median_no_vocal_energy = float(np.square(median_control_error).sum())
                resolution_report[mode] = {
                    "solver": solver,
                    "selection_counts": (
                        {name: int(np.count_nonzero(o2.labels == index))
                         for index, name in enumerate(member_names)}
                        if mode == "O2" else None
                    ),
                    "mean_weights": {
                        name: float(weights[..., index].mean())
                        for index, name in enumerate(member_names)
                    },
                    "artifact": artifact,
                    "removed_vocal": removed,
                    "rendered_artifact_pcm_sha256": _pcm(output),
                    "metrics": metrics,
                    "worst_identifiable_event": worst,
                    "seam_check": seam_check(output, view, routing),
                    "frequency_boundary_check": _frequency_boundary_check(
                        combined, frequency_ranges
                    ),
                    "no_vocal": {
                        "false_positive_energy_ratio": no_vocal_energy / max(
                            float(np.square(accompaniment.astype(np.float64)).sum()), 1e-12
                        ),
                        "candidate_over_median": _ratio(
                            no_vocal_energy, median_no_vocal_energy
                        ),
                        "artifact_pcm_sha256": _pcm(no_vocal_output),
                    },
                }
            work_report["resolutions"][resolution] = resolution_report
            if tile_seconds == 2.0:
                primary_view, primary_routing, primary_grid = view, routing, grid

        assert primary_view is not None and primary_routing is not None
        o1_index = work_report["resolutions"][PRIMARY_RESOLUTION]["O1_selected_index"]
        baseline_indices["O1_placeholder"] = o1_index
        for label, index in (
            ("O1", o1_index),
            ("median", indices["median_mdx_mel_bs"]),
            ("mdx", indices["mdx23c"]),
        ):
            raw = values[index]
            metrics, worst = _route_metrics(raw, accompaniment, vocal)
            work_report["baselines"][label] = {
                "member": member_names[index], "index": index,
                "artifact_pcm_sha256": _pcm(raw), "metrics": metrics,
                "worst_identifiable_event": worst,
            }
        for label, index in (
            ("O1_stft_identity", o1_index),
            ("median_stft_identity", indices["median_mdx_mel_bs"]),
        ):
            labels = np.full(primary_grid.Q.shape[:2], index, dtype=np.int32)
            weights = legacy_one_hot_weights(labels, len(members))
            transformed = _render(
                candidate_spectra, weights, primary_view, len(mixture), primary_routing
            )
            artifact = _write_route(
                layout=layout, source=source, mode=label.upper(), output=transformed,
                parents=members, plan=labels, config=primary_routing,
                code_commit=code_commit,
                execution={"oracle_diagnostic_only": True,
                           "transform_identity_control": True,
                           "routing_plan": labels.tolist()},
                truth_pcm={"accompaniment": _pcm(accompaniment), "vocal": _pcm(vocal)},
                solver_config={"solver": "fixed-one-hot-transform-identity/v2",
                               "selected_index": index},
                adapter_revision="certified-exact-fallback-route/v2",
            )
            combined = _combined_spectrum(
                candidate_spectra, weights, primary_view.time_frame_ranges,
                primary_view.frequency_bin_ranges,
            )
            work_report["transform_identity"][label] = {
                "member": member_names[index], "artifact": artifact,
                "metrics": _transform_identity_metrics(values[index], transformed),
                "seam_check": seam_check(transformed, primary_view, primary_routing),
                "frequency_boundary_check": _frequency_boundary_check(
                    combined, primary_view.frequency_bin_ranges
                ),
            }
        report["works"][work] = work_report
        print(json.dumps({"stage": "complete", "work": work}), flush=True)
    required = set(DEFAULT_WORKS)
    if required <= set(report["works"]):
        report["decision"] = _decision(report["works"])
    else:
        report["status"] = "needs_human_ab"
        report["decision"] = {
            "state": "incomplete_fragment",
            "missing_works": sorted(required - set(report["works"])),
            "learned_gate_authorized": False,
        }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != payload:
        raise RuntimeError(f"refusing to rewrite differing binding report: {args.output}")
    if not args.output.exists():
        args.output.write_text(payload)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--lib-root", required=True, type=Path)
    parser.add_argument("--htdemucs-045", required=True, type=Path)
    parser.add_argument("--htdemucs-955", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--work", action="append", choices=FULL_WORKS, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.work is None:
        args.work = list(DEFAULT_WORKS)
    result = run(args)
    print(json.dumps({
        "status": result["status"],
        "learned_gate_authorized": result["decision"]["learned_gate_authorized"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
