#!/usr/bin/env python3
"""Select and package the exact candidates-v2 production milestone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_extract import canon, identity, recipe as recipe_mod


VOICE_COEFFICIENT_CAP = 0.30
EVENT_HOLE_P90_CAP_DB = 18.0
OPPOSITE_AXIS_TOLERANCE_DB = 0.5
ACTIONABLE_AXIS_DB = 1.5
ARTIFACT_RATIO_SANITY_CAP = 1.0
STEREO_WIDTH_SANITY_CAP_DB = 20.0
REQUIRED_CANDIDATES = (
    "residual_mdx23c", "median_mdx_mel_bs",
    "geomedian_mdx_mel_bs", "convex_fusion_uniform",
)
REQUIRED_WORKS = (
    "bologna_verdi", "bologna_puccini", "bologna_donizetti",
    "aalto_mozart_dry", "aalto_mozart_hall",
)
METRICS = (
    "retained_voice_db_p90", "retained_voice_coef_p90",
    "event_hole_db_p90", "event_hole_db_max", "alpha_error_p90",
    "artifact_ratio_p90", "scale_dependent_sdr_db",
    "band_deficit_db/v2", "contiguous_hole_db/v2",
    "erb_envelope_dist_db/v2", "centroid_deviation/v2",
    "hf_ratio_deviation/v2", "transient_loss/v2", "transient_excess/v2",
    "stereo_width_dev_db/v2", "interchannel_coherence_dev/v2",
    "_available_tiles", "_total_tiles",
)


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


def _resolve(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if ":" in value:
        candidate = Path(value.split(":", 1)[1])
        if candidate.is_absolute() and candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def _read_exact(path: Path, frames: int | None = None) -> np.ndarray:
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (44_100, 2, "FLOAT"):
        raise ValueError(f"not 44.1-kHz stereo FLOAT: {path}: {info}")
    if frames is not None and info.frames != frames:
        raise ValueError(f"frame mismatch: {path}: {info.frames} != {frames}")
    value, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 44_100 or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid FLOAT samples: {path}")
    return value


def _pcm(value: np.ndarray) -> str:
    return identity.artifact_pcm_sha256(value, 44_100, ["FL", "FR"], len(value))


def _hardlink_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination) or _sha_file(source) == _sha_file(destination):
            return
        raise RuntimeError(f"refusing to replace different package file: {destination}")
    os.link(source, destination)


def _write_float_verified(path: Path, value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype="float32")
    if path.exists():
        reopened = _read_exact(path, len(value))
        if not np.array_equal(reopened, value):
            raise RuntimeError(f"refusing to replace different FLOAT output: {path}")
    else:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=path.parent)
        os.close(fd); temporary = Path(name)
        try:
            sf.write(temporary, value, 44_100, subtype="FLOAT", format="WAV")
            if not np.array_equal(_read_exact(temporary, len(value)), value):
                raise RuntimeError(f"FLOAT reopen mismatch: {temporary}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    reopened = _read_exact(path, len(value))
    return {
        "path": str(path), "container_sha256": _sha_file(path),
        "artifact_pcm_sha256": _pcm(reopened), "frames": len(reopened),
        "sample_rate_hz": 44_100, "channels": ["FL", "FR"], "subtype": "FLOAT",
    }


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to replace different package metadata: {path}")
        return
    path.write_text(text)


def _candidate_failure(metrics: dict[str, Any]) -> list[str]:
    failures = []
    required = [metrics.get(name) for name in METRICS]
    if any(value is None or not isinstance(value, (int, float))
           or not np.isfinite(value) for value in required):
        return ["missing or non-finite required metric"]
    if metrics["_available_tiles"] <= 0:
        failures.append("no identifiable source-coordinate tiles")
    if metrics["retained_voice_coef_p90"] > VOICE_COEFFICIENT_CAP:
        failures.append(
            f"retained_voice_coef_p90>{VOICE_COEFFICIENT_CAP}"
        )
    if metrics["event_hole_db_p90"] > EVENT_HOLE_P90_CAP_DB:
        failures.append(f"event_hole_db_p90>{EVENT_HOLE_P90_CAP_DB}")
    if metrics["artifact_ratio_p90"] > ARTIFACT_RATIO_SANITY_CAP:
        failures.append(f"artifact_ratio_p90>{ARTIFACT_RATIO_SANITY_CAP}")
    if metrics["stereo_width_dev_db/v2"] > STEREO_WIDTH_SANITY_CAP_DB:
        failures.append(f"stereo_width_dev_db/v2>{STEREO_WIDTH_SANITY_CAP_DB}")
    return failures


def _rank(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Frozen per-work critical-axis qualification and deterministic tie-break."""

    best_voice = min(row["metrics"]["retained_voice_db_p90"] for row in rows)
    best_hole = min(row["metrics"]["event_hole_db_p90"] for row in rows)
    for row in rows:
        metrics = row["metrics"]
        row["failures"] = _candidate_failure(metrics)
        row["critical_regret"] = {
            "retained_voice_db": metrics["retained_voice_db_p90"] - best_voice,
            "event_hole_db": metrics["event_hole_db_p90"] - best_hole,
        }
    feasible = [row for row in rows if not row["failures"]]
    qualification = "clears_all_frozen_caps"
    pool = feasible
    if not pool:
        qualification = "best_available_no_candidate_clears_all_frozen_caps"
        pool = rows
    frontier = [
        row for row in pool
        if row["critical_regret"]["retained_voice_db"] <= OPPOSITE_AXIS_TOLERANCE_DB
        and row["critical_regret"]["event_hole_db"] <= OPPOSITE_AXIS_TOLERANCE_DB
    ]
    if not frontier:
        frontier = pool

    def key(row):
        metrics = row["metrics"]
        regret = row["critical_regret"]
        normalized_worst = max(
            regret["retained_voice_db"] / ACTIONABLE_AXIS_DB,
            regret["event_hole_db"] / OPPOSITE_AXIS_TOLERANCE_DB,
        )
        return (
            normalized_worst,
            metrics["event_hole_db_max"], metrics["artifact_ratio_p90"],
            -metrics["scale_dependent_sdr_db"], metrics["band_deficit_db/v2"],
            metrics["transient_loss/v2"] + metrics["transient_excess/v2"],
            metrics["stereo_width_dev_db/v2"], row["candidate"],
        )
    ranked_frontier = sorted(frontier, key=key)
    remaining = sorted([row for row in pool if row not in frontier], key=key)
    rejected = sorted([row for row in rows if row not in pool], key=key)
    ranked = ranked_frontier + remaining + rejected
    return ranked, {
        "qualification": qualification,
        "frontier": [row["candidate"] for row in ranked_frontier],
        "voice_coefficient_cap": VOICE_COEFFICIENT_CAP,
        "event_hole_p90_cap_db": EVENT_HOLE_P90_CAP_DB,
        "opposite_axis_tolerance_db": OPPOSITE_AXIS_TOLERANCE_DB,
        "actionable_axis_db": ACTIONABLE_AXIS_DB,
        "artifact_ratio_sanity_cap": ARTIFACT_RATIO_SANITY_CAP,
        "stereo_width_sanity_cap_db": STEREO_WIDTH_SANITY_CAP_DB,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [f"# {report['work_id']} — candidates-v2 production milestone", ""]
    selected = report["selection"]
    lines += [
        f"Primary: **{selected['primary']}**  ",
        f"Alternate: **{selected['alternate']}**  ",
        f"Qualification: `{selected['policy']['qualification']}`", "",
        "| candidate | voice p90 dB | voice coef p90 | hole p90/max dB | artifact p90 | SDR dB | fullness dB | transient loss/excess | stereo width dB | caps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["candidates"]:
        m = row["metrics"]
        failures = ", ".join(row["failures"]) if row["failures"] else "PASS"
        lines.append(
            f"| {row['candidate']} | {m['retained_voice_db_p90']:.3f} | "
            f"{m['retained_voice_coef_p90']:.3f} | {m['event_hole_db_p90']:.3f}/"
            f"{m['event_hole_db_max']:.3f} | {m['artifact_ratio_p90']:.3f} | "
            f"{m['scale_dependent_sdr_db']:.3f} | {m['band_deficit_db/v2']:.3f} | "
            f"{m['transient_loss/v2']:.3f}/{m['transient_excess/v2']:.3f} | "
            f"{m['stereo_width_dev_db/v2']:.3f} | {failures} |"
        )
    lines += ["", "All audio is full-length, exact-grid, 44.1-kHz stereo FLOAT.", ""]
    return "\n".join(lines)


def _removed_recipe(source_input: dict, primary_recipe_id: str,
                    code_commit: str) -> tuple[str, dict[str, Any]]:
    recipe = {
        "schema": recipe_mod.SCHEMA, "canon": recipe_mod.CANON,
        "input_pcm": source_input,
        "operation": {"type": "mixture_minus_source", "target": "vocals",
                      "construction": "mixture_minus_source"},
        "model": {
            "model_id": "mixture-minus-selected-accompaniment",
            "weights_sha256": hashlib.sha256(primary_recipe_id.encode()).hexdigest(),
            "adapter": "audio-extract-residual",
            "adapter_revision": "residual-exact-grid-v1",
            "members": [primary_recipe_id],
        },
        "effective_config": {
            "model_sample_rate_hz": 44_100, "alignment": "source-grid-exact",
        },
        "software": {"audio_extract_commit": code_commit},
    }
    return identity.recipe_id(recipe), recipe


def _measurement_recipe(source_input: dict, parents: list[str],
                        code_commit: str, oracle_report_sha256: str) -> tuple[str, dict[str, Any]]:
    contract = {
        "critical_order": ["cap_rejection", "worst_critical_event_p90",
                           "secondary_fidelity"],
        "voice_coefficient_cap": str(VOICE_COEFFICIENT_CAP),
        "event_hole_p90_cap_db": str(EVENT_HOLE_P90_CAP_DB),
        "opposite_axis_tolerance_db": str(OPPOSITE_AXIS_TOLERANCE_DB),
        "artifact_ratio_sanity_cap": str(ARTIFACT_RATIO_SANITY_CAP),
        "stereo_width_sanity_cap_db": str(STEREO_WIDTH_SANITY_CAP_DB),
        "oracle_report_sha256": oracle_report_sha256,
    }
    contract_sha = hashlib.sha256(canon.canonicalize(contract)).hexdigest()
    recipe = {
        "schema": recipe_mod.SCHEMA, "canon": recipe_mod.CANON,
        "input_pcm": source_input,
        "operation": {"type": "measure", "target": "instrumental"},
        "model": {
            "model_id": "candidates-v2-exact-production-qualification",
            "weights_sha256": contract_sha,
            "adapter": "audio-extract-exact-metric-bank",
            "adapter_revision": "production-artifact-table/v1", "members": parents,
        },
        "effective_config": contract,
        "software": {"audio_extract_commit": code_commit},
    }
    return identity.recipe_id(recipe), recipe


def package_work(*, work: str, rows: list[dict[str, Any]], oracle_report: dict,
                 oracle_report_sha256: str, truth_root: Path, output_root: Path,
                 code_commit: str) -> dict[str, Any]:
    by_candidate = {row["candidate"]: row for row in rows}
    basis_metrics = {row["candidate_id"]: row["metrics"] for row in oracle_report["basis"]}
    if set(by_candidate) != set(REQUIRED_CANDIDATES) or set(basis_metrics) != set(REQUIRED_CANDIDATES):
        raise ValueError(f"incomplete four-candidate panel for {work}")
    scored = []
    for candidate in REQUIRED_CANDIDATES:
        row = by_candidate[candidate]
        path = _resolve(row["path"])
        value = _read_exact(path, row["frames"])
        if _pcm(value) != row["artifact_pcm_sha256"] or _sha_file(path) != row["container_sha256"]:
            raise ValueError(f"candidate identity mismatch: {work}/{candidate}")
        scored.append({
            "candidate": candidate, "recipe_id": row["recipe_id"],
            "source_path": str(path), "artifact_pcm_sha256": row["artifact_pcm_sha256"],
            "container_sha256": row["container_sha256"], "metrics": basis_metrics[candidate],
        })
    ranked, policy = _rank(scored)
    primary, alternate = ranked[0], ranked[1]
    directory = output_root / work
    directory.mkdir(parents=True, exist_ok=True)
    primary_path = directory / "accompaniment.primary.f32.wav"
    alternate_path = directory / "accompaniment.alternate.f32.wav"
    orchestra_path = directory / "exact-orchestra-target.f32.wav"
    voice_path = directory / "exact-voice-target.f32.wav"
    _hardlink_verified(Path(primary["source_path"]), primary_path)
    _hardlink_verified(Path(alternate["source_path"]), alternate_path)
    truth = truth_root / work
    _hardlink_verified(truth / "orchestra_only.wav", orchestra_path)
    _hardlink_verified(truth / "voice_ref.wav", voice_path)
    mixture = _read_exact(truth / "mix_with_voice.wav")
    primary_audio = _read_exact(primary_path, len(mixture))
    removed_audio = (mixture.astype(np.float64) - primary_audio.astype(np.float64)).astype("float32")
    removed_artifact = _write_float_verified(
        directory / "removed-vocal.primary.f32.wav", removed_audio
    )
    source_input = {
        "sha256": _pcm(mixture), "sample_rate_hz": 44_100,
        "channel_layout": ["FL", "FR"], "frames": len(mixture),
        "sample_format": "float32-le-interleaved",
    }
    removed_id, removed_recipe = _removed_recipe(
        source_input, primary["recipe_id"], code_commit
    )
    measurement_id, measurement_recipe = _measurement_recipe(
        source_input, [row["recipe_id"] for row in scored], code_commit,
        oracle_report_sha256,
    )
    report = {
        "schema": "audio-extract/candidates-v2-production-delivery/v1",
        "status": "complete", "work_id": work, "code_commit": code_commit,
        "oracle_code_commit": oracle_report["code_commit"],
        "oracle_report_sha256": oracle_report_sha256,
        "selection": {"primary": primary["candidate"],
                      "primary_recipe_id": primary["recipe_id"],
                      "alternate": alternate["candidate"],
                      "alternate_recipe_id": alternate["recipe_id"], "policy": policy},
        "measurement_recipe_id": measurement_id,
        "candidates": scored,
        "artifacts": {
            "accompaniment.primary.f32.wav": {
                "recipe_id": primary["recipe_id"], "container_sha256": _sha_file(primary_path),
                "artifact_pcm_sha256": _pcm(_read_exact(primary_path)),
            },
            "removed-vocal.primary.f32.wav": {
                "recipe_id": removed_id, **removed_artifact,
            },
            "accompaniment.alternate.f32.wav": {
                "recipe_id": alternate["recipe_id"], "container_sha256": _sha_file(alternate_path),
                "artifact_pcm_sha256": _pcm(_read_exact(alternate_path)),
            },
            "exact-orchestra-target.f32.wav": {
                "container_sha256": _sha_file(orchestra_path),
                "artifact_pcm_sha256": _pcm(_read_exact(orchestra_path)),
            },
            "exact-voice-target.f32.wav": {
                "container_sha256": _sha_file(voice_path),
                "artifact_pcm_sha256": _pcm(_read_exact(voice_path)),
            },
        },
    }
    _write_text(directory / "removed-vocal.primary.recipe.json",
                json.dumps(removed_recipe, indent=2, sort_keys=True) + "\n")
    _write_text(directory / "removed-vocal.primary.execution.json",
                json.dumps({**identity.execution_fingerprint(),
                            "parent_recipe_ids": [primary["recipe_id"]]},
                           indent=2, sort_keys=True) + "\n")
    _write_text(directory / "measurement.recipe.json",
                json.dumps(measurement_recipe, indent=2, sort_keys=True) + "\n")
    _write_text(directory / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_text(directory / "report.md", _markdown(report))
    manifest = {
        "schema": "audio-extract/delivery-manifest/v1", "work_id": work,
        "measurement_recipe_id": measurement_id, "files": report["artifacts"],
    }
    _write_text(directory / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_text(directory / "COMPLETE", "")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    code_commit = _git_commit()
    rows = [json.loads(line) for line in args.candidate_manifest.read_text().splitlines()]
    if len(rows) != 20:
        raise ValueError(f"expected 20 candidates-v2 rows, got {len(rows)}")
    reports = {}
    oracle_commits = set()
    for work in REQUIRED_WORKS:
        oracle_path = args.oracle_root / work / "oracle-envelope-report.json"
        oracle = json.loads(oracle_path.read_text())
        if oracle.get("work_id") != work or not oracle.get("code_commit"):
            raise ValueError(f"oracle report identity mismatch: {oracle_path}")
        oracle_commits.add(oracle["code_commit"])
        reports[work] = package_work(
            work=work, rows=[row for row in rows if row["work_id"] == work],
            oracle_report=oracle, oracle_report_sha256=_sha_file(oracle_path),
            truth_root=args.truth_root,
            output_root=args.output_root, code_commit=code_commit,
        )
        print(json.dumps({"work": work, "primary": reports[work]["selection"]["primary"],
                          "alternate": reports[work]["selection"]["alternate"]}), flush=True)
    if len(oracle_commits) != 1:
        raise ValueError(f"oracle reports span multiple code identities: {sorted(oracle_commits)}")
    summary = {
        "schema": "audio-extract/candidates-v2-production-delivery-run/v1",
        "status": "complete", "code_commit": code_commit,
        "oracle_code_commits": sorted(oracle_commits),
        "candidate_manifest_sha256": _sha_file(args.candidate_manifest),
        "oracle_root": str(args.oracle_root),
        "works": {work: report["selection"] for work, report in reports.items()},
    }
    _write_text(args.output_root / "report.json",
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_text(args.output_root / "COMPLETE", "")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--oracle-root", required=True, type=Path)
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
