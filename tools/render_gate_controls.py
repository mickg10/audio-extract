#!/usr/bin/env python3
"""Render immutable A-only or V-only controls for a bounded frozen-model gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from audio_extract.classical_baselines import _ensure_exact_source, _pinned_bundles
from audio_extract.model_lock import ModelLock
from audio_extract.separate import (
    render_candidate,
    render_ensemble_candidate,
    render_residual_candidate,
)
from audio_extract.storage import TrackLayout
from tools.render_oracle_no_vocal_basis import (
    BASIS_ORDER,
    _hardlink,
    _render_demucs,
    _sha_file,
)


CONTROL_FILES = {
    "no_vocal": ("orchestra_only.wav", "orchestra_only"),
    "vocal_only": ("voice_ref.wav", "featured_soloist_only"),
}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _member_record(layout: TrackLayout, name: str, accompaniment: dict,
                   bundle_sha256: str) -> dict:
    accompaniment_path = (
        layout.candidate_dir(accompaniment["recipe_id"]) / "output.f32.wav"
    )
    vocal_id = accompaniment["parent_vocal_recipe_id"]
    vocal_path = layout.candidate_dir(vocal_id) / "output.f32.wav"
    for path in (accompaniment_path, vocal_path):
        info = sf.info(path)
        if (info.samplerate, info.channels, info.subtype) != (44_100, 2, "FLOAT"):
            raise RuntimeError(f"control output is not exact FLOAT stereo: {path}: {info}")
    return {
        "name": name,
        "executed_bundle_hash": bundle_sha256,
        "vocal": {
            "recipe_id": vocal_id,
            "path": str(vocal_path),
            "artifact_pcm_sha256": (
                layout.candidate_dir(vocal_id) / "output.pcm.sha256"
            ).read_text().strip(),
            "container_sha256": _sha_file(vocal_path),
        },
        "accompaniment": {
            "recipe_id": accompaniment["recipe_id"],
            "path": str(accompaniment_path),
            "artifact_pcm_sha256": accompaniment["artifact_pcm_sha256"],
            "container_sha256": _sha_file(accompaniment_path),
        },
    }


def _ensemble_record(layout: TrackLayout, name: str, accompaniment: dict,
                     parent_rows: list[dict]) -> dict:
    path = layout.candidate_dir(accompaniment["recipe_id"]) / "output.f32.wav"
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (44_100, 2, "FLOAT"):
        raise RuntimeError(f"ensemble control output is not exact FLOAT stereo: {path}")
    return {
        "name": name,
        "executed_bundle_hashes": sorted(
            row["executed_bundle_hash"] for row in parent_rows
        ),
        "accompaniment": {
            "recipe_id": accompaniment["recipe_id"], "path": str(path),
            "artifact_pcm_sha256": accompaniment["artifact_pcm_sha256"],
            "container_sha256": _sha_file(path),
        },
        "parent_vocal_recipe_ids": list(accompaniment["parents"]),
    }


def run(args: argparse.Namespace) -> dict:
    if args.control not in CONTROL_FILES:
        raise ValueError(f"unknown control: {args.control}")
    requested = tuple(dict.fromkeys(args.members))
    unknown = sorted(set(requested) - set(BASIS_ORDER))
    if unknown or not requested:
        raise ValueError(f"invalid gate members: {unknown or requested}")

    code_commit = _git_commit()
    filename, source_role = CONTROL_FILES[args.control]
    track_id = f"{args.work}__gate_{args.control}"
    layout = TrackLayout(args.output_store, track_id)
    source_path = args.truth_root / args.work / filename
    source = _ensure_exact_source(layout, source_path)

    baseline_config = yaml.safe_load(args.baseline_config.read_text())
    bundles = _pinned_bundles(
        baseline_config, args.model_dir, layout.root / "model-lock.json"
    )
    lock = ModelLock(layout.root / "model-lock.json")
    bundle_dir = layout.root / "model-files"
    for name in requested:
        if name in baseline_config["models"]:
            spec = baseline_config["models"][name]
            _hardlink(args.model_dir / spec["weights"]["filename"], bundle_dir)
            _hardlink(args.model_dir / spec["config"]["filename"], bundle_dir)

    demucs_configs = {
        "htdemucs_04573f0d": args.demucs_045_config,
        "htdemucs_955717e8": args.demucs_955_config,
    }
    members = []
    for name in requested:
        print(json.dumps({
            "stage": "control", "control": args.control,
            "work": args.work, "member": name,
        }), flush=True)
        if name in demucs_configs:
            record = _render_demucs(
                config_path=demucs_configs[name], layout=layout, source=source,
                device=args.device, code_commit=code_commit, lock=lock,
                bundle_dir=bundle_dir, control=args.control, source_role=source_role,
            )
            lock.write()
            members.append(_member_record(
                layout, name, record, record["executed_bundle_hash"]
            ))
            continue

        spec, bundle = baseline_config["models"][name], bundles[name]
        vocal = render_candidate(
            layout, source, model_filename=spec["registry_alias"], target="vocals",
            construction="native_primary", overlap=int(baseline_config["overlap"]),
            code_commit=code_commit, model_dir=args.model_dir,
            executed_bundle_id=bundle["bundle_sha256"],
            expected_sha256=spec["weights"]["sha256"],
        )
        accompaniment = render_residual_candidate(
            layout, source, vocal_recipe_id=vocal["recipe_id"],
            code_commit=code_commit,
        )
        accompaniment["parent_vocal_recipe_id"] = vocal["recipe_id"]
        members.append(_member_record(
            layout, name, accompaniment, bundle["bundle_sha256"]
        ))

    if args.include_median or args.include_composites:
        median_names = ("mdx23c", "melband", "bs_roformer")
        by_name = {row["name"]: row for row in members}
        if not set(median_names) <= set(by_name):
            raise ValueError(
                "ensemble controls require mdx23c melband bs_roformer"
            )
        parents = [
            by_name[name]["vocal"]["recipe_id"] for name in median_names
        ]
        algorithms = [("median_mdx_mel_bs", "median")]
        if args.include_composites:
            algorithms.extend((
                ("geomedian_mdx_mel_bs", "stft_geometric_median"),
                ("convex_fusion_uniform", "mean"),
            ))
        for composite_name, algorithm in algorithms:
            ensemble = render_ensemble_candidate(
                layout, source, member_recipe_ids=parents, algo=algorithm,
                code_commit=code_commit,
            )
            members.append(_ensemble_record(
                layout, composite_name, ensemble,
                [by_name[name] for name in median_names],
            ))

    # Only the executed members must be resolvable from the combined lock.
    by_hash = {
        lock.resolve(logical_id)["bundle_sha256"]: logical_id
        for logical_id in lock.logical_ids()
    }
    for row in members:
        hashes = row.get("executed_bundle_hashes") or [row["executed_bundle_hash"]]
        for bundle_hash in hashes:
            if bundle_hash not in by_hash:
                raise RuntimeError(f"executed bundle is absent from the lock: {bundle_hash}")
            lock.verify(by_hash[bundle_hash], bundle_dir)

    source_audio, sr = sf.read(source_path, dtype="float32", always_2d=True)
    if sr != 44_100 or len(source_audio) != source["frames"]:
        raise RuntimeError("control source changed after immutable ingest")
    denominator = max(float(np.sum(source_audio.astype(np.float64) ** 2)), 1e-12)
    diagnostic_name = (
        "false_positive_energy_ratio" if args.control == "no_vocal"
        else "false_negative_energy_ratio"
    )
    for row in members:
        if "vocal" in row:
            estimate, _ = sf.read(
                row["vocal"]["path"], dtype="float32", always_2d=True
            )
        else:
            accompaniment, _ = sf.read(
                row["accompaniment"]["path"], dtype="float32", always_2d=True
            )
            estimate = source_audio.astype("float32") - accompaniment.astype("float32")
        error = estimate if args.control == "no_vocal" else source_audio - estimate
        row[diagnostic_name] = float(
            np.sum(error.astype(np.float64) ** 2) / denominator
        )

    report = {
        "schema": "audio-extract/classical-gate-controls/v1",
        "status": "complete", "work_id": args.work, "track_id": track_id,
        "control": args.control, "source_role": source_role,
        "code_commit": code_commit, "source": source, "members": members,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != payload:
        raise RuntimeError(f"refusing to rewrite differing manifest: {args.output}")
    if not args.output.exists():
        args.output.write_text(payload)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--work", required=True)
    parser.add_argument("--control", required=True, choices=sorted(CONTROL_FILES))
    parser.add_argument("--members", nargs="+", required=True, choices=BASIS_ORDER)
    parser.add_argument("--include-median", action="store_true")
    parser.add_argument(
        "--include-composites", action="store_true",
        help="also render median, STFT geometric median, and uniform mean",
    )
    parser.add_argument("--output-store", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--demucs-045-config", required=True, type=Path)
    parser.add_argument("--demucs-955-config", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({
        "status": report["status"], "control": report["control"],
        "members": len(report["members"]), "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
