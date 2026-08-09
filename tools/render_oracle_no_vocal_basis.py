#!/usr/bin/env python3
"""Render the five immutable Aalto no-vocal controls for oracle routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from audio_extract import identity, recipe as recipe_mod
from audio_extract.classical_baselines import _ensure_exact_source, _pinned_bundles
from audio_extract.model_lock import ModelLock, build_bundle
from audio_extract.separate import render_candidate, render_residual_candidate
from audio_extract.storage import ImmutableWriteError, TrackLayout
from audio_extract.train_classical import _apply_model_production, _load_model, _read_exact


BASIS_ORDER = (
    "htdemucs_04573f0d", "htdemucs_955717e8", "mdx23c", "melband", "bs_roformer"
)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _hardlink(source: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.exists():
        if _sha_file(destination) != _sha_file(source):
            raise RuntimeError(f"model-bundle filename collision: {destination}")
        return
    os.link(source, destination)


def _demucs_bundle(config_path: Path, model, base: dict, lock: ModelLock,
                   bundle_dir: Path) -> dict:
    try:
        demucs_version = version("demucs")
    except Exception:
        demucs_version = "unknown"
    checkpoint = Path(base["path"])
    signature = base["signature"]
    _hardlink(checkpoint, bundle_dir)
    _hardlink(config_path, bundle_dir)
    bundle = build_bundle(
        logical_id=f"htdemucs_{signature}_th", family="htdemucs", target_stem="vocals",
        registry_alias=checkpoint.name, files=[("weights", checkpoint), ("config", config_path)],
        adapter_version=demucs_version, adapter_revision="demucs-full-track-affine/v1",
        effective_defaults={
            "model_sample_rate_hz": 44_100, "normalization": "demucs-full-track-affine/v1",
            "shifts": 0, "split": True, "overlap_ppm": 250_000,
            "sources": list(model.sources), "vocal_source_index": 3,
        },
    )
    lock.add(bundle)
    return bundle


def _demucs_vocal_recipe(source: dict, *, signature: str, checkpoint_sha256: str,
                         bundle_id: str, code_commit: str) -> dict:
    return {
        "schema": recipe_mod.SCHEMA, "canon": recipe_mod.CANON,
        "input_pcm": {
            "sha256": source["input_pcm_sha256"], "sample_rate_hz": 44_100,
            "channel_layout": ["FL", "FR"], "frames": source["frames"],
            "sample_format": "float32-le-interleaved",
        },
        "operation": {"type": "separate", "target": "vocals",
                      "construction": "native_primary"},
        "model": {
            "model_id": f"HTDemucs-{signature}",
            "weights_sha256": checkpoint_sha256.removeprefix("sha256:"),
            "executed_bundle_id": bundle_id,
            "adapter": "demucs.apply_model",
            "adapter_revision": "demucs-full-track-affine/v1",
        },
        "effective_config": {
            "model_sample_rate_hz": 44_100, "normalization": "demucs-full-track-affine/v1",
            "shifts": 0, "split": True, "overlap_ppm": 250_000,
        },
        "software": {"audio_extract_commit": code_commit},
    }


def _render_demucs(
    *, config_path: Path, layout: TrackLayout, source: dict, device: str,
    code_commit: str, lock: ModelLock, bundle_dir: Path,
) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    signature = cfg["base_checkpoint"]["signature"]
    # Loading first is intentional: _load_model rehashes the pinned checkpoint and
    # rejects a bag or wrong four-source topology before a recipe can be published.
    model, base = _load_model(cfg, device, 0)
    bundle = _demucs_bundle(config_path, model, base, lock, bundle_dir)
    recipe = _demucs_vocal_recipe(
        source, signature=signature, checkpoint_sha256=base["sha256"],
        bundle_id=bundle["bundle_sha256"], code_commit=code_commit,
    )
    rid = identity.recipe_id(recipe)
    cdir = layout.candidate_dir(rid)
    cached = (cdir / "COMPLETE").is_file()
    if not cached:
        import torch

        audio = _read_exact(layout.source_dir / "canonical.f32.wav")
        model.eval()
        with torch.no_grad():
            output = _apply_model_production(
                model, audio.unsqueeze(0), device=device, split=True
            )[0, 3].cpu().T.numpy().astype("float32")
        if output.shape != (source["frames"], 2) or not np.all(np.isfinite(output)):
            raise RuntimeError(f"invalid HTDemucs no-vocal output: {output.shape}")
        artifact = identity.artifact_pcm_sha256(output, 44_100, ["FL", "FR"], len(output))
        fd, name = tempfile.mkstemp(suffix=".f32.wav"); os.close(fd)
        tmp = Path(name); sf.write(tmp, output, 44_100, subtype="FLOAT")
        try:
            layout.write_candidate(
                rid, recipe,
                {**identity.execution_fingerprint(), "control": "no_vocal",
                 "source_role": "orchestra_only", "checkpoint": base,
                 "executed_bundle_id": bundle["bundle_sha256"]},
                tmp, artifact,
            )
        except ImmutableWriteError:
            cached = True
        finally:
            tmp.unlink(missing_ok=True)
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    residual = render_residual_candidate(
        layout, source, vocal_recipe_id=rid, code_commit=code_commit
    )
    path = layout.candidate_dir(residual["recipe_id"]) / "output.f32.wav"
    return {
        "name": f"htdemucs_{signature}", "recipe_id": residual["recipe_id"],
        "parent_vocal_recipe_id": rid, "path": str(path),
        "artifact_pcm_sha256": residual["artifact_pcm_sha256"],
        "container_sha256": _sha_file(path), "executed_bundle_hash": bundle["bundle_sha256"],
        "cached": cached and residual["cached"],
    }


def run(args: argparse.Namespace) -> dict:
    code_commit = _git_commit()
    track_id = f"{args.work}__no_vocal_control"
    layout = TrackLayout(args.output_store, track_id)
    source_path = args.truth_root / args.work / "orchestra_only.wav"
    source = _ensure_exact_source(layout, source_path)
    baseline_config = yaml.safe_load(args.baseline_config.read_text())
    bundles = _pinned_bundles(
        baseline_config, args.model_dir, layout.root / "model-lock.json"
    )
    lock = ModelLock(layout.root / "model-lock.json")
    bundle_dir = layout.root / "model-files"
    for spec in baseline_config["models"].values():
        _hardlink(args.model_dir / spec["weights"]["filename"], bundle_dir)
        _hardlink(args.model_dir / spec["config"]["filename"], bundle_dir)

    members = []
    for config_path in (args.demucs_045_config, args.demucs_955_config):
        print(json.dumps({"stage": "demucs", "config": str(config_path)}), flush=True)
        members.append(_render_demucs(
            config_path=config_path, layout=layout, source=source, device=args.device,
            code_commit=code_commit, lock=lock, bundle_dir=bundle_dir,
        ))
        lock.write()

    for name in ("mdx23c", "melband", "bs_roformer"):
        print(json.dumps({"stage": "audio_separator", "member": name}), flush=True)
        spec, bundle = baseline_config["models"][name], bundles[name]
        vocal = render_candidate(
            layout, source, model_filename=spec["registry_alias"], target="vocals",
            construction="native_primary", overlap=int(baseline_config["overlap"]),
            code_commit=code_commit, model_dir=args.model_dir,
            executed_bundle_id=bundle["bundle_sha256"],
            expected_sha256=spec["weights"]["sha256"],
        )
        residual = render_residual_candidate(
            layout, source, vocal_recipe_id=vocal["recipe_id"], code_commit=code_commit
        )
        path = layout.candidate_dir(residual["recipe_id"]) / "output.f32.wav"
        members.append({
            "name": name, "recipe_id": residual["recipe_id"],
            "parent_vocal_recipe_id": vocal["recipe_id"], "path": str(path),
            "artifact_pcm_sha256": residual["artifact_pcm_sha256"],
            "container_sha256": _sha_file(path),
            "executed_bundle_hash": bundle["bundle_sha256"],
            "cached": vocal["cached"] and residual["cached"],
        })
    # One directory now re-hashes every weight/config/adapter-config in the
    # combined five-member lock; hard links avoid duplicating multi-GB weights.
    for logical_id in lock.logical_ids():
        lock.verify(logical_id, bundle_dir)
    if tuple(row["name"] for row in members) != BASIS_ORDER:
        raise RuntimeError(f"basis order changed: {[row['name'] for row in members]}")
    source_audio, _ = sf.read(source_path, dtype="float32", always_2d=True)
    denominator = max(float(np.sum(source_audio.astype(np.float64) ** 2)), 1e-12)
    for row in members:
        value, _ = sf.read(row["path"], dtype="float32", always_2d=True)
        removed = source_audio.astype(np.float64) - value.astype(np.float64)
        row["false_positive_energy_ratio"] = float(np.sum(removed * removed) / denominator)
    report = {
        "schema": "audio-extract/oracle-no-vocal-basis/v1", "status": "complete",
        "work_id": args.work, "track_id": track_id, "code_commit": code_commit,
        "source_pcm_sha256": source["input_pcm_sha256"], "members": members,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != text:
        raise RuntimeError(f"refusing to rewrite differing manifest: {args.output}")
    if not args.output.exists():
        args.output.write_text(text)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--work", default="aalto_mozart_dry")
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
    print(json.dumps({"status": report["status"], "members": len(report["members"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
