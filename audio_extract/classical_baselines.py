"""Regenerate hash-verified FLOAT classical baselines through the v2 DAG.

The pilot intentionally exposes only the two oracle-authorized Bologna Verdi
recipes: the MDX23C residual and the component-wise median of three pinned vocal
parents.  Parent separator candidates are first-class immutable nodes; no lossy
legacy candidate is used as an analysis input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import identity
from .manifest import Manifest
from .model_lock import ModelLock, build_bundle
from .separate import render_candidate, render_ensemble_candidate, render_residual_candidate
from .storage import TrackLayout


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _ensure_exact_source(layout: TrackLayout, truth_mix: Path) -> dict:
    samples, sr = sf.read(truth_mix, dtype="float32", always_2d=True)
    if (int(sr), samples.shape[1]) != (44_100, 2):
        raise ValueError(f"truth mixture must be 44.1 kHz stereo: {truth_mix}")
    frames = len(samples)
    channels = ["FL", "FR"]
    pcm = identity.artifact_pcm_sha256(samples, int(sr), channels, frames)
    record = {
        "track_id": layout.track_id,
        "original_filename": truth_mix.name,
        "source_path": str(truth_mix.resolve()),
        "source_blob_sha256": _sha_file(truth_mix),
        "input_pcm_sha256": pcm,
        "sample_rate_hz": int(sr),
        "channel_layout": channels,
        "frames": frames,
        "sample_format": "float32-le-interleaved",
    }
    layout.ensure()
    source_json = layout.source_dir / "source.json"
    canonical = layout.source_dir / "canonical.f32.wav"
    if source_json.exists() or canonical.exists():
        if not source_json.exists() or not canonical.exists():
            raise RuntimeError(f"incomplete immutable source directory: {layout.source_dir}")
        existing = json.loads(source_json.read_text())
        for key in ("input_pcm_sha256", "sample_rate_hz", "channel_layout", "frames"):
            if existing.get(key) != record[key]:
                raise RuntimeError(f"source identity mismatch for {key}: {existing.get(key)} != {record[key]}")
        reopened, reopened_sr = sf.read(canonical, dtype="float32", always_2d=True)
        if reopened_sr != sr or not np.array_equal(reopened, samples):
            raise RuntimeError("cached canonical source differs from pinned truth mixture")
        return existing
    sf.write(canonical, samples, int(sr), subtype="FLOAT")
    reopened, reopened_sr = sf.read(canonical, dtype="float32", always_2d=True)
    if reopened_sr != sr or not np.array_equal(reopened, samples):
        raise RuntimeError("canonical FLOAT write did not reopen bit-exactly")
    source_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with Manifest(layout.manifest_sqlite) as manifest:
        manifest.set_state(layout.track_id, "INGESTED")
    return record


def _pinned_bundles(config: dict, model_dir: Path, lock_path: Path) -> dict[str, dict]:
    try:
        adapter_version = version("audio-separator")
    except Exception:
        adapter_version = "unknown"
    lock = ModelLock(lock_path)
    bundles = {}
    for name, spec in config["models"].items():
        weights = model_dir / spec["weights"]["filename"]
        model_config = model_dir / spec["config"]["filename"]
        for role, path in (("weights", weights), ("config", model_config)):
            actual = _sha_file(path).removeprefix("sha256:")
            expected = spec[role]["sha256"].removeprefix("sha256:")
            if actual != expected:
                raise RuntimeError(f"{name} {role} hash mismatch: {actual} != {expected}")
        bundle = build_bundle(
            logical_id=spec["logical_id"], family=spec["family"],
            target_stem=spec["target"], registry_alias=spec["registry_alias"],
            files=[("weights", weights), ("config", model_config)],
            adapter_version=adapter_version,
            adapter_revision="audio-separator+audio-extract-adapter-v1",
            effective_defaults={
                "model_sample_rate_hz": 44_100,
                "segment_size": 256,
                "overlap_factor": int(config["overlap"]),
                "normalization_threshold": "1.000000",
                "amplification_threshold": "0.000000",
                "use_soundfile": True,
                "alignment": config["alignment"],
            },
        )
        lock.add(bundle)
        bundles[name] = bundle
    lock.write()
    return bundles


def _verified_row(layout: TrackLayout, work_id: str, candidate: str, record: dict,
                  bundles: dict[str, dict], code_commit: str) -> dict:
    cdir = layout.candidate_dir(record["recipe_id"])
    wav = cdir / "output.f32.wav"
    info = sf.info(wav)
    source = json.loads((layout.source_dir / "source.json").read_text())
    expected_grid = (source["frames"], source["sample_rate_hz"], 2, "FLOAT")
    if (info.frames, info.samplerate, info.channels, info.subtype) != expected_grid:
        raise RuntimeError(f"published candidate grid mismatch: {info} != {expected_grid}")
    reopened, sr = sf.read(wav, dtype="float32", always_2d=True)
    reopened_pcm = identity.artifact_pcm_sha256(
        reopened, int(sr), source["channel_layout"], len(reopened)
    )
    declared_pcm = (cdir / "output.pcm.sha256").read_text().strip()
    if reopened_pcm != declared_pcm or reopened_pcm != record["artifact_pcm_sha256"]:
        raise RuntimeError(
            f"reopen-and-rehash mismatch: {reopened_pcm}, {declared_pcm}, "
            f"{record['artifact_pcm_sha256']}"
        )
    recipe = json.loads((cdir / "recipe.json").read_text())
    parents = list(record["parents"])
    executed = []
    for parent_id in parents:
        parent_recipe = json.loads(
            (layout.candidate_dir(parent_id) / "recipe.json").read_text()
        )
        bundle_id = parent_recipe.get("model", {}).get("executed_bundle_id")
        if bundle_id:
            executed.append(bundle_id)
    known_bundle_ids = {b["bundle_sha256"] for b in bundles.values()}
    if not set(executed) <= known_bundle_ids:
        raise RuntimeError(f"candidate refers to an unknown executed bundle: {executed}")
    return {
        "schema": "audio-extract/classical-candidate/v2",
        "work_id": work_id,
        "candidate": candidate,
        "path": f"research6:{wav}",
        "recipe_id": record["recipe_id"],
        "artifact_pcm_sha256": reopened_pcm,
        "container_sha256": _sha_file(wav),
        "frames": info.frames,
        "channels": source["channel_layout"],
        "sr_hz": info.samplerate,
        "subtype": info.subtype,
        "parent_candidate_recipe_ids": parents,
        "executed_model_bundle_hashes": sorted(executed),
        "alignment_implementation": recipe["effective_config"]["alignment"],
        "code_commit": code_commit,
    }


def run_pilot(*, config_path: Path, truth_root: Path, lib_root: Path,
              model_dir: Path, manifest_out: Path, work_id: str) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema") != "audio-extract/classical-baselines/v2":
        raise ValueError("wrong classical baseline configuration schema")
    if work_id != "bologna_verdi":
        raise ValueError("the bounded pilot is preregistered for bologna_verdi only")
    code_commit = _git_commit()
    layout = TrackLayout(lib_root, work_id)
    source = _ensure_exact_source(layout, truth_root / work_id / "mix_with_voice.wav")
    bundles = _pinned_bundles(config, model_dir, layout.root / "model-lock.json")

    vocal_parents = {}
    for name in ("mdx23c", "melband", "bs_roformer"):
        spec, bundle = config["models"][name], bundles[name]
        vocal_parents[name] = render_candidate(
            layout, source, model_filename=spec["registry_alias"], target="vocals",
            construction="native_primary", overlap=int(config["overlap"]),
            code_commit=code_commit, model_dir=model_dir,
            executed_bundle_id=bundle["bundle_sha256"],
            expected_sha256=spec["weights"]["sha256"],
        )

    mdx = render_residual_candidate(
        layout, source, vocal_recipe_id=vocal_parents["mdx23c"]["recipe_id"],
        code_commit=code_commit,
    )
    median = render_ensemble_candidate(
        layout, source,
        member_recipe_ids=[vocal_parents[name]["recipe_id"]
                           for name in ("mdx23c", "melband", "bs_roformer")],
        algo="median", code_commit=code_commit,
    )
    mtimes = {
        rec["recipe_id"]: layout.candidate_dir(rec["recipe_id"]).joinpath(
            "output.f32.wav"
        ).stat().st_mtime_ns for rec in (mdx, median)
    }
    mdx_repeat = render_residual_candidate(
        layout, source, vocal_recipe_id=vocal_parents["mdx23c"]["recipe_id"],
        code_commit=code_commit,
    )
    median_repeat = render_ensemble_candidate(
        layout, source,
        member_recipe_ids=[vocal_parents[name]["recipe_id"]
                           for name in ("mdx23c", "melband", "bs_roformer")],
        algo="median", code_commit=code_commit,
    )
    for first, repeat in ((mdx, mdx_repeat), (median, median_repeat)):
        wav = layout.candidate_dir(first["recipe_id"]) / "output.f32.wav"
        if (repeat["recipe_id"] != first["recipe_id"] or not repeat["cached"]
                or wav.stat().st_mtime_ns != mtimes[first["recipe_id"]]):
            raise RuntimeError("repeat invocation rerendered an immutable candidate")

    rows = [
        _verified_row(layout, work_id, "residual_mdx23c", mdx, bundles, code_commit),
        _verified_row(layout, work_id, "median_mdx_mel_bs", median, bundles, code_commit),
    ]
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if manifest_out.exists() and manifest_out.read_text() != text:
        raise RuntimeError(f"refusing to rewrite differing pilot manifest: {manifest_out}")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_out.exists():
        manifest_out.write_text(text)
    return {
        "status": "pilot_complete",
        "work_id": work_id,
        "source_pcm_sha256": source["input_pcm_sha256"],
        "rows": rows,
        "repeat_cache_verified": True,
        "manifest": str(manifest_out),
        "command": " ".join(__import__("sys").argv),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("python -m audio_extract.classical_baselines")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--lib-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--work-id", default="bologna_verdi")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_pilot(
        config_path=args.config, truth_root=args.truth_root, lib_root=args.lib_root,
        model_dir=args.model_dir, manifest_out=args.manifest_out, work_id=args.work_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
