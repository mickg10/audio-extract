#!/usr/bin/env python3
"""Finalize and audit the frozen median-v2 production corpus.

This deliberately operates only through the v2 JSON CLI for new DAG nodes.  It
then reopens every selected artifact, verifies its decoded-PCM identity, and
writes a portable JSONL catalogue plus human-friendly symlinks.  Existing
source and candidate directories are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import soundfile as sf

from audio_extract import identity


MDX_BUNDLE = "ebb1c94fc330a788bb7a542478fc40a5a72160ac82d552e4f201161a2225e0ef"
ENSEMBLE_MODEL = "vocal-ensemble-median"
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _rid(candidate_dir: Path) -> str:
    return candidate_dir.name.replace("sha256_", "sha256:", 1)


def _candidate_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "candidates"
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "COMPLETE").is_file()
    )


def _index(run_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate_dir in _candidate_dirs(run_dir):
        recipe = _json(candidate_dir / "recipe.json")
        result[_rid(candidate_dir)] = (candidate_dir, recipe)
    return result


def _find_one(
    index: dict[str, tuple[Path, dict[str, Any]]],
    predicate: Any,
    label: str,
) -> str:
    matches = [rid for rid, (_, recipe) in index.items() if predicate(recipe)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {len(matches)}: {matches}")
    return matches[0]


def _run_cli(lib: Path, args: list[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "audio_extract.cli", "--lib", str(lib), *args, "--json"]
    print("RUN", " ".join(args), flush=True)
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=None)
    if process.returncode:
        raise RuntimeError(f"CLI failed ({process.returncode}): {' '.join(args)}")
    for line in reversed(process.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("ok") is True:
            print(line, flush=True)
            return value
    raise RuntimeError(f"CLI produced no success JSON: {' '.join(args)}")


def _candidate_id(result: dict[str, Any]) -> str:
    candidate = result.get("candidate")
    if not isinstance(candidate, dict) or not SHA256.fullmatch(str(candidate.get("recipe_id"))):
        raise RuntimeError(f"malformed candidate result: {result}")
    return str(candidate["recipe_id"])


def finalize(
    lib: Path,
    model_dir: Path,
    lock: Path,
    expected_runs: int,
) -> None:
    run_dirs = sorted(path for path in lib.iterdir() if (path / "source/source.json").is_file())
    if len(run_dirs) != expected_runs:
        raise RuntimeError(f"expected {expected_runs} ingested runs, found {len(run_dirs)}")

    models = ["mdx23c_hq_ckpt", "vocals_mel_band_roformer_ckpt", "viperx_1297_ckpt"]
    for number, run_dir in enumerate(run_dirs, 1):
        run_id = run_dir.name
        index = _index(run_dir)
        source = _json(run_dir / "source/source.json")
        channel_map: str | None = None
        if source["channel_layout"] == ["FC"]:
            channel_map = _find_one(
                index,
                lambda recipe: recipe["operation"]["type"] == "channel_map",
                "mono-to-stereo channel-map candidate",
            )

        separates = {
            recipe.get("model", {}).get("executed_bundle_id"): rid
            for rid, (_, recipe) in index.items()
            if recipe.get("operation", {}).get("type") == "separate"
        }
        if len(separates) < 3:
            print(f"[{number}/{len(run_dirs)}] separating {run_id}", flush=True)
            for model in models:
                args = [
                    "candidate", "render", "--run-id", run_id,
                    "--model", model, "--target", "vocals",
                    "--construction", "native_primary", "--overlap", "8",
                    "--model-dir", str(model_dir), "--lock", str(lock),
                ]
                if channel_map:
                    args.extend(["--input-recipe-id", channel_map])
                _run_cli(lib, args)
            index = _index(run_dir)

        vocal_ids = [
            rid for rid, (_, recipe) in index.items()
            if recipe.get("operation", {}).get("type") == "separate"
        ]
        if len(vocal_ids) != 3:
            raise RuntimeError(f"{run_id}: expected exactly 3 vocal candidates, found {len(vocal_ids)}")

        median_ids = [
            rid for rid, (_, recipe) in index.items()
            if recipe.get("model", {}).get("model_id") == ENSEMBLE_MODEL
            and recipe.get("operation", {}).get("target") == "instrumental"
        ]
        if not median_ids:
            args = [
                "candidate", "ensemble", "--run-id", run_id,
                "--members", ",".join(sorted(vocal_ids)), "--algo", "median",
            ]
            if channel_map:
                args.extend(["--mixture-recipe-id", channel_map])
            median_ids = [_candidate_id(_run_cli(lib, args))]
            index = _index(run_dir)
        if len(median_ids) != 1:
            raise RuntimeError(f"{run_id}: expected one median instrumental, found {median_ids}")

        mdx_vocal = _find_one(
            index,
            lambda recipe: recipe.get("model", {}).get("executed_bundle_id") == MDX_BUNDLE,
            "MDX vocal candidate",
        )
        residual_ids = [
            rid for rid, (_, recipe) in index.items()
            if recipe.get("model", {}).get("model_id") == "single-vocal-residual"
            and recipe.get("model", {}).get("members") == [mdx_vocal]
        ]
        if not residual_ids:
            args = ["candidate", "residual", "--run-id", run_id, "--vocal-recipe-id", mdx_vocal]
            if channel_map:
                args.extend(["--mixture-recipe-id", channel_map])
            residual_ids = [_candidate_id(_run_cli(lib, args))]
            index = _index(run_dir)
        if len(residual_ids) != 1:
            raise RuntimeError(f"{run_id}: expected one MDX residual, found {residual_ids}")

        for label, parent in (("median", median_ids[0]), ("mdx", residual_ids[0])):
            resamples = [
                rid for rid, (_, recipe) in index.items()
                if recipe.get("operation", {}).get("type") == "resample"
                and recipe.get("input_pcm", {}).get("parent_recipe_ids") == [parent]
                and recipe.get("effective_config", {}).get("output_rate_hz") == 48000
                and FULL_COMMIT.fullmatch(str(recipe.get("software", {}).get("audio_extract_commit", "")))
            ]
            # Multiple immutable resamples are possible across code revisions.  The current
            # invocation is deterministic and returns the one belonging to this checkout.
            result = _run_cli(
                lib,
                ["candidate", "resample", "--run-id", run_id,
                 "--parent-recipe-id", parent, "--target-rate-hz", "48000",
                 "--algo", "scipy_polyphase"],
            )
            current = _candidate_id(result)
            if current not in resamples:
                resamples.append(current)
            print(f"[{number}/{len(run_dirs)}] {run_id} {label}_48k={current}", flush=True)


def _read_audio(candidate_dir: Path) -> tuple[np.ndarray, sf.SoundFile]:
    wav = candidate_dir / "output.f32.wav"
    data, rate = sf.read(wav, dtype="float32", always_2d=True)
    info = sf.info(wav)
    if info.subtype != "FLOAT":
        raise RuntimeError(f"non-FLOAT candidate: {wav} ({info.subtype})")
    if int(rate) != info.samplerate or len(data) != info.frames or data.shape[1] != info.channels:
        raise RuntimeError(f"inconsistent decoded audio metadata: {wav}")
    return data, info


def _artifact_record(run_dir: Path, rid: str) -> dict[str, Any]:
    candidate_dir = run_dir / "candidates" / rid.replace(":", "_")
    recipe = _json(candidate_dir / "recipe.json")
    data, info = _read_audio(candidate_dir)
    channels = recipe["input_pcm"]["channel_layout"]
    actual_pcm = identity.artifact_pcm_sha256(data, int(info.samplerate), channels, len(data))
    declared_pcm = (candidate_dir / "output.pcm.sha256").read_text().strip()
    if actual_pcm != declared_pcm:
        raise RuntimeError(f"PCM hash mismatch for {run_dir.name}/{rid}")
    commit = str(recipe.get("software", {}).get("audio_extract_commit", ""))
    if not FULL_COMMIT.fullmatch(commit):
        raise RuntimeError(f"invalid code provenance for {run_dir.name}/{rid}: {commit}")
    unexpected = [
        path.name for path in candidate_dir.iterdir()
        if path.suffix.lower() in {".mp3", ".aac", ".m4a", ".ogg", ".opus", ".flac"}
    ]
    if unexpected:
        raise RuntimeError(f"lossy/compressed intermediate in {candidate_dir}: {unexpected}")
    return {
        "recipe_id": rid,
        "artifact_pcm_sha256": actual_pcm,
        "container_sha256": _sha_file(candidate_dir / "output.f32.wav"),
        "sample_rate_hz": int(info.samplerate),
        "channels": channels,
        "frames": int(info.frames),
        "sample_format": "float32",
        "code_commit": commit,
        "path": str(candidate_dir / "output.f32.wav"),
        "recipe": recipe,
    }


def _verify_lineage(
    run_dir: Path,
    index: dict[str, tuple[Path, dict[str, Any]]],
    rid: str,
    source_pcm: str,
    seen: set[str] | None = None,
) -> list[str]:
    seen = set() if seen is None else seen
    if rid in seen:
        raise RuntimeError(f"cycle in {run_dir.name} lineage at {rid}")
    seen.add(rid)
    if rid not in index:
        raise RuntimeError(f"missing candidate parent {run_dir.name}/{rid}")
    recipe = index[rid][1]
    parents = recipe.get("input_pcm", {}).get("parent_recipe_ids")
    if parents is None:
        model = recipe.get("model", {})
        parents = model.get("members") if isinstance(model.get("members"), list) else []
    result = [rid]
    for parent in parents:
        if parent in index:
            result.extend(_verify_lineage(run_dir, index, parent, source_pcm, seen))
        elif parent != source_pcm:
            raise RuntimeError(f"unresolved parent {run_dir.name}/{rid} -> {parent}")
    input_hash = recipe.get("input_pcm", {}).get("sha256")
    if input_hash != source_pcm and not any(
        (parent in index and (index[parent][0] / "output.pcm.sha256").read_text().strip() == input_hash)
        for parent in parents
    ):
        raise RuntimeError(f"input PCM is not bound to a parent for {run_dir.name}/{rid}")
    return result


def _control_metrics(source_path: Path, output_path: Path, source_channels: list[str]) -> dict[str, float]:
    source, source_rate = sf.read(source_path, dtype="float32", always_2d=True)
    output, output_rate = sf.read(output_path, dtype="float32", always_2d=True)
    if source_channels == ["FC"]:
        source = np.repeat(source, 2, axis=1)
    if source_rate != output_rate or source.shape != output.shape:
        raise RuntimeError(f"control grid mismatch: {source_path} vs {output_path}")
    source64 = source.astype(np.float64)
    output64 = output.astype(np.float64)
    delta = source64 - output64
    source_energy = float(np.sum(source64 * source64))
    delta_energy = float(np.sum(delta * delta))
    eps = np.finfo(np.float64).tiny
    return {
        "removed_energy_ratio": delta_energy / max(source_energy, eps),
        "removed_rms_db_relative_to_source": 10.0 * math.log10(max(delta_energy, eps) / max(source_energy, eps)),
        "retained_energy_ratio": float(np.sum(output64 * output64)) / max(source_energy, eps),
        "max_abs_difference": float(np.max(np.abs(delta))),
    }


def catalogue(
    lib: Path,
    input_root: Path,
    output_root: Path,
    lock_path: Path,
    expected_runs: int,
) -> None:
    lock = _json(lock_path)
    allowed_bundles = {bundle["bundle_sha256"]: bundle for bundle in lock["bundles"]}
    input_by_name = {path.name: path for path in input_root.rglob("*.mp3")}
    run_dirs = sorted(path for path in lib.iterdir() if (path / "source/source.json").is_file())
    if len(run_dirs) != expected_runs:
        raise RuntimeError(f"expected {expected_runs} runs, found {len(run_dirs)}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for number, run_dir in enumerate(run_dirs, 1):
        source = _json(run_dir / "source/source.json")
        source_path = run_dir / "source/canonical.f32.wav"
        source_data, source_info = _read_source(source_path)
        source_pcm = identity.artifact_pcm_sha256(
            source_data, int(source_info.samplerate), source["channel_layout"], len(source_data)
        )
        if source_pcm != source["input_pcm_sha256"]:
            raise RuntimeError(f"source PCM mismatch for {run_dir.name}")
        original = input_by_name.get(source["original_filename"])
        if original is None or _sha_file(original) != source["source_blob_sha256"]:
            raise RuntimeError(f"source blob mismatch or unavailable for {run_dir.name}")

        index = _index(run_dir)
        channel_maps = [rid for rid, (_, r) in index.items() if r["operation"]["type"] == "channel_map"]
        expected_map_count = 1 if source["channel_layout"] == ["FC"] else 0
        if len(channel_maps) != expected_map_count:
            raise RuntimeError(f"{run_dir.name}: expected {expected_map_count} channel maps, found {channel_maps}")

        vocals = {
            recipe.get("model", {}).get("executed_bundle_id"): rid
            for rid, (_, recipe) in index.items()
            if recipe.get("operation", {}).get("type") == "separate"
        }
        if set(vocals) != set(allowed_bundles):
            raise RuntimeError(f"{run_dir.name}: model bundle set differs from lock: {set(vocals)}")
        mdx_vocal = vocals[MDX_BUNDLE]
        median = _find_one(index, lambda r: r.get("model", {}).get("model_id") == ENSEMBLE_MODEL, "median")
        mdx = _find_one(
            index,
            lambda r: r.get("model", {}).get("model_id") == "single-vocal-residual"
            and r.get("model", {}).get("members") == [mdx_vocal],
            "MDX residual",
        )

        variants: dict[str, Any] = {}
        for label, parent in (("median", median), ("mdx", mdx)):
            resamples = [
                rid for rid, (_, r) in index.items()
                if r.get("operation", {}).get("type") == "resample"
                and r.get("input_pcm", {}).get("parent_recipe_ids") == [parent]
                and r.get("effective_config", {}).get("output_rate_hz") == 48000
            ]
            current_commit = os.environ.get("AUDIO_EXTRACT_CODE_COMMIT")
            if current_commit:
                resamples = [rid for rid in resamples if index[rid][1]["software"]["audio_extract_commit"] == current_commit]
            if len(resamples) != 1:
                raise RuntimeError(f"{run_dir.name}: expected one current {label} resample, found {resamples}")
            native_record = _artifact_record(run_dir, parent)
            delivery_record = _artifact_record(run_dir, resamples[0])
            if native_record["sample_rate_hz"] != 44100 or native_record["channels"] != ["FL", "FR"]:
                raise RuntimeError(f"invalid native grid for {run_dir.name}/{label}")
            expected_frames = math.ceil(native_record["frames"] * 48000 / 44100)
            if (
                delivery_record["sample_rate_hz"] != 48000
                or delivery_record["channels"] != ["FL", "FR"]
                or delivery_record["frames"] != expected_frames
            ):
                raise RuntimeError(f"invalid delivery grid for {run_dir.name}/{label}")
            lineage = sorted(set(_verify_lineage(run_dir, index, resamples[0], source_pcm)))
            link = output_root / f"{run_dir.name}__instrumental_{label}_48k_f32.wav"
            target = Path(delivery_record["path"])
            if link.is_symlink() and link.resolve() != target.resolve():
                link.unlink()
            if not link.exists():
                # A relative link remains valid both through the container's /data
                # mount and through the NAS's native /share/... dataset path.
                link.symlink_to(os.path.relpath(target, start=link.parent))
            variants[label] = {
                "native_44100_f32": {key: value for key, value in native_record.items() if key != "recipe"},
                "delivery_48000_f32": {key: value for key, value in delivery_record.items() if key != "recipe"},
                "delivery_symlink": str(link),
                "lineage_recipe_ids": lineage,
            }

        no_voice = "no-voice" in run_dir.name or "novoice" in run_dir.name
        controls = None
        if no_voice:
            controls = {
                label: _control_metrics(source_path, Path(record["native_44100_f32"]["path"]), source["channel_layout"])
                for label, record in variants.items()
            }
        rows.append({
            "schema": "audio-extract/median-v2-production-output/v1",
            "run_id": run_dir.name,
            "source": {
                **source,
                "canonical_path": str(source_path),
                "original_path": str(original),
                "canonical_container_sha256": _sha_file(source_path),
            },
            "channel_map_recipe_id": channel_maps[0] if channel_maps else None,
            "models": {
                bundle["logical_id"]: {
                    "bundle_sha256": bundle_id,
                    "vocal_recipe_id": vocals[bundle_id],
                    "files": bundle["files"],
                }
                for bundle_id, bundle in sorted(allowed_bundles.items())
            },
            "variants": variants,
            "no_voice_control_metrics": controls,
            "validation": {
                "source_blob_rehashed": True,
                "source_pcm_rehashed": True,
                "candidate_pcm_rehashed": True,
                "float32_wav": True,
                "exact_grid": True,
                "complete_markers": True,
                "explicit_parent_lineage": True,
                "model_lock_matched": True,
                "no_lossy_analysis_intermediate": True,
            },
        })
        print(f"AUDIT [{number}/{len(run_dirs)}] {run_dir.name} OK", flush=True)

    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_text(output_root / "outputs.jsonl", jsonl)
    total_frames = sum(row["variants"]["median"]["delivery_48000_f32"]["frames"] for row in rows)
    total_bytes = sum(
        Path(row["variants"][label]["delivery_48000_f32"]["path"]).stat().st_size
        for row in rows for label in ("median", "mdx")
    )
    summary = {
        "schema": "audio-extract/median-v2-production-summary/v1",
        "status": "needs_human_ab",
        "runs": len(rows),
        "outputs": len(rows) * 2,
        "sample_rate_hz": 48000,
        "channels": ["FL", "FR"],
        "sample_format": "float32",
        "median_total_duration_seconds": total_frames / 48000,
        "output_bytes": total_bytes,
        "catalogue_sha256": "sha256:" + hashlib.sha256(jsonl.encode()).hexdigest(),
        "model_lock_sha256": _sha_file(lock_path),
        "validation": {key: all(row["validation"][key] for row in rows) for key in rows[0]["validation"]},
        "no_voice_controls": {
            row["run_id"]: row["no_voice_control_metrics"]
            for row in rows if row["no_voice_control_metrics"] is not None
        },
    }
    _atomic_text(output_root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


def _read_source(path: Path) -> tuple[np.ndarray, sf.SoundFile]:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    info = sf.info(path)
    if info.subtype != "FLOAT" or int(rate) != info.samplerate:
        raise RuntimeError(f"invalid canonical source: {path}")
    return data, info


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=26)
    parser.add_argument("action", choices=("finalize", "catalogue", "all"))
    args = parser.parse_args()
    if args.action in {"finalize", "all"}:
        finalize(args.lib, args.model_dir, args.model_lock, args.expected_runs)
    if args.action in {"catalogue", "all"}:
        catalogue(args.lib, args.input_root, args.output_root, args.model_lock, args.expected_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
