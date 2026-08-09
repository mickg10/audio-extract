"""Materialize immutable float32 training triplets on one explicit sample grid.

The source directories must already be available locally.  This module performs
no SSH, credential handling, or download.  It refuses mismatched source grids;
there is no minimum-length truncation, clipping, or integer PCM intermediate.
Every accepted work records source hashes and explicit transform nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import canon, identity
from .challenges import SOLOIST_VS_REST_ROLES
from .resample import resample

SR = 44_100
SCHEMA = "audio-extract/classical-train-materialization/v1"
DOMAIN = b"audio-extract-classical-train-materialization-v1\0"


class GridMismatch(ValueError):
    """Sources cannot form an exact training triplet without an explicit repair."""


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"non-finite source samples: {path}")
    return audio, int(sr)


def _same_grid(named: dict[str, np.ndarray], rates: dict[str, int]) -> None:
    shapes = {name: tuple(value.shape) for name, value in named.items()}
    if len(set(shapes.values())) != 1 or len(set(rates.values())) != 1:
        raise GridMismatch(f"source grid mismatch: shapes={shapes}, rates={rates}")


def _stereo(audio: np.ndarray) -> tuple[np.ndarray, dict | None]:
    if audio.shape[1] == 2:
        return audio, None
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1), {
            "operation": "channel_construct",
            "method": "duplicate_mono_to_FL_FR",
            "input_channels": 1,
            "output_channels": 2,
        }
    raise GridMismatch(f"only mono or stereo sources are supported, got {audio.shape[1]} channels")


def _activity_mask(vocal: np.ndarray) -> np.ndarray:
    env = np.mean(np.abs(vocal), axis=1)
    window = max(1, round(0.05 * SR))
    smooth = np.convolve(env, np.ones(window) / window, mode="same")
    threshold = 0.05 * (float(smooth.max()) + 1e-12)
    return (smooth > threshold).astype("float32")


def _recipe_id(recipe: dict) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + canon.canonicalize(recipe)).hexdigest()


def _pcm_hash(audio: np.ndarray) -> str:
    return identity.artifact_pcm_sha256(
        np.asarray(audio, dtype="float32"), SR, ["FL", "FR"], int(audio.shape[0])
    )


def _demucs_full_track_affine(audio: np.ndarray) -> dict:
    """Compute the released Demucs affine with its exact torch reduction semantics."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exact training uses the train extra
        raise RuntimeError("Demucs-affine materialization requires the train extra") from exc
    from .demucs_affine import demucs_affine_from_audio

    tensor = torch.from_numpy(np.ascontiguousarray(audio.T, dtype="float32")).unsqueeze(0)
    affine = demucs_affine_from_audio(tensor)
    return {
        "implementation": "demucs-full-track-affine/v1",
        "epsilon": "0.00000001",
        # Canonical recipes forbid inexact JSON numbers. Nine significant decimal
        # digits round-trip every float32 statistic exactly.
        "mean": format(float(affine.mean[0, 0]), ".9g"),
        "scale": format(float(affine.scale[0, 0]), ".9g"),
    }


def _publish(output_root: Path, work: str, mixture: np.ndarray, accompaniment: np.ndarray,
             vocal: np.ndarray, recipe: dict) -> dict:
    _same_grid({"M": mixture, "A": accompaniment, "V": vocal}, {"M": SR, "A": SR, "V": SR})
    if mixture.shape[1] != 2:
        raise GridMismatch(f"materialized work must be stereo, got {mixture.shape}")
    residual = mixture.astype("float64") - accompaniment.astype("float64") - vocal.astype("float64")
    relative = np.sqrt(np.mean(residual ** 2)) / (np.sqrt(np.mean(mixture.astype("float64") ** 2)) + 1e-15)
    recipe = dict(recipe)
    recipe.update({"schema": SCHEMA, "work_id": work, "sample_rate_hz": SR,
                   "channel_layout": ["FL", "FR"], "frames": int(len(mixture)),
                   "sample_format": "float32-le-interleaved"})
    affines = {role: _demucs_full_track_affine(audio) for role, audio in
               (("M", mixture), ("A", accompaniment), ("V", vocal))}
    recipe["demucs_full_track_affine"] = affines
    recipe["recipe_id"] = _recipe_id(recipe)
    hashes = {role: _pcm_hash(audio) for role, audio in
              (("M", mixture), ("A", accompaniment), ("V", vocal))}
    final = output_root / work
    if final.exists():
        existing = json.loads((final / "recipe.json").read_text())
        if existing != recipe:
            raise RuntimeError(f"refusing to rewrite immutable materialization: {final}")
        for role, expected in hashes.items():
            audio, actual_sr = _read(final / f"{role}.f32.wav")
            if actual_sr != SR or _pcm_hash(audio) != expected:
                raise RuntimeError(f"immutable materialization verification failed: {final}/{role}")
        return {"work_id": work, "status": "verified_existing", "recipe_id": recipe["recipe_id"]}

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{work}.", dir=output_root))
    try:
        for role, audio in (("M", mixture), ("A", accompaniment), ("V", vocal)):
            sf.write(temporary / f"{role}.f32.wav", np.asarray(audio, dtype="float32"), SR,
                     subtype="FLOAT")
        np.save(temporary / "vocal_activity.npy", _activity_mask(vocal))
        (temporary / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
        report = {
            "work_id": work,
            "recipe_id": recipe["recipe_id"],
            "frames": int(len(mixture)),
            "M_eq_A_plus_V_db": float(20 * np.log10(relative + 1e-15)),
            "pcm_sha256": hashes,
            "demucs_full_track_affine": affines,
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, final)
        return {**report, "status": "materialized"}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def materialize_freidi(source_root: Path, output_root: Path, number: str) -> dict:
    paths = {"M": source_root / number / "mix.flac",
             "A": source_root / number / "accomp.flac",
             "V": source_root / number / "voice.flac"}
    arrays, rates = {}, {}
    for role, path in paths.items():
        arrays[role], rates[role] = _read(path)
    _same_grid(arrays, rates)
    operations = []
    if rates["M"] != SR:
        arrays = {role: resample(audio, rates[role], SR, "soxr_vhq")
                  for role, audio in arrays.items()}
        operations.append({"operation": "resample", "resampler": "soxr_vhq",
                           "source_rate_hz": rates["M"], "output_rate_hz": SR})
    _same_grid(arrays, {role: SR for role in arrays})
    arrays = {role: np.asarray(audio, dtype="float32") for role, audio in arrays.items()}
    stereo_nodes = []
    for role in arrays:
        arrays[role], node = _stereo(arrays[role])
        if node is not None:
            stereo_nodes.append({**node, "role": role})
    recipe = {
        "integrity_class": "same_performance_bleed",
        "parents": {role: {"path": str(path), "container_sha256": _file_sha(path)}
                    for role, path in paths.items()},
        "operations": operations + stereo_nodes,
    }
    return _publish(output_root, f"freidi_no{number}", arrays["M"], arrays["A"],
                    arrays["V"], recipe)


def materialize_cantoria(source_root: Path, output_root: Path, code: str) -> dict:
    mix_path = source_root / f"Cantoria_{code}_MixOrgan.wav"
    vocal_path = source_root / f"Cantoria_{code}_Mix.wav"
    mixture, mix_sr = _read(mix_path)
    vocal, vocal_sr = _read(vocal_path)
    _same_grid({"M": mixture, "V": vocal}, {"M": mix_sr, "V": vocal_sr})
    if mix_sr != SR:
        raise GridMismatch(f"Cantoria {code} unexpectedly has sample rate {mix_sr}")
    accompaniment = mixture - vocal
    arrays = {"M": mixture, "A": accompaniment, "V": vocal}
    stereo_nodes = []
    for role in arrays:
        arrays[role], node = _stereo(arrays[role])
        if node is not None:
            stereo_nodes.append({**node, "role": role})
    recipe = {
        "integrity_class": "linear_exact",
        "parents": {
            "M": {"path": str(mix_path), "container_sha256": _file_sha(mix_path)},
            "V": {"path": str(vocal_path), "container_sha256": _file_sha(vocal_path)},
        },
        "operations": [
            {"operation": "derive_source", "role": "A", "expression": "M-V"},
            *stereo_nodes,
        ],
    }
    return _publish(output_root, f"cantoria_{code}", arrays["M"], arrays["A"],
                    arrays["V"], recipe)


def materialize_exact_truth(truth_root: Path, output_root: Path, work: str) -> dict:
    paths = {
        "M": truth_root / work / "mix_with_voice.wav",
        "A": truth_root / work / "orchestra_only.wav",
        "V": truth_root / work / "voice_ref.wav",
    }
    arrays, rates = {}, {}
    for role, path in paths.items():
        arrays[role], rates[role] = _read(path)
    _same_grid(arrays, rates)
    if rates["M"] != SR or arrays["M"].shape[1] != 2:
        raise GridMismatch(f"exact truth must already be 44.1 kHz stereo: {work}")
    recipe = {
        "integrity_class": "linear_exact",
        "task": "soloist_vs_rest",
        "task_roles": {
            role: list(values) for role, values in SOLOIST_VS_REST_ROLES.items()
        },
        "parents": {role: {"path": str(path), "container_sha256": _file_sha(path)}
                    for role, path in paths.items()},
        "operations": [{"operation": "role_map", "mapping": {
            "mix_with_voice": "M", "orchestra_only": "A", "voice_ref": "V"
        }}],
    }
    return _publish(output_root, work, arrays["M"], arrays["A"], arrays["V"], recipe)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root",
                        help="root containing freidi/ and cantoria/ source directories")
    parser.add_argument("--truth-root",
                        help="materialize only Bologna/Aalto exact CV truth from this root")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_root)
    results = []
    if bool(args.source_root) == bool(args.truth_root):
        parser.error("provide exactly one of --source-root or --truth-root")
    if args.truth_root:
        truth = Path(args.truth_root)
        for work in ("bologna_verdi", "bologna_puccini", "bologna_donizetti",
                     "aalto_mozart_dry"):
            try:
                results.append(materialize_exact_truth(truth, output, work))
            except Exception as exc:
                results.append({"work_id": work, "status": "excluded",
                                "reason": f"{type(exc).__name__}: {exc}"})
    else:
        source = Path(args.source_root)
        for number in ("06", "08", "09"):
            try:
                results.append(materialize_freidi(source / "freidi", output, number))
            except Exception as exc:
                results.append({"work_id": f"freidi_no{number}", "status": "excluded",
                                "reason": f"{type(exc).__name__}: {exc}"})
        for code in ("CEA", "EJB1", "EJB2", "HCB", "LBM1", "LBM2", "LJT1", "LJT2",
                     "LNG", "RRC", "SSS", "THM", "VBP", "YSM"):
            try:
                results.append(materialize_cantoria(source / "cantoria", output, code))
            except Exception as exc:
                results.append({"work_id": f"cantoria_{code}", "status": "excluded",
                                "reason": f"{type(exc).__name__}: {exc}"})
    summary = {"schema": SCHEMA + "/report", "results": results,
               "materialized": sum(r["status"] in {"materialized", "verified_existing"}
                                   for r in results),
               "excluded": sum(r["status"] == "excluded" for r in results)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "materialization-report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["materialized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
