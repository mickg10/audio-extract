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

from . import alignment, canon, identity
from .cantolopera_training import SELECTION_SCHEMA, TASK
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


def _resolved_cantolopera_path(audio_root: Path, source_path: str) -> Path:
    """Resolve a staged source by filename without trusting the audit host path."""

    source = Path(source_path)
    if not source.name or source.name in {".", ".."}:
        raise ValueError(f"invalid Cantolopera source path: {source_path!r}")
    result = (audio_root / source.name).resolve()
    root = audio_root.resolve()
    if root not in result.parents:
        raise ValueError(f"Cantolopera source escapes audio root: {source_path!r}")
    return result


def _strict_soxr_vhq(audio: np.ndarray, source_rate: int, output_rate: int) -> tuple[np.ndarray, str]:
    """Run the declared backend or fail; recipe identity never hides a fallback."""

    import soxr

    result = soxr.resample(
        np.asarray(audio, dtype="float64"), source_rate, output_rate, quality="VHQ"
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("soxr_vhq produced non-finite samples")
    return np.asarray(result, dtype="float64"), str(soxr.__version__)


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


def materialize_cantolopera_pair(
    selected: dict, audio_root: Path, output_root: Path, *, selection_sha256: str
) -> dict:
    """Materialize one strict same-take pair as an explicit aligned M/A/V child."""

    if selected.get("schema") != SELECTION_SCHEMA:
        raise ValueError("wrong Cantolopera selection schema")
    assessment = selected.get("assessment") or {}
    if not assessment.get("eligible") or assessment.get("cohort") != "tier_a":
        raise ValueError(f"Cantolopera materializer refuses non-Tier-A pair {selected.get('pair_id')}")
    if selected.get("task") != TASK:
        raise ValueError(f"Cantolopera task must be {TASK}")
    if selected.get("split") not in {"train", "val"} or not selected.get("group_id"):
        raise ValueError("selected pair lacks frozen group/split facts")

    paths = {
        "M": _resolved_cantolopera_path(audio_root, selected["files"]["voice"]["path"]),
        "A": _resolved_cantolopera_path(audio_root, selected["files"]["orchestra"]["path"]),
    }
    expected = {
        "M": selected["files"]["voice"]["container_sha256"],
        "A": selected["files"]["orchestra"]["container_sha256"],
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if _file_sha(path) != expected[role]:
            raise ValueError(f"Cantolopera {role} container SHA-256 mismatch: {path}")
        info = sf.info(path)
        if (info.samplerate, info.channels, info.subtype) != (48_000, 2, "FLOAT"):
            raise GridMismatch(f"Cantolopera source must be 48 kHz stereo FLOAT: {path}")

    mixture, mix_rate = _read(paths["M"])
    accompaniment, accompaniment_rate = _read(paths["A"])
    _same_grid({"M": mixture, "A": accompaniment}, {"M": mix_rate, "A": accompaniment_rate})
    if mix_rate != 48_000 or mixture.shape[1] != 2:
        raise GridMismatch("Cantolopera source grid changed after validation")

    transform = assessment["transform"]
    fixed = transform["mixture_to_orchestra"]
    fixed_alignment = alignment.Alignment(
        delay_samples=int(fixed["delay_samples"]),
        fractional=float(fixed["fractional_samples"]),
        polarity=int(fixed["polarity"]),
        channel_swap=bool(fixed["channel_swap"]),
        confidence=1.0,
        residual_db=0.0,
    )
    aligned_mixture = alignment.apply_alignment(
        mixture, fixed_alignment, target_len=len(accompaniment)
    )
    fixed_gain = float(transform["orchestra_gain"])
    scaled_accompaniment = accompaniment.astype("float64") * fixed_gain
    aligned_mixture, soxr_version_m = _strict_soxr_vhq(aligned_mixture, 48_000, SR)
    scaled_accompaniment, soxr_version_a = _strict_soxr_vhq(
        scaled_accompaniment, 48_000, SR
    )
    if soxr_version_m != soxr_version_a:
        raise RuntimeError("soxr version changed inside one materialization")
    _same_grid(
        {"M": aligned_mixture, "A": scaled_accompaniment},
        {"M": SR, "A": SR},
    )
    mixture_f32 = np.asarray(aligned_mixture, dtype="float32")
    accompaniment_f32 = np.asarray(scaled_accompaniment, dtype="float32")
    vocal_f32 = np.asarray(mixture_f32 - accompaniment_f32, dtype="float32")
    if not all(np.all(np.isfinite(value)) for value in (
        mixture_f32, accompaniment_f32, vocal_f32
    )):
        raise ValueError("Cantolopera materialization produced non-finite audio")

    recipe = {
        "integrity_class": "same_take_paired_target",
        "task": TASK,
        "group_id": selected["group_id"],
        "split": selected["split"],
        "audit_sha256": selected["audit_sha256"],
        "selector_code_commit": selected["selector_code_commit"],
        "selection_sha256": selection_sha256,
        "parents": {
            role: {"path": str(path), "container_sha256": expected[role]}
            for role, path in paths.items()
        },
        "selection_transform": transform,
        "operations": [
            {
                "operation": "align",
                "role": "M",
                "reference_role": "A",
                **fixed,
                "implementation": "audio_extract.alignment.apply_alignment/v1",
                "boundary": "zero_pad_then_exact_target_frames",
            },
            {
                "operation": "fixed_gain",
                "role": "A",
                "gain": transform["orchestra_gain"],
            },
            {
                "operation": "resample",
                "roles": ["M", "A"],
                "resampler": "soxr_vhq",
                "backend": "python-soxr",
                "backend_version": soxr_version_m,
                "source_rate_hz": 48_000,
                "output_rate_hz": SR,
            },
            {"operation": "derive_source", "role": "V", "expression": "M-A"},
        ],
    }
    return _publish(
        output_root,
        selected["pair_id"],
        mixture_f32,
        accompaniment_f32,
        vocal_f32,
        recipe,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root",
                        help="root containing freidi/ and cantoria/ source directories")
    parser.add_argument("--truth-root",
                        help="materialize only Bologna/Aalto exact CV truth from this root")
    parser.add_argument("--cantolopera-selection",
                        help="Tier-A selection JSONL produced by the pair-audit selector")
    parser.add_argument("--cantolopera-audio-root",
                        help="local root containing completed Cantolopera WAV payloads")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_root)
    results = []
    modes = sum(bool(value) for value in (
        args.source_root, args.truth_root, args.cantolopera_selection
    ))
    if modes != 1:
        parser.error("provide exactly one of --source-root, --truth-root, or --cantolopera-selection")
    if bool(args.cantolopera_selection) != bool(args.cantolopera_audio_root):
        parser.error("Cantolopera selection and audio root are required together")
    report_name = "materialization-report.json"
    if args.cantolopera_selection:
        selection_path = Path(args.cantolopera_selection)
        selection_sha256 = _file_sha(selection_path)
        selected_rows = [
            json.loads(line)
            for line in selection_path.read_text().splitlines()
            if line.strip()
        ]
        for selected in selected_rows:
            if not selected.get("assessment", {}).get("eligible"):
                continue
            if selected.get("split") != args.split:
                continue
            try:
                results.append(materialize_cantolopera_pair(
                    selected,
                    Path(args.cantolopera_audio_root),
                    output,
                    selection_sha256=selection_sha256,
                ))
            except Exception as exc:
                results.append({
                    "work_id": selected.get("pair_id"),
                    "status": "excluded",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
        report_name = f"materialization-report-cantolopera-{args.split}.json"
    elif args.truth_root:
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
    report_path = output / report_name
    report_payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if report_path.exists() and report_path.read_text() != report_payload:
        raise RuntimeError(f"refusing to rewrite differing materialization report: {report_path}")
    if not report_path.exists():
        report_path.write_text(report_payload)
    print(json.dumps(summary, indent=2))
    return 0 if summary["materialized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
