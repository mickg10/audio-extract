#!/usr/bin/env python3
"""DEMUCS-AFFINE-PARITY-001: freeze and verify the production inference path."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from audio_extract import identity
from audio_extract.demucs_affine import (
    DemucsAffine,
    forward_demucs_production_affine,
    normalize_demucs_batch,
)
from audio_extract.train_classical import (
    SR,
    _apply_model_production,
    _load_model,
    _read_exact,
)

FULL_EXACT_TOLERANCE = 0.0
# Frozen from the immutable f62cb60 attempt before any model update.  The
# installed audio-separator 0.44.5 adapter divides by ref.std(), while released
# Demucs divides by ref.std() + 1e-8.  On the required complete Verdi work this
# sole affine difference measured max_abs=1.1644698679447174e-4 and
# RMS=3.929586696028066e-6.  The gate separately requires the adapter to be
# bit-exact with a raw-std reference, so these bounds cannot hide an inference
# or source-order mismatch.
ADAPTER_MAX_ABS_TOLERANCE = 1.25e-4
ADAPTER_RMS_TOLERANCE = 4e-6


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"parity grid mismatch: {reference.shape} != {candidate.shape}")
    difference = (reference.detach().cpu().float() - candidate.detach().cpu().float()).numpy()
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference.astype("float64") ** 2))),
        "finite": bool(np.all(np.isfinite(difference))),
        "shape": list(reference.shape),
    }


def _publish(path: Path, audio: torch.Tensor) -> dict:
    array = audio.detach().cpu().T.numpy().astype("float32")
    if path.exists():
        reopened, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SR or not np.array_equal(reopened, array):
            raise RuntimeError(f"refusing to rewrite differing parity artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, array, SR, subtype="FLOAT")
    info = sf.info(path)
    reopened, sr = sf.read(path, dtype="float32", always_2d=True)
    if (info.frames, info.samplerate, info.channels, info.subtype) != (
            len(array), SR, 2, "FLOAT") or not np.array_equal(reopened, array):
        raise RuntimeError(f"published parity artifact failed exact reopen: {path}")
    return {
        "path": str(path),
        "container_sha256": _sha_file(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            reopened, int(sr), ["FL", "FR"], len(reopened)
        ),
        "frames": info.frames,
        "sample_rate_hz": info.samplerate,
        "channels": ["FL", "FR"],
        "subtype": info.subtype,
    }


def _api_separator(model, *, device: str):
    from demucs.api import Separator

    separator = Separator.__new__(Separator)
    separator._model = model
    separator._samplerate = SR
    separator._audio_channels = 2
    separator._segment = None
    separator._shifts = 0
    separator._split = True
    separator._overlap = 0.25
    separator._device = device
    separator._jobs = 0
    separator._progress = False
    separator._callback = None
    separator._callback_arg = None
    return separator


def _api_sources(model, audio: torch.Tensor, *, device: str) -> torch.Tensor:
    _mix, stems = _api_separator(model, device=device).separate_tensor(audio, sr=SR)
    return torch.stack([stems[source] for source in model.sources], dim=0).unsqueeze(0)


def _adapter_sources(model, audio: torch.Tensor, *, device: str) -> torch.Tensor:
    from audio_separator.separator.architectures.demucs_separator import DemucsSeparator

    adapter = DemucsSeparator.__new__(DemucsSeparator)
    adapter.demucs_model_instance = model
    adapter.shifts = 0
    adapter.segments_enabled = True
    adapter.overlap = 0.25
    adapter.torch_device = device
    adapter.logger = logging.getLogger("demucs-affine-parity")
    raw = adapter.demix_demucs(audio.detach().cpu().numpy())
    # audio-separator swaps drums/bass internally and exposes the named order
    # [Bass, Drums, Other, Vocals]. Restore the model's declared source order.
    named = {"bass": raw[0], "drums": raw[1], "other": raw[2], "vocals": raw[3]}
    return torch.from_numpy(np.stack([named[source] for source in model.sources])).unsqueeze(0)


def _adapter_raw_std_reference(model, audio: torch.Tensor, *, device: str) -> torch.Tensor:
    """Official overlap inference with audio-separator 0.44.5's exact affine."""
    from demucs.apply import apply_model

    reference = audio.mean(dim=0)
    mean = reference.mean()
    scale = reference.std()
    normalized = ((audio - mean) / scale).unsqueeze(0)
    return apply_model(
        model, normalized, device=device, shifts=0, split=True, overlap=0.25
    ) * scale + mean


def _serialized_reload(model, config: dict, output_dir: Path, device: str):
    from demucs import states
    from omegaconf import OmegaConf

    path = output_dir / "model-step-000000.th"
    if not path.exists():
        torch.save(
            states.serialize_model(
                model, OmegaConf.create({"audio_extract": config}), half=False
            ), path,
        )
    return states.load_model(path, strict=True).to(device).eval(), path


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(0)
    model, base = _load_model(config, args.device, 0)
    model.eval()
    reloaded, export_path = _serialized_reload(model, config, output_dir, args.device)

    work_dir = args.truth_root / "bologna_verdi"
    full = _read_exact(work_dir / "mix_with_voice.wav")
    with torch.no_grad():
        trainer = _apply_model_production(
            model, full.unsqueeze(0), device=args.device, split=True
        ).cpu()
        api = _api_sources(model, full, device=args.device).cpu()
        exported = _apply_model_production(
            reloaded, full.unsqueeze(0), device=args.device, split=True
        ).cpu()
        adapter = _adapter_sources(reloaded, full, device=args.device).cpu()
        adapter_raw_std_reference = _adapter_raw_std_reference(
            reloaded, full, device=args.device
        ).cpu()

    full_comparisons = {
        "trainer_vs_demucs_api": _comparison(trainer, api),
        "trainer_vs_export_reload": _comparison(trainer, exported),
        "trainer_vs_audio_separator_adapter": _comparison(trainer, adapter),
        "audio_separator_adapter_vs_raw_std_reference": _comparison(
            adapter_raw_std_reference, adapter
        ),
    }

    recipe = json.loads(
        (Path(config["data"]["materialized_root"]) / "bologna_verdi" / "recipe.json").read_text()
    )
    affine_fact = recipe["demucs_full_track_affine"]["M"]
    full_affine = DemucsAffine(
        mean=torch.tensor([[float(affine_fact["mean"])]], device=args.device),
        scale=torch.tensor([[float(affine_fact["scale"])]], device=args.device),
    )
    crop = full[:, args.crop_start:args.crop_start + args.crop_frames].unsqueeze(0).to(args.device)
    with torch.no_grad():
        crop_trainer = forward_demucs_production_affine(model, crop, full_affine).cpu()
        crop_exported = forward_demucs_production_affine(reloaded, crop, full_affine).cpu()
        _, crop_local_affine = normalize_demucs_batch(crop)
        crop_local = forward_demucs_production_affine(
            model, crop, crop_local_affine
        ).cpu()
    crop_comparisons = {
        "trainer_vs_export_reload": _comparison(crop_trainer, crop_exported),
        "full_track_vs_crop_local_affine": _comparison(crop_trainer, crop_local),
        "persisted_affine": affine_fact,
        "crop_local_affine": {
            "mean": float(crop_local_affine.mean[0, 0]),
            "scale": float(crop_local_affine.scale[0, 0]),
        },
        "full_track_affine_reused": bool(
            not torch.equal(full_affine.mean, crop_local_affine.mean)
            or not torch.equal(full_affine.scale, crop_local_affine.scale)
        ),
    }

    artifacts = {"full": {}, "crop": {}}
    full_paths = {
        "trainer": trainer,
        "demucs_api": api,
        "export_reload": exported,
        "audio_separator_adapter": adapter,
    }
    for path_name, estimates in full_paths.items():
        artifacts["full"][path_name] = {
            source: _publish(output_dir / "full" / path_name / f"{source}.f32.wav",
                             estimates[0, index])
            for index, source in enumerate(model.sources)
        }
    for path_name, estimates in (("trainer", crop_trainer), ("export_reload", crop_exported)):
        artifacts["crop"][path_name] = {
            source: _publish(output_dir / "crop" / path_name / f"{source}.f32.wav",
                             estimates[0, index])
            for index, source in enumerate(model.sources)
        }

    failures = []
    for name in ("trainer_vs_demucs_api", "trainer_vs_export_reload"):
        if full_comparisons[name]["max_abs"] > FULL_EXACT_TOLERANCE:
            failures.append(f"{name} exceeded exact tolerance")
    adapter_cmp = full_comparisons["trainer_vs_audio_separator_adapter"]
    if (adapter_cmp["max_abs"] > ADAPTER_MAX_ABS_TOLERANCE
            or adapter_cmp["rms"] > ADAPTER_RMS_TOLERANCE):
        failures.append("audio-separator adapter exceeded frozen numerical tolerance")
    adapter_causal_cmp = full_comparisons[
        "audio_separator_adapter_vs_raw_std_reference"
    ]
    if adapter_causal_cmp["max_abs"] > FULL_EXACT_TOLERANCE:
        failures.append("audio-separator adapter differs from its raw-std reference")
    if crop_comparisons["trainer_vs_export_reload"]["max_abs"] > FULL_EXACT_TOLERANCE:
        failures.append("crop export/reload exceeded exact tolerance")
    if not crop_comparisons["full_track_affine_reused"]:
        failures.append("crop did not prove reuse of non-local full-track statistics")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "schema": "audio-extract/demucs-affine-parity/v1",
        "experiment": "DEMUCS-AFFINE-PARITY-001",
        "status": "pass" if not failures else "fail",
        "promotion_eligible": False,
        "source_commit": commit,
        "base_checkpoint": base,
        "source_order": list(model.sources),
        "full_work": "bologna_verdi",
        "full_frames": full.shape[-1],
        "crop_start": args.crop_start,
        "crop_frames": args.crop_frames,
        "tolerances": {
            "trainer_api_and_export_max_abs": FULL_EXACT_TOLERANCE,
            "audio_separator_adapter_max_abs": ADAPTER_MAX_ABS_TOLERANCE,
            "audio_separator_adapter_rms": ADAPTER_RMS_TOLERANCE,
            "audio_separator_raw_std_reference_max_abs": FULL_EXACT_TOLERANCE,
            "adapter_tolerance_basis": {
                "immutable_attempt_source_commit":
                    "f62cb60a07c03cc887de2b3f36af463207c0e7c2",
                "observed_max_abs": 0.00011644698679447174,
                "observed_rms": 3.929586696028066e-06,
                "cause": (
                    "audio-separator 0.44.5 uses ref.std(); released Demucs uses "
                    "ref.std() + 1e-8"
                ),
            },
        },
        "full_comparisons": full_comparisons,
        "crop_comparisons": crop_comparisons,
        "artifacts": artifacts,
        "export": {"path": str(export_path), "sha256": _sha_file(export_path)},
        "failures": failures,
    }
    report_path = output_dir / "report.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_path.exists() and report_path.read_text() != payload:
        raise RuntimeError(f"refusing to rewrite differing parity report: {report_path}")
    if not report_path.exists():
        report_path.write_text(payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-start", type=int, default=1_000_000)
    parser.add_argument("--crop-frames", type=int, default=343_980)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "experiment": report["experiment"], "status": report["status"],
        "failures": report["failures"], "report": str(args.output_dir / "report.json"),
    }, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
