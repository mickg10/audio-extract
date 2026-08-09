"""Pinned, named resamplers (v2.1 WP0 / §5.3).

Never rely on a library's hidden default resampling: one algorithm is selected by
name and recorded in the recipe. Registered algorithms map to a concrete backend;
the algorithm name (not just the rate) is part of candidate identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import canon, identity, recipe as recipe_mod
from .timebase import AudioGrid

RESAMPLERS = ("soxr_vhq", "scipy_polyphase", "ffmpeg_soxr_vhq")


def _soxr(x: np.ndarray, sr_in: int, sr_out: int, quality: str) -> np.ndarray:
    import soxr

    # x is (frames, channels); soxr resamples along axis 0.
    return soxr.resample(x, sr_in, sr_out, quality=quality)


def _scipy_polyphase(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    return resample_poly(x, up, down, axis=0)


def resample(x: np.ndarray, sr_in: int, sr_out: int, algo: str = "soxr_vhq") -> np.ndarray:
    """Resample ``x`` (frames, channels) from ``sr_in`` to ``sr_out`` with the named
    algorithm. No-op when the rates match. ``ffmpeg_soxr_vhq`` falls back to the
    in-process soxr backend here (the recipe still pins the name)."""
    if algo not in RESAMPLERS:
        raise ValueError(f"unknown resampler {algo!r}; registered: {RESAMPLERS}")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if sr_in == sr_out:
        return a
    if algo in ("soxr_vhq", "ffmpeg_soxr_vhq"):
        try:
            return _soxr(a, sr_in, sr_out, quality="VHQ")
        except Exception:
            return _scipy_polyphase(a, sr_in, sr_out)
    return _scipy_polyphase(a, sr_in, sr_out)


def resample_to_grid(x: np.ndarray, src: AudioGrid, dst_rate: int, algo: str = "soxr_vhq"):
    """Resample and return ``(array, new_grid, recipe_node)``. The recipe node
    records the exact resample step for the candidate DAG."""
    out = resample(x, src.sample_rate_hz, dst_rate, algo)
    grid = AudioGrid(sample_rate_hz=dst_rate, channels=src.channels, frames=int(out.shape[0]))
    node = {"operation": "resample", "resampler": algo,
            "source_rate_hz": src.sample_rate_hz, "output_rate_hz": dst_rate}
    return out, grid, node


def render_resample_candidate(layout, parent_recipe_id: str, *, target_rate_hz: int,
                              algo: str = "scipy_polyphase",
                              code_commit: str = "") -> dict:
    """Render a rate-conversion child without modifying its FLOAT parent.

    The production DDSP/player boundary requires 48 kHz stereo FLOAT WAV while
    separator candidates preserve the source grid.  This function makes that
    conversion an explicit immutable recipe node.  ``scipy_polyphase`` is the
    production default because it has no backend fallback or hidden negotiation.
    """
    import soundfile as sf

    from .manifest import Manifest
    from .storage import ImmutableWriteError

    if algo != "scipy_polyphase":
        raise ValueError("stored delivery resampling currently requires scipy_polyphase")
    if not isinstance(target_rate_hz, int) or target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be a positive integer")

    parent_dir = layout.candidate_dir(parent_recipe_id)
    parent_wav = parent_dir / "output.f32.wav"
    parent_recipe_path = parent_dir / "recipe.json"
    if not (parent_dir / "COMPLETE").is_file() or not parent_wav.is_file() or not parent_recipe_path.is_file():
        raise ValueError(f"resample parent {parent_recipe_id} is not complete")
    parent_recipe = json.loads(parent_recipe_path.read_text())
    audio, source_rate = sf.read(parent_wav, dtype="float32", always_2d=True)
    info = sf.info(parent_wav)
    if info.subtype != "FLOAT":
        raise ValueError(f"resample parent must be FLOAT, got {info.subtype}")
    source_rate = int(source_rate)
    if source_rate == target_rate_hz:
        raise ValueError("resample source and target rates are identical")

    parent_op = parent_recipe.get("operation", {}).get("type")
    if parent_op == "channel_map":
        channel_layout = parent_recipe.get("effective_config", {}).get("output_channel_layout")
    else:
        channel_layout = parent_recipe.get("input_pcm", {}).get("channel_layout")
    if not isinstance(channel_layout, list) or len(channel_layout) != audio.shape[1]:
        raise ValueError("resample parent recipe does not declare its output channel layout")
    parent_pcm = identity.artifact_pcm_sha256(
        audio, source_rate, channel_layout, len(audio)
    )
    declared_parent_pcm = (parent_dir / "output.pcm.sha256").read_text().strip()
    if parent_pcm != declared_parent_pcm:
        raise ValueError(
            f"resample parent PCM hash mismatch: {parent_pcm} != {declared_parent_pcm}"
        )

    divisor = math.gcd(source_rate, target_rate_hz)
    up, down = target_rate_hz // divisor, source_rate // divisor
    expected_frames = (len(audio) * up + down - 1) // down
    effective_config = {
        "resampler": algo,
        "source_rate_hz": source_rate,
        "output_rate_hz": target_rate_hz,
        "up_factor": up,
        "down_factor": down,
        "window": "kaiser_beta_5.0",
        "padtype": "constant",
        "frame_count_policy": "ceil(input_frames*up/down)",
        "output_frames": expected_frames,
    }
    transform_hash = hashlib.sha256(canon.canonicalize(effective_config)).hexdigest()
    recipe = {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": {
            "sha256": parent_pcm,
            "sample_rate_hz": source_rate,
            "channel_layout": channel_layout,
            "frames": len(audio),
            "sample_format": "float32-le-interleaved",
            "parent_recipe_ids": [parent_recipe_id],
        },
        "operation": {"type": "resample", "target": f"{target_rate_hz}_hz"},
        "model": {
            "model_id": f"{algo}/v1",
            "weights_sha256": transform_hash,
            "adapter": "audio-extract-resample",
            "adapter_revision": "scipy-resample-poly-v1",
        },
        "effective_config": effective_config,
        "software": {"audio_extract_commit": code_commit},
    }
    recipe_id = identity.recipe_id(recipe)
    candidate_dir = layout.candidate_dir(recipe_id)
    output_path = candidate_dir / "output.f32.wav"
    cached = output_path.exists()
    if cached:
        output, reopened_rate = sf.read(output_path, dtype="float32", always_2d=True)
        output_info = sf.info(output_path)
        if (len(output), int(reopened_rate), output.shape[1], output_info.subtype) != (
            expected_frames, target_rate_hz, len(channel_layout), "FLOAT"
        ):
            raise ValueError(f"cached resample grid mismatch: {output_info}")
        artifact = identity.artifact_pcm_sha256(
            output, target_rate_hz, channel_layout, len(output)
        )
        declared = (candidate_dir / "output.pcm.sha256").read_text().strip()
        if artifact != declared:
            raise ValueError(f"cached resample PCM hash mismatch: {artifact} != {declared}")
    else:
        output = resample(audio, source_rate, target_rate_hz, algo=algo)
        if output.shape != (expected_frames, len(channel_layout)) or not np.isfinite(output).all():
            raise RuntimeError(
                f"invalid resample output grid/content: {output.shape} != "
                f"{(expected_frames, len(channel_layout))}"
            )
        output = output.astype("float32")
        artifact = identity.artifact_pcm_sha256(
            output, target_rate_hz, channel_layout, len(output)
        )
        fd, temporary = tempfile.mkstemp(suffix=".f32.wav")
        os.close(fd)
        sf.write(temporary, output, target_rate_hz, subtype="FLOAT")
        try:
            layout.write_candidate(
                recipe_id, recipe, identity.execution_fingerprint(), Path(temporary), artifact
            )
        except ImmutableWriteError:
            cached = True

    record = {
        "recipe_id": recipe_id,
        "operation": "resample",
        "parents": [parent_recipe_id],
        "artifact_pcm_sha256": artifact,
        "sample_rate_hz": target_rate_hz,
        "channels": channel_layout,
        "frames": expected_frames,
        "sample_format": "float32",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cached": cached,
    }
    with Manifest(layout.manifest_sqlite) as manifest:
        manifest.upsert_candidate(record)
    return record
