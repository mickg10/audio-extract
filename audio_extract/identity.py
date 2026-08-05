"""The three v2 identities (docs/v2 §1.1).

* ``recipe_id`` — canonical description of the requested computation (cache key,
  DAG parentage).
* ``artifact_pcm_sha256`` — hash of the *decoded* output samples, independent of
  container metadata.
* ``execution_fingerprint`` — runtime facts that may explain nondeterminism.

Keeping them separate stops one string from being asked to mean both "same
recipe" and "same bytes".
"""

from __future__ import annotations

import hashlib
import platform
import struct
import sys
from typing import Any

from . import canon, recipe as recipe_mod

_RECIPE_DOMAIN = b"audio-extract-recipe-v2\x00"


def recipe_id(recipe_obj: dict[str, Any]) -> str:
    """``sha256:<hex>`` over the domain-separated canonical recipe.

    The recipe is normalized first, so an omitted default and an explicitly
    supplied default produce the same id.
    """
    normalized = recipe_mod.normalize_recipe(recipe_obj)
    digest = hashlib.sha256(_RECIPE_DOMAIN + canon.canonicalize(normalized)).hexdigest()
    return f"sha256:{digest}"


def channel_layout_descriptor(channel_layout: list[str]) -> bytes:
    """Deterministic byte descriptor of a channel layout (e.g. ``["FL","FR"]``)."""
    return canon.canonicalize(list(channel_layout))


def artifact_pcm_sha256(
    samples,
    sample_rate: int,
    channel_layout: list[str],
    frame_count: int | None = None,
) -> str:
    """Hash canonical decoded PCM: little-endian float32 interleaved samples,
    then ``sample_rate`` (u32 LE), the channel-layout descriptor, and
    ``frame_count`` (u64 LE).

    ``samples`` is a numpy array shaped ``(frames, channels)`` (interleaved on
    ``tobytes``) or ``(frames,)`` for mono. Imported lazily so the rest of the
    package has no hard numpy dependency.
    """
    import numpy as np

    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"samples must be 1-D or 2-D, got shape {arr.shape}")
    frames = arr.shape[0]
    if frame_count is not None and frame_count != frames:
        raise ValueError(f"frame_count {frame_count} != actual {frames}")
    if arr.shape[1] != len(channel_layout):
        raise ValueError(
            f"channel count {arr.shape[1]} != layout length {len(channel_layout)}"
        )

    pcm = np.ascontiguousarray(arr, dtype="<f4").tobytes()  # interleaved, row-major
    h = hashlib.sha256()
    h.update(pcm)
    h.update(struct.pack("<I", int(sample_rate)))
    h.update(channel_layout_descriptor(channel_layout))
    h.update(struct.pack("<Q", int(frames)))
    return f"sha256:{h.hexdigest()}"


def blob_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def execution_fingerprint() -> dict[str, Any]:
    """Best-effort snapshot of the runtime. Missing optional libs are recorded
    as ``None`` rather than raising."""

    def _ver(mod_name: str) -> str | None:
        try:
            mod = __import__(mod_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None

    torch_ver = _ver("torch")
    cuda = None
    try:  # torch may be absent on the Mac dev box
        import torch  # type: ignore

        cuda = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        cuda = None

    fp: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": _ver("numpy"),
        "torch": torch_ver,
        "cuda": cuda,
        "librosa": _ver("librosa"),
        "soundfile": _ver("soundfile"),
        "audio_separator": _ver("audio_separator"),
    }
    fp["fingerprint_sha256"] = "sha256:" + hashlib.sha256(canon.canonicalize(fp)).hexdigest()
    return fp
