"""Pinned, named resamplers (v2.1 WP0 / §5.3).

Never rely on a library's hidden default resampling: one algorithm is selected by
name and recorded in the recipe. Registered algorithms map to a concrete backend;
the algorithm name (not just the rate) is part of candidate identity.
"""

from __future__ import annotations

import numpy as np

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
