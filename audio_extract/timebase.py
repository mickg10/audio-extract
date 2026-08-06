"""Explicit audio time grids (v2.1 WP0 / §5).

Every measurement depends on knowing the exact sample-rate/channel/frame grid an
array lives on. Subtraction and ensembling are forbidden across differing grids —
:func:`grids_compatible` is the gate the residual/consensus paths must pass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioGrid:
    sample_rate_hz: int
    channels: tuple[str, ...]
    frames: int
    origin_sample: int = 0

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    def duration_s(self) -> float:
        return self.frames / self.sample_rate_hz if self.sample_rate_hz else 0.0


def grids_compatible(a: AudioGrid, b: AudioGrid, *, frame_tolerance: int = 0) -> tuple[bool, str]:
    """Return ``(ok, reason)``. Same sample rate + channel map, and frame counts
    within ``frame_tolerance`` (0 = exact). This must pass before any subtraction
    or ensemble (v2.1 §5.4/§5.6)."""
    if a.sample_rate_hz != b.sample_rate_hz:
        return False, f"sample-rate mismatch: {a.sample_rate_hz} vs {b.sample_rate_hz}"
    if a.channels != b.channels:
        return False, f"channel-map mismatch: {a.channels} vs {b.channels}"
    if abs(a.frames - b.frames) > frame_tolerance:
        return False, f"frame-count mismatch: {a.frames} vs {b.frames} (tol {frame_tolerance})"
    return True, ""


def map_sample(sample: int, grid_from: AudioGrid, grid_to: AudioGrid) -> int:
    """Map a sample index from one grid to another by the sample-rate ratio
    (e.g. a canonical-grid passage coordinate onto a model-rate grid)."""
    rel = sample - grid_from.origin_sample
    scaled = round(rel * grid_to.sample_rate_hz / grid_from.sample_rate_hz)
    return grid_to.origin_sample + scaled
