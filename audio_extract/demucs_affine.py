"""Exact affine pre/post-processing used by released Demucs inference/training.

Released Demucs inference normalizes a complete input track before overlap-add:

``ref = audio.mean(channels)``
``normalized = (audio - ref.mean()) / (ref.std() + 1e-8)``

Every predicted source is restored with the same scale and offset.  The official
``Wavset`` training loader similarly stores statistics from the *entire mixture
track* and applies those fixed statistics to every crop and source target.

Consequently:

* full-work inference may compute the affine from the full input directly;
* crop training must reuse full-track statistics, never crop-local statistics;
* mixture, accompaniment-only, and vocal-only control inputs each need their own
  full-track affine because production would normalize each input separately.

This module keeps that operation explicit, batched, differentiable, and
non-mutating.  It intentionally does not call ``apply_model``; callers may use a
direct forward pass for training or overlap-add inference for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemucsAffine:
    """Per-example statistics with shape ``(batch, 1)``."""

    mean: object
    scale: object


def demucs_affine_from_audio(audio, *, epsilon: float = 1e-8) -> DemucsAffine:
    """Compute production statistics from complete ``(B,C,T)`` inputs.

    For training crops, call this once on each complete M/A/V track (or load the
    same persisted values) and pass the resulting affine to
    :func:`normalize_demucs_batch` for every crop.
    """
    if getattr(audio, "ndim", None) != 3:
        raise ValueError(f"expected (batch, channels, frames), got {getattr(audio, 'shape', None)}")
    if audio.shape[1] < 1 or audio.shape[2] < 2:
        raise ValueError(f"invalid Demucs input shape: {tuple(audio.shape)}")
    reference = audio.mean(dim=1)
    mean = reference.mean(dim=-1, keepdim=True)
    scale = reference.std(dim=-1, keepdim=True) + float(epsilon)
    return DemucsAffine(mean=mean, scale=scale)


def normalize_demucs_batch(audio, affine: DemucsAffine | None = None):
    """Normalize ``(B,C,T)`` with explicit per-example full-track statistics.

    Passing ``affine=None`` is correct only when ``audio`` itself is the complete
    input used by production inference.  Crop training must pass a precomputed
    full-track affine.
    """
    if getattr(audio, "ndim", None) != 3:
        raise ValueError(f"expected (batch, channels, frames), got {getattr(audio, 'shape', None)}")
    if affine is None:
        affine = demucs_affine_from_audio(audio)
    if audio.shape[0] != affine.mean.shape[0] or affine.mean.shape != affine.scale.shape:
        raise ValueError(
            f"affine shape mismatch: audio={tuple(audio.shape)} "
            f"mean={tuple(affine.mean.shape)} scale={tuple(affine.scale.shape)}"
        )
    return (audio - affine.mean[:, None, :]) / affine.scale[:, None, :], affine


def restore_demucs_sources(estimates, affine: DemucsAffine):
    """Restore normalized ``(B,S,C,T)`` predictions to the original scale."""
    if getattr(estimates, "ndim", None) != 4:
        raise ValueError(
            f"expected (batch, sources, channels, frames), got {getattr(estimates, 'shape', None)}"
        )
    if estimates.shape[0] != affine.mean.shape[0]:
        raise ValueError(
            f"batch mismatch: estimates={estimates.shape[0]} affine={affine.mean.shape[0]}"
        )
    return estimates * affine.scale[:, None, None, :] + affine.mean[:, None, None, :]


def forward_demucs_production_affine(model, audio, affine: DemucsAffine | None = None):
    """Differentiable direct-forward path with production-equivalent affine handling."""
    normalized, affine = normalize_demucs_batch(audio, affine)
    return restore_demucs_sources(model(normalized), affine)
