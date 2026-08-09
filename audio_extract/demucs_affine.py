"""Exact affine pre/post-processing used by released Demucs inference.

The public Demucs API and the audio-separator Demucs adapter both normalize each
input independently before calling the neural network:

``ref = audio.mean(channels)``
``normalized = (audio - ref.mean()) / (ref.std() + 1e-8)``

Every predicted source is then restored with the same scale and offset.  A
continuation run that calls the pretrained model on raw waveform crops is not
training or evaluating the same function that production inference uses.

This module keeps that affine operation explicit, batched, differentiable, and
non-mutating.  It intentionally does not call ``apply_model``; the caller may
use a direct forward pass for training or overlap-add inference for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemucsAffine:
    """Per-example affine statistics with broadcast-ready shape ``(B, 1)``."""

    mean: object
    scale: object


def normalize_demucs_batch(audio, *, epsilon: float = 1e-8):
    """Normalize ``(batch, channels, frames)`` exactly like Demucs inference.

    Statistics are computed independently for every batch member.  This is
    crucial when a training batch concatenates a mixture, an accompaniment-only
    control, and a vocal-only control: each one must use its own production
    normalization.
    """
    if getattr(audio, "ndim", None) != 3:
        raise ValueError(f"expected (batch, channels, frames), got {getattr(audio, 'shape', None)}")
    if audio.shape[1] < 1 or audio.shape[2] < 2:
        raise ValueError(f"invalid Demucs input shape: {tuple(audio.shape)}")
    reference = audio.mean(dim=1)
    mean = reference.mean(dim=-1, keepdim=True)
    scale = reference.std(dim=-1, keepdim=True) + float(epsilon)
    normalized = (audio - mean[:, None, :]) / scale[:, None, :]
    return normalized, DemucsAffine(mean=mean, scale=scale)


def restore_demucs_sources(estimates, affine: DemucsAffine):
    """Restore normalized ``(batch, sources, channels, frames)`` predictions."""
    if getattr(estimates, "ndim", None) != 4:
        raise ValueError(
            f"expected (batch, sources, channels, frames), got {getattr(estimates, 'shape', None)}"
        )
    if estimates.shape[0] != affine.mean.shape[0]:
        raise ValueError(
            f"batch mismatch: estimates={estimates.shape[0]} affine={affine.mean.shape[0]}"
        )
    return (
        estimates * affine.scale[:, None, None, :]
        + affine.mean[:, None, None, :]
    )


def forward_demucs_production_affine(model, audio):
    """Differentiable direct-forward path matching production affine handling."""
    normalized, affine = normalize_demucs_batch(audio)
    return restore_demucs_sources(model(normalized), affine)
