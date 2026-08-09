"""Differentiable loss bank for classical/operatic vocal removal.

This module deliberately knows nothing about a separator architecture.  Callers
provide an accompaniment estimate and a vocal estimate on the exact target
sample grid.  For a four-source pretrained HTDemucs, accompaniment is the sum
of the drums, bass, and other outputs and vocals is the vocal output.

All inputs are float tensors shaped ``(batch, channels, frames)``.  Shape
mismatches are refused; this code never truncates, pads, aligns, or resamples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalLossConfig:
    waveform_l1: float = 1.0
    complex_stft: float = 0.5
    mixture_consistency: float = 0.1
    no_vocal_false_positive: float = 0.25
    vocal_only_false_negative: float = 0.25
    source_coordinate: float = 0.1
    stereo_coherence: float = 0.05
    event_weighted: float = 0.25
    stft_ffts: tuple[int, ...] = (512, 1024, 2048)
    source_coord_ridge: float = 1e-6
    source_coord_max_condition: float = 1e6
    eps: float = 1e-8


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training extra owns torch
        raise RuntimeError("classical training requires PyTorch") from exc
    return torch


def _same_shape(named: dict[str, object]) -> tuple[int, int, int]:
    shapes = {name: tuple(value.shape) for name, value in named.items()}
    first = next(iter(shapes.values()))
    if len(first) != 3:
        raise ValueError(f"audio tensors must be (batch, channels, frames), got {shapes}")
    if any(shape != first for shape in shapes.values()):
        raise ValueError(f"audio tensors must share one exact sample grid, got {shapes}")
    if min(first) <= 0:
        raise ValueError(f"audio tensors must be non-empty, got {shapes}")
    return first


def _finite(name: str, tensor: object) -> None:
    torch = _torch()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite samples")


def _complex_stft_l1(prediction: object, target: object, ffts: tuple[int, ...]):
    torch = _torch()
    frames = prediction.shape[-1]
    usable = tuple(n for n in ffts if 2 <= n <= frames)
    if not usable:
        raise ValueError(f"no STFT size fits {frames} frames")
    total = prediction.new_zeros(())
    flat_p = prediction.reshape(-1, frames)
    flat_t = target.reshape(-1, frames)
    for n_fft in usable:
        window = torch.hann_window(n_fft, device=prediction.device, dtype=prediction.dtype)
        p = torch.stft(flat_p, n_fft, n_fft // 4, window=window, return_complex=True)
        t = torch.stft(flat_t, n_fft, n_fft // 4, window=window, return_complex=True)
        # Complex distance retains phase.  Normalize per transform so high-energy
        # works do not dominate solely because of their mastering level.
        numerator = torch.view_as_real(p - t).abs().mean()
        denominator = torch.view_as_real(t).abs().mean().clamp_min(1e-8)
        total = total + numerator / denominator
    return total / len(usable)


def _mid_side_l1(prediction: object, target: object):
    if prediction.shape[1] != 2:
        return prediction.new_zeros(())
    p_mid = (prediction[:, 0] + prediction[:, 1]) * 0.5
    p_side = (prediction[:, 0] - prediction[:, 1]) * 0.5
    t_mid = (target[:, 0] + target[:, 1]) * 0.5
    t_side = (target[:, 0] - target[:, 1]) * 0.5
    return (p_mid - t_mid).abs().mean() + (p_side - t_side).abs().mean()


def _source_coordinate_loss(estimate: object, accompaniment: object, vocals: object,
                            cfg: ClassicalLossConfig):
    """Penalize accompaniment transfer, retained voice, and orthogonal residue.

    A differentiable two-source ridge fit is performed independently per batch
    item over all channels and frames.  Silent or ill-conditioned items are
    explicitly masked instead of producing a clean-looking zero label.
    """
    torch = _torch()
    batch = estimate.shape[0]
    y = estimate.reshape(batch, -1)
    a = accompaniment.reshape(batch, -1)
    v = vocals.reshape(batch, -1)
    aa = (a * a).sum(-1)
    vv = (v * v).sum(-1)
    av = (a * v).sum(-1)
    trace = aa + vv
    ridge = cfg.source_coord_ridge * (trace * 0.5).clamp_min(cfg.eps)
    gram = torch.stack(
        (torch.stack((aa + ridge, av), -1), torch.stack((av, vv + ridge), -1)),
        -2,
    )
    rhs = torch.stack(((a * y).sum(-1), (v * y).sum(-1)), -1).unsqueeze(-1)
    coeff = torch.linalg.solve(gram, rhs).squeeze(-1)
    alpha, beta = coeff[:, 0], coeff[:, 1]
    residue = y - alpha[:, None] * a - beta[:, None] * v
    residue_ratio = torch.sqrt((residue.square().sum(-1) + cfg.eps) / (aa + cfg.eps))

    raw_gram = torch.stack(
        (torch.stack((aa, av), -1), torch.stack((av, vv), -1)), -2
    )
    condition = torch.linalg.cond(raw_gram)
    valid = (
        (aa > cfg.eps)
        & (vv > cfg.eps)
        & torch.isfinite(condition)
        & (condition <= cfg.source_coord_max_condition)
    )
    per_item = (alpha - 1).abs() + beta.abs() + residue_ratio
    if bool(valid.any()):
        return per_item[valid].mean(), valid.sum()
    # Preserve a differentiable zero when no tile is identifiable.
    return estimate.sum() * 0.0, valid.sum()


def classical_separation_loss(
    accompaniment_estimate: object,
    vocal_estimate: object,
    mixture: object,
    accompaniment_target: object,
    vocal_target: object,
    *,
    no_vocal_accompaniment_estimate: object | None = None,
    no_vocal_vocal_estimate: object | None = None,
    vocal_only_accompaniment_estimate: object | None = None,
    vocal_only_vocal_estimate: object | None = None,
    event_weights: object | None = None,
    config: ClassicalLossConfig | None = None,
):
    """Return ``(total_loss, component_losses)`` on one exact-grid batch.

    Control estimates are optional, but their configured loss is reported as
    unavailable rather than silently counted as zero.  ``event_weights`` must be
    shaped ``(batch, 1, frames)`` or exactly like the audio and must come from
    exact source activity, never from the learned judge.
    """
    torch = _torch()
    cfg = config or ClassicalLossConfig()
    named = {
        "accompaniment_estimate": accompaniment_estimate,
        "vocal_estimate": vocal_estimate,
        "mixture": mixture,
        "accompaniment_target": accompaniment_target,
        "vocal_target": vocal_target,
    }
    _same_shape(named)
    for name, value in named.items():
        _finite(name, value)

    components: dict[str, object] = {}
    components["waveform_l1"] = (
        (accompaniment_estimate - accompaniment_target).abs().mean()
        + (vocal_estimate - vocal_target).abs().mean()
    )
    components["complex_stft"] = (
        _complex_stft_l1(accompaniment_estimate, accompaniment_target, cfg.stft_ffts)
        + _complex_stft_l1(vocal_estimate, vocal_target, cfg.stft_ffts)
    )
    components["mixture_consistency"] = (
        accompaniment_estimate + vocal_estimate - mixture
    ).abs().mean()
    source_coordinate, valid_count = _source_coordinate_loss(
        accompaniment_estimate, accompaniment_target, vocal_target, cfg
    )
    components["source_coordinate"] = source_coordinate
    components["source_coordinate_valid_items"] = valid_count
    components["stereo_coherence"] = (
        _mid_side_l1(accompaniment_estimate, accompaniment_target)
        + _mid_side_l1(vocal_estimate, vocal_target)
    )

    if (no_vocal_accompaniment_estimate is None) != (no_vocal_vocal_estimate is None):
        raise ValueError("both no-vocal control estimates must be supplied together")
    if no_vocal_accompaniment_estimate is not None:
        _same_shape({
            "no_vocal_accompaniment_estimate": no_vocal_accompaniment_estimate,
            "no_vocal_vocal_estimate": no_vocal_vocal_estimate,
            "accompaniment_target": accompaniment_target,
        })
        components["no_vocal_false_positive"] = (
            no_vocal_vocal_estimate.abs().mean()
            + (no_vocal_accompaniment_estimate - accompaniment_target).abs().mean()
        )
    else:
        components["no_vocal_false_positive"] = None

    if (vocal_only_accompaniment_estimate is None) != (vocal_only_vocal_estimate is None):
        raise ValueError("both vocal-only control estimates must be supplied together")
    if vocal_only_accompaniment_estimate is not None:
        _same_shape({
            "vocal_only_accompaniment_estimate": vocal_only_accompaniment_estimate,
            "vocal_only_vocal_estimate": vocal_only_vocal_estimate,
            "vocal_target": vocal_target,
        })
        components["vocal_only_false_negative"] = (
            vocal_only_accompaniment_estimate.abs().mean()
            + (vocal_only_vocal_estimate - vocal_target).abs().mean()
        )
    else:
        components["vocal_only_false_negative"] = None

    if event_weights is not None:
        try:
            weights = torch.broadcast_to(event_weights, accompaniment_target.shape)
        except RuntimeError as exc:
            raise ValueError(
                "event_weights must broadcast to (batch, channels, frames)"
            ) from exc
        _finite("event_weights", weights)
        if bool((weights < 0).any()):
            raise ValueError("event_weights must be non-negative")
        denom = weights.sum().clamp_min(cfg.eps)
        components["event_weighted"] = (
            ((accompaniment_estimate - accompaniment_target).abs() * weights).sum()
            + ((vocal_estimate - vocal_target).abs() * weights).sum()
        ) / denom
    else:
        components["event_weighted"] = None

    weighted = {
        "waveform_l1": cfg.waveform_l1,
        "complex_stft": cfg.complex_stft,
        "mixture_consistency": cfg.mixture_consistency,
        "no_vocal_false_positive": cfg.no_vocal_false_positive,
        "vocal_only_false_negative": cfg.vocal_only_false_negative,
        "source_coordinate": cfg.source_coordinate,
        "stereo_coherence": cfg.stereo_coherence,
        "event_weighted": cfg.event_weighted,
    }
    total = accompaniment_estimate.new_zeros(())
    for name, weight in weighted.items():
        value = components[name]
        if value is not None:
            total = total + float(weight) * value
    if not bool(torch.isfinite(total)):
        raise ValueError("classical separation loss is non-finite")
    return total, components
