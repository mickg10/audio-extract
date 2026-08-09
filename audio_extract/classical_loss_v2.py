"""Stable residual loss for classical/operatic featured-soloist removal.

This is a new, versioned contract.  It deliberately does not mutate the v1
``classical_separation_loss`` semantics, so previous experiments remain
reproducible.

For the supported construction,

``A_hat = M - V_hat`` and ``M = A + V``

there is only one independent source error.  Scoring both ``A_hat - A`` and
``V_hat - V`` duplicates the same residual and, when the two copies use
source-normalized spectral denominators, can make near-silent vocal targets
arbitrarily dominant.  This module scores the residual once, normalizes spectral
error to the mixture/reference scale, and requires explicit A-only and V-only
control forwards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalResidualLossConfig:
    residual_waveform: float = 1.0
    residual_complex_stft: float = 0.5
    no_vocal_false_positive: float = 1.0
    vocal_only_false_negative: float = 0.5
    source_coordinate: float = 0.1
    stereo_accompaniment: float = 0.05
    event_weighted_residual: float = 0.25
    stft_ffts: tuple[int, ...] = (512, 1024, 2048)
    waveform_reference_floor: float = 1e-3
    stft_reference_floor: float = 1e-3
    source_coord_ridge: float = 1e-6
    source_coord_max_condition: float = 1e6
    target_consistency_tolerance: float = 1e-5
    residual_consistency_tolerance: float = 1e-6
    eps: float = 1e-8


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training extra owns torch
        raise RuntimeError("classical residual training requires PyTorch") from exc
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


def _require_identity(name: str, lhs: object, rhs: object, tolerance: float) -> None:
    if tolerance < 0:
        raise ValueError(f"{name} tolerance must be non-negative")
    error = float((lhs.detach() - rhs.detach()).abs().max().cpu())
    scale = max(1.0, float(rhs.detach().abs().max().cpu()))
    limit = float(tolerance) * scale
    if error > limit:
        raise ValueError(f"{name} violated: max_abs={error} > tolerance={limit}")


def _normalized_waveform_l1(error: object, reference: object, floor: float):
    if floor <= 0:
        raise ValueError("waveform_reference_floor must be positive")
    numerator = error.abs().mean(dim=(-2, -1))
    denominator = reference.abs().mean(dim=(-2, -1)).clamp_min(float(floor))
    return (numerator / denominator).mean()


def _complex_stft_log_ratio(error: object, reference: object,
                            ffts: tuple[int, ...], floor: float):
    """Mixture/reference-normalized complex MR-STFT error.

    ``normalized=True`` keeps the absolute floor comparable across FFT sizes.
    ``log1p`` prevents one low-energy crop from dominating the optimizer state
    while preserving zero error and monotonicity.
    """
    torch = _torch()
    if floor <= 0:
        raise ValueError("stft_reference_floor must be positive")
    frames = error.shape[-1]
    usable = tuple(int(n) for n in ffts if 2 <= int(n) <= frames)
    if not usable:
        raise ValueError(f"no STFT size fits {frames} frames")
    flat_error = error.reshape(-1, frames)
    flat_reference = reference.reshape(-1, frames)
    total = error.new_zeros(())
    for n_fft in usable:
        window = torch.hann_window(n_fft, device=error.device, dtype=error.dtype)
        e = torch.stft(
            flat_error, n_fft=n_fft, hop_length=n_fft // 4, window=window,
            return_complex=True, normalized=True,
        )
        r = torch.stft(
            flat_reference, n_fft=n_fft, hop_length=n_fft // 4, window=window,
            return_complex=True, normalized=True,
        )
        numerator = e.abs().mean(dim=(-2, -1))
        denominator = r.abs().mean(dim=(-2, -1)).clamp_min(float(floor))
        total = total + torch.log1p(numerator / denominator).mean()
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
                            cfg: ClassicalResidualLossConfig):
    """Differentiable A/V ridge fit with an explicit identifiability mask."""
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
    # Conditioning is a mask/diagnostic, not a differentiable objective.
    condition = torch.linalg.cond(raw_gram.detach())
    valid = (
        (aa.detach() > cfg.eps)
        & (vv.detach() > cfg.eps)
        & torch.isfinite(condition)
        & (condition <= cfg.source_coord_max_condition)
    )
    per_item = (alpha - 1).abs() + beta.abs() + residue_ratio
    if bool(valid.any()):
        return per_item[valid].mean(), valid.sum()
    return estimate.sum() * 0.0, valid.sum()


def classical_residual_loss_v2(
    accompaniment_estimate: object,
    vocal_estimate: object,
    mixture: object,
    accompaniment_target: object,
    vocal_target: object,
    *,
    no_vocal_vocal_estimate: object,
    vocal_only_vocal_estimate: object,
    event_weights: object | None = None,
    config: ClassicalResidualLossConfig | None = None,
):
    """Return ``(total_loss, components)`` for a mixture-residual separator.

    The function requires the vocal outputs of real A-only and V-only forwards.
    It refuses source-grid, target-identity, and residual-construction mismatch.
    Missing controls therefore cannot be silently treated as clean zeros.
    """
    torch = _torch()
    cfg = config or ClassicalResidualLossConfig()
    named = {
        "accompaniment_estimate": accompaniment_estimate,
        "vocal_estimate": vocal_estimate,
        "mixture": mixture,
        "accompaniment_target": accompaniment_target,
        "vocal_target": vocal_target,
        "no_vocal_vocal_estimate": no_vocal_vocal_estimate,
        "vocal_only_vocal_estimate": vocal_only_vocal_estimate,
    }
    _same_shape(named)
    for name, value in named.items():
        _finite(name, value)

    _require_identity(
        "exact target identity M=A+V", mixture,
        accompaniment_target + vocal_target, cfg.target_consistency_tolerance,
    )
    _require_identity(
        "mixture-residual identity A_hat+V_hat=M",
        accompaniment_estimate + vocal_estimate, mixture,
        cfg.residual_consistency_tolerance,
    )

    # Under the two identities above, this is the one independent source error.
    source_error = vocal_estimate - vocal_target
    components: dict[str, object] = {
        "residual_waveform": _normalized_waveform_l1(
            source_error, mixture, cfg.waveform_reference_floor
        ),
        "residual_complex_stft": _complex_stft_log_ratio(
            source_error, mixture, cfg.stft_ffts, cfg.stft_reference_floor
        ),
        "no_vocal_false_positive": _normalized_waveform_l1(
            no_vocal_vocal_estimate, accompaniment_target,
            cfg.waveform_reference_floor,
        ),
        "vocal_only_false_negative": _normalized_waveform_l1(
            vocal_only_vocal_estimate - vocal_target, vocal_target,
            cfg.waveform_reference_floor,
        ),
        "stereo_accompaniment": _mid_side_l1(
            accompaniment_estimate, accompaniment_target
        ),
    }
    source_coordinate, valid_count = _source_coordinate_loss(
        accompaniment_estimate, accompaniment_target, vocal_target, cfg
    )
    components["source_coordinate"] = source_coordinate
    components["source_coordinate_valid_items"] = valid_count

    if event_weights is not None:
        try:
            weights = torch.broadcast_to(event_weights, source_error.shape)
        except RuntimeError as exc:
            raise ValueError(
                "event_weights must broadcast to (batch, channels, frames)"
            ) from exc
        _finite("event_weights", weights)
        if bool((weights < 0).any()):
            raise ValueError("event_weights must be non-negative")
        weight_sum = weights.sum(dim=(-2, -1)).clamp_min(cfg.eps)
        numerator = (source_error.abs() * weights).sum(dim=(-2, -1)) / weight_sum
        reference = (mixture.abs() * weights).sum(dim=(-2, -1)) / weight_sum
        components["event_weighted_residual"] = (
            numerator / reference.clamp_min(cfg.waveform_reference_floor)
        ).mean()
    else:
        components["event_weighted_residual"] = None

    weighted = {
        "residual_waveform": cfg.residual_waveform,
        "residual_complex_stft": cfg.residual_complex_stft,
        "no_vocal_false_positive": cfg.no_vocal_false_positive,
        "vocal_only_false_negative": cfg.vocal_only_false_negative,
        "source_coordinate": cfg.source_coordinate,
        "stereo_accompaniment": cfg.stereo_accompaniment,
        "event_weighted_residual": cfg.event_weighted_residual,
    }
    total = accompaniment_estimate.new_zeros(())
    for name, weight in weighted.items():
        value = components[name]
        if value is not None:
            total = total + float(weight) * value
    if not bool(torch.isfinite(total)):
        raise ValueError("classical residual loss v2 is non-finite")
    return total, components
