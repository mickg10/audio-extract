"""Stable residual loss for classical/operatic featured-soloist removal.

This is a new, versioned contract. It deliberately does not mutate the v1
``classical_separation_loss`` semantics, so previous experiments remain
reproducible.

For the supported construction,

``A_hat = M - V_hat`` and ``M = A + V``

there is only one independent source error. Scoring both ``A_hat - A`` and
``V_hat - V`` duplicates the same residual and, when the two copies use
source-normalized spectral denominators, can make near-silent vocal targets
arbitrarily dominant. This module scores the residual once, normalizes spectral
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
    identity_roundoff_ulps: float = 8.0
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


def _largest_epsilon(*tensors: object) -> float:
    """Largest machine epsilon among floating inputs.

    Identity checks are diagnostics, not loss terms. They must tolerate the
    arithmetic precision of the declared construction without allowing a fixed,
    dtype-blind threshold to abort mixed-precision training.
    """
    torch = _torch()
    values = []
    for tensor in tensors:
        if not tensor.is_floating_point():
            raise ValueError("audio identity tensors must be floating point")
        values.append(float(torch.finfo(tensor.dtype).eps))
    return max(values)


def _identity_limit(*, reference: object, operands: tuple[object, ...],
                    relative_tolerance: float, roundoff_ulps: float) -> float:
    if relative_tolerance < 0 or roundoff_ulps < 0:
        raise ValueError("identity tolerances must be non-negative")
    reference_scale = max(1.0, float(reference.detach().abs().max().cpu()))
    arithmetic = None
    for operand in operands:
        value = operand.detach().abs().to(dtype=_torch().float64)
        arithmetic = value if arithmetic is None else arithmetic + value
    arithmetic_scale = max(1.0, float(arithmetic.max().cpu()))
    roundoff = roundoff_ulps * _largest_epsilon(reference, *operands) * arithmetic_scale
    return max(float(relative_tolerance) * reference_scale, roundoff)


def _require_sum_identity(name: str, total: object, left: object, right: object,
                          relative_tolerance: float, roundoff_ulps: float) -> None:
    """Require ``total == left + right`` with a dtype-aware arithmetic bound."""
    expected = left.detach().to(dtype=_torch().float64) + right.detach().to(
        dtype=_torch().float64
    )
    observed = total.detach().to(dtype=_torch().float64)
    error = float((observed - expected).abs().max().cpu())
    limit = _identity_limit(
        reference=total, operands=(left, right),
        relative_tolerance=relative_tolerance, roundoff_ulps=roundoff_ulps,
    )
    if error > limit:
        raise ValueError(f"{name} violated: max_abs={error} > tolerance={limit}")


def _require_residual_construction(accompaniment: object, vocal: object,
                                   mixture: object, relative_tolerance: float,
                                   roundoff_ulps: float) -> None:
    """Require the supported construction directly: ``A_hat = M - V_hat``.

    Checking ``A_hat + V_hat == M`` after another rounded addition can reject a
    correct FP16/BF16 construction. Comparing to a high-precision subtraction of
    the already represented operands isolates the construction and derives the
    tolerance from their dtype and magnitudes.
    """
    expected = mixture.detach().to(dtype=_torch().float64) - vocal.detach().to(
        dtype=_torch().float64
    )
    observed = accompaniment.detach().to(dtype=_torch().float64)
    error = float((observed - expected).abs().max().cpu())
    limit = _identity_limit(
        reference=mixture, operands=(mixture, vocal),
        relative_tolerance=relative_tolerance, roundoff_ulps=roundoff_ulps,
    )
    if error > limit:
        raise ValueError(
            "mixture-residual identity A_hat=M-V_hat violated: "
            f"max_abs={error} > tolerance={limit}"
        )


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
    usable = tuple(int(n) for n in ffts if 4 <= int(n) <= frames)
    if not usable:
        raise ValueError(f"no STFT size with positive hop fits {frames} frames")
    flat_error = error.reshape(-1, frames)
    flat_reference = reference.reshape(-1, frames)
    total = error.new_zeros(())
    for n_fft in usable:
        hop_length = n_fft // 4
        window = torch.hann_window(n_fft, device=error.device, dtype=error.dtype)
        e = torch.stft(
            flat_error, n_fft=n_fft, hop_length=hop_length, window=window,
            return_complex=True, normalized=True,
        )
        r = torch.stft(
            flat_reference, n_fft=n_fft, hop_length=hop_length, window=window,
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

    _require_sum_identity(
        "exact target identity M=A+V", mixture,
        accompaniment_target, vocal_target,
        cfg.target_consistency_tolerance, cfg.identity_roundoff_ulps,
    )
    _require_residual_construction(
        accompaniment_estimate, vocal_estimate, mixture,
        cfg.residual_consistency_tolerance, cfg.identity_roundoff_ulps,
    )

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
