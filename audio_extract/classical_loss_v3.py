"""Mixed-precision-safe residual loss for operatic featured-soloist removal.

This version-separated draft keeps the production construction

    A_hat = M - V_hat

and scores the one independent source error once. All numerically sensitive
analysis is performed under an explicit identity-bearing precision policy:

    float64 inputs -> float64 analysis
    float16/bfloat16/float32 inputs -> float32 analysis

The differentiable cast keeps gradients connected to the original model dtype.
No ambient autocast behavior is part of the loss contract.
"""
from __future__ import annotations

from dataclasses import dataclass

_ANALYSIS_POLICY = "float64_if_any_input_float64_else_float32"


@dataclass(frozen=True)
class ClassicalResidualLossV3Config:
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
    analysis_dtype_policy: str = _ANALYSIS_POLICY
    eps: float = 1e-8

    def validate(self) -> None:
        for name in (
            "residual_waveform", "residual_complex_stft",
            "no_vocal_false_positive", "vocal_only_false_negative",
            "source_coordinate", "stereo_accompaniment",
            "event_weighted_residual", "source_coord_ridge",
            "target_consistency_tolerance", "residual_consistency_tolerance",
            "identity_roundoff_ulps",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "waveform_reference_floor", "stft_reference_floor",
            "source_coord_max_condition", "eps",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.source_coord_max_condition <= 1.0:
            raise ValueError("source_coord_max_condition must be greater than one")
        if self.analysis_dtype_policy != _ANALYSIS_POLICY:
            raise ValueError(
                f"unsupported analysis_dtype_policy {self.analysis_dtype_policy!r}"
            )


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("classical residual training requires PyTorch") from exc
    return torch


def _same_shape(named: dict[str, object]) -> tuple[int, int, int]:
    shapes = {name: tuple(value.shape) for name, value in named.items()}
    first = next(iter(shapes.values()))
    if len(first) != 3:
        raise ValueError(
            f"audio tensors must be (batch, channels, frames), got {shapes}"
        )
    if any(shape != first for shape in shapes.values()):
        raise ValueError(
            f"audio tensors must share one exact sample grid, got {shapes}"
        )
    if min(first) <= 0:
        raise ValueError(f"audio tensors must be non-empty, got {shapes}")
    return first


def _finite(name: str, tensor: object) -> None:
    torch = _torch()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite samples")


def _analysis_dtype(*tensors: object):
    torch = _torch()
    for tensor in tensors:
        if not tensor.is_floating_point():
            raise ValueError("audio loss tensors must be floating point")
    return (
        torch.float64
        if any(tensor.dtype == torch.float64 for tensor in tensors)
        else torch.float32
    )


def _largest_epsilon(*tensors: object) -> float:
    torch = _torch()
    return max(float(torch.finfo(tensor.dtype).eps) for tensor in tensors)


def _identity_limit(*, reference, operands, relative_tolerance, roundoff_ulps):
    torch = _torch()
    if relative_tolerance < 0.0 or roundoff_ulps < 0.0:
        raise ValueError("identity tolerances must be non-negative")
    reference_scale = max(1.0, float(reference.detach().abs().max().cpu()))
    arithmetic = None
    for operand in operands:
        value = operand.detach().abs().to(dtype=torch.float64)
        arithmetic = value if arithmetic is None else arithmetic + value
    arithmetic_scale = max(1.0, float(arithmetic.max().cpu()))
    roundoff = (
        float(roundoff_ulps) * _largest_epsilon(reference, *operands)
        * arithmetic_scale
    )
    return max(float(relative_tolerance) * reference_scale, roundoff)


def _require_sum_identity(
    name, total, left, right, relative_tolerance, roundoff_ulps,
):
    torch = _torch()
    expected = (
        left.detach().to(dtype=torch.float64)
        + right.detach().to(dtype=torch.float64)
    )
    observed = total.detach().to(dtype=torch.float64)
    error = float((observed - expected).abs().max().cpu())
    limit = _identity_limit(
        reference=total, operands=(left, right),
        relative_tolerance=relative_tolerance, roundoff_ulps=roundoff_ulps,
    )
    if error > limit:
        raise ValueError(
            f"{name} violated: max_abs={error} > tolerance={limit}"
        )


def _require_residual_construction(
    accompaniment, vocal, mixture, relative_tolerance, roundoff_ulps,
):
    torch = _torch()
    expected = (
        mixture.detach().to(dtype=torch.float64)
        - vocal.detach().to(dtype=torch.float64)
    )
    observed = accompaniment.detach().to(dtype=torch.float64)
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


def _normalized_waveform_l1(error, reference, floor, analysis_dtype):
    if floor <= 0.0:
        raise ValueError("waveform reference floor must be positive")
    err = error.to(dtype=analysis_dtype)
    ref = reference.to(dtype=analysis_dtype)
    numerator = err.abs().mean(dim=(-2, -1))
    denominator = ref.abs().mean(dim=(-2, -1)).clamp_min(float(floor))
    return (numerator / denominator).mean()


def _complex_stft_log_ratio(error, reference, ffts, floor, analysis_dtype):
    torch = _torch()
    if floor <= 0.0:
        raise ValueError("stft reference floor must be positive")
    frames = error.shape[-1]
    usable = tuple(int(value) for value in ffts if 4 <= int(value) <= frames)
    if not usable:
        raise ValueError(f"no STFT size with positive hop fits {frames} frames")
    flat_error = error.to(dtype=analysis_dtype).reshape(-1, frames)
    flat_reference = reference.to(dtype=analysis_dtype).reshape(-1, frames)
    total = flat_error.new_zeros(())
    for n_fft in usable:
        hop_length = n_fft // 4
        window = torch.hann_window(
            n_fft, device=error.device, dtype=analysis_dtype
        )
        error_stft = torch.stft(
            flat_error, n_fft=n_fft, hop_length=hop_length, window=window,
            center=True, pad_mode="constant", return_complex=True,
            normalized=True,
        )
        reference_stft = torch.stft(
            flat_reference, n_fft=n_fft, hop_length=hop_length, window=window,
            center=True, pad_mode="constant", return_complex=True,
            normalized=True,
        )
        numerator = error_stft.abs().mean(dim=(-2, -1))
        denominator = (
            reference_stft.abs().mean(dim=(-2, -1)).clamp_min(float(floor))
        )
        total = total + torch.log1p(numerator / denominator).mean()
    return total / len(usable)


def _normalized_mid_side_l1(prediction, target, floor, analysis_dtype):
    if prediction.shape[1] != 2:
        return prediction.to(dtype=analysis_dtype).sum() * 0.0
    prediction = prediction.to(dtype=analysis_dtype)
    target = target.to(dtype=analysis_dtype)
    p_mid = (prediction[:, 0] + prediction[:, 1]) * 0.5
    p_side = (prediction[:, 0] - prediction[:, 1]) * 0.5
    t_mid = (target[:, 0] + target[:, 1]) * 0.5
    t_side = (target[:, 0] - target[:, 1]) * 0.5
    numerator = (
        (p_mid - t_mid).abs().mean(dim=-1)
        + (p_side - t_side).abs().mean(dim=-1)
    )
    denominator = (
        t_mid.abs().mean(dim=-1) + t_side.abs().mean(dim=-1)
    ).clamp_min(float(floor))
    return (numerator / denominator).mean()


def _source_coordinate_loss(estimate, accompaniment, vocals, cfg, analysis_dtype):
    """Differentiable real A/V ridge fit with an analytic 2x2 solve."""
    torch = _torch()
    batch = estimate.shape[0]
    y = estimate.to(dtype=analysis_dtype).reshape(batch, -1)
    a = accompaniment.to(dtype=analysis_dtype).reshape(batch, -1)
    v = vocals.to(dtype=analysis_dtype).reshape(batch, -1)
    aa = (a * a).sum(-1)
    vv = (v * v).sum(-1)
    av = (a * v).sum(-1)
    ay = (a * y).sum(-1)
    vy = (v * y).sum(-1)
    trace = aa + vv
    ridge = float(cfg.source_coord_ridge) * (
        trace * 0.5
    ).clamp_min(float(cfg.eps))
    gaa = aa + ridge
    gvv = vv + ridge
    determinant = (gaa * gvv - av * av).clamp_min(float(cfg.eps))
    alpha = (gvv * ay - av * vy) / determinant
    beta = (gaa * vy - av * ay) / determinant
    residue = y - alpha[:, None] * a - beta[:, None] * v
    residue_ratio = torch.sqrt(
        (residue.square().sum(-1) + float(cfg.eps))
        / (aa + float(cfg.eps))
    )
    retained_voice_ratio = beta.abs() * torch.sqrt(
        (vv + float(cfg.eps)) / (aa + float(cfg.eps))
    )
    discriminant = torch.sqrt(
        ((aa.detach() - vv.detach()).square() + 4.0 * av.detach().square())
        .clamp_min(0.0)
    )
    largest = 0.5 * (trace.detach() + discriminant)
    smallest = 0.5 * (trace.detach() - discriminant)
    condition = largest / smallest.clamp_min(float(cfg.eps))
    valid = (
        (aa.detach() > float(cfg.eps))
        & (vv.detach() > float(cfg.eps))
        & torch.isfinite(condition)
        & (condition <= float(cfg.source_coord_max_condition))
    )
    per_item = (alpha - 1.0).abs() + retained_voice_ratio + residue_ratio
    if bool(valid.any()):
        return per_item[valid].mean(), valid.sum(), condition
    return estimate.to(dtype=analysis_dtype).sum() * 0.0, valid.sum(), condition


def classical_residual_loss_v3(
    accompaniment_estimate, vocal_estimate, mixture,
    accompaniment_target, vocal_target, *,
    no_vocal_vocal_estimate, vocal_only_vocal_estimate,
    event_weights=None, config=None,
):
    """Return ``(total, components)`` under the explicit analysis-dtype policy."""
    torch = _torch()
    cfg = config or ClassicalResidualLossV3Config()
    cfg.validate()
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
    analysis_dtype = _analysis_dtype(*named.values())
    source_error = vocal_estimate - vocal_target
    components = {
        "analysis_dtype": str(analysis_dtype),
        "analysis_dtype_policy": cfg.analysis_dtype_policy,
        "residual_waveform": _normalized_waveform_l1(
            source_error, mixture, cfg.waveform_reference_floor, analysis_dtype
        ),
        "residual_complex_stft": _complex_stft_log_ratio(
            source_error, mixture, cfg.stft_ffts,
            cfg.stft_reference_floor, analysis_dtype,
        ),
        "no_vocal_false_positive": _normalized_waveform_l1(
            no_vocal_vocal_estimate, accompaniment_target,
            cfg.waveform_reference_floor, analysis_dtype,
        ),
        "vocal_only_false_negative": _normalized_waveform_l1(
            vocal_only_vocal_estimate - vocal_target, vocal_target,
            cfg.waveform_reference_floor, analysis_dtype,
        ),
        "stereo_accompaniment": _normalized_mid_side_l1(
            accompaniment_estimate, accompaniment_target,
            cfg.waveform_reference_floor, analysis_dtype,
        ),
    }
    coordinate, valid_count, condition = _source_coordinate_loss(
        accompaniment_estimate, accompaniment_target, vocal_target,
        cfg, analysis_dtype,
    )
    components["source_coordinate"] = coordinate
    components["source_coordinate_valid_items"] = valid_count
    components["source_coordinate_condition"] = condition
    if event_weights is not None:
        weights = event_weights.to(dtype=analysis_dtype, device=source_error.device)
        try:
            weights = torch.broadcast_to(weights, source_error.shape)
        except RuntimeError as exc:
            raise ValueError(
                "event_weights must broadcast to (batch, channels, frames)"
            ) from exc
        _finite("event_weights", weights)
        if bool((weights < 0.0).any()):
            raise ValueError("event_weights must be non-negative")
        error = source_error.to(dtype=analysis_dtype)
        reference = mixture.to(dtype=analysis_dtype)
        weight_sum = weights.sum(dim=(-2, -1)).clamp_min(float(cfg.eps))
        numerator = (error.abs() * weights).sum(dim=(-2, -1)) / weight_sum
        denominator = (
            (reference.abs() * weights).sum(dim=(-2, -1)) / weight_sum
        ).clamp_min(float(cfg.waveform_reference_floor))
        components["event_weighted_residual"] = (numerator / denominator).mean()
    else:
        components["event_weighted_residual"] = None
    weights_by_name = {
        "residual_waveform": cfg.residual_waveform,
        "residual_complex_stft": cfg.residual_complex_stft,
        "no_vocal_false_positive": cfg.no_vocal_false_positive,
        "vocal_only_false_negative": cfg.vocal_only_false_negative,
        "source_coordinate": cfg.source_coordinate,
        "stereo_accompaniment": cfg.stereo_accompaniment,
        "event_weighted_residual": cfg.event_weighted_residual,
    }
    total = mixture.to(dtype=analysis_dtype).sum() * 0.0
    for name, weight in weights_by_name.items():
        value = components[name]
        if value is not None:
            total = total + float(weight) * value
    if not bool(torch.isfinite(total)):
        raise ValueError("classical residual loss v3 is non-finite")
    return total, components
