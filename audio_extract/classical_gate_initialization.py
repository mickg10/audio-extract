"""Deterministic trainable initialization for the bounded classical gate.

`SmoothResidualGate` uses a separate scalar amplitude to guarantee that decoded
step-zero audio is exactly the conservative parent. The spatial network itself
therefore need not be all-zero; all-zero convolutions permanently block spatial
gradients. This module initializes a trainable spatial path while resetting the
route amplitude to exact zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GateInitializationReport:
    seed: int
    final_weight_scale: float
    convolution_count: int
    nonzero_parameters: int
    total_parameters: int
    effective_amplitude: float


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - owned by the train extra
        raise RuntimeError("classical gate initialization requires PyTorch") from exc
    return torch


def _generator(tensor: object, seed: int):
    torch = _torch()
    device = tensor.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"deterministic teacher initialization supports cpu/cuda, got {device}"
        )
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def _fan_in(weight: object) -> int:
    shape = tuple(int(value) for value in weight.shape)
    if len(shape) < 2:
        raise ValueError(f"convolution weight has invalid shape: {shape}")
    receptive = math.prod(shape[2:]) if len(shape) > 2 else 1
    return shape[1] * receptive


def _reset_hidden_conv(layer: object, seed: int) -> None:
    torch = _torch()
    generator = _generator(layer.weight, seed)
    torch.nn.init.kaiming_uniform_(
        layer.weight, a=math.sqrt(5.0), generator=generator
    )
    if layer.bias is not None:
        fan_in = _fan_in(layer.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        torch.nn.init.uniform_(
            layer.bias, -bound, bound, generator=generator
        )


def prepare_gate_for_teacher_(
    gate: object,
    *,
    seed: int = 0,
    final_weight_scale: float = 1e-3,
) -> GateInitializationReport:
    """Initialize a gate before optimizer construction without changing audio.

    Every hidden convolution receives deterministic Kaiming initialization. The
    final convolution receives small nonzero weights and zero bias, so teacher
    loss reaches the full network immediately without saturating initial logits.
    The independent route amplitude is reset to exact zero last.
    """

    torch = _torch()
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("gate initialization seed must be a non-negative integer")
    scale = float(final_weight_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("final_weight_scale must be finite and positive")

    convolutions = [
        layer for layer in gate.network.modules()
        if isinstance(layer, torch.nn.Conv2d)
    ]
    if len(convolutions) < 2:
        raise ValueError("teacher gate requires at least two convolution layers")

    with torch.no_grad():
        for index, layer in enumerate(convolutions[:-1]):
            _reset_hidden_conv(layer, seed + 10_000 * index)
        final = convolutions[-1]
        generator = _generator(final.weight, seed + 10_000 * len(convolutions))
        torch.nn.init.normal_(
            final.weight, mean=0.0, std=scale, generator=generator
        )
        if final.bias is not None:
            torch.nn.init.zeros_(final.bias)
        gate.correction_amplitude.zero_()

    parameters = tuple(gate.network.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    nonzero = sum(
        int(torch.count_nonzero(parameter.detach()).item())
        for parameter in parameters
    )
    if total <= 0 or nonzero <= 0:
        raise RuntimeError("teacher initialization left the spatial gate degenerate")
    if float(gate.correction_amplitude.detach().cpu()) != 0.0:
        raise RuntimeError("teacher initialization violated exact step-zero amplitude")
    return GateInitializationReport(
        seed=seed,
        final_weight_scale=scale,
        convolution_count=len(convolutions),
        nonzero_parameters=nonzero,
        total_parameters=total,
        effective_amplitude=0.0,
    )
