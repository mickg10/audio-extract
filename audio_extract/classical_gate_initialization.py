"""Deterministic trainable initialization for the bounded classical gate.

`SmoothResidualGate` deliberately uses a separate scalar amplitude to guarantee
that the decoded route is exactly the conservative parent at step zero.  That
means the spatial network itself does not need every convolution initialized to
zero.  In fact, zeroing both convolutions makes the network permanently spatially
constant: the hidden activation is zero, the final weight receives no gradient,
and the first layer is disconnected.

This module initializes the spatial network deterministically while resetting the
route amplitude to exact zero.  The decoded step-zero output therefore remains
sample-identical to the parent, but teacher loss reaches every network parameter
on the first backward pass.
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
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def _reset_hidden_conv(layer: object, seed: int) -> None:
    torch = _torch()
    generator = _generator(layer.weight, seed)
    torch.nn.init.kaiming_uniform_(
        layer.weight, a=math.sqrt(5.0), generator=generator
    )
    if layer.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
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
    """Initialize a gate for teacher training without changing step-zero audio.

    The operation is deterministic for a fixed `(gate architecture, seed,
    final_weight_scale)` and is intended to run before optimizer construction.
    Every hidden convolution receives Kaiming initialization.  The final
    convolution receives small nonzero weights and zero bias, allowing gradient
    flow through the full network immediately without saturating initial logits.
    The scalar route amplitude is reset to exactly zero last.
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
