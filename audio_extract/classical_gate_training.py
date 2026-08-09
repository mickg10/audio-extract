"""Training utilities for the bounded frozen-separator gate.

The exact O2 route is a teacher available only on exact-reference training
works.  It must never become an inference input.  This module provides:

* a pairwise projection of multi-member O2 labels that masks cells owned by a
  third candidate rather than falsely calling them conservative;
* a teacher loss on the gate network logits, so the spatial network can learn
  while the decoded route remains exactly at its step-zero parent;
* a projected optimizer wrapper that keeps the scalar correction amplitude in
  ``[0, 1]`` after every update, preventing a negative first update from
  entering the lower-clamp dead zone permanently;
* an exact geometry check between the learned gate and the O2 teacher grid.

The separator members remain frozen.  The utilities do not run a separator and
do not consume clean references at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class PairwiseTeacherTargets:
    """Binary pairwise labels plus the cells for which they are meaningful."""

    target: object
    available: object
    parent_cells: int
    aggressive_cells: int
    masked_other_cells: int


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - owned by the train extra
        raise RuntimeError("classical gate training requires PyTorch") from exc
    return torch


def _value(source: object, name: str) -> Any:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


def require_teacher_geometry(gate_config: object, routing_config: object) -> None:
    """Require the learned gate and O2 teacher to use one exact cell grid."""

    fields = (
        "sample_rate_hz", "n_fft", "hop_length", "tile_seconds",
        "band_edges_hz",
    )
    differences = {}
    for name in fields:
        gate_value = _value(gate_config, name)
        teacher_value = _value(routing_config, name)
        if name == "band_edges_hz":
            gate_value = tuple(int(value) for value in gate_value)
            teacher_value = tuple(int(value) for value in teacher_value)
        elif name == "tile_seconds":
            gate_value = float(gate_value)
            teacher_value = float(teacher_value)
        else:
            gate_value = int(gate_value)
            teacher_value = int(teacher_value)
        if gate_value != teacher_value:
            differences[name] = {"gate": gate_value, "teacher": teacher_value}
    if differences:
        raise ValueError(f"gate/O2 teacher geometry differs: {differences}")


def pairwise_o2_targets(
    labels: object,
    *,
    parent_index: int,
    aggressive_index: int,
) -> PairwiseTeacherTargets:
    """Project K-way O2 labels onto one parent/aggressive experiment.

    A cell selected by any third member is masked.  Treating it as a parent label
    would teach the pairwise gate a fact that the exact teacher did not state.
    """

    torch = _torch()
    value = torch.as_tensor(labels)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or not value.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("O2 labels must be an integer (batch,time,frequency) tensor")
    parent_index = int(parent_index)
    aggressive_index = int(aggressive_index)
    if parent_index < 0 or aggressive_index < 0 or parent_index == aggressive_index:
        raise ValueError("parent and aggressive indices must be distinct non-negative values")
    if bool((value < 0).any()):
        raise ValueError("O2 labels must be non-negative")

    parent = value == parent_index
    aggressive = value == aggressive_index
    available = parent | aggressive
    target = aggressive.to(dtype=torch.float32).unsqueeze(1)
    available = available.unsqueeze(1)
    return PairwiseTeacherTargets(
        target=target,
        available=available,
        parent_cells=int(parent.sum().item()),
        aggressive_cells=int(aggressive.sum().item()),
        masked_other_cells=int((~available.squeeze(1)).sum().item()),
    )


def gate_teacher_logits(
    gate: object,
    mixture: object,
    conservative_vocal: object,
    aggressive_vocal: object,
) -> tuple[object, dict[str, object]]:
    """Run only the inference-available feature network, not the route output."""

    gate._validate_audio({
        "mixture": mixture,
        "conservative_vocal": conservative_vocal,
        "aggressive_vocal": aggressive_vocal,
    })
    mixture_spectrum, _ = gate._stft(mixture)
    conservative_spectrum, _ = gate._stft(conservative_vocal)
    aggressive_spectrum, _ = gate._stft(aggressive_vocal)
    delta = aggressive_spectrum - conservative_spectrum
    features, time_ranges, band_ranges = gate._coarse_features((
        mixture_spectrum, conservative_spectrum, aggressive_spectrum, delta
    ))
    logits = gate.network(features)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"pairwise gate logits must be (batch,1,time,band), got {logits.shape}")
    if not bool(_torch().isfinite(logits).all()):
        raise ValueError("gate teacher logits are non-finite")
    return logits, {
        "features": features,
        "time_ranges": time_ranges,
        "band_ranges": band_ranges,
    }


def pairwise_o2_teacher_loss(
    gate: object,
    mixture: object,
    conservative_vocal: object,
    aggressive_vocal: object,
    targets: PairwiseTeacherTargets,
    *,
    balance_classes: bool = True,
) -> tuple[object, dict[str, object]]:
    """Distill exact O2 labels into the spatial network on training works only."""

    torch = _torch()
    logits, details = gate_teacher_logits(
        gate, mixture, conservative_vocal, aggressive_vocal
    )
    target = torch.as_tensor(
        targets.target, device=logits.device, dtype=logits.dtype
    )
    available = torch.as_tensor(
        targets.available, device=logits.device, dtype=torch.bool
    )
    if target.shape != logits.shape or available.shape != logits.shape:
        raise ValueError(
            f"teacher target grid {target.shape}/{available.shape} != logits {logits.shape}"
        )
    if not bool(torch.isfinite(target).all()) or bool(((target < 0) | (target > 1)).any()):
        raise ValueError("teacher targets must be finite binary values")

    per_cell = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    count = int(available.sum().item())
    if count == 0:
        loss = logits.sum() * 0.0
        accuracy = logits.new_tensor(float("nan"))
    else:
        selected = per_cell[available]
        if balance_classes:
            selected_target = target[available]
            positive = int((selected_target >= 0.5).sum().item())
            negative = count - positive
            if positive and negative:
                weights = torch.where(
                    selected_target >= 0.5,
                    selected_target.new_tensor(0.5 / positive),
                    selected_target.new_tensor(0.5 / negative),
                )
                loss = (selected * weights).sum()
            else:
                loss = selected.mean()
        else:
            loss = selected.mean()
        prediction = logits[available] >= 0
        truth = target[available] >= 0.5
        accuracy = (prediction == truth).to(logits.dtype).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("pairwise O2 teacher loss is non-finite")
    return loss, {
        **details,
        "available_cells": count,
        "masked_other_cells": targets.masked_other_cells,
        "teacher_accuracy": accuracy,
        "balance_classes": bool(balance_classes),
    }


def project_gate_amplitude_(gate: object) -> dict[str, float]:
    """Project the raw scalar after every optimizer update.

    Projection after the update means the next forward begins exactly on a clamp
    boundary, where the existing gate has a usable derivative.  It never begins
    from a negative raw value whose lower-clamp derivative is zero.
    """

    torch = _torch()
    parameter = gate.correction_amplitude
    if tuple(parameter.shape) != () or not parameter.is_floating_point():
        raise ValueError("gate correction amplitude must be a floating scalar")
    before = float(parameter.detach().cpu())
    if not math.isfinite(before):
        raise ValueError("gate correction amplitude is non-finite")
    with torch.no_grad():
        parameter.clamp_(0.0, 1.0)
    after = float(parameter.detach().cpu())
    return {"raw_before_projection": before, "effective_amplitude": after}


class ProjectedGateOptimizer:
    """Optimizer adapter that makes amplitude projection non-optional."""

    def __init__(self, optimizer: object, gate: object):
        self.optimizer = optimizer
        self.gate = gate
        parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if id(gate.correction_amplitude) not in parameter_ids:
            raise ValueError("optimizer does not own the gate correction amplitude")
        project_gate_amplitude_(gate)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        result = self.optimizer.step(*args, **kwargs)
        projection = project_gate_amplitude_(self.gate)
        return result, projection

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        result = self.optimizer.load_state_dict(state_dict)
        project_gate_amplitude_(self.gate)
        return result

    @property
    def param_groups(self):
        return self.optimizer.param_groups
