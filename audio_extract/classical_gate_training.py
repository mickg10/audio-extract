"""Training utilities for the bounded frozen-separator gate.

The exact O2 route is a teacher available only on exact-reference training
works. It must never become an inference input. This module provides:

* immutable teacher geometry and basis identity carried to the loss boundary;
* pairwise projection of K-way O2 labels with both third-member and
  oracle-unavailable cells masked;
* a teacher loss on inference-available gate features;
* a post-step amplitude projector that preserves the native PyTorch optimizer
  API and clears outward momentum/first-moment state at active boundaries.

The separator members remain frozen. Clean references are not accepted by any
function in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TeacherGeometry:
    """Exact O2 analysis grid, including the realized cell partitions."""

    sample_rate_hz: int
    n_fft: int
    hop_length: int
    tile_seconds: float
    band_edges_hz: tuple[int, ...]
    time_ranges: tuple[tuple[int, int], ...]
    frequency_ranges: tuple[tuple[int, int], ...]

    def identity_dict(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": int(self.sample_rate_hz),
            "n_fft": int(self.n_fft),
            "hop_length": int(self.hop_length),
            "tile_seconds": format(float(self.tile_seconds), ".17g"),
            "band_edges_hz": list(self.band_edges_hz),
            "time_ranges": [list(value) for value in self.time_ranges],
            "frequency_ranges": [list(value) for value in self.frequency_ranges],
        }


@dataclass(frozen=True)
class PairwiseTeacherTargets:
    """Binary pairwise labels plus exact evidence and identity facts."""

    target: object
    available: object
    geometry: TeacherGeometry
    basis_ids: tuple[str, ...]
    parent_index: int
    aggressive_index: int
    parent_cells: int
    aggressive_cells: int
    unavailable_cells: int
    masked_other_cells: int
    teacher_sha256: str


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


def _ranges(values: Sequence[Sequence[int]], name: str) -> tuple[tuple[int, int], ...]:
    result = tuple((int(value[0]), int(value[1])) for value in values)
    if not result or any(start < 0 or stop <= start for start, stop in result):
        raise ValueError(f"{name} must contain non-empty increasing half-open ranges")
    if any(left[1] != right[0] for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must cover its axis contiguously without overlap")
    return result


def build_teacher_geometry(
    routing_config: object,
    *,
    time_ranges: Sequence[Sequence[int]],
    frequency_ranges: Sequence[Sequence[int]],
) -> TeacherGeometry:
    """Freeze the exact O2 config and realized STFT-cell ranges."""

    geometry = TeacherGeometry(
        sample_rate_hz=int(_value(routing_config, "sample_rate_hz")),
        n_fft=int(_value(routing_config, "n_fft")),
        hop_length=int(_value(routing_config, "hop_length")),
        tile_seconds=float(_value(routing_config, "tile_seconds")),
        band_edges_hz=tuple(int(value) for value in _value(
            routing_config, "band_edges_hz"
        )),
        time_ranges=_ranges(time_ranges, "time_ranges"),
        frequency_ranges=_ranges(frequency_ranges, "frequency_ranges"),
    )
    if geometry.sample_rate_hz <= 0 or geometry.n_fft < 4:
        raise ValueError("teacher geometry has invalid sample-rate/FFT facts")
    if geometry.hop_length <= 0 or geometry.hop_length > geometry.n_fft:
        raise ValueError("teacher geometry has invalid hop length")
    if not math.isfinite(geometry.tile_seconds) or geometry.tile_seconds <= 0:
        raise ValueError("teacher tile duration must be finite and positive")
    if (len(geometry.band_edges_hz) < 2
            or geometry.band_edges_hz[0] != 0
            or geometry.band_edges_hz[-1] != geometry.sample_rate_hz // 2
            or any(right <= left for left, right in zip(
                geometry.band_edges_hz, geometry.band_edges_hz[1:]
            ))):
        raise ValueError("teacher bands must increase from zero to Nyquist")
    return geometry


def require_teacher_geometry(
    gate_config: object,
    geometry: TeacherGeometry,
    *,
    time_ranges: Sequence[Sequence[int]] | None = None,
    frequency_ranges: Sequence[Sequence[int]] | None = None,
) -> None:
    """Require geometry equality at the actual teacher-loss boundary."""

    facts = {
        "sample_rate_hz": int(_value(gate_config, "sample_rate_hz")),
        "n_fft": int(_value(gate_config, "n_fft")),
        "hop_length": int(_value(gate_config, "hop_length")),
        "tile_seconds": float(_value(gate_config, "tile_seconds")),
        "band_edges_hz": tuple(int(value) for value in _value(
            gate_config, "band_edges_hz"
        )),
    }
    expected = {
        "sample_rate_hz": geometry.sample_rate_hz,
        "n_fft": geometry.n_fft,
        "hop_length": geometry.hop_length,
        "tile_seconds": geometry.tile_seconds,
        "band_edges_hz": geometry.band_edges_hz,
    }
    differences = {
        name: {"gate": facts[name], "teacher": expected[name]}
        for name in facts if facts[name] != expected[name]
    }
    if time_ranges is not None:
        actual = _ranges(time_ranges, "gate time_ranges")
        if actual != geometry.time_ranges:
            differences["time_ranges"] = {
                "gate": actual, "teacher": geometry.time_ranges
            }
    if frequency_ranges is not None:
        actual = _ranges(frequency_ranges, "gate frequency_ranges")
        if actual != geometry.frequency_ranges:
            differences["frequency_ranges"] = {
                "gate": actual, "teacher": geometry.frequency_ranges
            }
    if differences:
        raise ValueError(f"gate/O2 teacher geometry differs: {differences}")


def _teacher_hash(
    labels: object,
    available: object,
    geometry: TeacherGeometry,
    basis_ids: tuple[str, ...],
    parent_index: int,
    aggressive_index: int,
) -> str:
    torch = _torch()
    header = json.dumps({
        "schema": "audio-extract/pairwise-o2-teacher/v1",
        "geometry": geometry.identity_dict(),
        "basis_ids": list(basis_ids),
        "parent_index": parent_index,
        "aggressive_index": aggressive_index,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    label_bytes = torch.as_tensor(labels, dtype=torch.int32).detach().cpu().contiguous().numpy().tobytes()
    mask_bytes = torch.as_tensor(available, dtype=torch.uint8).detach().cpu().contiguous().numpy().tobytes()
    return "sha256:" + hashlib.sha256(header + label_bytes + mask_bytes).hexdigest()


def pairwise_o2_targets(
    labels: object,
    oracle_available: object,
    *,
    geometry: TeacherGeometry,
    basis_ids: Sequence[str],
    parent_index: int,
    aggressive_index: int,
) -> PairwiseTeacherTargets:
    """Project K-way O2 labels onto one parent/aggressive experiment.

    A cell selected by any third member is masked. A cell unavailable to exact
    source-coordinate evidence is also masked even when the MILP assigned it a
    smoothness-driven label.
    """

    torch = _torch()
    value = torch.as_tensor(labels)
    evidence = torch.as_tensor(oracle_available, dtype=torch.bool)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if evidence.ndim == 2:
        evidence = evidence.unsqueeze(0)
    if value.ndim != 3 or not value.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("O2 labels must be an integer (batch,time,frequency) tensor")
    if evidence.shape != value.shape:
        raise ValueError("oracle availability mask must match O2 labels")
    if value.shape[1:] != (
        len(geometry.time_ranges), len(geometry.frequency_ranges)
    ):
        raise ValueError("teacher labels do not match frozen geometry cell counts")

    basis = tuple(str(value) for value in basis_ids)
    if len(basis) < 2 or any(not item for item in basis) or len(set(basis)) != len(basis):
        raise ValueError("teacher basis IDs must be unique non-empty values")
    parent_index = int(parent_index)
    aggressive_index = int(aggressive_index)
    if not (0 <= parent_index < len(basis)):
        raise ValueError("parent index is outside teacher basis")
    if not (0 <= aggressive_index < len(basis)) or parent_index == aggressive_index:
        raise ValueError("aggressive index is outside teacher basis or equals parent")
    if bool((value < 0).any()) or bool((value >= len(basis)).any()):
        raise ValueError("O2 label is outside the declared teacher basis")

    parent = value == parent_index
    aggressive = value == aggressive_index
    pair = parent | aggressive
    available = evidence & pair
    target = aggressive.to(dtype=torch.float32).unsqueeze(1)
    available_4d = available.unsqueeze(1)
    return PairwiseTeacherTargets(
        target=target,
        available=available_4d,
        geometry=geometry,
        basis_ids=basis,
        parent_index=parent_index,
        aggressive_index=aggressive_index,
        parent_cells=int((parent & evidence).sum().item()),
        aggressive_cells=int((aggressive & evidence).sum().item()),
        unavailable_cells=int((~evidence).sum().item()),
        masked_other_cells=int((evidence & ~pair).sum().item()),
        teacher_sha256=_teacher_hash(
            value, evidence, geometry, basis, parent_index, aggressive_index
        ),
    )


def gate_teacher_logits(
    gate: object,
    mixture: object,
    conservative_vocal: object,
    aggressive_vocal: object,
) -> tuple[object, dict[str, object]]:
    """Run only inference-available features through the spatial network."""

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
        "time_ranges": tuple(tuple(value) for value in time_ranges),
        "frequency_ranges": tuple(tuple(value) for value in band_ranges),
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
    require_teacher_geometry(
        gate.config, targets.geometry,
        time_ranges=details["time_ranges"],
        frequency_ranges=details["frequency_ranges"],
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
    positive = int(((target >= 0.5) & available).sum().item())
    negative = count - positive
    if count == 0:
        loss = logits.sum() * 0.0
        accuracy = logits.new_tensor(float("nan"))
    else:
        selected = per_cell[available]
        if balance_classes and positive and negative:
            selected_target = target[available]
            weights = torch.where(
                selected_target >= 0.5,
                selected_target.new_tensor(0.5 / positive),
                selected_target.new_tensor(0.5 / negative),
            )
            loss = (selected * weights).sum()
        else:
            loss = selected.mean()
        prediction = logits[available] >= 0
        truth = target[available] >= 0.5
        accuracy = (prediction == truth).to(logits.dtype).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("pairwise O2 teacher loss is non-finite")
    return loss, {
        **details,
        "teacher_sha256": targets.teacher_sha256,
        "available_cells": count,
        "positive_cells": positive,
        "negative_cells": negative,
        "unavailable_cells": targets.unavailable_cells,
        "masked_other_cells": targets.masked_other_cells,
        "teacher_accuracy": accuracy,
        "balance_classes": bool(balance_classes),
    }


def _directional_state(optimizer: object, parameter: object) -> dict[str, object]:
    state = optimizer.state.get(parameter, {})
    return {
        key: state[key]
        for key in ("momentum_buffer", "exp_avg", "grad_avg")
        if key in state and hasattr(state[key], "shape")
            and tuple(state[key].shape) == tuple(parameter.shape)
    }


def _clear_outward_state_(
    optimizer: object,
    parameter: object,
    *,
    lower_boundary: bool,
    upper_boundary: bool,
) -> tuple[str, ...]:
    torch = _torch()
    cleared = []
    with torch.no_grad():
        for key, value in _directional_state(optimizer, parameter).items():
            outward = (
                (lower_boundary and bool((value > 0).any()))
                or (upper_boundary and bool((value < 0).any()))
            )
            if outward:
                value.zero_()
                cleared.append(key)
    return tuple(cleared)


def project_gate_amplitude_after_step_(
    optimizer: object,
    gate: object,
) -> dict[str, Any]:
    """Project amplitude and remove optimizer state that points out of bounds.

    Call this immediately after `optimizer.step()` or `scaler.step(optimizer)`.
    The original optimizer object remains available to schedulers, AMP, and
    frameworks; no optimizer return semantics are changed.
    """

    torch = _torch()
    parameter = gate.correction_amplitude
    if tuple(parameter.shape) != () or not parameter.is_floating_point():
        raise ValueError("gate correction amplitude must be a floating scalar")
    owners = [
        group for group in optimizer.param_groups
        if any(id(value) == id(parameter) for value in group["params"])
    ]
    if len(owners) != 1:
        raise ValueError("optimizer must own the gate amplitude in exactly one group")
    if float(owners[0].get("weight_decay", 0.0)) != 0.0:
        raise ValueError("gate amplitude optimizer group must use zero weight decay")

    before = float(parameter.detach().cpu())
    if not math.isfinite(before):
        raise ValueError("gate correction amplitude is non-finite")
    lower = before <= 0.0
    upper = before >= 1.0
    with torch.no_grad():
        parameter.clamp_(0.0, 1.0)
    after = float(parameter.detach().cpu())
    cleared = _clear_outward_state_(
        optimizer, parameter,
        lower_boundary=lower and after == 0.0,
        upper_boundary=upper and after == 1.0,
    )
    return {
        "raw_before_projection": before,
        "effective_amplitude": after,
        "active_boundary": "lower" if after == 0.0 else "upper" if after == 1.0 else None,
        "cleared_directional_state": list(cleared),
    }


class GateAmplitudeProjector:
    """State-light post-step hook; it is intentionally not an optimizer wrapper."""

    SCHEMA = "audio-extract/gate-amplitude-projector/v1"

    def __init__(self, optimizer: object, gate: object):
        self.optimizer = optimizer
        self.gate = gate
        project_gate_amplitude_after_step_(optimizer, gate)

    def after_step(self) -> dict[str, Any]:
        return project_gate_amplitude_after_step_(self.optimizer, self.gate)

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if dict(state_dict) != {"schema": self.SCHEMA}:
            raise ValueError("gate amplitude projector state mismatch")
        project_gate_amplitude_after_step_(self.optimizer, self.gate)
