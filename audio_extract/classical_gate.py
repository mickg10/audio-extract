"""Low-capacity conservative gate for frozen classical separator estimates.

The gate never runs or updates a separator.  Given a mixture and two immutable
stem estimates, it renders only the bounded correction

``S_hat = S_conservative + g * (S_aggressive - S_conservative)``.

For an accompaniment stem this is algebraically identical to the preregistered
vocal equation because ``V=M-A``.  Operating on accompaniment has the useful
precision property that the delivered step-0 output is bit-identical to its
stored conservative parent, without a subtract/add round trip.

``g`` is stereo-coherent and predicted on a deliberately low-resolution
time/frequency grid.  A separate scalar correction amplitude is initialized to
exactly zero, making the decoded step-0 output sample-identical to the
conservative parent while retaining a useful first-step gradient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


@dataclass(frozen=True)
class SmoothGateConfig:
    sample_rate_hz: int = 44_100
    n_fft: int = 2_048
    hop_length: int = 1_024
    tile_seconds: float = 2.0
    band_edges_hz: tuple[int, ...] = (
        0, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 22_050
    )
    feature_channels: int = 4
    hidden_channels: int = 8
    correction_weight: float = 0.05
    temporal_tv_weight: float = 0.02
    frequency_tv_weight: float = 0.02
    reference_floor: float = 1e-6
    adapter_revision: str = "smooth-residual-tf-gate/v1"

    def validate(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.n_fft < 4 or self.hop_length <= 0 or self.hop_length > self.n_fft:
            raise ValueError("invalid STFT grid")
        if self.tile_seconds <= 0 or not math.isfinite(self.tile_seconds):
            raise ValueError("tile_seconds must be finite and positive")
        edges = tuple(int(value) for value in self.band_edges_hz)
        if (len(edges) < 2 or edges[0] != 0
                or edges[-1] != self.sample_rate_hz // 2
                or any(right <= left for left, right in zip(edges, edges[1:]))):
            raise ValueError("band_edges_hz must increase from zero to Nyquist")
        if self.feature_channels != 4 or self.hidden_channels <= 0:
            raise ValueError("the v1 gate requires four features and positive hidden width")
        for name in (
            "correction_weight", "temporal_tv_weight", "frequency_tv_weight",
            "reference_floor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.reference_floor == 0:
            raise ValueError("reference_floor must be positive")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["band_edges_hz"] = list(self.band_edges_hz)
        for key, value in tuple(result.items()):
            if isinstance(value, float):
                result[key] = format(value, ".17g")
        return result


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - owned by the train extra
        raise RuntimeError("classical gate training requires PyTorch") from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class SmoothResidualGate:
    """Small trainable wrapper with an explicit, hashable tensor state."""

    def __init__(self, config: SmoothGateConfig | None = None):
        torch = _torch()
        self.config = config or SmoothGateConfig()
        self.config.validate()
        hidden = self.config.hidden_channels
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(4, hidden, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.correction_amplitude = torch.nn.Parameter(torch.zeros(()))
        for layer in self.network.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def parameters(self):
        yield self.correction_amplitude
        yield from self.network.parameters()

    def train(self, mode: bool = True):
        self.network.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self.network.to(*args, **kwargs)
        with _torch().no_grad():
            self.correction_amplitude.data = self.correction_amplitude.data.to(
                *args, **kwargs
            )
        return self

    def project_parameters(self) -> None:
        """Projected-optimizer constraint for the global correction amplitude.

        The raw parameter is kept in ``[0, 1]`` after every optimizer step.  The
        forward therefore does not contain a clamp with a dead region, and a
        state perturbed below zero can recover after projection.
        """
        with _torch().no_grad():
            self.correction_amplitude.clamp_(0.0, 1.0)

    def state_dict(self) -> dict[str, object]:
        return {
            "correction_amplitude": self.correction_amplitude.detach().clone(),
            **{f"network.{key}": value for key, value in self.network.state_dict().items()},
        }

    def load_state_dict(self, state: dict[str, object], *, strict: bool = True) -> None:
        torch = _torch()
        expected = {"correction_amplitude", *(
            f"network.{key}" for key in self.network.state_dict()
        )}
        if strict and set(state) != expected:
            raise ValueError(
                f"gate state keys differ: missing={sorted(expected-set(state))}, "
                f"extra={sorted(set(state)-expected)}"
            )
        with torch.no_grad():
            source = state["correction_amplitude"]
            if tuple(source.shape) != ():
                raise ValueError("correction_amplitude must be scalar")
            self.correction_amplitude.copy_(source)
        self.network.load_state_dict({
            key.removeprefix("network."): value
            for key, value in state.items() if key.startswith("network.")
        }, strict=strict)

    def _validate_audio(self, named: dict[str, object]) -> tuple[int, int, int]:
        torch = _torch()
        shapes = {name: tuple(value.shape) for name, value in named.items()}
        first = next(iter(shapes.values()))
        if len(first) != 3 or first[1] != 2 or first[2] <= 0:
            raise ValueError(f"gate audio must be (batch, 2, frames), got {shapes}")
        if any(shape != first for shape in shapes.values()):
            raise ValueError(f"gate inputs must share one exact grid, got {shapes}")
        for name, value in named.items():
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite floating samples")
        return first

    def _stft(self, value: object):
        torch = _torch()
        batch, channels, frames = value.shape
        window = torch.hann_window(
            self.config.n_fft, device=value.device, dtype=value.dtype
        )
        spectrum = torch.stft(
            value.reshape(batch * channels, frames), n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, win_length=self.config.n_fft,
            window=window, center=True, pad_mode="constant", normalized=True,
            return_complex=True,
        )
        return spectrum.reshape(batch, channels, *spectrum.shape[-2:]), window

    def _band_ranges(self, frequency_bins: int) -> tuple[tuple[int, int], ...]:
        import numpy as np

        frequencies = np.fft.rfftfreq(
            self.config.n_fft, d=1.0 / self.config.sample_rate_hz
        )
        if len(frequencies) != frequency_bins:
            raise ValueError("unexpected STFT frequency grid")
        result = []
        edges = self.config.band_edges_hz
        for index, (low, high) in enumerate(zip(edges, edges[1:])):
            start = int(np.searchsorted(frequencies, low, side="left"))
            side = "right" if index == len(edges) - 2 else "left"
            stop = int(np.searchsorted(frequencies, high, side=side))
            if stop <= start:
                raise ValueError(f"empty gate band {low}..{high}")
            result.append((start, stop))
        return tuple(result)

    def _coarse_features(self, spectra: tuple[object, ...]):
        torch = _torch()
        magnitudes = [value.abs().mean(dim=1) for value in spectra]
        reference = magnitudes[0].mean(dim=(-2, -1), keepdim=True).clamp_min(
            self.config.reference_floor
        )
        normalized = [torch.log1p(value / reference) for value in magnitudes]
        _, frequency_bins, time_frames = normalized[0].shape
        frames_per_tile = max(1, round(
            self.config.tile_seconds * self.config.sample_rate_hz
            / self.config.hop_length
        ))
        time_ranges = tuple(
            (start, min(start + frames_per_tile, time_frames))
            for start in range(0, time_frames, frames_per_tile)
        )
        cells = []
        for feature in normalized:
            rows = []
            for start, stop in time_ranges:
                rows.append(torch.stack([
                    feature[:, f0:f1, start:stop].mean(dim=(-2, -1))
                    for f0, f1 in self._band_ranges(frequency_bins)
                ], dim=-1))
            cells.append(torch.stack(rows, dim=-2))
        # (batch, feature, coarse_time, coarse_frequency)
        return torch.stack(cells, dim=1), time_ranges, self._band_ranges(frequency_bins)

    def __call__(self, mixture: object, conservative_estimate: object,
                 aggressive_estimate: object) -> tuple[object, dict[str, object]]:
        torch = _torch()
        batch, channels, frames = self._validate_audio({
            "mixture": mixture, "conservative_estimate": conservative_estimate,
            "aggressive_estimate": aggressive_estimate,
        })
        mixture_spectrum, window = self._stft(mixture)
        conservative_spectrum, _ = self._stft(conservative_estimate)
        aggressive_spectrum, _ = self._stft(aggressive_estimate)
        delta = aggressive_spectrum - conservative_spectrum
        features, time_ranges, band_ranges = self._coarse_features((
            mixture_spectrum, conservative_spectrum, aggressive_spectrum, delta
        ))
        spatial = torch.sigmoid(self.network(features))
        raw_amplitude = self.correction_amplitude
        raw_value = float(raw_amplitude.detach().cpu())
        if not 0.0 <= raw_value <= 1.0:
            raise ValueError(
                "gate correction amplitude is outside [0,1]; project after optimizer step"
            )
        amplitude = raw_amplitude
        coarse_gate = amplitude * spatial

        full_gate = torch.zeros(
            (batch, 1, conservative_spectrum.shape[-2], conservative_spectrum.shape[-1]),
            device=mixture.device, dtype=mixture.dtype,
        )
        for ti, (t0, t1) in enumerate(time_ranges):
            for fi, (f0, f1) in enumerate(band_ranges):
                full_gate[:, :, f0:f1, t0:t1] = coarse_gate[:, :, ti:ti + 1, fi:fi + 1]
        correction_spectrum = full_gate * delta
        correction = torch.istft(
            correction_spectrum.reshape(batch * channels, *correction_spectrum.shape[-2:]),
            n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            win_length=self.config.n_fft, window=window, center=True,
            normalized=True, length=frames,
        ).reshape(batch, channels, frames)
        estimate = conservative_estimate + correction
        return estimate, {
            "raw_amplitude": raw_amplitude.detach().clone(),
            "amplitude": amplitude.detach().clone(),
            "coarse_gate": coarse_gate,
            "full_gate": full_gate, "correction": correction,
        }


def gate_regularization(diagnostics: dict[str, object], aggressive_estimate: object,
                        conservative_estimate: object, config: SmoothGateConfig):
    """Correction magnitude plus low-resolution time/frequency TV penalties."""
    torch = _torch()
    correction = diagnostics["correction"]
    coarse = diagnostics["coarse_gate"]
    reference = (aggressive_estimate - conservative_estimate).square().mean().clamp_min(
        config.reference_floor ** 2
    )
    correction_ratio = correction.square().mean() / reference
    temporal_tv = (
        (coarse[:, :, 1:, :] - coarse[:, :, :-1, :]).abs().mean()
        if coarse.shape[-2] > 1 else coarse.sum() * 0.0
    )
    frequency_tv = (
        (coarse[:, :, :, 1:] - coarse[:, :, :, :-1]).abs().mean()
        if coarse.shape[-1] > 1 else coarse.sum() * 0.0
    )
    total = (
        config.correction_weight * correction_ratio
        + config.temporal_tv_weight * temporal_tv
        + config.frequency_tv_weight * frequency_tv
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("gate regularization is non-finite")
    return total, {
        "correction_ratio": correction_ratio,
        "temporal_tv": temporal_tv,
        "frequency_tv": frequency_tv,
    }


def gate_state_sha256(gate: SmoothResidualGate) -> str:
    """Hash exact state tensors in a filename-independent canonical order."""
    torch = _torch()
    digest = hashlib.sha256()
    digest.update(b"audio-extract/smooth-residual-gate-state/v1\0")
    for name, tensor in sorted(gate.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = _canonical_json({
            "name": name, "dtype": str(value.dtype), "shape": list(value.shape),
        })
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big")); digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    return "sha256:" + digest.hexdigest()


def gate_bundle_identity(gate: SmoothResidualGate) -> dict[str, str]:
    config_sha = "sha256:" + hashlib.sha256(
        _canonical_json(gate.config.identity_dict())
    ).hexdigest()
    state_sha = gate_state_sha256(gate)
    bundle_sha = "sha256:" + hashlib.sha256(_canonical_json({
        "weights": state_sha, "config": config_sha,
        "adapter_revision": gate.config.adapter_revision,
    })).hexdigest()
    return {
        "weights_sha256": state_sha, "config_sha256": config_sha,
        "adapter_revision": gate.config.adapter_revision,
        "bundle_sha256": bundle_sha,
    }
