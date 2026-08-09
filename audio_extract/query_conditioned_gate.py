"""Zero-initialized target-singer gate over frozen separator candidates.

This research component consumes precomputed low-resolution features and a
frozen singer embedding. It emits one shared stereo simplex weight vector per
time/frequency cell. It performs no STFT/ISTFT or singer embedding extraction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryGateConfig:
    feature_channels: int
    query_dim: int
    candidates: int
    hidden_channels: int = 64
    blocks: int = 3
    conservative_index: int = 0
    initial_parent_logit: float = 4.0

    def validate(self) -> None:
        for name in (
            "feature_channels", "query_dim", "candidates",
            "hidden_channels", "blocks",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= int(self.conservative_index) < int(self.candidates):
            raise ValueError("conservative_index is outside the candidate basis")


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - training extra owns torch
        raise RuntimeError("query-conditioned gate requires PyTorch") from exc
    return torch, nn


def build_query_conditioned_gate(config: QueryGateConfig):
    """Build a PyTorch gate lazily so the base package need not import torch."""

    torch, nn = _torch_modules()
    config.validate()

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            groups = 8 if channels % 8 == 0 else 1
            self.net = nn.Sequential(
                nn.GroupNorm(groups, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(groups, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, value):
            return value + self.net(value)

    class QueryConditionedConvexGate(nn.Module):
        """Predict target-singer-conditioned candidate weights."""

        def __init__(self):
            super().__init__()
            hidden = int(config.hidden_channels)
            self.config = config
            self.query_projection = nn.Sequential(
                nn.Linear(int(config.query_dim), hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
            )
            self.input_projection = nn.Conv2d(
                int(config.feature_channels) + hidden,
                hidden, kernel_size=3, padding=1,
            )
            self.blocks = nn.Sequential(*(
                ResidualBlock(hidden) for _ in range(int(config.blocks))
            ))
            self.head = nn.Conv2d(hidden, int(config.candidates), kernel_size=1)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, -float(config.initial_parent_logit))
            with torch.no_grad():
                self.head.bias[int(config.conservative_index)] = float(
                    config.initial_parent_logit
                )
            # Step zero uses scale 0. Training starts only after the experiment
            # scheduler sets a small positive scale.
            self.register_buffer(
                "route_scale", torch.tensor(0.0), persistent=True
            )

        def set_route_scale(self, value: float) -> None:
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError("route_scale must lie in [0,1]")
            self.route_scale.fill_(value)

        def forward(self, features, query_embedding):
            """Return `(weights, logits)`.

            `features`: `(batch, feature_channels, time, frequency)`.
            `query_embedding`: `(batch, query_dim)`.
            `weights`: `(batch, candidates, time, frequency)`.
            """

            if features.ndim != 4:
                raise ValueError(
                    "features must be (batch,channels,time,frequency)"
                )
            if query_embedding.ndim != 2:
                raise ValueError("query_embedding must be (batch,query_dim)")
            if features.shape[0] != query_embedding.shape[0]:
                raise ValueError("feature/query batch mismatch")
            if features.shape[1] != config.feature_channels:
                raise ValueError("feature channel mismatch")
            if query_embedding.shape[1] != config.query_dim:
                raise ValueError("query dimension mismatch")
            if not bool(torch.isfinite(features).all()) or not bool(
                torch.isfinite(query_embedding).all()
            ):
                raise ValueError("gate inputs must be finite")

            query = self.query_projection(query_embedding)
            query = query[:, :, None, None].expand(
                -1, -1, features.shape[2], features.shape[3]
            )
            hidden = self.input_projection(torch.cat((features, query), dim=1))
            logits = self.head(self.blocks(hidden))
            learned = torch.softmax(logits, dim=1)
            parent = torch.zeros_like(learned)
            parent[:, int(config.conservative_index)] = 1.0
            scale = self.route_scale.to(
                dtype=learned.dtype, device=learned.device
            )
            weights = parent + scale * (learned - parent)
            if not bool(torch.isfinite(weights).all()):
                raise RuntimeError("gate produced non-finite weights")
            return weights, logits

        def combine_complex_candidates(self, candidate_stft, weights):
            """Apply one shared stereo route to complex candidate spectra.

            `candidate_stft` has shape
            `(batch,candidates,audio_channels,frequency,time)`.
            """

            if candidate_stft.ndim != 5 or weights.ndim != 4:
                raise ValueError("invalid candidate/weight rank")
            if candidate_stft.shape[:2] != weights.shape[:2]:
                raise ValueError("candidate/weight basis mismatch")
            if candidate_stft.shape[1] != config.candidates:
                raise ValueError("candidate count mismatch")
            if candidate_stft.shape[-2] != weights.shape[-1]:
                raise ValueError("frequency dimension mismatch")
            if candidate_stft.shape[-1] != weights.shape[-2]:
                raise ValueError("time dimension mismatch")
            if not bool(torch.isfinite(candidate_stft.real).all()):
                raise ValueError("candidate STFT has non-finite real values")
            if torch.is_complex(candidate_stft) and not bool(
                torch.isfinite(candidate_stft.imag).all()
            ):
                raise ValueError("candidate STFT has non-finite imaginary values")

            # Exact bypass is stronger than relying on a one-hot reduction to
            # preserve step-zero decoded sample identity.
            if float(self.route_scale.detach().cpu()) == 0.0:
                return candidate_stft[:, int(config.conservative_index)]
            shared = weights.permute(0, 1, 3, 2).unsqueeze(2)
            return torch.sum(candidate_stft * shared, dim=1)

    return QueryConditionedConvexGate()


def route_regularization(
    weights,
    *,
    conservative_index: int,
    correction_weight: float = 1.0,
    temporal_weight: float = 1.0,
    frequency_weight: float = 1.0,
):
    """Return conservative-correction and smoothness penalties."""

    torch, _ = _torch_modules()
    if weights.ndim != 4:
        raise ValueError("weights must be (batch,candidates,time,frequency)")
    if not 0 <= conservative_index < weights.shape[1]:
        raise ValueError("invalid conservative index")
    for name, value in (
        ("correction_weight", correction_weight),
        ("temporal_weight", temporal_weight),
        ("frequency_weight", frequency_weight),
    ):
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    correction = 1.0 - weights[:, conservative_index]
    temporal = (
        (weights[:, :, 1:] - weights[:, :, :-1]).abs().mean()
        if weights.shape[2] > 1 else weights.new_zeros(())
    )
    frequency = (
        (weights[:, :, :, 1:] - weights[:, :, :, :-1]).abs().mean()
        if weights.shape[3] > 1 else weights.new_zeros(())
    )
    components = {
        "correction": correction.abs().mean(),
        "temporal_tv": temporal,
        "frequency_tv": frequency,
    }
    components["total"] = (
        float(correction_weight) * components["correction"]
        + float(temporal_weight) * components["temporal_tv"]
        + float(frequency_weight) * components["frequency_tv"]
    )
    if not bool(torch.isfinite(components["total"])):
        raise ValueError("route regularization is non-finite")
    return components
