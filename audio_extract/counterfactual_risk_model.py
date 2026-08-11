"""Identity-bound neural counterfactual-risk student for frozen separators.

The model consumes only inference-available, precomputed candidate features and
an optional frozen target-singer embedding.  It predicts non-negative monotone
risk quantiles plus per-metric availability probabilities for every frozen
candidate/time/frequency cell.  Clean accompaniment, vocal truth, and oracle
routes are deliberately absent from the inference signature.

The output can be converted into :class:`counterfactual_risk_router.RiskPanel`
and decoded by the deterministic Potts router.  Exact counterfactual labels are
used only by :func:`counterfactual_risk_training_loss`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .counterfactual_risk_router import RiskPanel, RiskPanelIdentity

MODEL_CONFIG_SCHEMA = "audio-extract/counterfactual-risk-model-config/v1"
MODEL_BUNDLE_SCHEMA = "audio-extract/counterfactual-risk-model-bundle/v1"
QUERY_CONDITION_SCHEMA = "audio-extract/counterfactual-risk-query/v1"
PANEL_CONSTRUCTION_SCHEMA = "audio-extract/counterfactual-risk-panel-construction/v1"


class RiskModelError(RuntimeError):
    """The risk model or its inference contract is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "").lower()
    if not result.startswith("sha256:") or len(result) != 71:
        raise ValueError(f"{name} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be sha256:<64 hex>") from exc
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - training extra owns torch
        raise RuntimeError(
            "counterfactual risk model requires PyTorch"
        ) from exc
    return torch, nn


@dataclass(frozen=True)
class RiskModelConfig:
    """Semantic identity of one risk student architecture."""

    candidate_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    feature_contract_sha256: str
    feature_channels: int
    query_encoder_sha256: str | None = None
    query_dim: int = 0
    quantile_levels: tuple[float, ...] = (0.5, 0.9, 0.95)
    hidden_channels: int = 64
    blocks: int = 3
    candidate_embedding_dim: int = 16
    query_projection_dim: int = 0
    availability_probability_threshold: float = 0.9
    initial_risk_logit: float = -4.0
    initial_availability_probability: float = 0.5

    def validate(self) -> None:
        if len(self.candidate_ids) < 2:
            raise ValueError("at least two frozen candidates are required")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique and ordered")
        for index, value in enumerate(self.candidate_ids):
            _sha(value, f"candidate_ids[{index}]")
        if not self.metric_names or any(
            not isinstance(name, str) or not name
            for name in self.metric_names
        ):
            raise ValueError("metric names must be non-empty strings")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric names must be unique and ordered")
        _sha(self.feature_contract_sha256, "feature_contract_sha256")
        for name in (
            "feature_channels",
            "hidden_channels",
            "blocks",
            "candidate_embedding_dim",
        ):
            if isinstance(getattr(self, name), bool) or int(
                getattr(self, name)
            ) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.query_dim, bool) or int(self.query_dim) < 0:
            raise ValueError("query_dim must be a non-negative integer")
        if isinstance(self.query_projection_dim, bool) or int(
            self.query_projection_dim
        ) < 0:
            raise ValueError(
                "query_projection_dim must be a non-negative integer"
            )
        query_enabled = self.query_encoder_sha256 is not None
        if query_enabled:
            _sha(self.query_encoder_sha256, "query_encoder_sha256")
            if self.query_dim < 1 or self.query_projection_dim < 1:
                raise ValueError(
                    "query-enabled models require positive query dimensions"
                )
        elif self.query_dim != 0 or self.query_projection_dim != 0:
            raise ValueError(
                "query-disabled models require zero query dimensions"
            )
        levels = tuple(float(value) for value in self.quantile_levels)
        if (
            not levels
            or any(not math.isfinite(value) or not 0 < value < 1 for value in levels)
            or any(right <= left for left, right in itertools.pairwise(levels))
        ):
            raise ValueError(
                "quantile levels must be finite, unique, and strictly increasing"
            )
        for name in (
            "availability_probability_threshold",
            "initial_availability_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly in (0,1)")
        if not math.isfinite(float(self.initial_risk_logit)):
            raise ValueError("initial_risk_logit must be finite")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": MODEL_CONFIG_SCHEMA,
            "candidate_ids": list(self.candidate_ids),
            "metric_names": list(self.metric_names),
            "feature_contract_sha256": self.feature_contract_sha256,
            "feature_channels": int(self.feature_channels),
            "query_encoder_sha256": self.query_encoder_sha256,
            "query_dim": int(self.query_dim),
            "quantile_levels": [
                float(value) for value in self.quantile_levels
            ],
            "hidden_channels": int(self.hidden_channels),
            "blocks": int(self.blocks),
            "candidate_embedding_dim": int(
                self.candidate_embedding_dim
            ),
            "query_projection_dim": int(self.query_projection_dim),
            "availability_probability_threshold": float(
                self.availability_probability_threshold
            ),
            "initial_risk_logit": float(self.initial_risk_logit),
            "initial_availability_probability": float(
                self.initial_availability_probability
            ),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class RiskModelOutput:
    """PyTorch tensors returned by the risk model.

    Shapes:
        quantiles:              (B,K,D,Q,T,F)
        availability_logits:    (B,K,D,T,F)
        hard_feature_available: (B,K,D,T,F)
    """

    quantiles: object
    availability_logits: object
    hard_feature_available: object


@dataclass(frozen=True)
class RiskTrainingLossConfig:
    quantile_weight: float = 1.0
    availability_weight: float = 0.25

    def validate(self) -> None:
        for name in ("quantile_weight", "availability_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.quantile_weight == 0 and self.availability_weight == 0:
            raise ValueError("at least one training loss weight must be positive")


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def build_counterfactual_risk_model(config: RiskModelConfig):
    """Build the PyTorch model lazily so the base package need not import it."""

    torch, nn = _torch()
    config.validate()
    candidates = len(config.candidate_ids)
    metrics = len(config.metric_names)
    quantiles = len(config.quantile_levels)
    hidden = int(config.hidden_channels)

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            groups = 8 if hidden % 8 == 0 else 1
            self.net = nn.Sequential(
                nn.GroupNorm(groups, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(groups, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, value):
            return value + self.net(value)

    class CounterfactualRiskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.candidate_embedding = nn.Parameter(
                torch.empty(candidates, int(config.candidate_embedding_dim))
            )
            nn.init.normal_(self.candidate_embedding, mean=0.0, std=0.02)
            if config.query_encoder_sha256 is not None:
                self.query_projection = nn.Sequential(
                    nn.Linear(int(config.query_dim), int(config.query_projection_dim)),
                    nn.LayerNorm(int(config.query_projection_dim)),
                    nn.SiLU(),
                )
            else:
                self.query_projection = None
            input_channels = (
                int(config.feature_channels)
                + int(config.candidate_embedding_dim)
                + int(config.query_projection_dim)
            )
            self.input_projection = nn.Conv2d(
                input_channels, hidden, kernel_size=3, padding=1
            )
            self.blocks = nn.Sequential(*(
                ResidualBlock() for _ in range(int(config.blocks))
            ))
            self.quantile_head = nn.Conv2d(
                hidden, metrics * quantiles, kernel_size=1
            )
            self.availability_head = nn.Conv2d(
                hidden, metrics, kernel_size=1
            )
            nn.init.zeros_(self.quantile_head.weight)
            nn.init.constant_(
                self.quantile_head.bias, float(config.initial_risk_logit)
            )
            nn.init.zeros_(self.availability_head.weight)
            nn.init.constant_(
                self.availability_head.bias,
                _logit(float(config.initial_availability_probability)),
            )

        def forward(
            self,
            features,
            hard_feature_available,
            query_embedding=None,
        ) -> RiskModelOutput:
            """Predict monotone risk quantiles without exposing truth.

            `features` is `(B,K,C,T,F)`.  `hard_feature_available` is
            `(B,K,D,T,F)` and represents non-learned input availability.  Frozen
            feature/query tensors are detached at the boundary; only this model's
            parameters receive gradients.
            """

            if features.ndim != 5:
                raise ValueError("features must be (B,K,C,T,F)")
            batch, candidate_count, channels, time, frequency = features.shape
            if candidate_count != candidates:
                raise ValueError("feature candidate order/count mismatch")
            if channels != int(config.feature_channels):
                raise ValueError("feature channel count mismatch")
            expected_available = (
                batch, candidates, metrics, time, frequency
            )
            if tuple(hard_feature_available.shape) != expected_available:
                raise ValueError(
                    "hard_feature_available must be (B,K,D,T,F)"
                )
            if hard_feature_available.dtype is not torch.bool:
                raise ValueError("hard_feature_available must be boolean")
            if not bool(torch.isfinite(features).all()):
                raise ValueError("features contain non-finite values")

            feature_value = features.detach()
            candidate = self.candidate_embedding[None, :, :, None, None]
            candidate = candidate.expand(
                batch, -1, -1, time, frequency
            )
            pieces = [feature_value, candidate]
            if self.query_projection is None:
                if query_embedding is not None:
                    raise ValueError(
                        "query-disabled model received a query embedding"
                    )
            else:
                if query_embedding is None:
                    raise ValueError(
                        "query-enabled model requires a query embedding"
                    )
                if tuple(query_embedding.shape) != (
                    batch, int(config.query_dim)
                ):
                    raise ValueError("query embedding shape mismatch")
                if not bool(torch.isfinite(query_embedding).all()):
                    raise ValueError("query embedding contains non-finite values")
                query = self.query_projection(query_embedding.detach())
                query = query[:, None, :, None, None].expand(
                    -1, candidates, -1, time, frequency
                )
                pieces.append(query)

            value = torch.cat(pieces, dim=2).reshape(
                batch * candidates, -1, time, frequency
            )
            hidden_value = self.blocks(self.input_projection(value))
            raw = self.quantile_head(hidden_value).reshape(
                batch, candidates, metrics, quantiles, time, frequency
            )
            positive = torch.nn.functional.softplus(raw)
            first = positive[:, :, :, :1]
            if quantiles > 1:
                increments = torch.cumsum(
                    positive[:, :, :, 1:], dim=3
                )
                predicted = torch.cat((first, first + increments), dim=3)
            else:  # pragma: no cover - default/config tests use multiple levels
                predicted = first
            availability_logits = self.availability_head(hidden_value).reshape(
                batch, candidates, metrics, time, frequency
            )
            if not bool(torch.isfinite(predicted).all()) or not bool(
                torch.isfinite(availability_logits).all()
            ):
                raise RiskModelError("risk model produced non-finite output")
            return RiskModelOutput(
                quantiles=predicted,
                availability_logits=availability_logits,
                hard_feature_available=hard_feature_available.detach(),
            )

    return CounterfactualRiskModel()


def model_state_sha256(model) -> str:
    """Hash exact config plus sorted state tensor names/dtypes/shapes/bytes."""

    torch, _ = _torch()
    config = model.config
    config.validate()
    digest = hashlib.sha256()
    digest.update(_canonical({
        "schema": MODEL_BUNDLE_SCHEMA,
        "config": config.identity_dict(),
    }))
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def query_condition_sha256(
    query_embedding,
    *,
    query_encoder_sha256: str,
) -> str:
    """Bind one represented query embedding to the frozen encoder identity."""

    torch, _ = _torch()
    encoder = _sha(query_encoder_sha256, "query_encoder_sha256")
    value = query_embedding.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if value.ndim != 1 or not bool(torch.isfinite(value).all()):
        raise ValueError("query condition must be one finite embedding vector")
    header = {
        "schema": QUERY_CONDITION_SCHEMA,
        "query_encoder_sha256": encoder,
        "shape": list(value.shape),
        "dtype": "float32-le",
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + value.numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()


def output_to_risk_panel(
    model,
    output: RiskModelOutput,
    *,
    source_pcm_sha256: str,
    batch_index: int = 0,
    upper_quantile: float = 0.95,
    query_condition_sha256_value: str | None = None,
) -> RiskPanel:
    """Convert one batch item into the identity-bound structured-router panel."""

    torch, _ = _torch()
    config = model.config
    config.validate()
    source = _sha(source_pcm_sha256, "source_pcm_sha256")
    levels = tuple(float(value) for value in config.quantile_levels)
    try:
        quantile_index = levels.index(float(upper_quantile))
    except ValueError as exc:
        raise ValueError(
            f"upper quantile {upper_quantile} is not in {levels}"
        ) from exc
    if not 0 <= int(batch_index) < output.quantiles.shape[0]:
        raise ValueError("batch index is outside model output")
    if config.query_encoder_sha256 is None:
        if query_condition_sha256_value is not None:
            raise ValueError(
                "query-disabled model cannot bind a query condition"
            )
    else:
        _sha(query_condition_sha256_value, "query_condition_sha256")

    risk = output.quantiles[
        batch_index, :, :, quantile_index
    ].detach().to(device="cpu", dtype=torch.float64)
    probability = torch.sigmoid(
        output.availability_logits[batch_index]
    ).detach().to(device="cpu", dtype=torch.float64)
    hard = output.hard_feature_available[batch_index].detach().cpu()
    available = hard & (
        probability >= float(config.availability_probability_threshold)
    )
    # K,D,T,F -> T,F,K,D
    upper = risk.permute(2, 3, 0, 1).contiguous().numpy()
    availability = available.permute(2, 3, 0, 1).contiguous().numpy()
    panel_identity = RiskPanelIdentity(
        source_pcm_sha256=source,
        candidate_ids=tuple(config.candidate_ids),
        metric_names=tuple(config.metric_names),
        feature_contract_sha256=config.feature_contract_sha256,
        risk_model_sha256=model_state_sha256(model),
        query_encoder_sha256=config.query_encoder_sha256,
        query_condition_sha256=query_condition_sha256_value,
    )
    panel = RiskPanel(panel_identity, upper, availability)
    panel.validate()
    return panel


def counterfactual_risk_training_loss(
    output: RiskModelOutput,
    target_risk,
    target_available,
    quantile_levels: Sequence[float],
    *,
    config: RiskTrainingLossConfig | None = None,
):
    """Quantile + availability supervision for exact counterfactual labels.

    Targets have shape `(B,K,D,T,F)`.  They are detached immediately.  Positive
    available labels must be finite and non-negative.  A batch with no available
    exact risk has a connected zero quantile loss; availability BCE still trains.
    """

    torch, _ = _torch()
    cfg = config or RiskTrainingLossConfig()
    cfg.validate()
    predicted = output.quantiles
    logits = output.availability_logits
    target = target_risk.detach().to(
        device=predicted.device, dtype=predicted.dtype
    )
    available = target_available.detach().to(
        device=predicted.device, dtype=torch.bool
    )
    expected = predicted.shape[:3] + predicted.shape[4:]
    if tuple(target.shape) != expected or tuple(available.shape) != expected:
        raise ValueError("risk targets must be (B,K,D,T,F)")
    if tuple(logits.shape) != expected:
        raise ValueError("availability logits differ from risk target geometry")
    if len(quantile_levels) != predicted.shape[3]:
        raise ValueError("quantile level count differs from model output")
    levels = torch.as_tensor(
        tuple(float(value) for value in quantile_levels),
        device=predicted.device,
        dtype=predicted.dtype,
    ).view(1, 1, 1, -1, 1, 1)
    if not bool(torch.isfinite(levels).all()) or bool(
        ((levels <= 0) | (levels >= 1)).any()
    ):
        raise ValueError("quantile levels must lie in (0,1)")
    if bool((levels[..., 1:, :, :] <= levels[..., :-1, :, :]).any()):
        raise ValueError("quantile levels must be strictly increasing")
    if bool((~torch.isfinite(target[available])).any()) or bool(
        (target[available] < 0).any()
    ):
        raise ValueError("available exact risks must be finite/non-negative")

    error = target.unsqueeze(3) - predicted
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    quantile_mask = available.unsqueeze(3).expand_as(pinball)
    if bool(quantile_mask.any()):
        quantile_loss = pinball[quantile_mask].mean()
    else:
        quantile_loss = predicted.sum() * 0.0
    availability_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, available.to(logits.dtype)
    )
    total = (
        float(cfg.quantile_weight) * quantile_loss
        + float(cfg.availability_weight) * availability_loss
    )
    if not bool(torch.isfinite(total)):
        raise RiskModelError("counterfactual risk loss is non-finite")
    return total, {
        "quantile": quantile_loss,
        "availability": availability_loss,
        "available_targets": int(available.sum().item()),
    }
