"""Identity-bound inference contract for counterfactual-risk models v2.

The architecture remains external.  This module binds exact candidate order,
feature/availability tensors, model state, query bytes, quantile selection, and
calibration policy to every produced risk panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json
import math
import re

import numpy as np

FEATURE_SCHEMA = "audio-extract/identity-bound-risk-features/v2"
QUERY_SCHEMA = "audio-extract/counterfactual-risk-query/v2"
PANEL_SCHEMA = "audio-extract/counterfactual-risk-panel-payload/v2"
STATE_SCHEMA = "audio-extract/counterfactual-risk-model-state/v2"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RiskModelV2Error(RuntimeError):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("counterfactual risk model v2 requires PyTorch") from exc
    return torch


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be canonical sha256:<64 lowercase hex>")
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise RiskModelV2Error(f"value is not canonical JSON: {exc}") from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _tensor_bytes(value) -> tuple[bytes, str, tuple[int, ...]]:
    torch = _torch()
    if not torch.is_tensor(value):
        raise TypeError("identity-bearing value must be a torch tensor")
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("identity-bearing tensor contains non-finite values")
    payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return payload, str(tensor.dtype), tuple(map(int, tensor.shape))


def tensor_sha256(value, *, component: str) -> str:
    payload, dtype, shape = _tensor_bytes(value)
    header = {
        "schema": FEATURE_SCHEMA,
        "component": component,
        "dtype": dtype,
        "shape": list(shape),
    }
    return "sha256:" + hashlib.sha256(_canonical(header) + payload).hexdigest()


def config_sha256(config: Any) -> str:
    if not hasattr(config, "identity_dict"):
        raise RiskModelV2Error("model config lacks identity_dict()")
    identity = config.identity_dict()
    if not isinstance(identity, Mapping):
        raise RiskModelV2Error("model config identity is not a mapping")
    return _mapping_sha({"schema": STATE_SCHEMA, "config": dict(identity)})


def model_state_sha256_v2(model) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    digest.update(
        _canonical(
            {
                "schema": STATE_SCHEMA,
                "config_sha256": config_sha256(model.config),
            }
        )
    )
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(
            _canonical(
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            )
        )
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def query_condition_sha256_v2(
    query_embedding,
    *,
    query_encoder_sha256: str,
) -> str:
    encoder = _sha(query_encoder_sha256, "query_encoder_sha256")
    payload, dtype, shape = _tensor_bytes(query_embedding)
    if len(shape) != 1:
        raise ValueError("query condition must be one embedding vector")
    return "sha256:" + hashlib.sha256(
        _canonical(
            {
                "schema": QUERY_SCHEMA,
                "query_encoder_sha256": encoder,
                "dtype": dtype,
                "shape": list(shape),
            }
        )
        + payload
    ).hexdigest()


def _tuple(config: Any, name: str) -> tuple[Any, ...]:
    value = getattr(config, name, ())
    return value if isinstance(value, tuple) else tuple(value or ())


def _validate_ids(values: tuple[str, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise RiskModelV2Error(f"{name} are not unique")
    for index, item in enumerate(values):
        _sha(item, f"{name}[{index}]")


def _validate_names(values: tuple[str, ...], name: str) -> None:
    if not values or any(
        not isinstance(item, str) or not item for item in values
    ):
        raise RiskModelV2Error(f"{name} are invalid")
    if len(set(values)) != len(values):
        raise RiskModelV2Error(f"{name} are not unique")


def _validate_levels(values: tuple[float, ...]) -> tuple[float, ...]:
    levels = tuple(map(float, values))
    if not levels or any(
        not math.isfinite(item) or not 0 < item < 1 for item in levels
    ):
        raise RiskModelV2Error("quantile levels are invalid")
    if any(right <= left for left, right in zip(levels, levels[1:])):
        raise RiskModelV2Error("quantile levels are invalid")
    return levels


@dataclass(frozen=True)
class IdentityBoundFeatureBatchV2:
    values: object
    hard_feature_available: object
    candidate_ids: tuple[str, ...]
    feature_contract_sha256: str

    def validate(self, config: Any) -> tuple[int, int, int, int, int]:
        torch = _torch()
        candidates = _tuple(config, "candidate_ids")
        metrics = _tuple(config, "metric_names")
        if self.candidate_ids != candidates:
            raise RiskModelV2Error(
                "feature candidate identities/order differ from model config"
            )
        _validate_ids(self.candidate_ids, "candidate_ids")
        _sha(self.feature_contract_sha256, "feature_contract_sha256")
        if self.feature_contract_sha256 != getattr(
            config,
            "feature_contract_sha256",
            None,
        ):
            raise RiskModelV2Error("feature contract differs from model config")
        if not torch.is_tensor(self.values) or self.values.ndim != 5:
            raise ValueError("features must be a torch tensor (B,K,C,T,F)")
        batch, count, channels, time, frequency = self.values.shape
        if count != len(candidates) or channels != int(
            getattr(config, "feature_channels")
        ):
            raise RiskModelV2Error("feature candidate/channel geometry differs")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("features contain non-finite values")
        expected = (batch, count, len(metrics), time, frequency)
        if (
            not torch.is_tensor(self.hard_feature_available)
            or tuple(self.hard_feature_available.shape) != expected
            or self.hard_feature_available.dtype is not torch.bool
        ):
            raise ValueError(
                "hard_feature_available must be boolean (B,K,D,T,F)"
            )
        return tuple(map(int, self.values.shape))

    @property
    def values_sha256(self) -> str:
        return tensor_sha256(self.values, component="candidate_features")

    @property
    def availability_sha256(self) -> str:
        return tensor_sha256(
            self.hard_feature_available,
            component="hard_feature_available",
        )


@dataclass(frozen=True)
class RiskModelOutputV2:
    quantiles: object
    availability_logits: object
    hard_feature_available: object
    candidate_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    feature_contract_sha256: str
    feature_values_sha256: str
    hard_feature_available_sha256: str
    model_config_sha256: str
    risk_model_state_sha256: str
    quantile_levels: tuple[float, ...]
    query_encoder_sha256: str | None
    query_condition_sha256s: tuple[str | None, ...]

    def validate(self) -> tuple[int, int, int, int, int, int]:
        torch = _torch()
        if not torch.is_tensor(self.quantiles) or self.quantiles.ndim != 6:
            raise RiskModelV2Error("quantiles must be (B,K,D,Q,T,F)")
        batch, candidates, metrics, levels, time, frequency = (
            self.quantiles.shape
        )
        expected = (batch, candidates, metrics, time, frequency)
        if tuple(self.availability_logits.shape) != expected:
            raise RiskModelV2Error("availability logits geometry differs")
        if tuple(self.hard_feature_available.shape) != expected or (
            self.hard_feature_available.dtype is not torch.bool
        ):
            raise RiskModelV2Error("hard feature availability geometry differs")
        if not bool(torch.isfinite(self.quantiles).all()) or not bool(
            torch.isfinite(self.availability_logits).all()
        ):
            raise RiskModelV2Error("model output contains non-finite values")
        if len(self.candidate_ids) != candidates or len(
            self.metric_names
        ) != metrics:
            raise RiskModelV2Error("output identity axes differ from tensors")
        _validate_ids(self.candidate_ids, "candidate_ids")
        _validate_names(self.metric_names, "metric_names")
        if len(_validate_levels(self.quantile_levels)) != levels:
            raise RiskModelV2Error("quantile levels differ from tensor axis")
        if len(self.query_condition_sha256s) != batch:
            raise RiskModelV2Error("query condition count differs from batch")
        for name in (
            "feature_contract_sha256",
            "feature_values_sha256",
            "hard_feature_available_sha256",
            "model_config_sha256",
            "risk_model_state_sha256",
        ):
            _sha(getattr(self, name), name)
        if self.query_encoder_sha256 is None:
            if any(item is not None for item in self.query_condition_sha256s):
                raise RiskModelV2Error(
                    "query-disabled output carries query identity"
                )
        else:
            _sha(self.query_encoder_sha256, "query_encoder_sha256")
            for item in self.query_condition_sha256s:
                _sha(item, "query_condition_sha256")
        return tuple(map(int, self.quantiles.shape))


@dataclass(frozen=True)
class RiskPanelPayloadV2:
    source_pcm_sha256: str
    upper: np.ndarray
    available: np.ndarray
    candidate_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    feature_contract_sha256: str
    feature_values_sha256: str
    hard_feature_available_sha256: str
    model_config_sha256: str
    risk_model_state_sha256: str
    query_encoder_sha256: str | None
    query_condition_sha256: str | None
    quantile_levels: tuple[float, ...]
    selected_quantile_level: float
    selected_quantile_index: int
    calibration_policy_sha256: str
    availability_probability_threshold: float

    def validate(self) -> tuple[int, int, int, int]:
        _sha(self.source_pcm_sha256, "source_pcm_sha256")
        for name in (
            "feature_contract_sha256",
            "feature_values_sha256",
            "hard_feature_available_sha256",
            "model_config_sha256",
            "risk_model_state_sha256",
            "calibration_policy_sha256",
        ):
            _sha(getattr(self, name), name)
        if self.query_encoder_sha256 is None:
            if self.query_condition_sha256 is not None:
                raise RiskModelV2Error(
                    "query-disabled panel carries query condition"
                )
        else:
            _sha(self.query_encoder_sha256, "query_encoder_sha256")
            _sha(self.query_condition_sha256, "query_condition_sha256")
        _validate_ids(self.candidate_ids, "candidate_ids")
        _validate_names(self.metric_names, "metric_names")
        levels = _validate_levels(self.quantile_levels)
        upper = np.asarray(self.upper, dtype=np.float64)
        available = np.asarray(self.available, dtype=bool)
        if upper.ndim != 4 or upper.shape != available.shape:
            raise RiskModelV2Error(
                "upper/available must share (time,band,candidate,metric)"
            )
        if upper.shape[2:] != (
            len(self.candidate_ids),
            len(self.metric_names),
        ):
            raise RiskModelV2Error("panel axes differ from identity")
        if np.any(~np.isfinite(upper[available])) or np.any(
            upper[available] < 0
        ):
            raise RiskModelV2Error(
                "available upper risks must be finite/non-negative"
            )
        level = float(self.selected_quantile_level)
        index = self.selected_quantile_index
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(levels)
        ):
            raise RiskModelV2Error("selected quantile index is invalid")
        if levels[index] != level:
            raise RiskModelV2Error(
                "selected quantile level/index are inconsistent"
            )
        threshold = float(self.availability_probability_threshold)
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise RiskModelV2Error("availability threshold is invalid")
        return tuple(map(int, upper.shape))

    def identity_dict(self) -> dict[str, Any]:
        shape = self.validate()
        mask = np.asarray(self.available, dtype=bool)
        upper = np.ascontiguousarray(
            np.where(mask, np.asarray(self.upper, dtype=np.float64), 0.0),
            dtype="<f8",
        )
        available = np.ascontiguousarray(mask, dtype=np.uint8)
        return {
            "schema": PANEL_SCHEMA,
            "source_pcm_sha256": self.source_pcm_sha256,
            "shape": list(shape),
            "candidate_ids": list(self.candidate_ids),
            "metric_names": list(self.metric_names),
            "feature_contract_sha256": self.feature_contract_sha256,
            "feature_values_sha256": self.feature_values_sha256,
            "hard_feature_available_sha256": (
                self.hard_feature_available_sha256
            ),
            "model_config_sha256": self.model_config_sha256,
            "risk_model_state_sha256": self.risk_model_state_sha256,
            "query_encoder_sha256": self.query_encoder_sha256,
            "query_condition_sha256": self.query_condition_sha256,
            "quantile_levels": list(map(float, self.quantile_levels)),
            "selected_quantile_level": float(self.selected_quantile_level),
            "selected_quantile_index": int(self.selected_quantile_index),
            "calibration_policy_sha256": self.calibration_policy_sha256,
            "availability_probability_threshold": float(
                self.availability_probability_threshold
            ),
            "upper_sha256": "sha256:"
            + hashlib.sha256(upper.tobytes()).hexdigest(),
            "availability_sha256": "sha256:"
            + hashlib.sha256(available.tobytes()).hexdigest(),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def run_identity_bound_inference_v2(
    model,
    feature_batch: IdentityBoundFeatureBatchV2,
    *,
    query_embedding=None,
) -> RiskModelOutputV2:
    torch = _torch()
    config = model.config
    batch, candidates, _, time, frequency = feature_batch.validate(config)
    metrics = _tuple(config, "metric_names")
    levels = _validate_levels(_tuple(config, "quantile_levels"))
    query_encoder = getattr(config, "query_encoder_sha256", None)
    if query_encoder is None:
        if query_embedding is not None:
            raise RiskModelV2Error("query-disabled model received a query")
        query_ids: tuple[str | None, ...] = (None,) * batch
    else:
        _sha(query_encoder, "query_encoder_sha256")
        expected_query = (batch, int(getattr(config, "query_dim")))
        if not torch.is_tensor(query_embedding) or tuple(
            query_embedding.shape
        ) != expected_query:
            raise RiskModelV2Error("query embedding shape differs")
        if not bool(torch.isfinite(query_embedding).all()):
            raise ValueError("query embedding contains non-finite values")
        query_ids = tuple(
            query_condition_sha256_v2(
                query_embedding[index],
                query_encoder_sha256=query_encoder,
            )
            for index in range(batch)
        )
    before = model_state_sha256_v2(model)
    raw = model(
        feature_batch.values,
        feature_batch.hard_feature_available,
        query_embedding,
    )
    if model_state_sha256_v2(model) != before:
        raise RiskModelV2Error("model state changed during inference")
    quantiles = raw.quantiles.detach().clone()
    logits = raw.availability_logits.detach().clone()
    hard = raw.hard_feature_available.detach().clone()
    if tuple(quantiles.shape) != (
        batch,
        candidates,
        len(metrics),
        len(levels),
        time,
        frequency,
    ):
        raise RiskModelV2Error("model quantile geometry differs from config")
    result = RiskModelOutputV2(
        quantiles=quantiles,
        availability_logits=logits,
        hard_feature_available=hard,
        candidate_ids=feature_batch.candidate_ids,
        metric_names=metrics,
        feature_contract_sha256=feature_batch.feature_contract_sha256,
        feature_values_sha256=feature_batch.values_sha256,
        hard_feature_available_sha256=feature_batch.availability_sha256,
        model_config_sha256=config_sha256(config),
        risk_model_state_sha256=before,
        quantile_levels=levels,
        query_encoder_sha256=query_encoder,
        query_condition_sha256s=query_ids,
    )
    result.validate()
    return result


def output_to_risk_panel_payload_v2(
    model,
    output: RiskModelOutputV2,
    *,
    source_pcm_sha256: str,
    calibration_policy_sha256: str,
    batch_index: int = 0,
    selected_quantile_level: float = 0.95,
) -> RiskPanelPayloadV2:
    torch = _torch()
    output.validate()
    source = _sha(source_pcm_sha256, "source_pcm_sha256")
    policy = _sha(calibration_policy_sha256, "calibration_policy_sha256")
    config = model.config
    if config_sha256(config) != output.model_config_sha256:
        raise RiskModelV2Error(
            "model config differs from inference-time output"
        )
    if model_state_sha256_v2(model) != output.risk_model_state_sha256:
        raise RiskModelV2Error(
            "model state differs from inference-time output"
        )
    if _tuple(config, "candidate_ids") != output.candidate_ids:
        raise RiskModelV2Error(
            "candidate order differs from inference-time output"
        )
    if _tuple(config, "metric_names") != output.metric_names:
        raise RiskModelV2Error("metric order differs from inference-time output")
    if getattr(config, "feature_contract_sha256") != (
        output.feature_contract_sha256
    ):
        raise RiskModelV2Error(
            "feature contract differs from inference-time output"
        )
    if getattr(config, "query_encoder_sha256", None) != (
        output.query_encoder_sha256
    ):
        raise RiskModelV2Error(
            "query encoder differs from inference-time output"
        )
    if not 0 <= int(batch_index) < output.quantiles.shape[0]:
        raise ValueError("batch index is outside model output")
    levels = _validate_levels(output.quantile_levels)
    try:
        quantile_index = levels.index(float(selected_quantile_level))
    except ValueError as exc:
        raise ValueError(
            f"selected quantile {selected_quantile_level} is not in {levels}"
        ) from exc
    risk = output.quantiles[
        batch_index,
        :,
        :,
        quantile_index,
    ].detach().cpu().to(torch.float64)
    probability = torch.sigmoid(
        output.availability_logits[batch_index]
    ).detach().cpu()
    hard = output.hard_feature_available[batch_index].detach().cpu()
    threshold = float(getattr(config, "availability_probability_threshold"))
    available = hard & (probability >= threshold)
    panel = RiskPanelPayloadV2(
        source_pcm_sha256=source,
        upper=risk.permute(2, 3, 0, 1).contiguous().numpy(),
        available=available.permute(2, 3, 0, 1).contiguous().numpy(),
        candidate_ids=output.candidate_ids,
        metric_names=output.metric_names,
        feature_contract_sha256=output.feature_contract_sha256,
        feature_values_sha256=output.feature_values_sha256,
        hard_feature_available_sha256=(
            output.hard_feature_available_sha256
        ),
        model_config_sha256=output.model_config_sha256,
        risk_model_state_sha256=output.risk_model_state_sha256,
        query_encoder_sha256=output.query_encoder_sha256,
        query_condition_sha256=(
            output.query_condition_sha256s[batch_index]
        ),
        quantile_levels=levels,
        selected_quantile_level=float(selected_quantile_level),
        selected_quantile_index=quantile_index,
        calibration_policy_sha256=policy,
        availability_probability_threshold=threshold,
    )
    panel.validate()
    return panel
