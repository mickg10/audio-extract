from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import torch

from audio_extract.counterfactual_risk_model_v2 import (
    IdentityBoundFeatureBatchV2,
    RiskModelV2Error,
    output_to_risk_panel_payload_v2,
    query_condition_sha256_v2,
    run_identity_bound_inference_v2,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


@dataclasses.dataclass(frozen=True)
class Config:
    candidate_ids: tuple[str, ...] = (sha("1"), sha("2"))
    metric_names: tuple[str, ...] = ("voice", "hole", "artifact", "hall")
    feature_contract_sha256: str = sha("3")
    feature_channels: int = 3
    query_encoder_sha256: str | None = sha("4")
    query_dim: int = 2
    quantile_levels: tuple[float, ...] = (0.5, 0.9, 0.95)
    availability_probability_threshold: float = 0.9

    def identity_dict(self):
        return {
            "candidate_ids": list(self.candidate_ids),
            "metric_names": list(self.metric_names),
            "feature_contract_sha256": self.feature_contract_sha256,
            "feature_channels": self.feature_channels,
            "query_encoder_sha256": self.query_encoder_sha256,
            "query_dim": self.query_dim,
            "quantile_levels": list(self.quantile_levels),
            "availability_probability_threshold": (
                self.availability_probability_threshold
            ),
        }


class FakeRiskModel(torch.nn.Module):
    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config()
        self.scale = torch.nn.Parameter(
            torch.tensor(0.25, dtype=torch.float64)
        )

    def forward(self, features, hard_feature_available, query_embedding=None):
        batch, candidates, _, time, frequency = features.shape
        metrics = len(self.config.metric_names)
        levels = len(self.config.quantile_levels)
        base = features.to(torch.float64).mean(dim=2, keepdim=True)
        base = base.expand(batch, candidates, metrics, time, frequency)
        if query_embedding is not None:
            query_term = query_embedding.to(torch.float64).sum(dim=1)
            base = base + query_term[:, None, None, None, None]
        base = torch.nn.functional.softplus(base * self.scale)
        quantiles = torch.stack(
            tuple(base + float(index) for index in range(levels)),
            dim=3,
        )
        logits = torch.full_like(base, 10.0)
        return SimpleNamespace(
            quantiles=quantiles,
            availability_logits=logits,
            hard_feature_available=hard_feature_available,
        )


def batch(
    *,
    candidate_ids: tuple[str, ...] | None = None,
    feature_contract_sha256: str | None = None,
):
    values = torch.arange(
        2 * 2 * 3 * 2 * 2,
        dtype=torch.float64,
    ).reshape(2, 2, 3, 2, 2) / 100.0
    available = torch.ones((2, 2, 4, 2, 2), dtype=torch.bool)
    return IdentityBoundFeatureBatchV2(
        values=values,
        hard_feature_available=available,
        candidate_ids=candidate_ids or Config().candidate_ids,
        feature_contract_sha256=(
            feature_contract_sha256 or Config().feature_contract_sha256
        ),
    )


def query():
    return torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
    )


def infer(model=None, feature_batch=None, query_embedding=None):
    return run_identity_bound_inference_v2(
        model or FakeRiskModel(),
        feature_batch or batch(),
        query_embedding=(
            query() if query_embedding is None else query_embedding
        ),
    )


def panel(model, output, **kwargs):
    options = dict(
        source_pcm_sha256=sha("5"),
        calibration_policy_sha256=sha("6"),
        batch_index=0,
        selected_quantile_level=0.95,
    )
    options.update(kwargs)
    return output_to_risk_panel_payload_v2(model, output, **options)


def test_candidate_order_is_bound_before_inference():
    model = FakeRiskModel()
    with pytest.raises(RiskModelV2Error, match="candidate identities/order"):
        run_identity_bound_inference_v2(
            model,
            batch(
                candidate_ids=tuple(reversed(model.config.candidate_ids))
            ),
            query_embedding=query(),
        )


def test_feature_contract_is_bound_before_inference():
    with pytest.raises(RiskModelV2Error, match="feature contract"):
        run_identity_bound_inference_v2(
            FakeRiskModel(),
            batch(feature_contract_sha256=sha("f")),
            query_embedding=query(),
        )


def test_output_carries_inference_time_model_and_query_provenance():
    model = FakeRiskModel()
    output = infer(model)
    assert output.risk_model_state_sha256.startswith("sha256:")
    assert output.model_config_sha256.startswith("sha256:")
    assert all(
        item and item.startswith("sha256:")
        for item in output.query_condition_sha256s
    )
    result = panel(model, output)
    assert result.query_condition_sha256 == (
        output.query_condition_sha256s[0]
    )
    assert "query_condition_sha256_value" not in inspect.signature(
        output_to_risk_panel_payload_v2
    ).parameters


def test_post_inference_weight_change_is_refused_at_panel_boundary():
    model = FakeRiskModel()
    output = infer(model)
    with torch.no_grad():
        model.scale.add_(1.0)
    with pytest.raises(RiskModelV2Error, match="model state differs"):
        panel(model, output)


def test_same_shaped_output_from_another_model_is_refused():
    first = FakeRiskModel()
    output = infer(first)
    second = FakeRiskModel()
    with torch.no_grad():
        second.scale.fill_(0.75)
    with pytest.raises(RiskModelV2Error, match="model state differs"):
        panel(second, output)


def test_query_identity_preserves_float64_precision():
    first = torch.tensor([1.0, 1.0 + 2.0**-40], dtype=torch.float64)
    second = torch.tensor([1.0, 1.0 + 2.0**-39], dtype=torch.float64)
    assert first.to(torch.float32).equal(second.to(torch.float32))
    first_sha = query_condition_sha256_v2(
        first,
        query_encoder_sha256=sha("4"),
    )
    second_sha = query_condition_sha256_v2(
        second,
        query_encoder_sha256=sha("4"),
    )
    assert first_sha != second_sha


def test_selected_quantile_and_calibration_policy_are_panel_identity():
    model = FakeRiskModel()
    output = infer(model)
    median = panel(model, output, selected_quantile_level=0.5)
    upper = panel(model, output, selected_quantile_level=0.95)
    other_policy = panel(
        model,
        output,
        selected_quantile_level=0.95,
        calibration_policy_sha256=sha("7"),
    )
    assert median.selected_quantile_index == 0
    assert upper.selected_quantile_index == 2
    assert median.selected_quantile_level == 0.5
    assert upper.selected_quantile_level == 0.95
    assert median.sha256 != upper.sha256
    assert upper.sha256 != other_policy.sha256


def test_feature_payload_and_hard_availability_are_identity_bearing():
    model = FakeRiskModel()
    first_batch = batch()
    first = infer(model, first_batch)
    changed_values = first_batch.values.clone()
    changed_values[0, 0, 0, 0, 0] += 1.0
    second = infer(
        model,
        dataclasses.replace(first_batch, values=changed_values),
    )
    assert first.feature_values_sha256 != second.feature_values_sha256

    changed_available = first_batch.hard_feature_available.clone()
    changed_available[0, 0, 0, 0, 0] = False
    third = infer(
        model,
        dataclasses.replace(
            first_batch,
            hard_feature_available=changed_available,
        ),
    )
    assert first.hard_feature_available_sha256 != (
        third.hard_feature_available_sha256
    )


def test_output_tensors_are_cloned_from_model_return_values():
    model = FakeRiskModel()
    feature_batch = batch()
    output = infer(model, feature_batch)
    original = output.quantiles.clone()
    feature_batch.values.add_(1000.0)
    assert torch.equal(output.quantiles, original)


def test_query_disabled_contract_refuses_query_and_emits_none_identity():
    config = dataclasses.replace(
        Config(),
        query_encoder_sha256=None,
        query_dim=0,
    )
    model = FakeRiskModel(config)
    with pytest.raises(RiskModelV2Error, match="query-disabled"):
        run_identity_bound_inference_v2(
            model,
            batch(),
            query_embedding=query(),
        )
    output = run_identity_bound_inference_v2(model, batch())
    result = panel(model, output)
    assert result.query_encoder_sha256 is None
    assert result.query_condition_sha256 is None


def test_unknown_quantile_is_refused():
    model = FakeRiskModel()
    output = infer(model)
    with pytest.raises(ValueError, match="not in"):
        panel(model, output, selected_quantile_level=0.8)


def test_panel_recomputes_selected_quantile_level_from_index():
    model = FakeRiskModel()
    output = infer(model)
    value = panel(model, output)
    forged = dataclasses.replace(value, selected_quantile_index=0)
    with pytest.raises(RiskModelV2Error, match="inconsistent"):
        forged.validate()
