import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audio_extract.counterfactual_risk_model import (
    RiskModelConfig,
    RiskTrainingLossConfig,
    build_counterfactual_risk_model,
    counterfactual_risk_training_loss,
    model_state_sha256,
    output_to_risk_panel,
    query_condition_sha256,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def config(*, query: bool = True, availability_threshold: float = 0.4):
    return RiskModelConfig(
        candidate_ids=(sha("1"), sha("2"), sha("3")),
        metric_names=("voice", "hole", "artifact"),
        feature_contract_sha256=sha("4"),
        feature_channels=5,
        query_encoder_sha256=sha("5") if query else None,
        query_dim=7 if query else 0,
        query_projection_dim=6 if query else 0,
        quantile_levels=(0.5, 0.9, 0.95),
        hidden_channels=16,
        blocks=2,
        candidate_embedding_dim=4,
        availability_probability_threshold=availability_threshold,
        initial_risk_logit=-4.0,
        initial_availability_probability=0.5,
    )


def inputs(*, query: bool = True, batch: int = 2, time: int = 4, freq: int = 6):
    torch.manual_seed(101)
    features = torch.randn(batch, 3, 5, time, freq, requires_grad=True)
    available = torch.ones(batch, 3, 3, time, freq, dtype=torch.bool)
    embedding = (
        torch.randn(batch, 7, requires_grad=True) if query else None
    )
    return features, available, embedding


def test_query_free_default_contract_is_valid():
    value = RiskModelConfig(
        candidate_ids=(sha("1"), sha("2")),
        metric_names=("voice",),
        feature_contract_sha256=sha("3"),
        feature_channels=4,
    )
    value.validate()
    assert value.query_dim == 0
    assert value.query_projection_dim == 0


def test_forward_shapes_nonnegative_and_monotone_quantiles():
    model = build_counterfactual_risk_model(config())
    features, available, query = inputs()
    result = model(features, available, query)
    assert result.quantiles.shape == (2, 3, 3, 3, 4, 6)
    assert result.availability_logits.shape == (2, 3, 3, 4, 6)
    assert result.hard_feature_available.shape == (2, 3, 3, 4, 6)
    assert torch.all(result.quantiles >= 0)
    assert torch.all(result.quantiles[:, :, :, 1:] >= result.quantiles[:, :, :, :-1])
    assert torch.all(result.quantiles[:, :, :, 2] > result.quantiles[:, :, :, 0])


def test_frozen_feature_and_query_inputs_receive_no_gradient():
    model = build_counterfactual_risk_model(config())
    features, available, query = inputs(batch=1)
    result = model(features, available, query)
    loss = result.quantiles.mean() + result.availability_logits.mean()
    loss.backward()
    assert features.grad is None
    assert query.grad is None
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(
        gradient is not None and torch.isfinite(gradient).all()
        and float(gradient.abs().sum()) > 0
        for gradient in gradients
    )


def test_query_presence_and_feature_contract_are_strict():
    query_model = build_counterfactual_risk_model(config(query=True))
    features, available, query = inputs(query=True)
    with pytest.raises(ValueError, match="requires a query"):
        query_model(features, available, None)
    with pytest.raises(ValueError, match="query embedding shape"):
        query_model(features, available, query[:, :-1])
    with pytest.raises(ValueError, match="candidate order/count"):
        query_model(features[:, :2], available[:, :2], query)
    with pytest.raises(ValueError, match="feature channel"):
        query_model(features[:, :, :4], available, query)

    plain_model = build_counterfactual_risk_model(config(query=False))
    plain_features, plain_available, _ = inputs(query=False)
    plain_model(plain_features, plain_available, None)
    with pytest.raises(ValueError, match="query-disabled"):
        plain_model(plain_features, plain_available, torch.randn(2, 1))


def test_query_can_change_predictions_after_head_is_nonzero():
    model = build_counterfactual_risk_model(config())
    with torch.no_grad():
        model.quantile_head.weight.normal_(0, 0.05)
        model.availability_head.weight.normal_(0, 0.05)
    features = torch.zeros(2, 3, 5, 3, 4)
    available = torch.ones(2, 3, 3, 3, 4, dtype=torch.bool)
    query = torch.stack((torch.zeros(7), torch.ones(7)))
    output = model(features, available, query)
    assert not torch.allclose(output.quantiles[0], output.quantiles[1])
    assert not torch.allclose(
        output.availability_logits[0], output.availability_logits[1]
    )


def test_model_state_hash_is_deterministic_and_parameter_sensitive():
    torch.manual_seed(12)
    first = build_counterfactual_risk_model(config())
    torch.manual_seed(12)
    second = build_counterfactual_risk_model(config())
    assert model_state_sha256(first) == model_state_sha256(second)
    with torch.no_grad():
        second.candidate_embedding[0, 0] += 0.001
    assert model_state_sha256(first) != model_state_sha256(second)


def test_query_condition_hash_is_encoder_and_value_sensitive():
    query = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    first = query_condition_sha256(query, query_encoder_sha256=sha("1"))
    repeat = query_condition_sha256(query.clone(), query_encoder_sha256=sha("1"))
    changed_value = query_condition_sha256(
        query + 0.01, query_encoder_sha256=sha("1")
    )
    changed_encoder = query_condition_sha256(
        query, query_encoder_sha256=sha("2")
    )
    assert first == repeat
    assert first != changed_value
    assert first != changed_encoder


def test_output_conversion_binds_source_model_query_and_axes():
    model = build_counterfactual_risk_model(config())
    features, available, query = inputs(batch=1, time=2, freq=3)
    output = model(features, available, query)
    condition = query_condition_sha256(
        query[0], query_encoder_sha256=model.config.query_encoder_sha256
    )
    panel = output_to_risk_panel(
        model,
        output,
        source_pcm_sha256=sha("9"),
        batch_index=0,
        upper_quantile=0.95,
        query_condition_sha256_value=condition,
    )
    assert panel.upper.shape == (2, 3, 3, 3)
    assert panel.available.shape == panel.upper.shape
    assert panel.identity.source_pcm_sha256 == sha("9")
    assert panel.identity.candidate_ids == model.config.candidate_ids
    assert panel.identity.metric_names == model.config.metric_names
    assert panel.identity.query_condition_sha256 == condition
    assert panel.identity.risk_model_sha256 == model_state_sha256(model)
    panel.validate()


def test_hard_input_availability_cannot_be_overridden_by_model_confidence():
    model = build_counterfactual_risk_model(config(availability_threshold=0.1))
    features, available, query = inputs(batch=1, time=2, freq=2)
    available[:, 1, 2, 0, 1] = False
    with torch.no_grad():
        model.availability_head.bias.fill_(20.0)
    output = model(features, available, query)
    condition = query_condition_sha256(
        query[0], query_encoder_sha256=model.config.query_encoder_sha256
    )
    panel = output_to_risk_panel(
        model,
        output,
        source_pcm_sha256=sha("8"),
        query_condition_sha256_value=condition,
    )
    assert not panel.available[0, 1, 1, 2]


def test_training_loss_is_finite_and_backpropagates_quantiles_and_availability():
    model = build_counterfactual_risk_model(config())
    features, available, query = inputs(batch=1, time=2, freq=3)
    output = model(features, available, query)
    torch.manual_seed(55)
    target = torch.rand(1, 3, 3, 2, 3)
    target_available = torch.ones_like(target, dtype=torch.bool)
    loss, parts = counterfactual_risk_training_loss(
        output,
        target,
        target_available,
        model.config.quantile_levels,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["available_targets"] == target.numel()
    assert float(parts["quantile"]) > 0
    assert float(parts["availability"]) > 0
    assert model.quantile_head.bias.grad is not None
    assert model.availability_head.bias.grad is not None


def test_unavailable_nan_targets_are_masked_and_zero_target_batch_is_connected():
    model = build_counterfactual_risk_model(config())
    features, available, query = inputs(batch=1, time=2, freq=2)
    output = model(features, available, query)
    target = torch.full((1, 3, 3, 2, 2), float("nan"))
    target_available = torch.zeros_like(target, dtype=torch.bool)
    loss, parts = counterfactual_risk_training_loss(
        output,
        target,
        target_available,
        model.config.quantile_levels,
        config=RiskTrainingLossConfig(
            quantile_weight=1.0,
            availability_weight=0.0,
        ),
    )
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.0)
    assert float(parts["quantile"].detach()) == pytest.approx(0.0)
    assert parts["available_targets"] == 0
    assert model.quantile_head.bias.grad is not None
    assert torch.all(model.quantile_head.bias.grad == 0)


def test_available_negative_or_nonfinite_training_targets_are_refused():
    model = build_counterfactual_risk_model(config())
    features, available, query = inputs(batch=1, time=1, freq=1)
    output = model(features, available, query)
    target = torch.zeros(1, 3, 3, 1, 1)
    target_available = torch.ones_like(target, dtype=torch.bool)
    target[0, 0, 0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="finite/non-negative"):
        counterfactual_risk_training_loss(
            output, target, target_available, model.config.quantile_levels
        )
    target[0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite/non-negative"):
        counterfactual_risk_training_loss(
            output, target, target_available, model.config.quantile_levels
        )


def test_panel_risks_match_requested_quantile_and_axis_permutation():
    model = build_counterfactual_risk_model(config())
    features, available, query = inputs(batch=1, time=2, freq=3)
    output = model(features, available, query)
    condition = query_condition_sha256(
        query[0], query_encoder_sha256=model.config.query_encoder_sha256
    )
    panel = output_to_risk_panel(
        model,
        output,
        source_pcm_sha256=sha("7"),
        upper_quantile=0.9,
        query_condition_sha256_value=condition,
    )
    expected = (
        output.quantiles[0, :, :, 1]
        .detach()
        .to(dtype=torch.float64)
        .permute(2, 3, 0, 1)
        .numpy()
    )
    assert np.array_equal(panel.upper, expected)
