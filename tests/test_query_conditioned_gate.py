import pytest

torch = pytest.importorskip("torch")

from audio_extract.query_conditioned_gate import (
    QueryGateConfig,
    build_query_conditioned_gate,
    route_regularization,
)


def gate():
    torch.manual_seed(2)
    return build_query_conditioned_gate(QueryGateConfig(
        feature_channels=5,
        query_dim=8,
        candidates=3,
        hidden_channels=16,
        blocks=2,
        conservative_index=1,
    ))


def inputs(batch=2, time=4, frequency=6):
    return (
        torch.randn(batch, 5, time, frequency),
        torch.randn(batch, 8),
    )


def test_step_zero_weights_are_exact_parent():
    model = gate()
    features, query = inputs()
    weights, _ = model(features, query)
    assert torch.equal(weights[:, 1], torch.ones_like(weights[:, 1]))
    assert torch.equal(weights[:, 0], torch.zeros_like(weights[:, 0]))
    assert torch.equal(weights[:, 2], torch.zeros_like(weights[:, 2]))


def test_step_zero_complex_output_is_bit_exact_parent():
    model = gate()
    features, query = inputs(batch=1, time=3, frequency=5)
    weights, _ = model(features, query)
    candidates = torch.randn(1, 3, 2, 5, 3, dtype=torch.complex64)
    result = model.combine_complex_candidates(candidates, weights)
    assert torch.equal(result, candidates[:, 1])


def test_positive_scale_stays_on_simplex_and_uses_shared_stereo_route():
    model = gate()
    model.set_route_scale(0.5)
    features, query = inputs(batch=1, time=3, frequency=5)
    weights, _ = model(features, query)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]))

    candidates = torch.zeros(1, 3, 2, 5, 3, dtype=torch.complex64)
    candidates[:, 0, 0] = 1; candidates[:, 0, 1] = 2
    candidates[:, 1, 0] = 3; candidates[:, 1, 1] = 6
    candidates[:, 2, 0] = 5; candidates[:, 2, 1] = 10
    result = model.combine_complex_candidates(candidates, weights)
    assert torch.allclose(result[:, 1], 2 * result[:, 0])


def test_query_can_change_route_after_head_is_unfrozen():
    model = gate()
    model.set_route_scale(1.0)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)
    features = torch.zeros(2, 5, 4, 6)
    query = torch.stack((torch.zeros(8), torch.ones(8)))
    weights, _ = model(features, query)
    assert not torch.allclose(weights[0], weights[1])


def test_only_gate_parameters_receive_gradients_from_frozen_candidates():
    model = gate()
    model.set_route_scale(1.0)
    features, query = inputs(batch=1, time=3, frequency=5)
    candidates = torch.randn(
        1, 3, 2, 5, 3, dtype=torch.complex64, requires_grad=False
    )
    weights, _ = model(features, query)
    loss = model.combine_complex_candidates(candidates, weights).abs().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert candidates.grad is None


def test_route_regularization_is_zero_for_parent_and_positive_for_switching():
    parent = torch.zeros(1, 3, 4, 5)
    parent[:, 1] = 1
    clean = route_regularization(parent, conservative_index=1)
    assert float(clean["total"]) == pytest.approx(0.0)

    switched = parent.clone()
    switched[:, :, 2:, 2:] = 0
    switched[:, 2, 2:, 2:] = 1
    changed = route_regularization(switched, conservative_index=1)
    assert float(changed["correction"]) > 0
    assert float(changed["temporal_tv"]) > 0
    assert float(changed["frequency_tv"]) > 0


def test_bad_shapes_and_route_scale_are_refused():
    model = gate()
    features, query = inputs()
    with pytest.raises(ValueError):
        model.set_route_scale(1.1)
    with pytest.raises(ValueError):
        model(features[:, :4], query)
    weights, _ = model(features, query)
    with pytest.raises(ValueError):
        model.combine_complex_candidates(
            torch.randn(2, 2, 2, 6, 4, dtype=torch.complex64), weights
        )
