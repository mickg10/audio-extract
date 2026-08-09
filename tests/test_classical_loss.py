import pytest

torch = pytest.importorskip("torch")

from audio_extract.classical_loss import ClassicalLossConfig, classical_separation_loss


CFG = ClassicalLossConfig(stft_ffts=(64, 128), eps=1e-10)


def sources(frames=512):
    t = torch.linspace(0, 1, frames)
    a = torch.stack((torch.sin(2 * torch.pi * 7 * t),
                     0.7 * torch.sin(2 * torch.pi * 11 * t)), 0).unsqueeze(0)
    v = torch.stack((0.4 * torch.cos(2 * torch.pi * 17 * t),
                     0.3 * torch.cos(2 * torch.pi * 19 * t)), 0).unsqueeze(0)
    return a, v, a + v


def test_perfect_estimate_has_near_zero_loss_and_two_source_coordinate_inputs():
    a, v, m = sources()
    loss, parts = classical_separation_loss(a, v, m, a, v, config=CFG)
    assert float(loss) < 1e-4
    assert int(parts["source_coordinate_valid_items"]) == 1


def test_retained_voice_in_accompaniment_increases_source_coordinate_loss():
    a, v, m = sources()
    _, clean = classical_separation_loss(a, v, m, a, v, config=CFG)
    _, leaked = classical_separation_loss(a + 0.5 * v, 0.5 * v, m, a, v, config=CFG)
    assert float(leaked["source_coordinate"]) > float(clean["source_coordinate"]) + 0.4


def test_no_vocal_and_vocal_only_controls_are_supervised():
    a, v, m = sources()
    _, parts = classical_separation_loss(
        a, v, m, a, v,
        no_vocal_accompaniment_estimate=a,
        no_vocal_vocal_estimate=torch.zeros_like(a),
        vocal_only_accompaniment_estimate=torch.zeros_like(v),
        vocal_only_vocal_estimate=v,
        config=CFG,
    )
    assert float(parts["no_vocal_false_positive"]) == pytest.approx(0.0)
    assert float(parts["vocal_only_false_negative"]) == pytest.approx(0.0)


def test_event_weights_must_be_finite_nonnegative_and_broadcastable():
    a, v, m = sources()
    with pytest.raises(ValueError, match="non-negative"):
        classical_separation_loss(a, v, m, a, v, event_weights=-torch.ones(1, 1, 512), config=CFG)
    with pytest.raises(ValueError, match="broadcast"):
        classical_separation_loss(a, v, m, a, v, event_weights=torch.ones(2, 5), config=CFG)


def test_shape_mismatch_is_refused_instead_of_truncated():
    a, v, m = sources()
    with pytest.raises(ValueError, match="exact sample grid"):
        classical_separation_loss(a[..., :-1], v, m, a, v, config=CFG)


def test_loss_backpropagates_with_finite_gradients():
    a, v, m = sources()
    ah = (0.9 * a + 0.1 * v).clone().requires_grad_()
    vh = (m - ah.detach()).clone().requires_grad_()
    weights = torch.ones(1, 1, a.shape[-1])
    weights[..., a.shape[-1] // 2:] = 2
    loss, _ = classical_separation_loss(ah, vh, m, a, v, event_weights=weights, config=CFG)
    loss.backward()
    assert bool(torch.isfinite(ah.grad).all())
    assert bool(torch.isfinite(vh.grad).all())
