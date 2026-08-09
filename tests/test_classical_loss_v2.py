import pytest

torch = pytest.importorskip("torch")

from audio_extract.classical_loss_v2 import (
    ClassicalResidualLossConfig,
    _complex_stft_log_ratio,
    _require_residual_construction,
    classical_residual_loss_v2,
)


def sources(frames=512, vocal_scale=1.0):
    t = torch.linspace(0, 1, frames)
    a = torch.stack((torch.sin(2 * torch.pi * 7 * t),
                     0.7 * torch.sin(2 * torch.pi * 11 * t)), 0).unsqueeze(0)
    v = vocal_scale * torch.stack((0.4 * torch.cos(2 * torch.pi * 17 * t),
                                   0.3 * torch.cos(2 * torch.pi * 19 * t)), 0).unsqueeze(0)
    return a, v, a + v


def cfg(**kwargs):
    base = dict(stft_ffts=(64, 128), waveform_reference_floor=1e-3,
                stft_reference_floor=1e-3, eps=1e-10)
    base.update(kwargs)
    return ClassicalResidualLossConfig(**base)


def test_perfect_estimate_and_controls_have_near_zero_loss():
    a, v, m = sources()
    loss, parts = classical_residual_loss_v2(
        a, v, m, a, v,
        no_vocal_vocal_estimate=torch.zeros_like(a),
        vocal_only_vocal_estimate=v,
        config=cfg(),
    )
    assert float(loss) < 1e-5
    assert int(parts["source_coordinate_valid_items"]) == 1


def test_primary_residual_waveform_is_scored_once_not_twice():
    a, v, m = sources()
    error = 0.05 * torch.ones_like(v)
    vh = v + error
    ah = m - vh
    config = cfg(
        residual_waveform=1.0, residual_complex_stft=0.0,
        no_vocal_false_positive=0.0, vocal_only_false_negative=0.0,
        source_coordinate=0.0, stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, parts = classical_residual_loss_v2(
        ah, vh, m, a, v,
        no_vocal_vocal_estimate=torch.zeros_like(a),
        vocal_only_vocal_estimate=v,
        config=config,
    )
    expected = error.abs().mean() / m.abs().mean().clamp_min(
        config.waveform_reference_floor
    )
    assert float(parts["residual_waveform"]) == pytest.approx(
        float(expected), rel=1e-6
    )
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)


def test_near_silent_vocal_target_cannot_blow_up_spectral_gradient():
    a, v, m = sources(vocal_scale=1e-9)
    vh = (0.02 * torch.sin(torch.linspace(0, 20, v.shape[-1]))
          .repeat(1, 2, 1)).requires_grad_()
    ah = m - vh
    config = cfg(
        residual_waveform=0.0, residual_complex_stft=1.0,
        no_vocal_false_positive=0.0, vocal_only_false_negative=0.0,
        source_coordinate=0.0, stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, _ = classical_residual_loss_v2(
        ah, vh, m, a, v,
        no_vocal_vocal_estimate=torch.zeros_like(a),
        vocal_only_vocal_estimate=v,
        config=config,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss.detach()) < 10.0
    assert bool(torch.isfinite(vh.grad).all())
    assert float(vh.grad.abs().max()) < 10.0


def test_exact_silence_uses_absolute_floor_and_stays_finite():
    frames = 512
    m = torch.zeros(1, 2, frames)
    a = torch.zeros_like(m)
    v = torch.zeros_like(m)
    vh = (0.02 * torch.sin(torch.linspace(0, 20, frames))
          .repeat(1, 2, 1)).requires_grad_()
    ah = m - vh
    config = cfg(
        residual_waveform=0.0, residual_complex_stft=1.0,
        no_vocal_false_positive=0.0, vocal_only_false_negative=0.0,
        source_coordinate=0.0, stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, _ = classical_residual_loss_v2(
        ah, vh, m, a, v,
        no_vocal_vocal_estimate=torch.zeros_like(a),
        vocal_only_vocal_estimate=v,
        config=config,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss.detach()) < 10.0
    assert bool(torch.isfinite(vh.grad).all())
    assert float(vh.grad.abs().max()) < 10.0


def test_low_precision_residual_roundoff_is_accepted_but_real_mismatch_is_not():
    mixture = torch.tensor([[[1.0, -0.75, 0.125, 4.0]]], dtype=torch.float16)
    vocal = torch.tensor([[[0.8, -0.3, 0.75, -3.0]]], dtype=torch.float16)
    accompaniment = mixture - vocal
    _require_residual_construction(
        accompaniment, vocal, mixture,
        relative_tolerance=1e-6, roundoff_ulps=8.0,
    )
    with pytest.raises(ValueError, match="mixture-residual identity"):
        _require_residual_construction(
            accompaniment + torch.tensor(0.25, dtype=torch.float16),
            vocal, mixture,
            relative_tolerance=1e-6, roundoff_ulps=8.0,
        )


def test_stft_sizes_that_would_produce_zero_hop_are_rejected():
    signal = torch.zeros(1, 1, 16)
    with pytest.raises(ValueError, match="positive hop"):
        _complex_stft_log_ratio(signal, signal, (2, 3), 1e-3)


def test_residual_identity_mismatch_is_refused():
    a, v, m = sources()
    with pytest.raises(ValueError, match="mixture-residual identity"):
        classical_residual_loss_v2(
            a + 0.1, v, m, a, v,
            no_vocal_vocal_estimate=torch.zeros_like(a),
            vocal_only_vocal_estimate=v,
            config=cfg(),
        )


def test_target_identity_mismatch_is_refused():
    a, v, m = sources()
    with pytest.raises(ValueError, match="exact target identity"):
        classical_residual_loss_v2(
            a, v, m + 0.1, a, v,
            no_vocal_vocal_estimate=torch.zeros_like(a),
            vocal_only_vocal_estimate=v,
            config=cfg(),
        )


def test_controls_are_independent_and_required_by_signature():
    a, v, m = sources()
    no_vocal_error = 0.1 * a
    vocal_only_error = -0.2 * v
    config = cfg(
        residual_waveform=0.0, residual_complex_stft=0.0,
        no_vocal_false_positive=1.0, vocal_only_false_negative=1.0,
        source_coordinate=0.0, stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, parts = classical_residual_loss_v2(
        a, v, m, a, v,
        no_vocal_vocal_estimate=no_vocal_error,
        vocal_only_vocal_estimate=v + vocal_only_error,
        config=config,
    )
    expected_no_vocal = no_vocal_error.abs().mean() / a.abs().mean()
    expected_vocal_only = vocal_only_error.abs().mean() / v.abs().mean()
    assert float(parts["no_vocal_false_positive"]) == pytest.approx(
        float(expected_no_vocal), rel=1e-6
    )
    assert float(parts["vocal_only_false_negative"]) == pytest.approx(
        float(expected_vocal_only), rel=1e-6
    )
    assert float(loss) == pytest.approx(
        float(expected_no_vocal + expected_vocal_only), rel=1e-6
    )


def test_event_weight_is_finite_nonnegative_and_scored_once():
    a, v, m = sources()
    error = torch.zeros_like(v)
    error[..., v.shape[-1] // 2:] = 0.1
    vh = v + error
    ah = m - vh
    weights = torch.ones(1, 1, v.shape[-1])
    weights[..., v.shape[-1] // 2:] = 3.0
    config = cfg(
        residual_waveform=0.0, residual_complex_stft=0.0,
        no_vocal_false_positive=0.0, vocal_only_false_negative=0.0,
        source_coordinate=0.0, stereo_accompaniment=0.0,
        event_weighted_residual=1.0,
    )
    loss, parts = classical_residual_loss_v2(
        ah, vh, m, a, v,
        no_vocal_vocal_estimate=torch.zeros_like(a),
        vocal_only_vocal_estimate=v,
        event_weights=weights,
        config=config,
    )
    assert float(loss) == pytest.approx(
        float(parts["event_weighted_residual"]), rel=1e-6
    )
    with pytest.raises(ValueError, match="non-negative"):
        classical_residual_loss_v2(
            ah, vh, m, a, v,
            no_vocal_vocal_estimate=torch.zeros_like(a),
            vocal_only_vocal_estimate=v,
            event_weights=-weights,
            config=config,
        )
