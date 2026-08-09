import torch
import pytest

from audio_extract.classical_loss_v3 import (
    ClassicalResidualLossV3Config,
    classical_residual_loss_v3,
)


def _cfg(**kwargs):
    values = dict(
        stft_ffts=(64, 128),
        waveform_reference_floor=1e-5,
        stft_reference_floor=1e-5,
        eps=1e-10,
    )
    values.update(kwargs)
    return ClassicalResidualLossV3Config(**values)


def _sources(frames=512, dtype=torch.float32):
    generator = torch.Generator().manual_seed(1947)
    accompaniment = torch.randn(
        1, 2, frames, generator=generator, dtype=torch.float32
    )
    vocal = 0.3 * torch.randn(
        1, 2, frames, generator=generator, dtype=torch.float32
    )
    return (
        accompaniment.to(dtype),
        vocal.to(dtype),
        (accompaniment + vocal).to(dtype),
    )


def _loss(
    a_hat, v_hat, mixture, a, v, *, config, no_vocal=None, vocal_only=None
):
    return classical_residual_loss_v3(
        a_hat,
        v_hat,
        mixture,
        a,
        v,
        no_vocal_vocal_estimate=(
            torch.zeros_like(a) if no_vocal is None else no_vocal
        ),
        vocal_only_vocal_estimate=(v if vocal_only is None else vocal_only),
        config=config,
    )


def test_residual_construction_has_one_independent_source_error():
    a, v, mixture = _sources(dtype=torch.float64)
    error = 0.02 * torch.sin(
        torch.linspace(0, 8, mixture.shape[-1], dtype=torch.float64)
    )
    error = error.repeat(1, 2, 1)
    v_hat = v + error
    a_hat = mixture - v_hat
    assert torch.equal(a_hat - a, -(v_hat - v))

    config = _cfg(
        residual_waveform=1.0,
        residual_complex_stft=0.0,
        no_vocal_false_positive=0.0,
        vocal_only_false_negative=0.0,
        source_coordinate=0.0,
        stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, parts = _loss(a_hat, v_hat, mixture, a, v, config=config)
    expected = error.abs().mean() / mixture.abs().mean()
    assert float(parts["residual_waveform"]) == pytest.approx(
        float(expected), rel=1e-10
    )
    assert float(loss) == pytest.approx(float(expected), rel=1e-10)


def test_normalized_residual_loss_is_gain_invariant_above_floor():
    a, v, mixture = _sources(dtype=torch.float64)
    error = 0.015 * torch.cos(
        torch.linspace(0, 10, mixture.shape[-1], dtype=torch.float64)
    )
    error = error.repeat(1, 2, 1)
    config = _cfg(
        residual_waveform=1.0,
        residual_complex_stft=0.7,
        no_vocal_false_positive=0.0,
        vocal_only_false_negative=0.0,
        source_coordinate=0.0,
        stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    values = []
    for gain in (0.25, 1.0, 4.0):
        ag, vg, mg, eg = a * gain, v * gain, mixture * gain, error * gain
        loss, _ = _loss(
            mg - (vg + eg), vg + eg, mg, ag, vg, config=config
        )
        values.append(float(loss))
    assert values[0] == pytest.approx(values[1], rel=2e-9, abs=2e-9)
    assert values[2] == pytest.approx(values[1], rel=2e-9, abs=2e-9)


def test_no_vocal_control_gradient_moves_estimate_toward_zero():
    a, v, mixture = _sources()
    estimate = torch.full_like(a, 0.1, requires_grad=True)
    config = _cfg(
        residual_waveform=0.0,
        residual_complex_stft=0.0,
        no_vocal_false_positive=1.0,
        vocal_only_false_negative=0.0,
        source_coordinate=0.0,
        stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, _ = _loss(
        a, v, mixture, a, v, config=config, no_vocal=estimate
    )
    loss.backward()
    assert torch.all(estimate.grad > 0)


def test_vocal_only_control_gradient_moves_estimate_toward_target():
    a, v, _ = _sources()
    v = v.abs() + 0.05
    mixture = a + v
    estimate = torch.zeros_like(v, requires_grad=True)
    config = _cfg(
        residual_waveform=0.0,
        residual_complex_stft=0.0,
        no_vocal_false_positive=0.0,
        vocal_only_false_negative=1.0,
        source_coordinate=0.0,
        stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, _ = _loss(
        a, v, mixture, a, v, config=config, vocal_only=estimate
    )
    loss.backward()
    assert torch.all(estimate.grad < 0)


def test_source_coordinate_beta_is_weighted_by_audibility():
    a, v, _ = _sources(dtype=torch.float64)
    config = _cfg(
        residual_waveform=0.0,
        residual_complex_stft=0.0,
        no_vocal_false_positive=0.0,
        vocal_only_false_negative=0.0,
        source_coordinate=1.0,
        stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    losses = []
    for scale in (0.01, 1.0):
        target = v * scale
        mixture = a + target
        # Same coefficient beta=0.5 in the accompaniment estimate.
        accompaniment_estimate = a + 0.5 * target
        vocal_estimate = mixture - accompaniment_estimate
        _, parts = _loss(
            accompaniment_estimate,
            vocal_estimate,
            mixture,
            a,
            target,
            config=config,
        )
        losses.append(float(parts["source_coordinate"]))
    assert losses[1] > losses[0] * 10.0


def test_analysis_policy_is_explicit_in_components():
    for dtype, expected in (
        (torch.float16, "torch.float32"),
        (torch.bfloat16, "torch.float32"),
        (torch.float32, "torch.float32"),
        (torch.float64, "torch.float64"),
    ):
        a, v, mixture = _sources(dtype=dtype)
        loss, parts = _loss(a, v, mixture, a, v, config=_cfg())
        assert parts["analysis_dtype"] == expected
        assert parts["analysis_dtype_policy"] == (
            "float64_if_any_input_float64_else_float32"
        )
        assert torch.isfinite(loss)
