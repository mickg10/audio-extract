import torch
import pytest

from audio_extract.classical_loss_v3 import (
    ClassicalResidualLossV3Config,
    classical_residual_loss_v3,
)


def _signals(dtype=torch.float32, frames=512, vocal_scale=1.0):
    t = torch.linspace(0, 1, frames, dtype=torch.float32)
    accompaniment = torch.stack((
        torch.sin(2 * torch.pi * 7 * t),
        0.7 * torch.sin(2 * torch.pi * 11 * t),
    ), 0).unsqueeze(0).to(dtype=dtype)
    vocal = (
        vocal_scale
        * torch.stack((
            0.4 * torch.cos(2 * torch.pi * 17 * t),
            0.3 * torch.cos(2 * torch.pi * 19 * t),
        ), 0).unsqueeze(0)
    ).to(dtype=dtype)
    return accompaniment, vocal, accompaniment + vocal


def _config(**kwargs):
    values = dict(
        stft_ffts=(64, 128),
        waveform_reference_floor=1e-3,
        stft_reference_floor=1e-3,
        eps=1e-10,
    )
    values.update(kwargs)
    return ClassicalResidualLossV3Config(**values)


def _run(dtype, *, config=None, perturb=0.0):
    accompaniment, vocal, mixture = _signals(dtype=dtype)
    estimate = (vocal + perturb).detach().clone().requires_grad_()
    loss, parts = classical_residual_loss_v3(
        mixture - estimate,
        estimate,
        mixture,
        accompaniment,
        vocal,
        no_vocal_vocal_estimate=torch.zeros_like(accompaniment),
        vocal_only_vocal_estimate=vocal,
        config=config or _config(),
    )
    loss.backward()
    return loss, parts, estimate


def test_perfect_estimate_is_near_zero():
    loss, parts, estimate = _run(torch.float32)
    assert float(loss) < 1e-5
    assert parts["analysis_dtype"] == "torch.float32"
    assert torch.isfinite(estimate.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_mixed_precision_forward_backward_is_supported(dtype):
    loss, parts, estimate = _run(dtype, perturb=0.01)
    assert parts["analysis_dtype"] == "torch.float32"
    assert torch.isfinite(loss)
    assert torch.isfinite(estimate.grad).all()
    assert float(estimate.grad.abs().max()) < 100.0


def test_float64_uses_float64_analysis():
    loss, parts, estimate = _run(torch.float64, perturb=0.01)
    assert parts["analysis_dtype"] == "torch.float64"
    assert loss.dtype == torch.float64
    assert torch.isfinite(estimate.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_matches_float32_on_same_represented_inputs(dtype):
    accompaniment, vocal, mixture = _signals(dtype=dtype)
    perturb = torch.full_like(vocal, 0.01)
    vocal_estimate = (vocal + perturb).detach().clone().requires_grad_()
    accompaniment_estimate = mixture - vocal_estimate
    config = _config(
        target_consistency_tolerance=3e-3,
        residual_consistency_tolerance=3e-3,
    )
    loss_low, _ = classical_residual_loss_v3(
        accompaniment_estimate, vocal_estimate, mixture,
        accompaniment, vocal,
        no_vocal_vocal_estimate=torch.zeros_like(accompaniment),
        vocal_only_vocal_estimate=vocal,
        config=config,
    )
    loss_low.backward()
    grad_low = vocal_estimate.grad.detach().float()

    a32, v32, m32 = accompaniment.float(), vocal.float(), mixture.float()
    vh32 = (vocal + perturb).float().detach().clone().requires_grad_()
    loss_32, _ = classical_residual_loss_v3(
        accompaniment_estimate.float(), vh32, m32, a32, v32,
        no_vocal_vocal_estimate=torch.zeros_like(a32),
        vocal_only_vocal_estimate=v32,
        config=config,
    )
    loss_32.backward()
    assert float(loss_low) == pytest.approx(float(loss_32), rel=3e-4, abs=3e-5)
    assert torch.allclose(grad_low, vh32.grad, rtol=2e-2, atol=2e-3)


def test_near_silent_target_has_finite_bounded_gradient():
    accompaniment, vocal, mixture = _signals(
        dtype=torch.float16, vocal_scale=1e-4
    )
    estimate = (
        0.02 * torch.sin(torch.linspace(
            0, 20, vocal.shape[-1], dtype=torch.float32
        )).repeat(1, 2, 1)
    ).to(dtype=torch.float16).requires_grad_()
    config = _config(
        residual_waveform=0.0,
        residual_complex_stft=1.0,
        no_vocal_false_positive=0.0,
        vocal_only_false_negative=0.0,
        source_coordinate=0.0,
        stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
        target_consistency_tolerance=3e-3,
        residual_consistency_tolerance=3e-3,
    )
    loss, _ = classical_residual_loss_v3(
        mixture - estimate, estimate, mixture, accompaniment, vocal,
        no_vocal_vocal_estimate=torch.zeros_like(accompaniment),
        vocal_only_vocal_estimate=vocal,
        config=config,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss) < 10.0
    assert torch.isfinite(estimate.grad).all()
    assert float(estimate.grad.abs().max()) < 10.0


def test_residual_identity_is_refused():
    accompaniment, vocal, mixture = _signals()
    with pytest.raises(ValueError, match="A_hat=M-V_hat"):
        classical_residual_loss_v3(
            accompaniment + 0.25, vocal, mixture, accompaniment, vocal,
            no_vocal_vocal_estimate=torch.zeros_like(accompaniment),
            vocal_only_vocal_estimate=vocal,
            config=_config(),
        )


def test_source_coordinate_uses_audibility_weighted_beta():
    accompaniment, vocal, mixture = _signals(vocal_scale=1e-3)
    estimate = (vocal * 2.0).detach().clone().requires_grad_()
    config = _config(
        residual_waveform=0.0,
        residual_complex_stft=0.0,
        no_vocal_false_positive=0.0,
        vocal_only_false_negative=0.0,
        source_coordinate=1.0,
        stereo_accompaniment=0.0,
        event_weighted_residual=0.0,
    )
    loss, parts = classical_residual_loss_v3(
        mixture - estimate, estimate, mixture, accompaniment, vocal,
        no_vocal_vocal_estimate=torch.zeros_like(accompaniment),
        vocal_only_vocal_estimate=vocal,
        config=config,
    )
    assert torch.isfinite(loss)
    assert float(parts["source_coordinate"]) < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_autocast_does_not_define_analysis_precision():
    device = torch.device("cuda")
    accompaniment, vocal, mixture = (
        value.to(device=device) for value in _signals()
    )
    estimate = vocal.detach().clone().requires_grad_()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, parts = classical_residual_loss_v3(
            mixture - estimate, estimate, mixture, accompaniment, vocal,
            no_vocal_vocal_estimate=torch.zeros_like(accompaniment),
            vocal_only_vocal_estimate=vocal,
            config=_config(),
        )
    loss.backward()
    assert parts["analysis_dtype"] == "torch.float32"
    assert torch.isfinite(estimate.grad).all()
