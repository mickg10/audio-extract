import pytest

torch = pytest.importorskip("torch")

from audio_extract.classical_gate import (
    SmoothGateConfig,
    SmoothResidualGate,
    gate_bundle_identity,
    gate_regularization,
)


def _audio(frames=4096):
    torch.manual_seed(13)
    mixture = 0.1 * torch.randn(1, 2, frames)
    conservative = 0.02 * torch.randn(1, 2, frames)
    aggressive = conservative + 0.01 * torch.randn(1, 2, frames)
    return mixture, conservative, aggressive


def _config(**kwargs):
    values = dict(
        n_fft=128, hop_length=32, tile_seconds=0.01,
        band_edges_hz=(0, 2_000, 8_000, 22_050), hidden_channels=3,
    )
    values.update(kwargs)
    return SmoothGateConfig(**values)


def test_step_zero_decoded_output_is_exact_conservative_parent():
    mixture, conservative, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    output, diagnostics = gate(mixture, conservative, aggressive)
    assert torch.equal(output, conservative)
    assert torch.count_nonzero(diagnostics["correction"]) == 0
    assert torch.count_nonzero(diagnostics["coarse_gate"]) == 0
    assert float(diagnostics["amplitude"].detach()) == 0.0


def test_gate_is_bounded_stereo_coherent_and_trainable_from_zero():
    mixture, conservative, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    output, diagnostics = gate(mixture, conservative, aggressive)
    target = aggressive
    loss = (output - target).square().mean()
    loss.backward()
    assert gate.correction_amplitude.grad is not None
    assert float(gate.correction_amplitude.grad) < 0.0

    with torch.no_grad():
        gate.correction_amplitude.fill_(0.4)
    output, diagnostics = gate(mixture, conservative, aggressive)
    full = diagnostics["full_gate"]
    assert full.shape[1] == 1
    assert float(full.detach().min()) >= 0.0
    assert float(full.detach().max()) <= 1.0
    assert not torch.equal(output, conservative)


def test_regularization_is_finite_and_zero_at_step_zero():
    mixture, conservative, aggressive = _audio()
    config = _config()
    gate = SmoothResidualGate(config)
    _, diagnostics = gate(mixture, conservative, aggressive)
    total, parts = gate_regularization(
        diagnostics, aggressive, conservative, config
    )
    assert float(total.detach()) == 0.0
    assert all(float(value.detach()) == 0.0 for value in parts.values())


def test_state_reload_and_identity_bind_weights_config_and_adapter():
    gate = SmoothResidualGate(_config())
    before = gate_bundle_identity(gate)
    clone = SmoothResidualGate(_config())
    clone.load_state_dict(gate.state_dict())
    assert gate_bundle_identity(clone) == before
    with torch.no_grad():
        clone.correction_amplitude.fill_(0.25)
    assert gate_bundle_identity(clone)["weights_sha256"] != before["weights_sha256"]
    changed = SmoothResidualGate(_config(temporal_tv_weight=0.03))
    assert gate_bundle_identity(changed)["config_sha256"] != before["config_sha256"]


def test_gate_refuses_grid_and_nonfinite_inputs():
    mixture, conservative, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    with pytest.raises(ValueError, match="exact grid"):
        gate(mixture, conservative[..., :-1], aggressive)
    broken = mixture.clone()
    broken[..., 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        gate(broken, conservative, aggressive)


def test_config_refuses_bad_band_or_stft_grid():
    with pytest.raises(ValueError, match="Nyquist"):
        SmoothGateConfig(band_edges_hz=(0, 1000)).validate()
    with pytest.raises(ValueError, match="STFT"):
        SmoothGateConfig(n_fft=2).validate()
