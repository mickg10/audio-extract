import pytest

torch = pytest.importorskip("torch")

from audio_extract.classical_gate import SmoothGateConfig, SmoothResidualGate
from audio_extract.classical_gate_initialization import prepare_gate_for_teacher_
from audio_extract.classical_gate_training import (
    build_teacher_geometry,
    gate_teacher_logits,
    pairwise_o2_targets,
    pairwise_o2_teacher_loss,
)


BASIS = ("parent", "aggressive")


def _config():
    return SmoothGateConfig(
        n_fft=128,
        hop_length=32,
        tile_seconds=0.01,
        band_edges_hz=(0, 2_000, 8_000, 22_050),
        hidden_channels=4,
    )


def _audio(frames=4096):
    torch.manual_seed(47)
    mixture = 0.1 * torch.randn(1, 2, frames)
    parent = 0.02 * torch.randn(1, 2, frames)
    aggressive = parent + 0.01 * torch.randn(1, 2, frames)
    return mixture, parent, aggressive


def _targets(gate, mixture, parent, aggressive, labels):
    _, details = gate_teacher_logits(gate, mixture, parent, aggressive)
    geometry = build_teacher_geometry(
        gate.config,
        time_ranges=details["time_ranges"],
        frequency_ranges=details["frequency_ranges"],
    )
    return pairwise_o2_targets(
        labels, torch.ones_like(labels, dtype=torch.bool),
        geometry=geometry, basis_ids=BASIS,
        parent_index=0, aggressive_index=1,
    )


def test_unprepared_all_zero_network_is_spatially_gradient_blocked():
    gate = SmoothResidualGate(_config())
    mixture, parent, aggressive = _audio()
    logits, _ = gate_teacher_logits(gate, mixture, parent, aggressive)
    labels = torch.ones(
        logits.shape[0], logits.shape[2], logits.shape[3], dtype=torch.int64
    )
    targets = _targets(gate, mixture, parent, aggressive, labels)
    loss, _ = pairwise_o2_teacher_loss(
        gate, mixture, parent, aggressive, targets
    )
    loss.backward()
    gradients = [parameter.grad for parameter in gate.network.parameters()]
    # Only the final bias can move when both convolutions and hidden activation
    # are exactly zero. This proves why explicit preparation is required.
    nonzero = [
        gradient is not None and float(gradient.abs().sum()) > 0
        for gradient in gradients
    ]
    assert nonzero == [False, False, False, True]


def test_prepared_network_has_first_backward_gradient_for_every_parameter():
    gate = SmoothResidualGate(_config())
    report = prepare_gate_for_teacher_(gate, seed=19)
    mixture, parent, aggressive = _audio()

    output, diagnostics = gate(mixture, parent, aggressive)
    assert torch.equal(output, parent)
    assert float(diagnostics["amplitude"].detach()) == 0.0
    assert report.nonzero_parameters > 0

    logits, _ = gate_teacher_logits(gate, mixture, parent, aggressive)
    labels = torch.zeros(
        logits.shape[0], logits.shape[2], logits.shape[3], dtype=torch.int64
    )
    labels[:, ::2] = 1
    targets = _targets(gate, mixture, parent, aggressive, labels)
    loss, _ = pairwise_o2_teacher_loss(
        gate, mixture, parent, aggressive, targets
    )
    loss.backward()
    for parameter in gate.network.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0
    assert gate.correction_amplitude.grad is None


def test_teacher_initialization_is_deterministic_and_identity_bearing():
    first = SmoothResidualGate(_config())
    second = SmoothResidualGate(_config())
    report_a = prepare_gate_for_teacher_(first, seed=123)
    report_b = prepare_gate_for_teacher_(second, seed=123)
    assert report_a == report_b
    assert first.state_dict().keys() == second.state_dict().keys()
    for key in first.state_dict():
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])

    third = SmoothResidualGate(_config())
    prepare_gate_for_teacher_(third, seed=124)
    assert any(
        not torch.equal(first.state_dict()[key], third.state_dict()[key])
        for key in first.state_dict()
        if key != "correction_amplitude"
    )


def test_initialization_rejects_invalid_settings():
    gate = SmoothResidualGate(_config())
    with pytest.raises(ValueError, match="seed"):
        prepare_gate_for_teacher_(gate, seed=-1)
    with pytest.raises(ValueError, match="final_weight_scale"):
        prepare_gate_for_teacher_(gate, final_weight_scale=0.0)
