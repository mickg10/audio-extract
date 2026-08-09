import pytest

torch = pytest.importorskip("torch")

from audio_extract.classical_gate import SmoothGateConfig, SmoothResidualGate
from audio_extract.classical_gate_initialization import prepare_gate_for_teacher_
from audio_extract.classical_gate_training import (
    GateAmplitudeProjector,
    build_teacher_geometry,
    gate_teacher_logits,
    pairwise_o2_targets,
    pairwise_o2_teacher_loss,
    project_gate_amplitude_after_step_,
    require_teacher_geometry,
)
from audio_extract.oracle_routing import RoutingConfig


BASIS = ("045", "955", "mdx", "mel", "bs")


def _audio(frames=4096):
    torch.manual_seed(31)
    mixture = 0.1 * torch.randn(1, 2, frames)
    parent = 0.02 * torch.randn(1, 2, frames)
    aggressive = parent + 0.01 * torch.randn(1, 2, frames)
    return mixture, parent, aggressive


def _config(**kwargs):
    values = dict(
        n_fft=128,
        hop_length=32,
        tile_seconds=0.01,
        band_edges_hz=(0, 2_000, 8_000, 22_050),
        hidden_channels=3,
    )
    values.update(kwargs)
    return SmoothGateConfig(**values)


def _geometry(gate, mixture, parent, aggressive, *, routing_config=None):
    _, details = gate_teacher_logits(gate, mixture, parent, aggressive)
    return build_teacher_geometry(
        routing_config or gate.config,
        time_ranges=details["time_ranges"],
        frequency_ranges=details["frequency_ranges"],
    )


def _targets(gate, mixture, parent, aggressive, labels, available=None,
             *, geometry=None):
    geometry = geometry or _geometry(gate, mixture, parent, aggressive)
    if available is None:
        available = torch.ones_like(labels, dtype=torch.bool)
    return pairwise_o2_targets(
        labels, available,
        geometry=geometry, basis_ids=BASIS,
        parent_index=0, aggressive_index=1,
    )


def test_pairwise_projection_masks_third_members_and_unavailable_cells():
    gate = SmoothResidualGate(_config())
    mixture, parent, aggressive = _audio()
    labels = torch.tensor([
        [0, 1, 2, 1],
        [4, 0, 1, 3],
    ])
    available = torch.tensor([
        [True, True, True, False],
        [True, False, True, True],
    ])
    result = _targets(
        gate, mixture, parent, aggressive, labels, available
    )
    assert result.target.shape == (1, 1, 2, 4)
    assert result.available.shape == result.target.shape
    assert result.parent_cells == 1
    assert result.aggressive_cells == 2
    assert result.unavailable_cells == 2
    assert result.masked_other_cells == 3
    assert result.target[result.available].tolist() == [0.0, 1.0, 1.0]
    assert result.teacher_sha256.startswith("sha256:")


def test_teacher_loss_trains_prepared_network_while_route_is_parent():
    mixture, parent, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    prepare_gate_for_teacher_(gate, seed=9)
    output, diagnostics = gate(mixture, parent, aggressive)
    assert torch.equal(output, parent)
    assert float(diagnostics["amplitude"].detach()) == 0.0

    logits, _ = gate_teacher_logits(gate, mixture, parent, aggressive)
    labels = torch.zeros(
        logits.shape[0], logits.shape[2], logits.shape[3], dtype=torch.int64
    )
    labels[:, ::2] = 1
    targets = _targets(gate, mixture, parent, aggressive, labels)
    loss, report = pairwise_o2_teacher_loss(
        gate, mixture, parent, aggressive, targets
    )
    loss.backward()
    for parameter in gate.network.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0
    assert gate.correction_amplitude.grad is None
    assert report["available_cells"] == labels.numel()
    assert report["positive_cells"] + report["negative_cells"] == labels.numel()


def test_teacher_loss_enforces_geometry_even_when_tensor_shape_matches():
    mixture, parent, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    logits, details = gate_teacher_logits(gate, mixture, parent, aggressive)
    labels = torch.zeros(
        logits.shape[0], logits.shape[2], logits.shape[3], dtype=torch.int64
    )
    wrong_config = SmoothGateConfig(
        n_fft=gate.config.n_fft,
        hop_length=gate.config.hop_length * 2,
        tile_seconds=gate.config.tile_seconds,
        band_edges_hz=gate.config.band_edges_hz,
        hidden_channels=gate.config.hidden_channels,
    )
    wrong_geometry = build_teacher_geometry(
        wrong_config,
        time_ranges=details["time_ranges"],
        frequency_ranges=details["frequency_ranges"],
    )
    targets = _targets(
        gate, mixture, parent, aggressive, labels,
        geometry=wrong_geometry,
    )
    with pytest.raises(ValueError, match="geometry differs"):
        pairwise_o2_teacher_loss(
            gate, mixture, parent, aggressive, targets
        )


def test_sgd_momentum_recovers_immediately_after_outward_projection():
    gate = SmoothResidualGate(_config())
    optimizer = torch.optim.SGD(
        [{"params": list(gate.parameters()), "weight_decay": 0.0}],
        lr=0.25, momentum=0.9,
    )
    projector = GateAmplitudeProjector(optimizer, gate)

    optimizer.zero_grad()
    gate.correction_amplitude.backward()
    optimizer.step()
    first = projector.after_step()
    assert first["raw_before_projection"] < 0
    assert first["cleared_directional_state"] == ["momentum_buffer"]
    assert float(gate.correction_amplitude.detach()) == 0.0

    optimizer.zero_grad()
    (-0.1 * gate.correction_amplitude).backward()
    optimizer.step()
    projector.after_step()
    assert float(gate.correction_amplitude.detach()) > 0.0


def test_adam_first_moment_is_cleared_at_lower_boundary():
    gate = SmoothResidualGate(_config())
    optimizer = torch.optim.Adam(
        [{"params": list(gate.parameters()), "weight_decay": 0.0}], lr=0.1
    )
    projector = GateAmplitudeProjector(optimizer, gate)
    optimizer.zero_grad()
    gate.correction_amplitude.backward()
    optimizer.step()
    report = projector.after_step()
    assert "exp_avg" in report["cleared_directional_state"]
    assert float(gate.correction_amplitude.detach()) == 0.0

    optimizer.zero_grad()
    (-0.1 * gate.correction_amplitude).backward()
    optimizer.step()
    projector.after_step()
    assert float(gate.correction_amplitude.detach()) > 0.0


def test_projector_preserves_native_optimizer_scheduler_and_step_api():
    gate = SmoothResidualGate(_config())
    optimizer = torch.optim.SGD(
        [{"params": list(gate.parameters()), "weight_decay": 0.0}], lr=0.1
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    projector = GateAmplitudeProjector(optimizer, gate)
    optimizer.zero_grad()
    (-gate.correction_amplitude).backward()
    result = optimizer.step()
    assert result is None
    projector.after_step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_projection_refuses_nonfinite_ownership_and_weight_decay():
    gate = SmoothResidualGate(_config())
    wrong = torch.optim.SGD(gate.network.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="exactly one group"):
        project_gate_amplitude_after_step_(wrong, gate)

    decayed = torch.optim.SGD(
        [{"params": list(gate.parameters()), "weight_decay": 1e-4}], lr=0.1
    )
    with pytest.raises(ValueError, match="zero weight decay"):
        project_gate_amplitude_after_step_(decayed, gate)

    valid = torch.optim.SGD(
        [{"params": list(gate.parameters()), "weight_decay": 0.0}], lr=0.1
    )
    with torch.no_grad():
        gate.correction_amplitude.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        project_gate_amplitude_after_step_(valid, gate)


def test_projector_state_is_small_and_resume_checked():
    gate = SmoothResidualGate(_config())
    optimizer = torch.optim.Adam(
        [{"params": list(gate.parameters()), "weight_decay": 0.0}], lr=1e-3
    )
    projector = GateAmplitudeProjector(optimizer, gate)
    state = projector.state_dict()
    assert state == {"schema": GateAmplitudeProjector.SCHEMA}
    with torch.no_grad():
        gate.correction_amplitude.fill_(2.0)
    projector.load_state_dict(state)
    assert float(gate.correction_amplitude.detach()) == 1.0
    with pytest.raises(ValueError, match="state mismatch"):
        projector.load_state_dict({"schema": "wrong"})


def test_teacher_geometry_must_match_exact_o2_grid():
    exact = SmoothGateConfig(
        hop_length=1024,
        tile_seconds=2.0,
        band_edges_hz=(
            0, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 22_050
        ),
    )
    geometry = build_teacher_geometry(
        RoutingConfig(),
        time_ranges=((0, 2), (2, 4)),
        frequency_ranges=((0, 2), (2, 4)),
    )
    require_teacher_geometry(exact, geometry)
    with pytest.raises(ValueError, match="geometry differs"):
        require_teacher_geometry(SmoothGateConfig(), geometry)


def test_teacher_loss_masks_all_unknown_cells_without_clean_labels():
    mixture, parent, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    logits, _ = gate_teacher_logits(gate, mixture, parent, aggressive)
    labels = torch.full(
        (logits.shape[0], logits.shape[2], logits.shape[3]),
        4,
        dtype=torch.int64,
    )
    targets = _targets(gate, mixture, parent, aggressive, labels)
    loss, report = pairwise_o2_teacher_loss(
        gate, mixture, parent, aggressive, targets
    )
    assert float(loss.detach()) == 0.0
    assert report["available_cells"] == 0
    assert report["masked_other_cells"] == labels.numel()
