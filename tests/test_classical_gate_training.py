import pytest

torch = pytest.importorskip("torch")

from audio_extract.classical_gate import SmoothGateConfig, SmoothResidualGate
from audio_extract.classical_gate_training import (
    ProjectedGateOptimizer,
    gate_teacher_logits,
    pairwise_o2_targets,
    pairwise_o2_teacher_loss,
    project_gate_amplitude_,
    require_teacher_geometry,
)
from audio_extract.oracle_routing import RoutingConfig


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


def test_pairwise_projection_masks_cells_selected_by_third_members():
    labels = torch.tensor([
        [0, 1, 2, 1],
        [4, 0, 1, 3],
    ])
    result = pairwise_o2_targets(
        labels,
        parent_index=0,
        aggressive_index=1,
    )
    assert result.target.shape == (1, 1, 2, 4)
    assert result.available.shape == result.target.shape
    assert result.parent_cells == 2
    assert result.aggressive_cells == 3
    assert result.masked_other_cells == 3
    assert result.target[result.available].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_teacher_loss_trains_spatial_network_while_route_is_step_zero_parent():
    mixture, parent, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    output, diagnostics = gate(mixture, parent, aggressive)
    assert torch.equal(output, parent)
    assert float(diagnostics["amplitude"].detach()) == 0.0

    logits, _ = gate_teacher_logits(gate, mixture, parent, aggressive)
    labels = torch.zeros(logits.shape[0], logits.shape[2], logits.shape[3], dtype=torch.int64)
    labels[:, ::2] = 1
    targets = pairwise_o2_targets(labels, parent_index=0, aggressive_index=1)
    loss, report = pairwise_o2_teacher_loss(
        gate, mixture, parent, aggressive, targets
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in gate.network.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)
    assert gate.correction_amplitude.grad is None
    assert report["available_cells"] == labels.numel()


def test_projected_optimizer_recovers_after_update_that_would_go_negative():
    gate = SmoothResidualGate(_config())
    optimizer = ProjectedGateOptimizer(
        torch.optim.SGD(gate.parameters(), lr=0.25), gate
    )

    # A positive raw gradient would move the unconstrained scalar negative.
    optimizer.zero_grad()
    gate.correction_amplitude.backward()
    _, projection = optimizer.step()
    assert projection["raw_before_projection"] < 0
    assert float(gate.correction_amplitude.detach()) == 0.0

    # The following update begins at the boundary, not inside the dead zone, and
    # can immediately move toward a useful positive correction.
    optimizer.zero_grad()
    (-gate.correction_amplitude).backward()
    optimizer.step()
    assert float(gate.correction_amplitude.detach()) > 0.0


def test_projection_refuses_nonfinite_and_clamps_both_bounds():
    gate = SmoothResidualGate(_config())
    with torch.no_grad():
        gate.correction_amplitude.fill_(1.5)
    assert project_gate_amplitude_(gate)["effective_amplitude"] == 1.0
    with torch.no_grad():
        gate.correction_amplitude.fill_(-0.5)
    assert project_gate_amplitude_(gate)["effective_amplitude"] == 0.0
    with torch.no_grad():
        gate.correction_amplitude.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        project_gate_amplitude_(gate)


def test_optimizer_must_own_amplitude_and_projects_loaded_state():
    gate = SmoothResidualGate(_config())
    with pytest.raises(ValueError, match="does not own"):
        ProjectedGateOptimizer(
            torch.optim.SGD(gate.network.parameters(), lr=0.1), gate
        )
    wrapped = ProjectedGateOptimizer(
        torch.optim.Adam(gate.parameters(), lr=1e-3), gate
    )
    state = wrapped.state_dict()
    with torch.no_grad():
        gate.correction_amplitude.fill_(2.0)
    wrapped.load_state_dict(state)
    assert float(gate.correction_amplitude.detach()) == 1.0


def test_teacher_geometry_must_match_exact_o2_grid():
    exact = SmoothGateConfig(
        hop_length=1024,
        tile_seconds=2.0,
        band_edges_hz=(0, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 22_050),
    )
    require_teacher_geometry(exact, RoutingConfig())
    with pytest.raises(ValueError, match="geometry differs"):
        require_teacher_geometry(SmoothGateConfig(), RoutingConfig())


def test_teacher_loss_masks_all_unknown_cells_without_inventing_clean_labels():
    mixture, parent, aggressive = _audio()
    gate = SmoothResidualGate(_config())
    logits, _ = gate_teacher_logits(gate, mixture, parent, aggressive)
    labels = torch.full(
        (logits.shape[0], logits.shape[2], logits.shape[3]),
        4,
        dtype=torch.int64,
    )
    targets = pairwise_o2_targets(labels, parent_index=0, aggressive_index=1)
    loss, report = pairwise_o2_teacher_loss(
        gate, mixture, parent, aggressive, targets
    )
    assert float(loss.detach()) == 0.0
    assert report["available_cells"] == 0
    assert report["masked_other_cells"] == labels.numel()
