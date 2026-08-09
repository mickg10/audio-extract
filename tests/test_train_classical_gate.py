import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from audio_extract.train_classical_gate import (
    MANIFEST_SCHEMA,
    GateDataset,
    _forward_loss,
    audio_record,
    load_manifest,
)
from audio_extract.classical_gate import SmoothGateConfig, SmoothResidualGate
from audio_extract.classical_loss_v2 import ClassicalResidualLossConfig


def _write(path: Path, value: np.ndarray) -> dict:
    sf.write(path, value.astype("float32"), 44100, subtype="FLOAT")
    return audio_record(path)


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    frames = 2048
    rng = np.random.default_rng(4)
    a = (0.05 * rng.standard_normal((frames, 2))).astype("float32")
    v = (0.02 * rng.standard_normal((frames, 2))).astype("float32")
    m = (a + v).astype("float32")
    records = {
        "mixture": _write(tmp_path / "M.wav", m),
        "accompaniment": _write(tmp_path / "A.wav", a),
        "vocal": _write(tmp_path / "V.wav", v),
    }
    estimates = {}
    for control, source in (("mixture", m), ("no_vocal", a), ("vocal_only", v)):
        estimates[control] = {}
        for member, scale, digit in (("conservative", 0.9, "1"), ("aggressive", 0.98, "2")):
            path = tmp_path / f"{control}-{member}.wav"
            record = _write(path, (scale * source).astype("float32"))
            record["recipe_id"] = "sha256:" + digit * 64
            record["executed_bundle_hash"] = digit * 64
            estimates[control][member] = record
    activity_path = tmp_path / "activity.npy"
    np.save(activity_path, np.ones(frames, dtype="float32"))
    document = {
        "schema": MANIFEST_SCHEMA,
        "members": {"conservative": "c", "aggressive": "a"},
        "works": [{
            "work_id": "opera", "group_id": "independent-opera",
            "truth": records, "estimates": estimates,
            "vocal_activity_path": str(activity_path),
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path, document


def test_manifest_rehashes_all_exact_float_inputs_and_dataset_reads_one_grid(tmp_path):
    path, _ = _manifest(tmp_path)
    document = load_manifest(path)
    dataset = GateDataset(document, ["opera"], 512)
    batch = dataset.sample(np.random.default_rng(1))
    assert batch["truth"]["mixture"].shape == (2, 512)
    assert batch["activity"].shape == (1, 512)
    assert set(batch["estimates"]) == {"mixture", "no_vocal", "vocal_only"}


def test_manifest_refuses_mutated_audio_and_grid_mismatch(tmp_path):
    path, document = _manifest(tmp_path)
    sf.write(
        document["works"][0]["truth"]["mixture"]["path"],
        np.zeros((2048, 2), dtype="float32"), 44100, subtype="FLOAT",
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_manifest(path)


def test_manifest_requires_distinct_members_and_complete_controls(tmp_path):
    path, document = _manifest(tmp_path)
    document["members"]["aggressive"] = "c"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="distinct"):
        load_manifest(path)

    _, document = _manifest(tmp_path)
    del document["works"][0]["estimates"]["vocal_only"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="mixture/A-only/V-only"):
        load_manifest(path)


def test_training_forward_uses_real_controls_and_backpropagates_at_step_zero(tmp_path):
    path, _ = _manifest(tmp_path)
    batch = GateDataset(load_manifest(path), ["opera"], 512).sample(
        np.random.default_rng(2)
    )
    gate = SmoothResidualGate(SmoothGateConfig(
        n_fft=64, hop_length=16, tile_seconds=0.01,
        band_edges_hz=(0, 2_000, 8_000, 22_050), hidden_channels=3,
    ))
    loss, components, regularization, controls, output = _forward_loss(
        gate, batch, device="cpu",
        loss_config=ClassicalResidualLossConfig(stft_ffts=(64, 128)),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(
        output["accompaniment"].detach()[0],
        batch["estimates"]["mixture"]["conservative"],
    )
    assert gate.correction_amplitude.grad is not None
    assert set(controls) == {"mixture", "no_vocal", "vocal_only"}
    assert components["no_vocal_false_positive"] is not None
    assert float(regularization.detach()) == 0.0
