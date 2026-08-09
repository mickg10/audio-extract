from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "demucs_affine_parity.py"
_SPEC = importlib.util.spec_from_file_location("demucs_affine_parity_tool", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_PARITY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PARITY)


def test_comparison_reports_exact_and_nonexact_results():
    reference = torch.zeros(1, 2, 2, 8)

    exact = _PARITY._comparison(reference, reference.clone())
    assert exact == {
        "max_abs": 0.0,
        "rms": 0.0,
        "finite": True,
        "shape": [1, 2, 2, 8],
    }

    candidate = reference.clone()
    candidate[0, 0, 0, 0] = 0.25
    changed = _PARITY._comparison(reference, candidate)
    assert changed["max_abs"] == 0.25
    assert changed["rms"] > 0
    assert changed["finite"] is True


def test_comparison_refuses_grid_mismatch():
    try:
        _PARITY._comparison(torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 7))
    except ValueError as exc:
        assert "parity grid mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("grid mismatch was accepted")


def test_publish_reopens_float_artifact_and_is_immutable(tmp_path):
    path = tmp_path / "other.f32.wav"
    audio = torch.linspace(-0.5, 0.5, 32, dtype=torch.float32).reshape(2, 16)

    first = _PARITY._publish(path, audio)
    second = _PARITY._publish(path, audio.clone())

    assert first == second
    assert first["subtype"] == "FLOAT"
    assert first["frames"] == 16
    assert first["sample_rate_hz"] == _PARITY.SR
    reopened, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    assert sample_rate == _PARITY.SR
    assert np.array_equal(reopened, audio.T.numpy())

    changed = audio.clone()
    changed[0, 0] += 0.125
    try:
        _PARITY._publish(path, changed)
    except RuntimeError as exc:
        assert "refusing to rewrite differing parity artifact" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("immutable artifact was rewritten")
