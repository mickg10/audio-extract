from __future__ import annotations

import importlib.util
from pathlib import Path


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "gate2_compare.py"
_SPEC = importlib.util.spec_from_file_location("gate2_compare_tool", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE)


def _facts(*, voice: float, hole: float, hole_max: float | None = None,
           alpha: float = 0.2, artifact: float = 0.1,
           width: float = 0.1, coherence: float = 0.01,
           no_vocal_fp: float = 0.01, available: int = 10,
           total: int = 12) -> dict:
    return {
        "metrics": {
            "retained_voice_db_p90": voice,
            "event_hole_db_p90": hole,
            "event_hole_db_max": hole if hole_max is None else hole_max,
            "alpha_error_p90": alpha,
            "artifact_ratio_p90": artifact,
            "stereo_width_dev_db/v2": width,
            "interchannel_coherence_dev/v2": coherence,
            "_available_tiles": available,
            "_total_tiles": total,
        },
        "no_vocal_false_positive_energy_ratio": no_vocal_fp,
    }


def _report(step: int, works: dict[str, dict]) -> dict:
    return {"step": step, "works": works}


def test_fold_v_goes_to_500_when_preregistered_gate_passes():
    baseline = _report(0, {
        "bologna_verdi": _facts(voice=-7.0, hole=2.0, hole_max=3.0),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0, hole_max=3.0),
    })
    candidate = _report(100, {
        "bologna_verdi": _facts(voice=-7.7, hole=2.4, hole_max=3.2),
        "aalto_mozart_dry": _facts(voice=-20.8, hole=2.3, hole_max=3.1),
    })

    result = _GATE.evaluate("V", baseline, candidate)

    assert result["decision"] == "GO_TO_500"
    assert result["failures"] == []


def test_fold_d_goes_to_500_when_holes_improve_and_voice_is_stable():
    baseline = _report(0, {
        "bologna_donizetti": _facts(voice=-19.0, hole=18.8, hole_max=22.0),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0, hole_max=3.0),
    })
    candidate = _report(100, {
        "bologna_donizetti": _facts(voice=-18.7, hole=18.1, hole_max=21.1),
        "aalto_mozart_dry": _facts(voice=-20.8, hole=2.1, hole_max=3.1),
    })

    result = _GATE.evaluate("D", baseline, candidate)

    assert result["decision"] == "GO_TO_500"


def test_gate_stops_when_target_improvement_is_too_small():
    baseline = _report(0, {
        "bologna_verdi": _facts(voice=-7.0, hole=2.0),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0),
    })
    candidate = _report(100, {
        "bologna_verdi": _facts(voice=-7.2, hole=2.0),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0),
    })

    result = _GATE.evaluate("V", baseline, candidate)

    assert result["decision"] == "STOP_AT_100"
    assert "Verdi retained-voice p90 improvement <0.5 dB" in result["failures"]


def test_gate_stops_on_no_vocal_false_positive_regression():
    baseline = _report(0, {
        "bologna_verdi": _facts(voice=-7.0, hole=2.0, no_vocal_fp=0.01),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0, no_vocal_fp=0.01),
    })
    candidate = _report(100, {
        "bologna_verdi": _facts(voice=-7.7, hole=2.0, no_vocal_fp=0.011),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0, no_vocal_fp=0.01),
    })

    result = _GATE.evaluate("V", baseline, candidate)

    assert result["decision"] == "STOP_AT_100"
    assert any("no-vocal false-positive" in failure for failure in result["failures"])


def test_gate_refuses_missing_or_incoherent_evidence():
    baseline = _report(0, {
        "bologna_verdi": _facts(voice=-7.0, hole=2.0),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0),
    })
    candidate = _report(100, {
        "bologna_verdi": _facts(voice=-7.7, hole=2.0, hole_max=1.0),
        "aalto_mozart_dry": _facts(voice=-21.0, hole=2.0),
    })

    try:
        _GATE.evaluate("V", baseline, candidate)
    except _GATE.GateError as exc:
        assert "max < p90" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("incoherent evidence was accepted")
