import numpy as np

from audio_extract import selector as sel


def _cells(level: float, n: int = 12, u: float = 0.02, spike: float | None = None):
    vals = [(level, u)] * n
    if spike is not None:
        vals[0] = (spike, u)
    return {"orchestral_theft": vals, "event_hole": vals, "vocal_leakage": vals,
            "fullness": vals, "generic_quality": vals}


def test_pav_isotonic_monotone():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.3, 0.2, 0.8, 0.7])   # violations at 2 and 4
    xs, ys = sel.pav_isotonic(x, y)
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:]))


def test_severity_map_calibrates():
    m = sel.SeverityMap.fit([0.0, 3.0, 6.0, 12.0], [0.0, 0.3, 0.6, 1.0])
    assert m.apply(-1.0) == 0.0 and m.apply(20.0) == 1.0
    assert 0.25 < m.apply(3.0) < 0.35
    assert m.apply(6.0) > m.apply(3.0)         # monotone


def test_cvar_small_n_is_tail_mean():
    assert sel.cvar([0.1, 0.2, 0.9], alpha=0.90) == 0.9     # n=3 -> worst element
    assert abs(sel.cvar([0.0, 1.0, 1.0, 0.0], alpha=0.5) - 1.0) < 1e-12


def test_risk_worst_defect_dominates():
    cells = {"orchestral_theft": [(0.1, 0.0)] * 5, "event_hole": [(0.8, 0.0)] * 5}
    risk, per = sel.candidate_risk(cells, lam=0.0)
    assert risk == per["event_hole"] == 0.8


def test_hard_gate_blocks_low_risk_candidate():
    decision = sel.select({
        "good_but_thief": {"cells": _cells(0.05), "gates": {"orchestral_theft": 0.9}},
        "ok": {"cells": _cells(0.30), "gates": {}},
    }, n_boot=50)
    assert decision["status"] == "final"
    assert decision["candidate_id"] == "ok"    # the thief cannot win despite low risk
    assert decision["candidates"]["good_but_thief"]["gates_passed"] is False


def test_clear_winner_rule():
    decision = sel.select({
        "a": {"cells": _cells(0.10), "gates": {}},
        "b": {"cells": _cells(0.55), "gates": {}},
    }, n_boot=50)
    assert decision["status"] == "final" and decision["mode"] == "clear_winner"
    assert decision["candidate_id"] == "a"


def test_overlap_yields_best_safe_not_certainty():
    decision = sel.select({
        "a": {"cells": _cells(0.30, spike=0.5), "gates": {}},
        "b": {"cells": _cells(0.32, spike=0.5), "gates": {}},
    }, n_boot=50)
    assert decision["status"] == "final" and decision["mode"] == "best_safe"
    assert "not 'certain'" in decision["reason"]


def test_all_over_ceiling_no_acceptable():
    decision = sel.select({
        "a": {"cells": _cells(0.9), "gates": {}},
        "b": {"cells": _cells(0.95), "gates": {}},
    }, n_boot=50)
    assert decision["status"] == "no_acceptable_candidate"
    assert decision["best_available"] == "a"


def test_no_gate_passer_reports_best_available():
    decision = sel.select({
        "a": {"cells": _cells(0.2), "gates": {"vocal_leakage": 0.99}},
    }, n_boot=50)
    assert decision["status"] == "no_acceptable_candidate"
    assert decision["best_available"] == "a"
    assert decision["failed_gates"] == ["vocal_leakage"]


def test_lexicographic_priorities():
    # equal max-risk, but 'a' concentrates damage in a LOW-priority tier
    a = {"orchestral_theft": [(0.1, 0.0)] * 6, "generic_quality": [(0.5, 0.0)] * 6}
    b = {"orchestral_theft": [(0.5, 0.0)] * 6, "generic_quality": [(0.1, 0.0)] * 6}
    ka = sel.lexicographic_key(dict((k, sel.cvar([s for s, _ in v])) for k, v in a.items()))
    kb = sel.lexicographic_key(dict((k, sel.cvar([s for s, _ in v])) for k, v in b.items()))
    assert ka < kb        # theft (tier 1) hurts more than generic quality (tier 4)
