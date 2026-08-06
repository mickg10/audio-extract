"""v3 Phase C (grouped benchmark) + Phases K/L (LTT calibration + constrained
feasibility selector) tests."""
import numpy as np

from audio_extract import benchmark as bm
from audio_extract import fixtures as fx
from audio_extract import selector as sel

SR = 44100


# --- Phase C ---------------------------------------------------------------
def test_group_atomic_splits():
    g1 = bm.GroupKey(work="tosca", performer="callas")
    g2 = bm.GroupKey(work="boheme", performer="pavarotti")
    cases = [{"id": f"c{i}", "group": g1} for i in range(10)] + \
            [{"id": f"d{i}", "group": g2} for i in range(10)]
    splits = bm.split_cases(cases)
    # every clip of one work lands in exactly ONE split — never divided
    for split_cases_ in splits.values():
        works = {c["group"].work for c in split_cases_}
        for other in splits.values():
            if other is split_cases_:
                continue
            assert not (works & {c["group"].work for c in other})
    assert bm.assign_split(g1) == bm.assign_split(g1)   # deterministic


def test_tier2_mastered_transfer():
    orch = fx.synth_orchestra(SR, 2.0)
    vocal, _ = fx.synth_vocal(SR, 2.0)
    n = min(len(orch), len(vocal))
    mix = orch[:n] + vocal[:n]
    case = bm.build_tier2_case(mix, orch[:n], SR,
                               bm.GroupKey(work="w"), chain_id="chain-A")
    assert case.tier == 2 and case.reference_exact is False       # claims discipline
    assert "NOT exact" in case.meta["claim"]
    assert np.max(np.abs(case.mixture)) <= 0.99                    # limiter ceiling
    assert not np.allclose(case.mixture[:n], mix[:n], atol=1e-3)   # chain changed audio
    again = bm.build_tier2_case(mix, orch[:n], SR, bm.GroupKey(work="w"), chain_id="chain-A")
    assert np.allclose(case.mixture, again.mixture)                # deterministic
    other = bm.build_tier2_case(mix, orch[:n], SR, bm.GroupKey(work="w"), chain_id="chain-B")
    assert not np.allclose(case.mixture, other.mixture)            # held-out chains differ


# --- Phase K: Learn Then Test ---------------------------------------------
def _calib(n_good=40, n_bad=10, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_good):   # good cases: low predicted severity, truly fine
        rows.append((f"g{i}", float(rng.uniform(0.05, 0.35) + rng.normal(0, noise)), False))
    for i in range(n_bad):    # bad cases: high predicted severity, truly bad
        rows.append((f"b{i}", float(rng.uniform(0.6, 0.95)), True))
    return rows


def test_ltt_controls_risk():
    res = sel.learn_then_test(_calib(), target_risk=0.10, delta=0.05)
    assert res["certifiable"]
    assert 0.3 <= res["tau"] < 0.6         # certifies the good range, excludes the bad
    assert res["risk_ucb"] <= 0.10


def test_ltt_small_n_is_conservative():
    res = sel.learn_then_test(_calib(n_good=4, n_bad=1), target_risk=0.10)
    assert not res["certifiable"]           # 5 groups cannot support a 10% UCB claim


def test_ltt_group_atomic():
    # 50 clips of ONE group must count as one unit, not fifty
    rows = [("same_group", 0.2, False)] * 50
    res = sel.learn_then_test(rows, target_risk=0.10)
    assert res["n_groups"] == 1
    assert not res["certifiable"]


# --- Phase L: constrained-feasibility selector -----------------------------
def _taus(tau=0.5):
    return {d: {"tau": tau, "risk_ucb": 0.05, "n_groups": 50, "certifiable": True}
            for d in sel.CRITICAL_DEFECTS}


def _cand(level, secondary, n=10, u=0.01, evidence_split=True):
    cells = {d: [(level, u)] * n for d in sel.CRITICAL_DEFECTS}
    ev = {}
    if evidence_split:  # two independent evidence families, each sufficient
        ev = {"exact_ref": {d: [(level, u)] * (n // 2) for d in sel.CRITICAL_DEFECTS},
              "assays": {d: [(level, u)] * (n // 2) for d in sel.CRITICAL_DEFECTS}}
    return {"cells": cells, "secondary": secondary, "evidence": ev, "gates": {}}


def test_v3_final_when_feasible_robust_separated():
    # a's secondary 0.10 vs b's 0.50 -> separation 0.40 >= margin 0.10 -> final
    d = sel.select_v3({"a": _cand(0.2, 0.10), "b": _cand(0.3, 0.50)}, _taus(),
                      epsilon_regret=0.10)
    assert d["status"] == "final" and d["candidate_id"] == "a"
    assert d["certification"] == "autonomous_proxy_certified"
    assert d["separation"]["margin"] == 0.40


def test_v3_unseparated_runnerup_does_not_certify():
    # near-tie on secondary (0.30 vs 0.32) -> separation 0.02 < margin 0.10.
    # The OLD regret check (star-runner <= eps) passed this vacuously; it must not.
    d = sel.select_v3({"a": _cand(0.2, 0.30), "b": _cand(0.2, 0.32)}, _taus(),
                      epsilon_regret=0.10, probe_budget_left=False)
    assert d["status"] == "no_acceptable_candidate"
    assert "runner" in d["reason"] or "separation" in str(d)
    # with probe budget, it asks for a discriminating probe instead of certifying
    d2 = sel.select_v3({"a": _cand(0.2, 0.30), "b": _cand(0.2, 0.32)}, _taus(),
                       epsilon_regret=0.10, probe_budget_left=True)
    assert d2["status"] == "needs_probe"


def test_v3_infeasible_routes_to_probe_then_abstain():
    hot = {"a": _cand(0.9, 0.1)}
    probe = sel.select_v3(hot, _taus(), probe_budget_left=True)
    assert probe["status"] == "needs_probe"
    final = sel.select_v3(hot, _taus(), probe_budget_left=False)
    assert final["status"] == "no_acceptable_candidate"


def test_v3_uncertifiable_tau_blocks_final():
    taus = _taus()
    taus["orchestral_theft"]["certifiable"] = False   # calibration failed for theft
    d = sel.select_v3({"a": _cand(0.1, 0.1)}, taus, probe_budget_left=False)
    assert d["status"] == "no_acceptable_candidate"


def test_v3_missing_critical_evidence_is_infeasible():
    c = _cand(0.1, 0.1)
    del c["cells"]["orchestral_theft"]                # no theft evidence at all
    d = sel.select_v3({"a": c}, _taus(), probe_budget_left=False)
    assert d["status"] == "no_acceptable_candidate"
    assert "orchestral_theft" in d["candidates"]["a"]["infeasible_defects"]


def test_v3_out_of_domain_abstains():
    d = sel.select_v3({"a": _cand(0.1, 0.1)}, _taus(),
                      distribution_flag="out_of_domain")
    assert d["status"] == "no_acceptable_candidate"
    assert "calibration domain" in d["reason"]


def test_v3_loeo_instability_blocks_final():
    c = _cand(0.2, 0.1, evidence_split=False)
    # one family is clean, the other is the ONLY thing keeping severity low;
    # removing the clean family leaves a hot one -> instability
    c["evidence"] = {"clean": {d: [(0.1, 0.01)] * 5 for d in sel.CRITICAL_DEFECTS},
                     "hot": {d: [(0.9, 0.01)] * 5 for d in sel.CRITICAL_DEFECTS}}
    d = sel.select_v3({"a": c}, _taus(), probe_budget_left=False)
    assert d["status"] == "no_acceptable_candidate"
    assert "evidence" in d["reason"]
