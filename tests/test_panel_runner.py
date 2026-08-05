from audio_extract import fixtures as fx
from audio_extract import panel_runner as pr

SR = 44100
DUR = 3.0


def _base():
    vocal, activity = fx.synth_vocal(SR, DUR)
    orch = fx.synth_orchestra(SR, DUR)
    n = min(len(vocal), len(orch), len(activity))
    return vocal[:n], orch[:n], activity[:n]


def _score(cands: dict, vocal, activity, reference=None):
    ref = reference if reference is not None else pr.consensus_reference(list(cands.values()))
    return [
        pr.Scored(rid, pr.score_candidate(arr, SR, reference=ref, vocal_ref=vocal, activity=activity))
        for rid, arr in cands.items()
    ]


def test_clean_candidate_ranks_first():
    vocal, orch, activity = _base()
    garbage = fx.inject_pumping(
        fx.spectral_hole(fx.add_vocal_bleed(orch, vocal, -6.0), SR, 2000, 4000, 12.0),
        activity, 12.0,
    )
    # Score against ground truth (clean orchestra). Note bleed can *mask* pumping,
    # so damaged candidates may stay Pareto-incomparable — the robust signal is the
    # scalarized ranking, which must put the clean candidate first.
    scored = _score({"clean": orch, "bleed": fx.add_vocal_bleed(orch, vocal, -8.0),
                     "holes": fx.spectral_hole(orch, SR, 2000, 4000, 10.0), "garbage": garbage},
                    vocal, activity, reference=orch)
    ranked = pr.rank_by_scalarized(scored)
    front_ids = {s.recipe_id for s in pr.pareto_frontier(scored)}
    assert ranked[0].recipe_id == "clean"          # best overall
    assert ranked[-1].recipe_id in {"garbage", "holes", "bleed"}  # a defective one is worst
    assert "clean" in front_ids                    # clean is Pareto-optimal


def test_tradeoff_yields_multi_member_frontier():
    vocal, orch, activity = _base()
    x = fx.spectral_hole(fx.add_vocal_bleed(orch, vocal, -6.0), SR, 2000, 4000, 12.0)   # high leak, big hole
    y = fx.spectral_hole(fx.add_vocal_bleed(orch, vocal, -24.0), SR, 2000, 4000, 2.0)   # low leak, small hole
    scored = _score({"x": x, "y": y}, vocal, activity)
    front_ids = {s.recipe_id for s in pr.pareto_frontier(scored)}
    assert front_ids == {"x", "y"}   # neither dominates the other


def test_successive_halving_narrows():
    vocal, orch, activity = _base()
    scored = _score({
        "clean": orch,
        "bleed": fx.add_vocal_bleed(orch, vocal, -8.0),
        "holes": fx.spectral_hole(orch, SR, 2000, 4000, 10.0),
        "pump": fx.inject_pumping(orch, activity, 10.0),
    }, vocal, activity)
    kept = pr.successive_halving(scored, keep=2)
    assert len(kept) == 2
    assert kept[0].recipe_id == "clean"


def test_dominates_semantics():
    a = {k: 0.0 for k in pr.AXES}
    b = {k: 1.0 for k in pr.AXES}
    assert pr.dominates(a, b)
    assert not pr.dominates(b, a)
    assert not pr.dominates(a, a)  # equal does not dominate
