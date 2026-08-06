from audio_extract import probes as pb


def _cand(mean, width=0.05):
    return {"mean": dict(mean),
            "lower": {k: v - width for k, v in mean.items()},
            "upper": {k: v + width for k, v in mean.items()}}


def test_probe_targets_overlapping_axis():
    # theft is far apart (decided); event_hole overlaps heavily -> probe event_hole
    a = _cand({"orchestral_theft": 0.1, "event_hole": 0.40}, width=0.10)
    b = _cand({"orchestral_theft": 0.6, "event_hole": 0.42}, width=0.10)
    rec = pb.recommend_probe(a, b)
    assert rec.axis == "event_hole"
    assert rec.probe in ("run_track_remix_challenge", "run_vocal_intervention",
                         "run_alternate_model_family")
    assert rec.rank_flip_probability > 0.5


def test_probe_dedup_moves_to_next_best():
    a = _cand({"event_hole": 0.40}, width=0.10)
    b = _cand({"event_hole": 0.42}, width=0.10)
    first = pb.recommend_probe(a, b)
    second = pb.recommend_probe(a, b, executed={first.probe})
    assert second.probe != first.probe


def test_theft_uncertainty_routes_to_theft_assay():
    a = _cand({"orchestral_theft": 0.30}, width=0.15)
    b = _cand({"orchestral_theft": 0.33}, width=0.15)
    rec = pb.recommend_probe(a, b)
    assert rec.probe == "run_no_vocal_theft_assay"


def test_deterministic_fallback_when_nothing_overlaps():
    # clearly separated on every axis -> fallback rule still returns a probe
    a = _cand({"vocal_leakage": 0.1}, width=0.01)
    b = _cand({"vocal_leakage": 0.9}, width=0.01)
    rec = pb.recommend_probe(a, b)
    assert rec.probe in pb.PROBE_VOCABULARY
    assert "fallback" in rec.reason


def test_fallback_rules_cover_axes():
    for axis, probe in pb.FALLBACK_RULES.items():
        assert probe in pb.PROBE_VOCABULARY
