from audio_extract import evidence as ev
from audio_extract.selector import CRITICAL_DEFECTS


def _good_profile(**kw):
    base = dict(pairwise_accuracy=0.95, calibration_error=0.05, repeatability=0.98,
                order_sensitivity=0.02, ood_failure_rate=0.05, calibrated=True)
    base.update(kw)
    return ev.ReliabilityProfile(**base)


def test_every_critical_defect_has_primary_mapping():
    for d in CRITICAL_DEFECTS:
        assert ev.AXIS_SOURCES[d]["primary"], d


def test_uncalibrated_source_is_diagnostic_only():
    out = ev.fuse("vocal_leakage",
                  {"exact_target_leakage": [(0.2, 0.02)] * 5,
                   "saj_precision": [(0.8, 0.02)] * 5},
                  {"exact_target_leakage": _good_profile(),
                   "saj_precision": ev.ReliabilityProfile(calibrated=False)})
    assert "saj_precision" in out["diagnostic_only"]
    assert out["measured"] and abs(out["cells"][0][0] - 0.2) < 1e-9


def test_secondaries_never_overturn_primary():
    # two loud secondaries disagree with the primary: severity STAYS at the
    # primary's value; only uncertainty inflates (no majority vote, §12.3)
    out = ev.fuse("vocal_leakage",
                  {"exact_target_leakage": [(0.2, 0.02)] * 5,
                   "saj_precision": [(0.9, 0.02)] * 5,
                   "harmonic_f0": [(0.9, 0.02)] * 5},
                  {"exact_target_leakage": _good_profile(),
                   "saj_precision": _good_profile(pairwise_accuracy=0.8),
                   "harmonic_f0": _good_profile(pairwise_accuracy=0.8)})
    s, u = out["cells"][0]
    assert abs(s - 0.2) < 1e-9          # severity is the primary's
    assert u > 0.02                      # disagreement widened uncertainty


def test_redundant_pair_counts_once():
    dep = {("saj_precision", "harmonic_f0"): {"redundant": True, "rank_correlation": 0.97}}
    out = ev.fuse("vocal_leakage",
                  {"exact_target_leakage": [(0.3, 0.02)] * 4,
                   "saj_precision": [(0.5, 0.02)] * 4,
                   "harmonic_f0": [(0.5, 0.02)] * 4},
                  {"exact_target_leakage": _good_profile(),
                   "saj_precision": _good_profile(pairwise_accuracy=0.9),
                   "harmonic_f0": _good_profile(pairwise_accuracy=0.8)},
                  dependence=dep)
    assert out["dropped_redundant"] == ["harmonic_f0"]   # less reliable member dropped
    assert "saj_precision" in out["secondaries"]


def test_no_primary_means_unmeasured():
    out = ev.fuse("orchestral_theft",
                  {"orchestra_response": [(0.1, 0.01)] * 5},   # secondary only
                  {"orchestra_response": _good_profile()})
    assert out["measured"] is False and out["cells"] == []   # selector -> infeasible


def test_dependence_profile_detects_redundancy():
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    b = [0.12, 0.19, 0.33, 0.41, 0.52, 0.58]     # same ordering
    c = [0.6, 0.1, 0.5, 0.2, 0.4, 0.3]           # unrelated ordering
    assert ev.dependence_profile(a, b)["redundant"] is True
    assert ev.dependence_profile(a, c)["redundant"] is False


def test_primary_disagreement_widens_uncertainty():
    profs = {"exact_event_deficit": _good_profile(),
             "local_envelope_deficit": _good_profile()}
    # promote the secondary to primary-like by using two primaries of one axis:
    out_agree = ev.fuse("event_hole",
                        {"exact_event_deficit": [(0.3, 0.02)] * 4},
                        profs)
    out2 = ev.fuse("event_hole",
                   {"exact_event_deficit": [(0.3, 0.02)] * 4,
                    "local_envelope_deficit": [(0.7, 0.02)] * 4},
                   profs)
    assert out2["cells"][0][1] > out_agree["cells"][0][1]
