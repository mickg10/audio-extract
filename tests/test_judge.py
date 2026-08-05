import numpy as np

from audio_extract import fixtures as fx
from audio_extract import judge as jd

SR = 44100
DUR = 2.5


def test_bradley_terry_orders_transitive_wins():
    items = ["A", "B", "C"]
    pairs = [("A", "B")] * 3 + [("B", "C")] * 3 + [("A", "C")] * 3
    s = jd.bradley_terry(items, pairs)
    assert s["A"] > s["B"] > s["C"]


def test_logistic_ranker_separates():
    fm = {"A": np.array([2.0, 0.0]), "B": np.array([0.0, 0.0]), "C": np.array([-2.0, 0.0])}
    pairs = [("A", "B"), ("B", "C"), ("A", "C")] * 4
    w = jd.fit_logistic_ranker(fm, pairs)
    assert jd.prob_a_beats_b(w, fm["A"], fm["C"]) > 0.7
    assert jd.prob_a_beats_b(w, fm["C"], fm["A"]) < 0.3


def _calibration_inputs():
    vocal, _ = fx.synth_vocal(SR, DUR)
    orch = fx.synth_orchestra(SR, DUR)
    n = min(len(vocal), len(orch))
    clean = orch[:n]
    vocal = vocal[:n]
    # monotonic-damage ladder via spectral holes (summed cost rises with severity)
    d = [fx.spectral_hole(clean, SR, 2000, 4000, sev) for sev in (3.0, 6.0, 12.0)]
    ladders = [[clean, d[0], d[1], d[2]]]
    ab_pairs = [(clean, d[2]), (clean, d[1]), (d[0], d[2])]
    human = [(clean, d[2], "a"), (clean, d[1], "a"), (d[0], d[2], "a")]
    clips = [clean, d[1]]
    return dict(vocal=vocal, clean=clean, ladders=ladders, ab_pairs=ab_pairs,
               repeat_pairs=ab_pairs, human_labeled=human, clips=clips)


def test_metric_judge_passes_calibration():
    ci = _calibration_inputs()
    judge = jd.MetricJudge(ci["clean"], SR, vocal_ref=ci["vocal"])
    rep = jd.calibrate(judge, SR, ab_pairs=ci["ab_pairs"], repeat_pairs=ci["repeat_pairs"],
                       ladders=ci["ladders"], human_labeled=ci["human_labeled"], clips=ci["clips"])
    assert rep.passed is True
    assert rep.verdict == "selector"
    assert rep.metrics["defect_monotonicity"] >= 0.90
    assert rep.metrics["loudness_bias"] <= 0.10


def test_noisy_judge_fails_calibration():
    ci = _calibration_inputs()
    truth = jd.MetricJudge(ci["clean"], SR, vocal_ref=ci["vocal"])
    noisy = jd.NoisyJudge(truth, accuracy=0.5, seed=1)
    rep = jd.calibrate(noisy, SR, ab_pairs=ci["ab_pairs"], repeat_pairs=ci["repeat_pairs"],
                       ladders=ci["ladders"], human_labeled=ci["human_labeled"], clips=ci["clips"])
    assert rep.passed is False
    assert rep.verdict == "annotation_only"


def test_promotion_thresholds_present():
    for k in ("ab_order_consistency", "repeated_pair_agreement", "defect_monotonicity",
              "human_agreement", "max_loudness_bias"):
        assert k in jd.PROMOTION
