
from audio_extract import fixtures as fx
from audio_extract import judge as jd

SR = 44100
DUR = 2.0


def _inputs():
    vocal, _ = fx.synth_vocal(SR, DUR)
    orch = fx.synth_orchestra(SR, DUR)
    n = min(len(vocal), len(orch))
    clean, vocal = orch[:n], vocal[:n]
    dmg = [fx.spectral_hole(clean, SR, 2000, 4000, d) for d in (3.0, 6.0, 12.0)]
    bleed = [fx.add_vocal_bleed(clean, vocal, g) for g in (-24.0, -12.0, -6.0)]
    # exact-target-labeled pairs: truth = less-damaged wins (severity is known)
    exact = [(clean, dmg[2], "a"), (dmg[0], dmg[2], "a"), (clean, dmg[1], "a"),
             (clean, bleed[2], "a"), (bleed[0], bleed[2], "a")]
    ladders = {"fullness": [clean, dmg[0], dmg[1], dmg[2]],
               "vocal_leakage": [clean, bleed[0], bleed[1], bleed[2]]}
    return clean, vocal, exact, ladders


def test_metric_judge_calibrates_autonomously_by_class():
    clean, vocal, exact, ladders = _inputs()
    judge = jd.MetricJudge(clean, SR, vocal_ref=vocal)
    rep = jd.calibrate_autonomous(
        judge, SR, judge_id="metric-judge/v1", calibration_id="c1",
        exact_pairs=exact, remix_pairs=exact, ladders_by_defect=ladders,
        clips=[clean],
        thresholds={**jd.PROMOTION_V2, "min_total_pairs": 4},  # tiny synthetic set
    )
    assert rep.synthetic_pair_accuracy == 1.0
    assert rep.monotonicity_by_defect["fullness"] == 1.0
    assert rep.monotonicity_by_defect["vocal_leakage"] == 1.0
    assert rep.passed_axes == {"fullness", "vocal_leakage"}   # per-class reporting
    assert rep.gain_invariance == 1.0
    assert rep.passed and rep.verdict == "selector"


def test_noisy_judge_fails_autonomous_gate():
    clean, vocal, exact, ladders = _inputs()
    truth = jd.MetricJudge(clean, SR, vocal_ref=vocal)
    noisy = jd.NoisyJudge(truth, accuracy=0.5, seed=3)
    rep = jd.calibrate_autonomous(
        noisy, SR, judge_id="noisy", calibration_id="c1",
        exact_pairs=exact, remix_pairs=exact, ladders_by_defect=ladders,
        clips=[clean],
        thresholds={**jd.PROMOTION_V2, "min_total_pairs": 4},
    )
    assert rep.passed is False and rep.verdict == "annotation_only"


def test_min_pairs_enforced():
    clean, vocal, exact, ladders = _inputs()
    judge = jd.MetricJudge(clean, SR, vocal_ref=vocal)
    rep = jd.calibrate_autonomous(
        judge, SR, judge_id="m", calibration_id="c",
        exact_pairs=exact, remix_pairs=exact, ladders_by_defect=ladders,
        clips=[clean])                       # default min_total_pairs=200 unmet
    assert rep.passed is False               # a perfect judge on 10 pairs is unproven


def test_tie_answers_supported():
    class TieJudge:
        name = "tie"
        def compare(self, a, b, sr, **ctx):
            return "tie"

    clean, vocal, exact, ladders = _inputs()
    rep = jd.calibrate_autonomous(
        TieJudge(), SR, judge_id="tie", calibration_id="c",
        exact_pairs=exact, remix_pairs=exact, ladders_by_defect=ladders, clips=[clean],
        thresholds={**jd.PROMOTION_V2, "min_total_pairs": 4})
    assert rep.synthetic_pair_accuracy == 0.0   # ties are conservative misses
    assert rep.ood_coverage == 1.0               # but ties on identical inputs = honest
    assert rep.passed is False
