"""WP5 metric-v2 tests, including the review's NEGATIVE controls: a written
diminuendo must not score as a hole; vocal bleed must raise uncertainty, never
improve the score."""
import numpy as np

from audio_extract import dsp
from audio_extract import fixtures as fx
from audio_extract import metrics_v2 as m2

SR = 44100
HOP = 441  # 10 ms


def _material():
    vocal, activity = fx.synth_vocal(SR, 4.0)
    orch = fx.synth_orchestra(SR, 4.0)
    n = min(len(vocal), len(orch), len(activity))
    return vocal[:n], orch[:n], activity[:n]


def _events_from_activity(activity):
    a = activity > 0.1
    edges = np.flatnonzero(np.diff(a.astype(int)))
    spans, start = [], None
    if a[0]:
        start = 0
    for e in edges:
        if a[e + 1] and start is None:
            start = e + 1
        elif not a[e + 1] and start is not None:
            spans.append((start // HOP, (e + 1) // HOP))
            start = None
    if start is not None:
        spans.append((start // HOP, len(a) // HOP))
    return spans


def _envs(cand, clean, vocal):
    return (dsp.band_envelope_db(cand, SR, dsp.PUMP_BANDS),
            dsp.band_envelope_db(clean, SR, dsp.PUMP_BANDS),
            dsp.band_envelope_db(vocal, SR, dsp.PUMP_BANDS))


def _get(obs_list, name):
    return next(o for o in obs_list if o["metric"] == name)


def test_event_conditioned_pumping_detected():
    vocal, orch, activity = _material()
    events = _events_from_activity(activity)
    pumped = fx.inject_pumping(orch, activity, 12.0)
    ce, ee, ve = _envs(pumped, orch, vocal * 1e-6)  # negligible vocal -> no masking
    res = m2.event_holes(ce, ee, ve, events)
    assert _get(res, "event_hole_depth/v2")["value"] > 3.0


def test_negative_control_diminuendo_not_a_hole():
    # A WRITTEN diminuendo (present in candidate AND expected consensus) with no
    # vocal events must not register as pumping.
    _, orch, _ = _material()
    n = len(orch)
    ramp = np.linspace(1.0, 0.15, n).reshape(-1, 1)
    dim = orch * ramp                     # the score says decrescendo
    ce, ee, ve = _envs(dim, dim, orch * 1e-6)   # consensus shows the SAME diminuendo
    events = [(int(0.3 * n / HOP), int(0.6 * n / HOP))]
    res = m2.event_holes(ce, ee, ve, events)
    d = _get(res, "event_hole_depth/v2")
    assert (d["value"] or 0.0) < 1.5      # no meaningful hole


def test_bleed_masks_to_uncertainty_not_improvement():
    vocal, orch, activity = _material()
    events = _events_from_activity(activity)
    pumped = fx.inject_pumping(orch, activity, 12.0)
    bleedy = fx.add_vocal_bleed(pumped, vocal, 0.0)   # loud bleed over the holes
    ce, ee, ve = _envs(bleedy, orch, vocal)
    res = m2.event_holes(ce, ee, ve, events)
    unk = _get(res, "masked_hole_uncertainty/v2")["value"]
    assert unk > 0.1                       # contaminated cells become UNKNOWN
    clean_res = m2.event_holes(*_envs(pumped, orch, vocal * 1e-6), events)
    assert _get(clean_res, "masked_hole_uncertainty/v2")["value"] < unk


def test_fullness_level_fit_removes_gain_confound():
    _, orch, _ = _material()
    quieter = orch * 0.5                  # pure global gain, no missing content
    res = m2.fullness_v2(quieter, orch, SR)
    assert _get(res, "band_deficit_db/v2")["value"] < 0.6
    holed = fx.spectral_hole(quieter, SR, 2000, 4000, 12.0)
    res2 = m2.fullness_v2(holed, orch, SR)
    assert _get(res2, "band_deficit_db/v2")["value"] > _get(res, "band_deficit_db/v2")["value"]


def test_loo_family_consensus_balances_families():
    T = 50
    a = np.zeros((2, T)); b = np.zeros((2, T)) + 0.0   # family X members ~0 dB
    c = np.ones((2, T)) * 10.0                          # family Y member at 10 dB
    envs = {"a": a, "b": b, "c": c, "me": np.ones((2, T)) * 99}
    fams = {"a": "X", "b": "X", "c": "Y", "me": "X"}
    exp, info = m2.loo_family_consensus(envs, fams, exclude="me")
    assert info["families"] == 2 and info["members"] == 3
    assert abs(float(exp.mean()) - 5.0) < 1e-9          # X(0) and Y(10) weigh EQUALLY
    none_exp, info0 = m2.loo_family_consensus({"me": a}, {"me": "X"}, exclude="me")
    assert none_exp is None and info0["families"] == 0


def test_transient_v2_direction():
    _, orch, _ = _material()
    bed = np.zeros((SR * 2, 2)); bed[:: SR // 4] = 1.0
    smeared = fx.smear_transients(bed, SR, 8.0)
    res = m2.transient_v2(smeared, bed, SR)
    assert _get(res, "transient_loss/v2")["value"] > _get(res, "transient_excess/v2")["value"]
    res2 = m2.transient_v2(bed, smeared, SR)   # reversed: candidate ADDS transients
    assert _get(res2, "transient_excess/v2")["value"] > _get(res2, "transient_loss/v2")["value"]


def test_stereo_v2_bounded():
    _, orch, _ = _material()
    mono = fx.narrow_stereo(orch, 1.0)
    res = m2.stereo_v2(mono, orch)
    dev = _get(res, "stereo_width_dev_db/v2")["value"]
    assert 0 < dev <= 80.0                  # bounded even for pure mono


def test_missing_evidence_unavailable():
    res = m2.event_holes(np.zeros((2, 10)), np.zeros((2, 10)), np.zeros((2, 10)), [])
    d = _get(res, "event_hole_depth/v2")
    assert d["available"] is False and d["value"] is None


def test_near_silent_band_gated():
    # a huge deficit confined to a near-silent target band must NOT dominate (§8.1)
    B, T = 6, 200
    expected = np.full((B, T), -6.0)
    expected[5, :] = -70.0                 # band 5 is ~silent (64 dB below peak)
    vocal = np.full((B, T), -120.0)
    events = [(40, 120)]
    # candidate perfectly matches except a giant "hole" in the silent band
    cand = expected.copy()
    cand[5, 60:100] = -120.0               # 50 dB deficit, but inaudible band
    res = m2.event_holes(cand, expected, vocal, events)
    assert (m2_get := _get(res, "event_hole_depth/v2"))["value"] is None or m2_get["value"] < 5.0
    # the SAME deficit in a loud band IS counted
    cand2 = expected.copy()
    cand2[2, 60:100] = -30.0               # 24 dB deficit in an audible band
    res2 = m2.event_holes(cand2, expected, vocal, events)
    assert _get(res2, "event_hole_depth/v2")["value"] > 10.0


def test_aggregate_shape():
    agg = m2.aggregate([1.0, 2.0, 8.0, 3.0])
    assert agg["worst"] >= agg["p90"] >= agg["median"]
    assert agg["boot_lower"] <= agg["median"] <= agg["boot_upper"]
    assert m2.aggregate([]) == {"available": False}
