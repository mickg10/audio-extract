"""WP4 challenge-engine tests. The load-bearing property: the remix target is
exact BY CONSTRUCTION (linear add), so an ideal estimator recovers it perfectly —
whatever the hall model's quality."""
import numpy as np

from audio_extract import challenges as ch
from audio_extract import fixtures as fx

SR = 44100


def _material():
    vocal, _ = fx.synth_vocal(SR, 3.0)
    orch = fx.synth_orchestra(SR, 3.0)
    n = min(len(vocal), len(orch))
    return vocal[:n], orch[:n]


def _cases(max_cases=4):
    vocal, orch = _material()
    return ch.build_track_remix_cases([("ctl0", orch)], [("voc0", vocal)], SR,
                                      max_cases=max_cases, seed=3), vocal, orch


def test_remix_target_exact_by_construction():
    cases, _, _ = _cases()
    assert len(cases) >= 2
    for c in cases:
        # mixture - injected == target, to float precision: the target is EXACT
        assert np.allclose(c.mixture - c.injected_vocal, c.target, atol=1e-12)


def test_ideal_estimator_recovers_target():
    cases, _, _ = _cases()
    c = cases[0]
    ideal = c.mixture - c.injected_vocal          # estimator with perfect vocal knowledge
    err = ch.exact_reference_error(ideal, c.target, SR)
    assert err["si_sdr_db"] > 40                   # essentially perfect recovery
    assert err["stft_distance"] < 0.05


def test_exact_reference_error_monotone_with_damage():
    cases, _, _ = _cases()
    c = cases[0]
    ideal = c.mixture - c.injected_vocal
    vals = []
    for depth in (0.0, 6.0, 12.0):
        est = fx.spectral_hole(ideal, SR, 2000, 4000, depth) if depth else ideal
        vals.append(ch.exact_reference_error(est, c.target, SR)["band_envelope_err_db"])
    assert vals[0] < vals[1] < vals[2]


def test_challenge_ids_unique_and_stable():
    cases, _, _ = _cases()
    ids = [c.challenge_id for c in cases]
    assert len(set(ids)) == len(ids)
    again, _, _ = _cases()
    assert [c.challenge_id for c in again] == ids   # deterministic


def test_theft_assay_scales():
    _, orch = _material()
    zero = ch.run_no_vocal_theft_assay(lambda a: np.zeros_like(a), [("c", orch)], SR)
    tenth = ch.run_no_vocal_theft_assay(lambda a: 0.1 * a, [("c", orch)], SR)
    half = ch.run_no_vocal_theft_assay(lambda a: 0.5 * a, [("c", orch)], SR)
    assert zero[0]["theft_broadband"] < 1e-6
    assert abs(tenth[0]["theft_broadband"] - 0.1) < 1e-3
    assert abs(half[0]["theft_broadband"] - 0.5) < 1e-3


def test_bleed_assay():
    vocal, _ = _material()
    r = ch.run_vocal_only_bleed_assay(lambda v: 0.25 * v, [("v", vocal)], SR)
    assert abs(r[0]["bleed_broadband"] - 0.25) < 1e-3


def test_orchestra_intervention_pass_through_ordering():
    _, orch = _material()
    probe = ch.make_orchestral_probe("brass_onset", SR, 1.0)
    identity = lambda m: m                       # passes the probe through exactly
    half = lambda m: 0.5 * m                     # halves it
    absorb = lambda m: np.zeros_like(m)          # absorbs it entirely
    e_id = ch.orchestra_intervention_error(identity, orch, probe)
    e_half = ch.orchestra_intervention_error(half, orch, probe)
    e_abs = ch.orchestra_intervention_error(absorb, orch, probe)
    assert e_id < 1e-9
    assert e_id < e_half < e_abs
    assert abs(e_half - 0.5) < 1e-6 and abs(e_abs - 1.0) < 1e-6


def test_vocal_intervention_invariance():
    vocal, orch = _material()
    dv = 0.2 * vocal
    constant = lambda m: orch[: len(m)]          # ignores the input -> fully invariant
    identity = lambda m: m                       # leaks the added vocal -> variant
    assert ch.vocal_intervention_invariance(constant, orch, dv) < 1e-9
    assert ch.vocal_intervention_invariance(identity, orch, dv) > 0.01


def test_probes_build_all_kinds():
    for kind in ch.PROBE_KINDS:
        p = ch.make_orchestral_probe(kind, SR, 0.5, level=0.05)
        assert p.shape == (SR // 2, 2)
        assert abs(np.max(np.abs(p)) - 0.05) < 1e-9
    side = ch.make_orchestral_probe("side_hf_air", SR, 0.5)
    assert np.allclose(side[:, 0], -side[:, 1])   # pure side content


def test_symmetric_source_response_semantics():
    _, orch = _material()
    probe = ch.make_orchestral_probe("brass_onset", SR, 1.0)
    identity = lambda m: m
    absorb = lambda m: np.zeros_like(m)
    r_id = ch.symmetric_source_response(identity, orch, probe)
    r_ab = ch.symmetric_source_response(absorb, orch, probe)
    assert all(row["pass_through_err"] < 1e-9 for row in r_id["per_alpha"])   # J == δ exactly
    assert r_id["stability"] < 1e-9                                            # linear across α
    assert all(row["response_ratio"] < 1e-9 for row in r_ab["per_alpha"])      # fully invariant
    assert all(abs(row["pass_through_err"] - 1.0) < 1e-6 for row in r_ab["per_alpha"])


def test_audit_pair_integrity_classes():
    vocal, orch = _material()
    full = orch + vocal
    # linear pair: base IS the true accompaniment of full
    a1 = ch.audit_pair(full, orch, SR, solo_inactive_mask=(np.abs(vocal).sum(axis=1) < 1e-9))
    assert a1["recommended_integrity"] == "linear_exact"
    # same-take, master differs: base with gain + small delay
    base2 = np.vstack([np.zeros((30, 2)), orch * 0.7])[: len(orch)]
    a2 = ch.audit_pair(full, base2, SR, solo_inactive_mask=(np.abs(vocal).sum(axis=1) < 1e-9))
    assert a2["recommended_integrity"] in ("linear_exact", "same_take_paired_target")
    assert abs(a2["delay_samples"]) >= 25
    # different length -> different edit, never a mute
    a3 = ch.audit_pair(full, orch[: len(orch) // 2], SR)
    assert a3["recommended_integrity"] == "matched_program"
    # unrelated recording -> matched_program
    other = fx.synth_orchestra(SR, 3.0, seed=99)[: len(full)]
    a4 = ch.audit_pair(full, other, SR)
    assert a4["recommended_integrity"] == "matched_program"


def test_ontology_constants():
    assert "soloist_vs_rest" in ch.EVALUATION_TASKS
    assert set(ch.PAIR_INTEGRITY) == {"linear_exact", "same_take_paired_target",
                                      "same_performance_bleed", "matched_program"}


def test_rt60_estimate_in_range():
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * 1.5)) / SR
    tail = (np.exp(-t / (1.0 / 6.91)) * rng.standard_normal(len(t))).reshape(-1, 1)
    rt = ch.estimate_rt60_from_tail(tail, SR)
    assert 0.5 < rt < 2.0                          # true RT60 = 1.0 s, crude fit ok
