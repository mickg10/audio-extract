"""Oracle P0s: certified path (frozen calibration -> select_v3 -> immutable
decision -> decision-bound render) and its refusals."""
import json

import numpy as np
import soundfile as sf

from audio_extract import cli_autonomous as auto
from audio_extract import fixtures as fx
from audio_extract import identity
from audio_extract.storage import TrackLayout

SR = 44100


def _frozen_calibration(tmp_path):
    """A minimal but real-schema calibration artifact: orchestral_theft certifiable
    at 0.1 with an identity-ish severity map; event_hole/fullness present."""
    def knots():
        xs = list(np.linspace(0, 1, 8)); return {"xs": xs, "ys": xs}
    doc = {"schema": "audio-extract/calibration/v1",
           "defects": {
               "orchestral_theft": {"map_knots": knots(),
                   "learn_then_test_unit=work_model": {
                       "0.1": {"tau": 0.5, "risk_ucb": 0.08, "n_groups": 58, "certifiable": True}}},
               "event_hole": {"map_knots": knots(),
                   "learn_then_test_unit=work_model": {
                       "0.1": {"tau": 0.6, "risk_ucb": 0.09, "n_groups": 40, "certifiable": True}}},
               "fullness": {"map_knots": knots(),
                   "learn_then_test_unit=work_model": {
                       "0.1": {"tau": 0.7, "risk_ucb": 0.09, "n_groups": 40, "certifiable": True}}},
           }}
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(doc))
    return p


def _run_with_challenge_results(tmp_path, theft_by_recipe):
    """Seed a run + stored challenge_result rows for two recipes, keyed by recipe id."""
    from audio_extract.manifest_v2 import ManifestV2

    layout = TrackLayout(tmp_path, "trk").ensure()
    mix = fx.synth_orchestra(SR, 3.0)
    sf.write(str(layout.source_dir / "canonical.f32.wav"), mix.astype("float32"), SR, subtype="FLOAT")
    (layout.source_dir / "source.json").write_text(json.dumps({
        "track_id": "trk", "sample_rate_hz": SR, "channel_layout": ["FL", "FR"],
        "frames": len(mix), "input_pcm_sha256": "sha256:in"}))
    layout.passages_dir.mkdir(parents=True, exist_ok=True)
    (layout.passages_dir / "passages.v1.json").write_text(json.dumps({
        "schema": "audio-extract/passages/v1",
        "passages": [{"passage_id": "p0", "start_sample": 0, "end_sample": len(mix) // 2,
                      "tags": ["voice_dominant"], "features": {}, "detectors": {}}]}))
    with ManifestV2(layout.manifest_sqlite) as m2:
        for i, (recipe_id, theft) in enumerate(theft_by_recipe.items()):
            m2.add_challenge_case(challenge_id=f"sha256:case{i}", challenge_type="track_remix",
                                  mixture_node_id="m", target_node_id="t", recipe={}, klass={})
            m2.add_challenge_result(challenge_id=f"sha256:case{i}", candidate_recipe_id=recipe_id,
                                    result={"exact_reference": {"si_sdr_db": 15.0,
                                            "band_envelope_err_db": 3.0, "stereo_width_err": 0.01},
                                            "exact_labels": {"event_hole_depth_db": 4.0,
                                            "vocal_interference_ratio": 0.1}})
            m2.add_challenge_result(challenge_id="theft_assay", candidate_recipe_id=recipe_id,
                                    result={"theft_mean": theft})
        # write the candidate audio so render can find it
        for recipe_id in theft_by_recipe:
            arr = (mix * 0.9).astype("float32")
            wav = tmp_path / f"{recipe_id.replace(':','_')}.wav"
            sf.write(str(wav), arr, SR, subtype="FLOAT")
            layout.write_candidate(recipe_id, {"operation": {"type": "mixture_minus_source",
                                   "construction": "mixture_minus_primary", "target": "instrumental"}},
                                   {}, wav, identity.artifact_pcm_sha256(arr, SR, ["FL", "FR"], len(arr)))
    return layout


def _seed_multi(tmp_path, recipes, n_cases):
    """A run with `recipes` scored over `n_cases` distinct challenges each."""
    from audio_extract.manifest_v2 import ManifestV2

    layout = TrackLayout(tmp_path, "trk").ensure()
    mix = fx.synth_orchestra(SR, 2.0)
    (layout.source_dir / "source.json").write_text(json.dumps({
        "track_id": "trk", "sample_rate_hz": SR, "channel_layout": ["FL", "FR"],
        "frames": len(mix), "input_pcm_sha256": "sha256:in"}))
    with ManifestV2(layout.manifest_sqlite) as m2:
        for i in range(n_cases):
            m2.add_challenge_case(challenge_id=f"sha256:case{i}", challenge_type="track_remix",
                                  mixture_node_id="m", target_node_id="t", recipe={}, klass={})
            for rid in recipes:
                m2.add_challenge_result(challenge_id=f"sha256:case{i}", candidate_recipe_id=rid,
                    result={"exact_reference": {"si_sdr_db": 12.0, "band_envelope_err_db": 3.0,
                            "stereo_width_err": 0.01},
                            "exact_labels": {"event_hole_depth_db": 4.0,
                            "vocal_interference_ratio": 0.1}})
        for rid in recipes:
            m2.add_challenge_result(challenge_id="theft_assay", candidate_recipe_id=rid,
                                    result={"theft_mean": 0.1})
    return layout


def test_cells_aggregate_by_recipe_not_challenge(tmp_path):
    # P0 §1: ONE recipe × FOUR challenges -> exactly ONE candidate with FOUR cells
    layout = _seed_multi(tmp_path, ["sha256:recipeA"], n_cases=4)
    cands = auto.severity_cells_from_store(layout)
    assert set(cands) == {"sha256:recipeA"}                     # not four challenge-keyed pseudo-candidates
    assert len(cands["sha256:recipeA"]["cells"]["event_hole"]) == 4


def test_two_recipes_same_challenge_two_candidates(tmp_path):
    layout = _seed_multi(tmp_path, ["sha256:recipeA", "sha256:recipeB"], n_cases=1)
    cands = auto.severity_cells_from_store(layout)
    assert set(cands) == {"sha256:recipeA", "sha256:recipeB"}   # two candidates, not one-per-challenge


def test_certification_inputs_are_recipe_ids_not_challenge_ids(tmp_path):
    calib = _frozen_calibration(tmp_path)
    layout = _seed_multi(tmp_path, ["sha256:recipeA", "sha256:recipeB"], n_cases=3)
    d = auto.select_autonomous(layout, calibration_path=calib)
    ids = set(d["certification_inputs"]["candidate_recipe_ids"])
    assert ids == {"sha256:recipeA", "sha256:recipeB"}
    assert not any(k.startswith("sha256:case") for k in ids)    # never a challenge id


def test_preview_never_certifies(tmp_path):
    layout = _run_with_challenge_results(tmp_path, {"sha256:good": 0.1, "sha256:bad": 0.2})
    d = auto.select_autonomous(layout)                 # no calibration -> preview
    assert d["rollout_level"] == "engineering_preview"
    assert d.get("certified") is False
    assert d.get("certification") == "none"


def test_certified_path_uses_v3_and_records_inputs(tmp_path):
    calib = _frozen_calibration(tmp_path)
    layout = _run_with_challenge_results(tmp_path, {"sha256:a": 0.1, "sha256:b": 0.15})
    d = auto.select_autonomous(layout, calibration_path=calib, task="soloist_vs_rest",
                               domain="anechoic-dry")
    assert d["selector_version"] == auto.CERTIFIED_SELECTOR
    ci = d["certification_inputs"]
    assert ci["calibration_sha256"].startswith("sha256:")
    assert ci["task"] == "soloist_vs_rest" and ci["domain"] == "anechoic-dry"
    assert ci["candidate_recipe_ids"]                  # keyed by recipe id, not model name
    assert d["rollout_level"] in ("exact_benchmark_qualified", "engineering_preview")


def test_render_refuses_without_decision(tmp_path):
    layout = _run_with_challenge_results(tmp_path, {"sha256:a": 0.1})
    # a bogus decision id is refused
    res = auto.finalize_from_decision(layout, SR, "sd_nonexistent")
    assert res["status"] == "refused" and "not found" in res["reason"]


def test_render_refuses_stale_calibration(tmp_path):
    calib = _frozen_calibration(tmp_path)
    layout = _run_with_challenge_results(tmp_path, {"sha256:a": 0.1, "sha256:b": 0.9})
    d = auto.select_autonomous(layout, calibration_path=calib)
    did = d["decision_id"]
    other = tmp_path / "other_calib.json"
    other.write_text(json.dumps({"schema": "x", "defects": {}}))   # different hash
    res = auto.finalize_from_decision(layout, SR, did, calibration_path=other)
    assert res["status"] == "refused"
    assert any("hash mismatch" in c for c in res["checks_failed"])


def test_legacy_selector_cannot_emit_certified_final(tmp_path):
    # the preview path relabels certification=none even when it says final
    layout = _run_with_challenge_results(tmp_path, {"sha256:a": 0.05, "sha256:b": 0.9})
    d = auto.select_autonomous(layout)     # legacy select under the hood
    assert d.get("certified") is not True
