import copy

from audio_extract import planning as pl


def _recipe(**overrides):
    r = {
        "schema": "audio-extract/recipe/v2", "canon": "rfc8785+jcs-schema-v1",
        "input_pcm": {"sha256": "sha256:a", "sample_rate_hz": 44100,
                      "channel_layout": ["FL", "FR"], "frames": 100,
                      "sample_format": "float32-le-interleaved"},
        "operation": {"type": "separate", "target": "vocals", "construction": "native_primary"},
        "model": {"model_id": "m", "weights_sha256": "w" * 64,
                  "adapter": "a", "adapter_revision": "r"},
        "effective_config": {"model_sample_rate_hz": 44100, "segment_samples": 1000,
                             "overlap_factor": 8},
        "software": {"audio_extract_commit": "c1"},
    }
    for path, val in overrides.items():
        sec, key = path.split(".")
        r[sec][key] = val
    return r


def test_recipe_diff_one_variable():
    parent = _recipe()
    child = _recipe(**{"effective_config.overlap_factor": 4})
    changed = pl.material_changes(parent, child)
    assert changed == {"effective_config.overlap_factor"}
    ok, _, _ = pl.validate_experiment(parent, child)
    assert ok


def test_omitted_default_is_not_a_change():
    parent = _recipe()
    child = copy.deepcopy(parent)
    child["effective_config"].pop("batch_size", None)   # omitted default == explicit default
    assert pl.material_changes(parent, child) == set()
    ok, reason, _ = pl.validate_experiment(parent, child)
    assert not ok and "nothing material" in reason      # a duplicate is not an experiment


def test_two_variable_change_rejected_unless_combined():
    parent = _recipe()
    child = _recipe(**{"effective_config.overlap_factor": 4,
                       "operation.construction": "mixture_minus_primary"})
    ok, reason, changed = pl.validate_experiment(parent, child)
    assert not ok and len(changed) == 2
    ok2, _, _ = pl.validate_experiment(parent, child, allow_combined=True)
    assert ok2


def test_software_change_is_not_material():
    parent = _recipe()
    child = _recipe(**{"software.audio_extract_commit": "c2"})
    assert pl.material_changes(parent, child) == set()  # provenance, not an experiment


def test_planner_response_validation():
    ok, _ = pl.validate_planner_response(
        {"status": "experiments", "actions": [
            {"type": "propose_model_variant", "parent_recipe_id": "sha256:p"}]})
    assert ok
    ok, r = pl.validate_planner_response(
        {"status": "experiments", "actions": [{"type": "propose_model_variant"}]})
    assert not ok and "parent_recipe_id" in r
    ok, r = pl.validate_planner_response({"status": "final", "candidate_id": "x"})
    assert not ok and "selector" in r                    # planner may NOT finalize
    ok, _ = pl.validate_planner_response(
        {"status": "no_more_useful_probes", "reason": "done"})
    assert ok


def test_deterministic_fallback_planner():
    a = {"mean": {"event_hole": 0.4}, "lower": {"event_hole": 0.3}, "upper": {"event_hole": 0.5}}
    b = {"mean": {"event_hole": 0.42}, "lower": {"event_hole": 0.32}, "upper": {"event_hole": 0.52}}
    p = pl.DeterministicFallbackPlanner()
    r1 = p.propose({"top_pair": [a, b], "best_recipe_id": "sha256:x"})
    assert r1["status"] == "experiments"
    assert r1["actions"][0]["type"] == "run_registered_experiment"
    assert pl.validate_planner_response(r1)[0]
    r_none = pl.DeterministicFallbackPlanner().propose({"top_pair": []})
    assert r_none["status"] == "no_more_useful_probes"
