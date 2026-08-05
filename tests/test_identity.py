import numpy as np
import pytest

from audio_extract import identity, recipe as recipe_mod


def test_recipe_id_stable_under_key_reorder(base_recipe):
    r1 = base_recipe()
    r2 = base_recipe()
    # reinsert top-level keys in reverse order
    r2 = {k: r2[k] for k in reversed(list(r2.keys()))}
    assert identity.recipe_id(r1) == identity.recipe_id(r2)


def test_omitted_default_equals_explicit_default(base_recipe):
    explicit = base_recipe()
    omitted = base_recipe()
    # These carry the materialized defaults; dropping them must not change identity.
    for key in ("batch_size", "pitch_shift_semitones", "input_peak_normalization",
                "output_peak_normalization", "output_sample_format"):
        omitted["effective_config"].pop(key, None)
    assert identity.recipe_id(explicit) == identity.recipe_id(omitted)


def test_overlap_changes_id(base_recipe):
    r1 = base_recipe()
    r2 = base_recipe()
    r2["effective_config"]["overlap_factor"] = 4
    assert identity.recipe_id(r1) != identity.recipe_id(r2)


def test_construction_changes_id(base_recipe):
    r1 = base_recipe()
    r2 = base_recipe()
    r2["operation"]["construction"] = "mixture_minus_primary"
    assert identity.recipe_id(r1) != identity.recipe_id(r2)


def test_model_hash_changes_id(base_recipe):
    r1 = base_recipe()
    r2 = base_recipe()
    r2["model"]["weights_sha256"] = "0" * 64
    assert identity.recipe_id(r1) != identity.recipe_id(r2)


def test_code_revision_changes_id(base_recipe):
    r1 = base_recipe()
    r2 = base_recipe()
    r2["software"]["audio_extract_commit"] = "deadbeef"
    assert identity.recipe_id(r1) != identity.recipe_id(r2)


def test_resampler_addition_changes_id(base_recipe):
    r1 = base_recipe()
    r2 = base_recipe()
    r2["effective_config"]["resampler"] = "soxr_hq"
    assert identity.recipe_id(r1) != identity.recipe_id(r2)


def test_unknown_top_level_field_rejected(base_recipe):
    r = base_recipe()
    r["surprise"] = 1
    with pytest.raises(recipe_mod.RecipeError):
        identity.recipe_id(r)


def test_bad_operation_type_rejected(base_recipe):
    r = base_recipe()
    r["operation"]["type"] = "teleport"
    with pytest.raises(recipe_mod.RecipeError):
        identity.recipe_id(r)


def test_recipe_id_is_prefixed_hex(base_recipe):
    rid = identity.recipe_id(base_recipe())
    assert rid.startswith("sha256:")
    assert len(rid) == len("sha256:") + 64


def test_artifact_pcm_sha256_deterministic():
    rng = np.zeros((100, 2), dtype="float32")
    rng[:, 0] = np.linspace(-1, 1, 100)
    a = identity.artifact_pcm_sha256(rng, 44100, ["FL", "FR"], 100)
    b = identity.artifact_pcm_sha256(rng.copy(), 44100, ["FL", "FR"], 100)
    assert a == b and a.startswith("sha256:")


def test_artifact_pcm_sha256_changes_with_samples():
    x = np.zeros((100, 2), dtype="float32")
    y = x.copy()
    y[0, 0] = 0.5
    assert identity.artifact_pcm_sha256(x, 44100, ["FL", "FR"]) != \
        identity.artifact_pcm_sha256(y, 44100, ["FL", "FR"])


def test_artifact_pcm_sha256_changes_with_sample_rate():
    x = np.zeros((100, 2), dtype="float32")
    assert identity.artifact_pcm_sha256(x, 44100, ["FL", "FR"]) != \
        identity.artifact_pcm_sha256(x, 48000, ["FL", "FR"])


def test_artifact_pcm_sha256_layout_mismatch_raises():
    x = np.zeros((100, 2), dtype="float32")
    with pytest.raises(ValueError):
        identity.artifact_pcm_sha256(x, 44100, ["FL", "FR", "FC"])
