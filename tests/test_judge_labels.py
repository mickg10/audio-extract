import json

import numpy as np
import pytest

from audio_extract import judge_labels as jl


RID = "sha256:" + "1" * 64
CID = "sha256:" + "2" * 64


def test_complex_ridge_recovers_source_coordinates():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((256, 2)) + 1j * rng.standard_normal((256, 2))
    v = rng.standard_normal((256, 2)) + 1j * rng.standard_normal((256, 2))
    alpha = 0.82 + 0.11j
    beta = -0.17 + 0.06j
    y = alpha * a + beta * v

    label = jl.fit_source_coordinates(y, a, v, ridge_relative=0.0)

    assert label.available
    assert complex(label.alpha_real, label.alpha_imag) == pytest.approx(alpha, abs=1e-10)
    assert complex(label.beta_real, label.beta_imag) == pytest.approx(beta, abs=1e-10)
    assert label.alpha_error_abs == pytest.approx(abs(alpha - 1.0), abs=1e-10)
    assert label.retained_voice_coefficient == pytest.approx(abs(beta), abs=1e-10)
    assert label.orthogonal_artifact_ratio < 1e-10
    payload = label.to_dict()
    assert payload["schema"] == jl.LABEL_SCHEMA
    assert payload["label_id"].startswith("sha256:")


def test_dual_voice_targets_and_orthogonal_artifact():
    # Equal-energy, mutually orthogonal coordinates make the expected values exact.
    a = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0, 0.0])
    artifact = np.array([0.0, 0.0, 0.1, 0.0])
    y = a + 0.2 * v + artifact

    label = jl.fit_source_coordinates(y, a, v, ridge_relative=0.0)

    assert label.available
    assert label.retained_voice_coefficient == pytest.approx(0.2)
    assert label.retained_voice_energy_ratio == pytest.approx(0.04)
    assert label.retained_voice_energy_db == pytest.approx(10 * np.log10(0.04))
    assert label.orthogonal_artifact_ratio == pytest.approx(0.1)


def test_collinear_sources_are_masked_not_zero_labeled():
    a = np.linspace(-1.0, 1.0, 128)
    v = a.copy()
    y = 0.8 * a

    label = jl.fit_source_coordinates(y, a, v, max_condition=1e4)

    assert label.available is False
    assert label.mask_reason == "ill_conditioned_sources"
    assert label.uncertainty == 1.0
    assert label.retained_voice_coefficient is None
    assert label.orthogonal_artifact_ratio is None


def test_local_tiles_keep_frame_accounting_and_unavailable_uncertainty():
    a = np.zeros((10, 2))
    v = np.zeros((10, 2))
    a[:, 0] = np.arange(1, 11)
    v[:, 1] = np.arange(11, 21)
    y = a + 0.1 * v

    labels = jl.local_source_coordinate_labels(
        y, a, v, tile_frames=4, hop_frames=4, min_tile_frames=2,
        ridge_relative=0.0,
    )

    assert [(x.start_frame, x.end_frame) for x in labels] == [(0, 4), (4, 8), (8, 10)]
    assert all(x.available for x in labels)
    assert jl.label_uncertainty(labels) < 1.0

    bad = jl.fit_source_coordinates(np.ones(8), np.ones(8), np.ones(8))
    assert jl.label_uncertainty([labels[0], bad]) == 1.0


def test_exact_label_inputs_refuse_shape_drift_nonfinite_and_bad_weights():
    with pytest.raises(jl.LabelInputError, match="identical shapes"):
        jl.fit_source_coordinates(np.ones(8), np.ones(7), np.ones(8))
    broken = np.ones(8)
    broken[2] = np.nan
    with pytest.raises(jl.LabelInputError, match="non-finite"):
        jl.fit_source_coordinates(broken, np.ones(8), np.arange(8.0))
    with pytest.raises(jl.LabelInputError, match="non-negative"):
        jl.fit_source_coordinates(
            np.ones(8), np.arange(8.0), np.arange(8.0)[::-1],
            weights=np.array([1, 1, 1, -1, 1, 1, 1, 1]),
        )


def _row(**overrides):
    a = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    label = jl.fit_source_coordinates(a + 0.1 * v, a, v, ridge_relative=0.0)
    values = dict(
        split="train",
        split_group="production-a",
        work_id="work-a",
        corpus_id="corpus-a",
        donor_ids=("singer-a", "orchestra-a"),
        source_lineage_ids=("source-a",),
        additional_group_ids=("session:session-a", "venue:venue-a"),
        candidate_recipe_id=RID,
        challenge_id=CID,
        task="soloist_vs_rest",
        reference_grade="linear_exact",
        pair_integrity="linear_exact",
        threshold_eligible=True,
        label_uncertainty=jl.label_uncertainty([label]),
        labels=(label,),
        metadata={"condition": "dry"},
    )
    values.update(overrides)
    return jl.JudgeDatasetRow.build(**values)


def test_row_identity_is_order_stable_and_rows_are_immutable():
    first = _row(metadata={"b": 2, "a": 1})
    same = _row(metadata={"a": 1, "b": 2})
    changed = _row(metadata={"a": 1, "b": 3})
    assert first == same
    assert first.row_id == same.row_id
    assert first.example_key == changed.example_key
    assert first.row_id != changed.row_id

    builder = jl.JudgeDatasetBuilder()
    assert builder.add(first) is first
    assert builder.add(same) is first
    with pytest.raises(jl.ImmutableRowError):
        builder.add(changed)


@pytest.mark.parametrize(
    "override",
    [
        {"split_group": "other", "work_id": "work-a"},
        {"split_group": "other", "work_id": "work-b", "corpus_id": "corpus-a"},
        {
            "split_group": "other", "work_id": "work-b", "corpus_id": "corpus-b",
            "donor_ids": ("singer-a",), "source_lineage_ids": ("source-b",),
        },
        {
            "split_group": "other", "work_id": "work-b", "corpus_id": "corpus-b",
            "donor_ids": ("singer-b",), "source_lineage_ids": ("source-a",),
        },
        {
            "split_group": "other", "work_id": "work-b", "corpus_id": "corpus-b",
            "donor_ids": ("singer-b",), "source_lineage_ids": ("source-b",),
            "additional_group_ids": ("session:session-a",),
        },
    ],
)
def test_builder_refuses_group_leakage_across_splits(override):
    builder = jl.JudgeDatasetBuilder()
    builder.add(_row())
    with pytest.raises(jl.SplitLeakageError):
        builder.add(_row(
            split="test",
            candidate_recipe_id="sha256:" + "3" * 64,
            challenge_id="sha256:" + "4" * 64,
            **override,
        ))


def test_lossy_preview_allowed_for_weak_training_but_not_threshold_calibration():
    weak = _row(
        reference_grade="lossy_preview",
        pair_integrity="same_take_paired_target",
        threshold_eligible=False,
    )
    builder = jl.JudgeDatasetBuilder()
    builder.add(weak)
    assert builder.rows == (weak,)

    with pytest.raises(jl.ThresholdEligibilityError):
        _row(
            split="calibration",
            reference_grade="lossy_preview",
            pair_integrity="same_take_paired_target",
            threshold_eligible=False,
        )
    with pytest.raises(jl.ThresholdEligibilityError):
        _row(reference_grade="lossy_preview", threshold_eligible=True)


def test_calibration_export_and_jsonl_are_deterministic():
    row = _row(split="calibration")
    builder = jl.JudgeDatasetBuilder()
    builder.add(row)
    assert builder.threshold_calibration_rows() == (row,)
    payload = builder.to_dict()
    assert payload["schema"] == jl.DATASET_SCHEMA
    assert payload["split_counts"]["calibration"] == 1
    assert json.loads(builder.to_jsonl()) == row.to_dict()


def test_row_requires_hash_identities():
    with pytest.raises(ValueError, match="sha256"):
        _row(candidate_recipe_id="model.ckpt")


def test_row_requires_at_least_one_label():
    with pytest.raises(ValueError, match="at least one"):
        _row(labels=())
