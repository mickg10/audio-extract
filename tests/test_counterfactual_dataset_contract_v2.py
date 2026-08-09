import dataclasses

import pytest

from audio_extract.counterfactual_dataset_contract import (
    ArrayArtifact,
    CandidateArtifact,
    CandidateRole,
    CounterfactualDatasetError,
    GROUP_AXES,
    GroupIdentity,
    SourceGridIdentity,
    SplitAssignment,
    TASK_ID,
)
from audio_extract.counterfactual_dataset_contract_v2 import (
    CounterfactualDatasetContractV2,
    CounterfactualSplitContractV2,
    WorkBlockV2,
)


def sha(value: int | str) -> str:
    if isinstance(value, int):
        return "sha256:" + f"{value:064x}"
    return "sha256:" + value * 64


def array(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    axes: tuple[str, ...],
):
    return ArrayArtifact(
        path=f"arrays/{name}.npy",
        container_sha256=sha(abs(hash((name, "container"))) % (1 << 256)),
        payload_sha256=sha(abs(hash((name, "payload"))) % (1 << 256)),
        dtype=dtype,
        shape=shape,
        axes=axes,
    )


def roles() -> tuple[CandidateRole, ...]:
    return (
        CandidateRole("residual_mdx", sha(11), sha(12)),
        CandidateRole("median_panel", sha(13), sha(14)),
    )


def group(
    seed: int,
    *,
    source_family: str | None = None,
) -> GroupIdentity:
    values = [sha(seed + index) for index in range(7)]
    if source_family is not None:
        values[0] = source_family
    return GroupIdentity(*values)


def block(
    seed: int,
    *,
    source_family: str | None = None,
) -> WorkBlockV2:
    mixture = sha(seed + 100)
    candidate_axis = roles()
    cells = 4
    metrics = 5
    return WorkBlockV2(
        block_id=sha(seed + 200),
        task_id=TASK_ID,
        group=group(seed, source_family=source_family),
        source=SourceGridIdentity(
            mixture_pcm_sha256=mixture,
            accompaniment_pcm_sha256=sha(seed + 101),
            vocal_pcm_sha256=sha(seed + 102),
            frames=44100,
            sample_rate_hz=44100,
            channels=("FL", "FR"),
        ),
        candidates=tuple(
            CandidateArtifact(
                role=item.role,
                recipe_id=sha(seed + 300 + index),
                artifact_pcm_sha256=sha(seed + 400 + index),
                source_mixture_pcm_sha256=mixture,
            )
            for index, item in enumerate(candidate_axis)
        ),
        arrays={
            "features": array(
                f"features-{seed}",
                "<f4",
                (cells, 8),
                ("cell", "feature"),
            ),
            "exact_risks": array(
                f"risks-{seed}",
                "<f8",
                (cells, 2, metrics),
                ("cell", "candidate", "metric"),
            ),
            "risk_available": array(
                f"available-{seed}",
                "|b1",
                (cells, 2, metrics),
                ("cell", "candidate", "metric"),
            ),
            "fallback_mode": array(
                f"fallback-{seed}",
                "<i4",
                (cells, 2),
                ("cell", "candidate"),
            ),
            "cell_measure": array(
                f"measure-{seed}",
                "<f8",
                (cells,),
                ("cell",),
            ),
            "time_sample_ranges": array(
                f"times-{seed}",
                "<i8",
                (cells, 2),
                ("cell", "endpoint"),
            ),
            "frequency_band_index": array(
                f"bands-{seed}",
                "<i4",
                (cells,),
                ("cell",),
            ),
        },
    )


def dataset(*, blocks: tuple[WorkBlockV2, ...] | None = None):
    return CounterfactualDatasetContractV2(
        task_id=TASK_ID,
        candidate_axis=roles(),
        critical_metrics=("voice", "hole", "artifact"),
        secondary_metrics=("hall", "stereo"),
        feature_contract_sha256=sha(1),
        risk_contract_sha256=sha(2),
        cell_grid_contract_sha256=sha(3),
        group_contract_sha256=sha(4),
        blocks=blocks or (block(1000), block(2000), block(3000)),
    )


def split(value: CounterfactualDatasetContractV2, **kwargs):
    options = dict(
        dataset_sha256=value.sha256,
        assignments=tuple(
            SplitAssignment(item.block_id, role)
            for item, role in zip(
                value.blocks,
                ("train", "calibration", "test"),
            )
        ),
    )
    options.update(kwargs)
    return CounterfactualSplitContractV2(**options)


def test_complete_metric_axis_includes_secondary_labels_and_availability():
    value = dataset()
    value.validate()
    assert value.metric_axis == (
        "voice",
        "hole",
        "artifact",
        "hall",
        "stereo",
    )
    assert value.blocks[0].arrays["exact_risks"].shape[-1] == len(
        value.metric_axis
    )
    assert value.blocks[0].arrays["risk_available"].shape[-1] == len(
        value.metric_axis
    )
    assert value.sha256.startswith("sha256:")


def test_critical_only_arrays_are_refused_when_secondary_metrics_are_declared():
    value = dataset()
    first = value.blocks[0]
    arrays = dict(first.arrays)
    arrays["exact_risks"] = array(
        "critical-only-risks",
        "<f8",
        (4, 2, 3),
        ("cell", "candidate", "metric"),
    )
    arrays["risk_available"] = array(
        "critical-only-available",
        "|b1",
        (4, 2, 3),
        ("cell", "candidate", "metric"),
    )
    broken = dataclasses.replace(
        value,
        blocks=(
            dataclasses.replace(first, arrays=arrays),
            *value.blocks[1:],
        ),
    )
    with pytest.raises(
        CounterfactualDatasetError,
        match="exact_risks shape differs",
    ):
        broken.validate()


def test_every_group_axis_is_mandatory_in_canonical_order():
    value = dataset()
    with pytest.raises(
        CounterfactualDatasetError,
        match="every mandatory grouping axis",
    ):
        split(
            value,
            enforced_group_axes=("work_sha256",),
        ).validate(value)
    with pytest.raises(
        CounterfactualDatasetError,
        match="every mandatory grouping axis",
    ):
        split(
            value,
            enforced_group_axes=tuple(reversed(GROUP_AXES)),
        ).validate(value)


def test_shared_source_family_across_train_and_calibration_is_refused():
    shared = sha(999)
    value = dataset(
        blocks=(
            block(1000, source_family=shared),
            block(2000, source_family=shared),
            block(3000),
        )
    )
    with pytest.raises(
        CounterfactualDatasetError,
        match="source_family_sha256",
    ):
        split(value).validate(value)


def test_split_validates_all_group_axes_and_hashes_identity():
    value = dataset()
    contract = split(value)
    contract.validate(value)
    assert contract.enforced_group_axes == GROUP_AXES
    assert contract.sha256(value).startswith("sha256:")


def test_secondary_metric_axis_is_identity_bearing():
    first = dataset()
    second = dataclasses.replace(
        first,
        secondary_metrics=("hall", "stereo", "transient"),
    )
    blocks = []
    for source in second.blocks:
        arrays = dict(source.arrays)
        arrays["exact_risks"] = array(
            f"expanded-risks-{source.block_id[-8:]}",
            "<f8",
            (4, 2, 6),
            ("cell", "candidate", "metric"),
        )
        arrays["risk_available"] = array(
            f"expanded-available-{source.block_id[-8:]}",
            "|b1",
            (4, 2, 6),
            ("cell", "candidate", "metric"),
        )
        blocks.append(dataclasses.replace(source, arrays=arrays))
    second = dataclasses.replace(second, blocks=tuple(blocks))
    second.validate()
    assert first.sha256 != second.sha256


def test_duplicate_or_overlapping_metric_names_are_refused():
    with pytest.raises(CounterfactualDatasetError, match="duplicates"):
        dataclasses.replace(
            dataset(),
            secondary_metrics=("hall", "hall"),
        ).validate()
    with pytest.raises(CounterfactualDatasetError, match="duplicates"):
        dataclasses.replace(
            dataset(),
            secondary_metrics=("voice",),
        ).validate()


def test_incomplete_or_duplicate_split_assignments_are_refused():
    value = dataset()
    assignments = (
        SplitAssignment(value.blocks[0].block_id, "train"),
        SplitAssignment(value.blocks[0].block_id, "calibration"),
        SplitAssignment(value.blocks[2].block_id, "test"),
    )
    with pytest.raises(CounterfactualDatasetError, match="more than once"):
        CounterfactualSplitContractV2(
            value.sha256,
            assignments,
        ).validate(value)
    with pytest.raises(CounterfactualDatasetError, match="does not cover"):
        CounterfactualSplitContractV2(
            value.sha256,
            assignments[:1],
        ).validate(value)
