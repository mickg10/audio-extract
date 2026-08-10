"""Counterfactual-risk dataset contract v2.

V2 stores exact labels and availability for the complete ordered metric axis
(critical plus secondary) and makes every preregistered grouping axis mandatory
for train/calibration/test isolation.  The v1 contract remains available for
review history.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .counterfactual_dataset_contract import (
    ArrayArtifact,
    CandidateArtifact,
    CandidateRole,
    CounterfactualDatasetError,
    EXACT_LABEL_ROLES,
    FEATURE_INPUT_ROLES,
    GROUP_AXES,
    GroupIdentity,
    SourceGridIdentity,
    SplitAssignment,
    TASK_ID,
    _mapping_sha,
    _sha,
)

DATASET_SCHEMA_V2 = "audio-extract/counterfactual-risk-dataset/v2"
BLOCK_SCHEMA_V2 = "audio-extract/counterfactual-risk-block/v2"
SPLIT_SCHEMA_V2 = "audio-extract/counterfactual-risk-splits/v2"
REQUIRED_ARRAYS_V2 = {
    "features": ("cell", "feature"),
    "exact_risks": ("cell", "candidate", "metric"),
    "risk_available": ("cell", "candidate", "metric"),
    "fallback_mode": ("cell", "candidate"),
    "cell_measure": ("cell",),
    "time_sample_ranges": ("cell", "endpoint"),
    "frequency_band_index": ("cell",),
}


@dataclass(frozen=True)
class WorkBlockV2:
    block_id: str
    task_id: str
    group: GroupIdentity
    source: SourceGridIdentity
    candidates: tuple[CandidateArtifact, ...]
    arrays: Mapping[str, ArrayArtifact]
    feature_input_roles: tuple[str, ...] = FEATURE_INPUT_ROLES
    exact_label_roles: tuple[str, ...] = EXACT_LABEL_ROLES

    def validate(
        self,
        *,
        candidate_axis: Sequence[CandidateRole],
        metric_count: int,
    ) -> None:
        _sha(self.block_id, "block_id")
        if self.task_id != TASK_ID:
            raise CounterfactualDatasetError("work block task differs")
        self.group.validate()
        self.source.validate()
        if self.feature_input_roles != FEATURE_INPUT_ROLES:
            raise CounterfactualDatasetError("feature input roles differ")
        if self.exact_label_roles != EXACT_LABEL_ROLES:
            raise CounterfactualDatasetError("exact label roles differ")
        if len(self.candidates) != len(candidate_axis):
            raise CounterfactualDatasetError("candidate count differs from panel")
        for artifact, role in zip(self.candidates, candidate_axis):
            artifact.validate(
                role=role,
                mixture_pcm_sha256=self.source.mixture_pcm_sha256,
            )
        if set(self.arrays) != set(REQUIRED_ARRAYS_V2):
            raise CounterfactualDatasetError("work block array set differs")
        for name, axes in REQUIRED_ARRAYS_V2.items():
            self.arrays[name].validate(name)
            if self.arrays[name].axes != axes:
                raise CounterfactualDatasetError(f"{name} axes differ")
        cells = self.arrays["features"].shape[0]
        candidates = len(candidate_axis)
        expected = {
            "exact_risks": (cells, candidates, metric_count),
            "risk_available": (cells, candidates, metric_count),
            "fallback_mode": (cells, candidates),
            "cell_measure": (cells,),
            "time_sample_ranges": (cells, 2),
            "frequency_band_index": (cells,),
        }
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise CounterfactualDatasetError(
                    f"{name} shape differs: {self.arrays[name].shape} != {shape}"
                )
        if self.arrays["features"].dtype not in {"<f4", "<f8"}:
            raise CounterfactualDatasetError("features must be floating")
        if self.arrays["exact_risks"].dtype not in {"<f4", "<f8"}:
            raise CounterfactualDatasetError("exact risks must be floating")
        if self.arrays["risk_available"].dtype != "|b1":
            raise CounterfactualDatasetError("risk availability must be bool")
        if self.arrays["fallback_mode"].dtype not in {"<i4", "<i8"}:
            raise CounterfactualDatasetError("fallback mode must be integer")

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": BLOCK_SCHEMA_V2,
            "block_id": self.block_id,
            "task_id": self.task_id,
            "group": self.group.to_dict(),
            "source": self.source.to_dict(),
            "candidates": [row.to_dict() for row in self.candidates],
            "arrays": {
                name: self.arrays[name].to_dict()
                for name in sorted(self.arrays)
            },
            "feature_input_roles": list(self.feature_input_roles),
            "exact_label_roles": list(self.exact_label_roles),
        }


@dataclass(frozen=True)
class CounterfactualDatasetContractV2:
    task_id: str
    candidate_axis: tuple[CandidateRole, ...]
    critical_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    feature_contract_sha256: str
    risk_contract_sha256: str
    cell_grid_contract_sha256: str
    group_contract_sha256: str
    blocks: tuple[WorkBlockV2, ...]

    @property
    def metric_axis(self) -> tuple[str, ...]:
        return self.critical_metrics + self.secondary_metrics

    def validate(self) -> None:
        if self.task_id != TASK_ID:
            raise CounterfactualDatasetError("dataset task differs")
        if len(self.candidate_axis) < 2:
            raise CounterfactualDatasetError(
                "at least two candidates are required"
            )
        roles = []
        for role in self.candidate_axis:
            role.validate()
            roles.append(role.role)
        if len(set(roles)) != len(roles):
            raise CounterfactualDatasetError("candidate roles are not unique")
        if not self.critical_metrics:
            raise CounterfactualDatasetError("critical metric axis is empty")
        if any(
            not isinstance(name, str) or not name
            for name in self.metric_axis
        ):
            raise CounterfactualDatasetError(
                "metric axis contains an invalid name"
            )
        if len(set(self.metric_axis)) != len(self.metric_axis):
            raise CounterfactualDatasetError("metric axis contains duplicates")
        for field in (
            "feature_contract_sha256",
            "risk_contract_sha256",
            "cell_grid_contract_sha256",
            "group_contract_sha256",
        ):
            _sha(getattr(self, field), field)
        if not self.blocks:
            raise CounterfactualDatasetError("dataset has no work blocks")
        seen = set()
        for block in self.blocks:
            block.validate(
                candidate_axis=self.candidate_axis,
                metric_count=len(self.metric_axis),
            )
            if block.block_id in seen:
                raise CounterfactualDatasetError("duplicate block_id")
            seen.add(block.block_id)

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": DATASET_SCHEMA_V2,
            "task_id": self.task_id,
            "candidate_axis": [row.to_dict() for row in self.candidate_axis],
            "critical_metrics": list(self.critical_metrics),
            "secondary_metrics": list(self.secondary_metrics),
            "metric_axis": list(self.metric_axis),
            "feature_contract_sha256": self.feature_contract_sha256,
            "risk_contract_sha256": self.risk_contract_sha256,
            "cell_grid_contract_sha256": self.cell_grid_contract_sha256,
            "group_contract_sha256": self.group_contract_sha256,
            "blocks": [block.identity_dict() for block in self.blocks],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class CounterfactualSplitContractV2:
    dataset_sha256: str
    assignments: tuple[SplitAssignment, ...]
    enforced_group_axes: tuple[str, ...] = GROUP_AXES

    def validate(self, dataset: CounterfactualDatasetContractV2) -> None:
        dataset.validate()
        _sha(self.dataset_sha256, "split dataset SHA")
        if self.dataset_sha256 != dataset.sha256:
            raise CounterfactualDatasetError("split names a different dataset")
        if self.enforced_group_axes != GROUP_AXES:
            raise CounterfactualDatasetError(
                "every mandatory grouping axis must be enforced in canonical order"
            )
        blocks = {block.block_id: block for block in dataset.blocks}
        roles: dict[str, str] = {}
        for assignment in self.assignments:
            assignment.validate()
            if assignment.block_id not in blocks:
                raise CounterfactualDatasetError("split names an unknown block")
            if assignment.block_id in roles:
                raise CounterfactualDatasetError(
                    "block is assigned more than once"
                )
            roles[assignment.block_id] = assignment.role
        if set(roles) != set(blocks):
            raise CounterfactualDatasetError(
                "split does not cover every block"
            )
        observed = {
            role: {axis: set() for axis in GROUP_AXES}
            for role in ("train", "calibration", "test")
        }
        for block_id, role in roles.items():
            group = blocks[block_id].group
            for axis in GROUP_AXES:
                value = getattr(group, axis)
                for other in observed:
                    if other != role and value in observed[other][axis]:
                        raise CounterfactualDatasetError(
                            f"group leakage on {axis}: {value} appears in "
                            f"{role} and {other}"
                        )
                observed[role][axis].add(value)

    def identity_dict(
        self,
        dataset: CounterfactualDatasetContractV2,
    ) -> dict[str, Any]:
        self.validate(dataset)
        return {
            "schema": SPLIT_SCHEMA_V2,
            "dataset_sha256": self.dataset_sha256,
            "enforced_group_axes": list(self.enforced_group_axes),
            "assignments": [asdict(row) for row in self.assignments],
        }

    def sha256(self, dataset: CounterfactualDatasetContractV2) -> str:
        return _mapping_sha(self.identity_dict(dataset))
