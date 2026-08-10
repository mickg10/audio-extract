"""Immutable dataset contract for D0/R0 opera routing experiments.

The contract separates inference features from exact offline labels and binds the
ordered candidate, metric, cell, source, and grouping axes. It contains no
feature extractor or model training code.

A valid work block contains:

* inference features derived only from the mixture and frozen candidate outputs;
* exact candidate x defect labels derived from the clean accompaniment/vocal
  references;
* an availability/fallback mode for every cell and candidate;
* immutable source and candidate PCM identities;
* mandatory source-family/work/singer/session/ensemble/room/mastering IDs.

The grouped split validator refuses leakage across every preregistered grouping
axis. A string such as ``unknown`` is not accepted as an identity; upstream data
preparation must provide a content-derived canonical SHA identity, including a
canonical sentinel identity when a concept is genuinely absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

DATASET_SCHEMA = "audio-extract/counterfactual-risk-dataset/v1"
SPLIT_SCHEMA = "audio-extract/counterfactual-risk-splits/v1"
BLOCK_SCHEMA = "audio-extract/counterfactual-risk-block/v1"
ARRAY_SCHEMA = "audio-extract/immutable-array/v1"
TASK_ID = "soloist_vs_rest"
FEATURE_INPUT_ROLES = ("mixture", "candidate_accompaniments")
EXACT_LABEL_ROLES = ("accompaniment", "vocal")
REQUIRED_ARRAYS = {
    "features": ("cell", "feature"),
    "exact_risks": ("cell", "candidate", "critical_metric"),
    "risk_available": ("cell", "candidate", "critical_metric"),
    "fallback_mode": ("cell", "candidate"),
    "cell_measure": ("cell",),
    "time_sample_ranges": ("cell", "endpoint"),
    "frequency_band_index": ("cell",),
}
GROUP_AXES = (
    "source_family_sha256",
    "work_sha256",
    "singer_sha256",
    "session_sha256",
    "ensemble_sha256",
    "room_sha256",
    "mastering_sha256",
)
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class CounterfactualDatasetError(ValueError):
    """Dataset identity, shape, provenance, or split is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise CounterfactualDatasetError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _relative_path(value: Any, name: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != text
    ):
        raise CounterfactualDatasetError(
            f"{name} must be a normalized relative POSIX path"
        )
    return text


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CounterfactualDatasetError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ArrayArtifact:
    """Content and semantic identity for one immutable array file."""

    path: str
    container_sha256: str
    payload_sha256: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]

    def validate(self, name: str) -> None:
        _relative_path(self.path, f"{name}.path")
        _sha(self.container_sha256, f"{name}.container_sha256")
        _sha(self.payload_sha256, f"{name}.payload_sha256")
        if self.dtype not in {"<f4", "<f8", "|b1", "<i4", "<i8"}:
            raise CounterfactualDatasetError(
                f"{name}.dtype is not one of the frozen representations"
            )
        if not self.shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.shape
        ):
            raise CounterfactualDatasetError(f"{name}.shape is invalid")
        if len(self.axes) != len(self.shape) or any(
            not isinstance(axis, str) or not axis for axis in self.axes
        ):
            raise CounterfactualDatasetError(f"{name}.axes are invalid")
        if len(set(self.axes)) != len(self.axes):
            raise CounterfactualDatasetError(f"{name}.axes are not unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate("array")
        return {
            "schema": ARRAY_SCHEMA,
            "path": self.path,
            "container_sha256": self.container_sha256,
            "payload_sha256": self.payload_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
        }


@dataclass(frozen=True)
class GroupIdentity:
    """Independent and correlated source-family identities for one work block."""

    source_family_sha256: str
    work_sha256: str
    singer_sha256: str
    session_sha256: str
    ensemble_sha256: str
    room_sha256: str
    mastering_sha256: str

    def validate(self) -> None:
        for axis in GROUP_AXES:
            _sha(getattr(self, axis), axis)

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {axis: getattr(self, axis) for axis in GROUP_AXES}


@dataclass(frozen=True)
class SourceGridIdentity:
    mixture_pcm_sha256: str
    accompaniment_pcm_sha256: str
    vocal_pcm_sha256: str
    frames: int
    sample_rate_hz: int
    channels: tuple[str, ...]
    sample_format: str = "float32-le-interleaved"

    def validate(self) -> None:
        for name in (
            "mixture_pcm_sha256",
            "accompaniment_pcm_sha256",
            "vocal_pcm_sha256",
        ):
            _sha(getattr(self, name), name)
        _positive_int(self.frames, "source frames")
        _positive_int(self.sample_rate_hz, "source sample rate")
        if self.channels != ("FL", "FR"):
            raise CounterfactualDatasetError(
                "counterfactual source grid must be stereo FL/FR"
            )
        if self.sample_format != "float32-le-interleaved":
            raise CounterfactualDatasetError("source sample format is not FLOAT")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mixture_pcm_sha256": self.mixture_pcm_sha256,
            "accompaniment_pcm_sha256": self.accompaniment_pcm_sha256,
            "vocal_pcm_sha256": self.vocal_pcm_sha256,
            "frames": self.frames,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": list(self.channels),
            "sample_format": self.sample_format,
        }


@dataclass(frozen=True)
class CandidateRole:
    """One stable semantic vertex in the global frozen candidate panel."""

    role: str
    model_bundle_sha256: str
    construction_sha256: str

    def validate(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise CounterfactualDatasetError("candidate role must be non-empty")
        _sha(self.model_bundle_sha256, f"candidate {self.role} model bundle")
        _sha(self.construction_sha256, f"candidate {self.role} construction")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CandidateArtifact:
    """Scene-local candidate bytes for one stable candidate role."""

    role: str
    recipe_id: str
    artifact_pcm_sha256: str
    source_mixture_pcm_sha256: str

    def validate(self, *, role: CandidateRole, mixture_pcm_sha256: str) -> None:
        role.validate()
        if self.role != role.role:
            raise CounterfactualDatasetError(
                f"candidate artifact role differs: {self.role} != {role.role}"
            )
        _sha(self.recipe_id, f"candidate {self.role} recipe")
        _sha(self.artifact_pcm_sha256, f"candidate {self.role} PCM")
        _sha(
            self.source_mixture_pcm_sha256,
            f"candidate {self.role} source mixture",
        )
        if self.source_mixture_pcm_sha256 != mixture_pcm_sha256:
            raise CounterfactualDatasetError(
                f"candidate {self.role} does not derive from the exact mixture"
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class WorkBlock:
    """One exact work/session block with inference and privileged arrays."""

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
        critical_metric_count: int,
    ) -> None:
        _sha(self.block_id, "block_id")
        if self.task_id != TASK_ID:
            raise CounterfactualDatasetError(
                f"work block task differs: {self.task_id!r}"
            )
        self.group.validate()
        self.source.validate()
        if self.feature_input_roles != FEATURE_INPUT_ROLES:
            raise CounterfactualDatasetError(
                "features may use only mixture and candidate accompaniments"
            )
        if self.exact_label_roles != EXACT_LABEL_ROLES:
            raise CounterfactualDatasetError(
                "exact labels must use accompaniment and vocal references"
            )
        if len(self.candidates) != len(candidate_axis):
            raise CounterfactualDatasetError(
                "block candidate count differs from the global panel"
            )
        for artifact, role in zip(self.candidates, candidate_axis):
            artifact.validate(
                role=role,
                mixture_pcm_sha256=self.source.mixture_pcm_sha256,
            )
        if not isinstance(self.arrays, Mapping) or set(self.arrays) != set(
            REQUIRED_ARRAYS
        ):
            raise CounterfactualDatasetError(
                "work block array set differs from the frozen contract"
            )
        for name, expected_axes in REQUIRED_ARRAYS.items():
            array = self.arrays[name]
            array.validate(name)
            if array.axes != expected_axes:
                raise CounterfactualDatasetError(
                    f"{name} axes differ: {array.axes} != {expected_axes}"
                )
        features = self.arrays["features"]
        cell_count = features.shape[0]
        candidate_count = len(candidate_axis)
        expected_shapes = {
            "exact_risks": (
                cell_count,
                candidate_count,
                critical_metric_count,
            ),
            "risk_available": (
                cell_count,
                candidate_count,
                critical_metric_count,
            ),
            "fallback_mode": (cell_count, candidate_count),
            "cell_measure": (cell_count,),
            "time_sample_ranges": (cell_count, 2),
            "frequency_band_index": (cell_count,),
        }
        for name, expected in expected_shapes.items():
            if self.arrays[name].shape != expected:
                raise CounterfactualDatasetError(
                    f"{name} shape differs: {self.arrays[name].shape} != {expected}"
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
            "schema": BLOCK_SCHEMA,
            "block_id": self.block_id,
            "task_id": self.task_id,
            "group": self.group.to_dict(),
            "source": self.source.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "arrays": {
                name: self.arrays[name].to_dict()
                for name in sorted(self.arrays)
            },
            "feature_input_roles": list(self.feature_input_roles),
            "exact_label_roles": list(self.exact_label_roles),
        }


@dataclass(frozen=True)
class CounterfactualDatasetContract:
    task_id: str
    candidate_axis: tuple[CandidateRole, ...]
    critical_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    feature_contract_sha256: str
    risk_contract_sha256: str
    cell_grid_contract_sha256: str
    group_contract_sha256: str
    blocks: tuple[WorkBlock, ...]

    def validate(self) -> None:
        if self.task_id != TASK_ID:
            raise CounterfactualDatasetError("dataset task is not soloist_vs_rest")
        if len(self.candidate_axis) < 2:
            raise CounterfactualDatasetError(
                "dataset needs at least two frozen candidates"
            )
        roles = []
        for role in self.candidate_axis:
            role.validate()
            roles.append(role.role)
        if len(set(roles)) != len(roles):
            raise CounterfactualDatasetError("candidate roles are not unique")
        for name, metrics in (
            ("critical_metrics", self.critical_metrics),
            ("secondary_metrics", self.secondary_metrics),
        ):
            if any(not isinstance(metric, str) or not metric for metric in metrics):
                raise CounterfactualDatasetError(f"{name} contains an invalid name")
            if len(set(metrics)) != len(metrics):
                raise CounterfactualDatasetError(f"{name} contains duplicates")
        if not self.critical_metrics:
            raise CounterfactualDatasetError("critical metric axis is empty")
        if set(self.critical_metrics) & set(self.secondary_metrics):
            raise CounterfactualDatasetError(
                "critical and secondary metric axes overlap"
            )
        for name in (
            "feature_contract_sha256",
            "risk_contract_sha256",
            "cell_grid_contract_sha256",
            "group_contract_sha256",
        ):
            _sha(getattr(self, name), name)
        if not self.blocks:
            raise CounterfactualDatasetError("dataset has no work blocks")
        seen = set()
        for block in self.blocks:
            block.validate(
                candidate_axis=self.candidate_axis,
                critical_metric_count=len(self.critical_metrics),
            )
            if block.block_id in seen:
                raise CounterfactualDatasetError("duplicate dataset block_id")
            seen.add(block.block_id)

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": DATASET_SCHEMA,
            "task_id": self.task_id,
            "candidate_axis": [role.to_dict() for role in self.candidate_axis],
            "critical_metrics": list(self.critical_metrics),
            "secondary_metrics": list(self.secondary_metrics),
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
class SplitAssignment:
    block_id: str
    role: str

    def validate(self) -> None:
        _sha(self.block_id, "split block_id")
        if self.role not in {"train", "calibration", "test"}:
            raise CounterfactualDatasetError(
                f"unknown split role: {self.role!r}"
            )


@dataclass(frozen=True)
class CounterfactualSplitContract:
    dataset_sha256: str
    enforced_group_axes: tuple[str, ...]
    assignments: tuple[SplitAssignment, ...]

    def validate(self, dataset: CounterfactualDatasetContract) -> None:
        dataset.validate()
        _sha(self.dataset_sha256, "split dataset SHA")
        if self.dataset_sha256 != dataset.sha256:
            raise CounterfactualDatasetError(
                "split contract names a different dataset"
            )
        if not self.enforced_group_axes or any(
            axis not in GROUP_AXES for axis in self.enforced_group_axes
        ):
            raise CounterfactualDatasetError(
                "split enforced_group_axes are invalid"
            )
        if len(set(self.enforced_group_axes)) != len(
            self.enforced_group_axes
        ):
            raise CounterfactualDatasetError(
                "split enforced_group_axes contain duplicates"
            )
        by_block = {block.block_id: block for block in dataset.blocks}
        roles = {}
        for assignment in self.assignments:
            assignment.validate()
            if assignment.block_id in roles:
                raise CounterfactualDatasetError(
                    "dataset block is assigned more than once"
                )
            if assignment.block_id not in by_block:
                raise CounterfactualDatasetError(
                    "split assignment names an unknown block"
                )
            roles[assignment.block_id] = assignment.role
        if set(roles) != set(by_block):
            raise CounterfactualDatasetError(
                "split assignments do not cover every dataset block"
            )
        role_values: dict[str, dict[str, set[str]]] = {
            role: {axis: set() for axis in self.enforced_group_axes}
            for role in ("train", "calibration", "test")
        }
        for block_id, role in roles.items():
            group = by_block[block_id].group
            for axis in self.enforced_group_axes:
                value = getattr(group, axis)
                for other_role in role_values:
                    if other_role != role and value in role_values[other_role][axis]:
                        raise CounterfactualDatasetError(
                            f"group leakage on {axis}: {value} appears in "
                            f"{role} and {other_role}"
                        )
                role_values[role][axis].add(value)

    def identity_dict(
        self, dataset: CounterfactualDatasetContract
    ) -> dict[str, Any]:
        self.validate(dataset)
        return {
            "schema": SPLIT_SCHEMA,
            "dataset_sha256": self.dataset_sha256,
            "enforced_group_axes": list(self.enforced_group_axes),
            "assignments": [asdict(value) for value in self.assignments],
        }

    def sha256(self, dataset: CounterfactualDatasetContract) -> str:
        return _mapping_sha(self.identity_dict(dataset))
