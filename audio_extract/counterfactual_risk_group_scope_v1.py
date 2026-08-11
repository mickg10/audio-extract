"""Source-family exchangeability scope for grouped split-conformal D0/R0.

Cells, query variants, resolutions, and crops are not independent calibration
units.  This module derives immutable train/calibration/test subset manifests
from the exact grouped dataset and split, counts only unique source-family
certificates, and binds the statistical scope to the grouped linear student's
existing calibration certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json
import math
import re

from .counterfactual_risk_dataset_contract_v2 import (
    DatasetManifestV2,
    SplitManifestV2,
)
from .group_conformal_risk_student import GroupedLinearRiskStudent

SUBSET_SCHEMA = "audio-extract/counterfactual-risk-group-subset/v1"
SCOPE_SCHEMA = "audio-extract/counterfactual-risk-exchangeability-scope/v1"
BOUND_MODEL_SCHEMA = "audio-extract/bound-grouped-risk-student/v1"
_GROUP_UNIT = "source_family_sha256/v2"
_SCORE_MODE = "max_cells_candidates_metrics/v1"
_GROUP_WEIGHT_MODE = "one_source_family_one_vote/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupScopeError(ValueError):
    """A grouped subset, exchangeability scope, or bound student is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupScopeError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupScopeError(
            f"{name} must be 40 lowercase hexadecimal characters"
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


@dataclass(frozen=True)
class GroupSubsetManifestV1:
    partition: str
    dataset_sha256: str
    split_sha256: str
    group_family_sha256s: tuple[str, ...]
    source_family_sha256s: tuple[str, ...]
    ordered_row_ids: tuple[str, ...]

    @classmethod
    def build(
        cls,
        dataset: DatasetManifestV2,
        split: SplitManifestV2,
        *,
        partition: str,
    ) -> "GroupSubsetManifestV1":
        split.validate(dataset)
        mapping = {
            "train": split.train_group_family_sha256s,
            "calibration": split.calibration_group_family_sha256s,
            "test": split.test_group_family_sha256s,
        }
        if partition not in mapping:
            raise GroupScopeError(f"unknown grouped partition {partition!r}")
        groups = tuple(mapping[partition])
        group_set = set(groups)
        rows = tuple(
            row for row in dataset.rows
            if row.group_family.sha256 in group_set
        )
        result = cls(
            partition=partition,
            dataset_sha256=dataset.sha256,
            split_sha256=split.sha256(dataset),
            group_family_sha256s=groups,
            source_family_sha256s=tuple(
                sorted(
                    {
                        row.group_family.source_family_sha256
                        for row in rows
                    }
                )
            ),
            ordered_row_ids=tuple(row.row_id for row in rows),
        )
        result.validate(dataset, split)
        return result

    def validate(
        self, dataset: DatasetManifestV2, split: SplitManifestV2
    ) -> None:
        split.validate(dataset)
        if self.partition not in {"train", "calibration", "test"}:
            raise GroupScopeError("unknown subset partition")
        _sha(self.dataset_sha256, "subset dataset_sha256")
        _sha(self.split_sha256, "subset split_sha256")
        if self.dataset_sha256 != dataset.sha256:
            raise GroupScopeError("subset names a different dataset")
        if self.split_sha256 != split.sha256(dataset):
            raise GroupScopeError("subset names a different split")
        expected = GroupSubsetManifestV1.build_unchecked(
            dataset, split, partition=self.partition
        )
        fields = (
            "group_family_sha256s",
            "source_family_sha256s",
            "ordered_row_ids",
        )
        different = sorted(
            name for name in fields
            if getattr(self, name) != getattr(expected, name)
        )
        if different:
            raise GroupScopeError(
                f"subset differs from exact dataset/split fields: {different}"
            )
        if not self.group_family_sha256s or not self.source_family_sha256s:
            raise GroupScopeError("grouped subset is empty")
        for name, values in (
            ("group_family_sha256s", self.group_family_sha256s),
            ("source_family_sha256s", self.source_family_sha256s),
            ("ordered_row_ids", self.ordered_row_ids),
        ):
            for index, value in enumerate(values):
                _sha(value, f"{name}[{index}]")
        if self.source_family_sha256s != tuple(
            sorted(set(self.source_family_sha256s))
        ):
            raise GroupScopeError(
                "source-family calibration units must be unique and canonical"
            )

    @classmethod
    def build_unchecked(
        cls,
        dataset: DatasetManifestV2,
        split: SplitManifestV2,
        *,
        partition: str,
    ) -> "GroupSubsetManifestV1":
        mapping = {
            "train": split.train_group_family_sha256s,
            "calibration": split.calibration_group_family_sha256s,
            "test": split.test_group_family_sha256s,
        }
        groups = tuple(mapping[partition])
        group_set = set(groups)
        rows = tuple(
            row for row in dataset.rows
            if row.group_family.sha256 in group_set
        )
        return cls(
            partition=partition,
            dataset_sha256=dataset.sha256,
            split_sha256=split.sha256(dataset),
            group_family_sha256s=groups,
            source_family_sha256s=tuple(
                sorted(
                    {
                        row.group_family.source_family_sha256
                        for row in rows
                    }
                )
            ),
            ordered_row_ids=tuple(row.row_id for row in rows),
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": SUBSET_SCHEMA,
            "partition": self.partition,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "group_family_sha256s": list(self.group_family_sha256s),
            "source_family_sha256s": list(self.source_family_sha256s),
            "ordered_row_ids": list(self.ordered_row_ids),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class GroupCalibrationScopeV1:
    train: GroupSubsetManifestV1
    calibration: GroupSubsetManifestV1
    test: GroupSubsetManifestV1
    source_registry_sha256: str
    partition_registry_sha256: str
    feature_registry_sha256: str
    metric_scales_sha256: str
    target_coverage: float
    exchangeability_policy_sha256: str
    hyperparameter_selection_manifest_sha256: str
    query_quality_strata_policy_sha256: str
    source_commit: str
    verifier_commit: str
    group_unit: str = _GROUP_UNIT
    group_score_mode: str = _SCORE_MODE
    group_weight_mode: str = _GROUP_WEIGHT_MODE
    status: str = "passed"

    @classmethod
    def build(
        cls,
        dataset: DatasetManifestV2,
        split: SplitManifestV2,
        *,
        source_registry_sha256: str,
        partition_registry_sha256: str,
        feature_registry_sha256: str,
        metric_scales_sha256: str,
        target_coverage: float,
        exchangeability_policy_sha256: str,
        hyperparameter_selection_manifest_sha256: str,
        query_quality_strata_policy_sha256: str,
        source_commit: str,
        verifier_commit: str,
    ) -> "GroupCalibrationScopeV1":
        result = cls(
            train=GroupSubsetManifestV1.build(
                dataset, split, partition="train"
            ),
            calibration=GroupSubsetManifestV1.build(
                dataset, split, partition="calibration"
            ),
            test=GroupSubsetManifestV1.build(
                dataset, split, partition="test"
            ),
            source_registry_sha256=source_registry_sha256,
            partition_registry_sha256=partition_registry_sha256,
            feature_registry_sha256=feature_registry_sha256,
            metric_scales_sha256=metric_scales_sha256,
            target_coverage=target_coverage,
            exchangeability_policy_sha256=exchangeability_policy_sha256,
            hyperparameter_selection_manifest_sha256=(
                hyperparameter_selection_manifest_sha256
            ),
            query_quality_strata_policy_sha256=(
                query_quality_strata_policy_sha256
            ),
            source_commit=source_commit,
            verifier_commit=verifier_commit,
        )
        result.validate(dataset, split)
        return result

    def validate(
        self, dataset: DatasetManifestV2, split: SplitManifestV2
    ) -> None:
        if self.status != "passed":
            raise GroupScopeError("calibration scope status must be passed")
        if self.group_unit != _GROUP_UNIT:
            raise GroupScopeError("calibration unit must be the source-family SHA")
        if self.group_score_mode != _SCORE_MODE:
            raise GroupScopeError("unknown simultaneous group score mode")
        if self.group_weight_mode != _GROUP_WEIGHT_MODE:
            raise GroupScopeError("unknown calibration group weight mode")
        self.train.validate(dataset, split)
        self.calibration.validate(dataset, split)
        self.test.validate(dataset, split)
        if (
            self.train.partition,
            self.calibration.partition,
            self.test.partition,
        ) != ("train", "calibration", "test"):
            raise GroupScopeError("scope subset roles are misassigned")
        source_sets = (
            set(self.train.source_family_sha256s),
            set(self.calibration.source_family_sha256s),
            set(self.test.source_family_sha256s),
        )
        if (
            source_sets[0] & source_sets[1]
            or source_sets[0] & source_sets[2]
            or source_sets[1] & source_sets[2]
        ):
            raise GroupScopeError(
                "source-family exchangeability units cross scope partitions"
            )
        for name, value in (
            ("source_registry_sha256", self.source_registry_sha256),
            ("partition_registry_sha256", self.partition_registry_sha256),
            ("feature_registry_sha256", self.feature_registry_sha256),
            ("metric_scales_sha256", self.metric_scales_sha256),
            (
                "exchangeability_policy_sha256",
                self.exchangeability_policy_sha256,
            ),
            (
                "hyperparameter_selection_manifest_sha256",
                self.hyperparameter_selection_manifest_sha256,
            ),
            (
                "query_quality_strata_policy_sha256",
                self.query_quality_strata_policy_sha256,
            ),
        ):
            _sha(value, name)
        coverage = float(self.target_coverage)
        if not math.isfinite(coverage) or not 0.5 < coverage < 1.0:
            raise GroupScopeError(
                "target coverage must lie strictly between 0.5 and 1"
            )
        groups = len(self.calibration.source_family_sha256s)
        rank = math.ceil((groups + 1) * coverage)
        if rank > groups:
            raise GroupScopeError(
                "insufficient independent source families for finite coverage"
            )
        _git(self.source_commit, "scope source_commit")
        _git(self.verifier_commit, "scope verifier_commit")
        if dataset.source_commit != self.source_commit:
            raise GroupScopeError(
                "scope and exact dataset use different source commits"
            )

    @property
    def calibration_group_count(self) -> int:
        return len(self.calibration.source_family_sha256s)

    @property
    def conformal_rank(self) -> int:
        return math.ceil(
            (self.calibration_group_count + 1) * self.target_coverage
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": SCOPE_SCHEMA,
            "status": self.status,
            "group_unit": self.group_unit,
            "group_score_mode": self.group_score_mode,
            "group_weight_mode": self.group_weight_mode,
            "train_subset_sha256": self.train.sha256,
            "calibration_subset_sha256": self.calibration.sha256,
            "test_subset_sha256": self.test.sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "partition_registry_sha256": self.partition_registry_sha256,
            "feature_registry_sha256": self.feature_registry_sha256,
            "metric_scales_sha256": self.metric_scales_sha256,
            "target_coverage": float(self.target_coverage),
            "calibration_group_count": self.calibration_group_count,
            "conformal_rank": self.conformal_rank,
            "exchangeability_policy_sha256": (
                self.exchangeability_policy_sha256
            ),
            "hyperparameter_selection_manifest_sha256": (
                self.hyperparameter_selection_manifest_sha256
            ),
            "query_quality_strata_policy_sha256": (
                self.query_quality_strata_policy_sha256
            ),
            "source_commit": self.source_commit,
            "verifier_commit": self.verifier_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class BoundGroupedRiskStudentV1:
    student: GroupedLinearRiskStudent
    scope: GroupCalibrationScopeV1

    def validate(
        self, dataset: DatasetManifestV2, split: SplitManifestV2
    ) -> None:
        self.student.validate()
        self.scope.validate(dataset, split)
        calibration = self.student.calibration
        first = dataset.rows[0]
        comparisons = {
            "training subset": (
                calibration.training_manifest_sha256,
                self.scope.train.sha256,
            ),
            "calibration subset": (
                calibration.calibration_manifest_sha256,
                self.scope.calibration.sha256,
            ),
            "split": (
                calibration.split_manifest_sha256,
                split.sha256(dataset),
            ),
            "candidate panel": (
                self.student.candidate_panel_sha256,
                first.candidate_panel.sha256,
            ),
            "feature contract": (
                self.student.feature_contract_sha256,
                first.feature_contract_sha256,
            ),
            "metric contract": (
                self.student.metric_contract_sha256,
                first.metric_contract_sha256,
            ),
            "target coverage": (
                float(calibration.target_coverage),
                float(self.scope.target_coverage),
            ),
            "calibration group count": (
                calibration.calibration_group_count,
                self.scope.calibration_group_count,
            ),
            "conformal rank": (
                calibration.conformal_rank,
                self.scope.conformal_rank,
            ),
            "group score mode": (
                calibration.group_score_mode,
                self.scope.group_score_mode,
            ),
            "code commit": (
                calibration.code_commit,
                self.scope.source_commit,
            ),
        }
        different = sorted(
            name for name, (actual, expected) in comparisons.items()
            if actual != expected
        )
        if different:
            raise GroupScopeError(
                f"student calibration differs from frozen grouped scope: {different}"
            )

    def identity_dict(
        self, dataset: DatasetManifestV2, split: SplitManifestV2
    ) -> dict[str, Any]:
        self.validate(dataset, split)
        return {
            "schema": BOUND_MODEL_SCHEMA,
            "student_sha256": self.student.sha256,
            "scope_sha256": self.scope.sha256,
            "dataset_sha256": dataset.sha256,
            "split_sha256": split.sha256(dataset),
        }

    def sha256(
        self, dataset: DatasetManifestV2, split: SplitManifestV2
    ) -> str:
        return _mapping_sha(self.identity_dict(dataset, split))
