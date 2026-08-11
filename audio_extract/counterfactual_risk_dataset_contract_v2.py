"""Stable-slot grouped exact dataset contract for the D0/R0 comparison.

V1 incorrectly treated source-specific candidate recipe/PCM identities as the
cross-work candidate panel identity.  In the real system those artifacts vary by
source.  V2 separates a stable ordered candidate *slot* (model/adapter/
construction/query semantics) from its verified per-source artifact binding.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .counterfactual_risk_dataset_contract_v1 import (
    CellGeometry,
    CounterfactualRiskDatasetError,
    GroupFamilyIdentity,
)

DATASET_SCHEMA = "audio-extract/counterfactual-risk-dataset/v2"
ROW_SCHEMA = "audio-extract/counterfactual-risk-row/v2"
SPLIT_SCHEMA = "audio-extract/counterfactual-risk-split/v2"
INFERENCE_SCHEMA = "audio-extract/counterfactual-risk-inference-row/v2"
PANEL_SCHEMA = "audio-extract/counterfactual-candidate-panel/v2"
ARTIFACT_BINDING_SCHEMA = "audio-extract/candidate-artifact-binding/v1"
PURPOSE = "D0_R0_EQUAL_COMPUTE"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class CounterfactualRiskDatasetV2Error(CounterfactualRiskDatasetError):
    """The stable-slot exact dataset, split, or projection is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise CounterfactualRiskDatasetV2Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise CounterfactualRiskDatasetV2Error(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CounterfactualRiskDatasetV2Error(
            f"{name} must be a non-empty, trim-stable string"
        )
    return value


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


def _array_sha(value: np.ndarray, *, component: str, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    header = {
        "schema": ROW_SCHEMA,
        "component": component,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def _features(value: Any) -> np.ndarray:
    if np.asarray(value).dtype == np.bool_:
        raise CounterfactualRiskDatasetV2Error(
            "features must be numeric and not boolean"
        )
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 1 or not np.all(np.isfinite(result)):
        raise CounterfactualRiskDatasetV2Error(
            "features must be a non-empty finite one-dimensional array"
        )
    return result


def _risks(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if np.asarray(value).dtype == np.bool_:
        raise CounterfactualRiskDatasetV2Error(
            "exact risks must be numeric and not boolean"
        )
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise CounterfactualRiskDatasetV2Error(
            f"exact risk shape differs: {result.shape} != {shape}"
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise CounterfactualRiskDatasetV2Error(
            "exact risks must be finite and non-negative"
        )
    return result


def _availability(value: Any, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype != np.bool_:
        raise CounterfactualRiskDatasetV2Error(
            "availability must be a boolean array matching exact risks"
        )
    return np.asarray(raw, dtype=bool)


def _canonical_hash_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    for index, value in enumerate(result):
        _sha(value, f"{name}[{index}]")
    if len(set(result)) != len(result):
        raise CounterfactualRiskDatasetV2Error(f"{name} contains duplicates")
    if result != tuple(sorted(result)):
        raise CounterfactualRiskDatasetV2Error(
            f"{name} must be in canonical lexical order"
        )
    return result


@dataclass(frozen=True)
class CandidateSlot:
    """Stable cross-source candidate semantics."""

    slot_id: str
    model_bundle_sha256: str
    adapter_bundle_sha256: str
    construction_contract_sha256: str
    query_contract_sha256: str
    output_role: str = "instrumental"

    def validate(self) -> None:
        _text(self.slot_id, "candidate slot_id")
        if self.output_role != "instrumental":
            raise CounterfactualRiskDatasetV2Error(
                "D0/R0 candidate slots must target instrumental output"
            )
        for name, value in (
            ("model_bundle_sha256", self.model_bundle_sha256),
            ("adapter_bundle_sha256", self.adapter_bundle_sha256),
            (
                "construction_contract_sha256",
                self.construction_contract_sha256,
            ),
            ("query_contract_sha256", self.query_contract_sha256),
        ):
            _sha(value, name)

    def identity_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "slot_id": self.slot_id,
            "model_bundle_sha256": self.model_bundle_sha256,
            "adapter_bundle_sha256": self.adapter_bundle_sha256,
            "construction_contract_sha256": (
                self.construction_contract_sha256
            ),
            "query_contract_sha256": self.query_contract_sha256,
            "output_role": self.output_role,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class CandidatePanel:
    slots: tuple[CandidateSlot, ...]

    def validate(self) -> None:
        if len(self.slots) < 2:
            raise CounterfactualRiskDatasetV2Error(
                "candidate panel requires at least two ordered slots"
            )
        for slot in self.slots:
            slot.validate()
        slot_ids = tuple(slot.slot_id for slot in self.slots)
        slot_shas = tuple(slot.sha256 for slot in self.slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise CounterfactualRiskDatasetV2Error(
                "candidate panel slot IDs must be unique"
            )
        if len(set(slot_shas)) != len(slot_shas):
            raise CounterfactualRiskDatasetV2Error(
                "candidate panel slot identities must be unique"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PANEL_SCHEMA,
            "ordered_slots": [slot.identity_dict() for slot in self.slots],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class CandidateArtifactBinding:
    """Verified realization of one stable slot on one source family."""

    slot_id: str
    slot_sha256: str
    recipe_id: str
    artifact_pcm_sha256: str
    recipe_semantic_sha256: str
    recipe_slot_projection_sha256: str
    source_family_sha256: str
    verifier_commit: str
    status: str = "passed"

    def validate(
        self,
        *,
        expected_slot: CandidateSlot,
        expected_source_family_sha256: str,
    ) -> None:
        if self.status != "passed":
            raise CounterfactualRiskDatasetV2Error(
                "candidate artifact binding status must be passed"
            )
        _text(self.slot_id, "artifact slot_id")
        if self.slot_id != expected_slot.slot_id:
            raise CounterfactualRiskDatasetV2Error(
                "candidate artifact slot_id differs from panel order"
            )
        for name, value in (
            ("slot_sha256", self.slot_sha256),
            ("recipe_id", self.recipe_id),
            ("artifact_pcm_sha256", self.artifact_pcm_sha256),
            ("recipe_semantic_sha256", self.recipe_semantic_sha256),
            (
                "recipe_slot_projection_sha256",
                self.recipe_slot_projection_sha256,
            ),
            ("source_family_sha256", self.source_family_sha256),
        ):
            _sha(value, name)
        if self.slot_sha256 != expected_slot.sha256:
            raise CounterfactualRiskDatasetV2Error(
                "candidate artifact names a different stable slot identity"
            )
        if self.recipe_slot_projection_sha256 != expected_slot.sha256:
            raise CounterfactualRiskDatasetV2Error(
                "verified recipe slot projection differs from the panel slot"
            )
        if self.source_family_sha256 != expected_source_family_sha256:
            raise CounterfactualRiskDatasetV2Error(
                "candidate artifact names a different source family"
            )
        _git(self.verifier_commit, "artifact binding verifier_commit")

    def identity_dict(
        self,
        *,
        expected_slot: CandidateSlot,
        expected_source_family_sha256: str,
    ) -> dict[str, Any]:
        self.validate(
            expected_slot=expected_slot,
            expected_source_family_sha256=expected_source_family_sha256,
        )
        return {
            "schema": ARTIFACT_BINDING_SCHEMA,
            "status": self.status,
            "slot_id": self.slot_id,
            "slot_sha256": self.slot_sha256,
            "recipe_id": self.recipe_id,
            "artifact_pcm_sha256": self.artifact_pcm_sha256,
            "recipe_semantic_sha256": self.recipe_semantic_sha256,
            "recipe_slot_projection_sha256": (
                self.recipe_slot_projection_sha256
            ),
            "source_family_sha256": self.source_family_sha256,
            "verifier_commit": self.verifier_commit,
        }


@dataclass(frozen=True)
class CounterfactualRiskRowV2:
    group_family: GroupFamilyIdentity
    mixture_pcm_sha256: str
    accompaniment_truth_pcm_sha256: str
    vocal_truth_pcm_sha256: str
    cell: CellGeometry
    candidate_panel: CandidatePanel
    candidate_artifacts: tuple[CandidateArtifactBinding, ...]
    metric_names: tuple[str, ...]
    metric_units: tuple[str, ...]
    metric_directions: tuple[str, ...]
    features: np.ndarray
    exact_risks: np.ndarray
    available: np.ndarray
    feature_contract_sha256: str
    metric_contract_sha256: str
    route_policy_sha256: str

    def validate(self) -> None:
        self.group_family.validate()
        self.cell.validate()
        self.candidate_panel.validate()
        for name, value in (
            ("mixture_pcm_sha256", self.mixture_pcm_sha256),
            (
                "accompaniment_truth_pcm_sha256",
                self.accompaniment_truth_pcm_sha256,
            ),
            ("vocal_truth_pcm_sha256", self.vocal_truth_pcm_sha256),
            ("feature_contract_sha256", self.feature_contract_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
            ("route_policy_sha256", self.route_policy_sha256),
        ):
            _sha(value, name)

        if len(self.candidate_artifacts) != len(self.candidate_panel.slots):
            raise CounterfactualRiskDatasetV2Error(
                "candidate artifact axis differs from the stable panel"
            )
        artifact_pairs = []
        for slot, artifact in zip(
            self.candidate_panel.slots, self.candidate_artifacts
        ):
            artifact.validate(
                expected_slot=slot,
                expected_source_family_sha256=(
                    self.group_family.source_family_sha256
                ),
            )
            artifact_pairs.append((artifact.recipe_id, artifact.artifact_pcm_sha256))
        if len(set(artifact_pairs)) != len(artifact_pairs):
            raise CounterfactualRiskDatasetV2Error(
                "per-source candidate recipe/PCM identities are duplicated"
            )

        metric_count = len(self.metric_names)
        if metric_count < 1 or not (
            len(self.metric_units) == metric_count
            and len(self.metric_directions) == metric_count
        ):
            raise CounterfactualRiskDatasetV2Error(
                "metric names, units and directions are empty or misaligned"
            )
        if len(set(self.metric_names)) != metric_count:
            raise CounterfactualRiskDatasetV2Error("metric names must be unique")
        for index, (name, unit, direction) in enumerate(
            zip(self.metric_names, self.metric_units, self.metric_directions)
        ):
            _text(name, f"metric_names[{index}]")
            _text(unit, f"metric_units[{index}]")
            if direction != "lower_is_better":
                raise CounterfactualRiskDatasetV2Error(
                    "every exact defect metric must declare lower_is_better"
                )

        features = _features(self.features)
        shape = (len(self.candidate_panel.slots), metric_count)
        risks = _risks(self.exact_risks, shape)
        available = _availability(self.available, shape)
        if np.any(risks[~available] != 0.0):
            raise CounterfactualRiskDatasetV2Error(
                "unavailable exact-risk entries must be stored as zero"
            )
        if features.shape != np.asarray(self.features).shape:
            raise CounterfactualRiskDatasetV2Error(
                "feature representation is invalid"
            )

    @property
    def feature_sha256(self) -> str:
        self.validate()
        return _array_sha(
            _features(self.features), component="features", dtype="<f8"
        )

    @property
    def exact_risks_sha256(self) -> str:
        self.validate()
        return _array_sha(
            np.asarray(self.exact_risks, dtype=np.float64),
            component="exact_risks",
            dtype="<f8",
        )

    @property
    def availability_sha256(self) -> str:
        self.validate()
        return _array_sha(
            np.asarray(self.available, dtype=np.uint8),
            component="availability",
            dtype="u1",
        )

    def artifact_documents(self) -> list[dict[str, Any]]:
        self.validate()
        return [
            artifact.identity_dict(
                expected_slot=slot,
                expected_source_family_sha256=(
                    self.group_family.source_family_sha256
                ),
            )
            for slot, artifact in zip(
                self.candidate_panel.slots, self.candidate_artifacts
            )
        ]

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": ROW_SCHEMA,
            "group_family_sha256": self.group_family.sha256,
            "group_family": self.group_family.identity_dict(),
            "mixture_pcm_sha256": self.mixture_pcm_sha256,
            "accompaniment_truth_pcm_sha256": (
                self.accompaniment_truth_pcm_sha256
            ),
            "vocal_truth_pcm_sha256": self.vocal_truth_pcm_sha256,
            "cell_sha256": self.cell.sha256,
            "cell": self.cell.identity_dict(),
            "candidate_panel_sha256": self.candidate_panel.sha256,
            "candidate_artifacts": self.artifact_documents(),
            "metric_names": list(self.metric_names),
            "metric_units": list(self.metric_units),
            "metric_directions": list(self.metric_directions),
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "route_policy_sha256": self.route_policy_sha256,
            "feature_sha256": self.feature_sha256,
            "exact_risks_sha256": self.exact_risks_sha256,
            "availability_sha256": self.availability_sha256,
        }

    @property
    def row_id(self) -> str:
        return _mapping_sha(self.identity_dict())

    def to_record(self) -> dict[str, Any]:
        result = self.identity_dict()
        result["row_id"] = self.row_id
        result["features"] = _features(self.features).tolist()
        result["exact_risks"] = np.asarray(
            self.exact_risks, dtype=np.float64
        ).tolist()
        result["available"] = np.asarray(self.available, dtype=bool).tolist()
        return result

    def to_inference_record(self) -> dict[str, Any]:
        """Project one row without clean truth or exact labels."""

        self.validate()
        return {
            "schema": INFERENCE_SCHEMA,
            "row_id": self.row_id,
            "group_family_sha256": self.group_family.sha256,
            "work_id": self.group_family.work_id,
            "recording_session_id": self.group_family.recording_session_id,
            "target_singer_id": self.group_family.target_singer_id,
            "source_family_sha256": self.group_family.source_family_sha256,
            "query_condition_sha256": (
                self.group_family.query_condition_sha256
            ),
            "mixture_pcm_sha256": self.mixture_pcm_sha256,
            "cell_sha256": self.cell.sha256,
            "cell": self.cell.identity_dict(),
            "candidate_panel_sha256": self.candidate_panel.sha256,
            "candidate_artifacts": self.artifact_documents(),
            "feature_contract_sha256": self.feature_contract_sha256,
            "features": _features(self.features).tolist(),
        }


@dataclass(frozen=True)
class DatasetManifestV2:
    rows: tuple[CounterfactualRiskRowV2, ...]
    source_commit: str
    dataset_name: str = "d0-r0-grouped-exact-v2"
    purpose: str = PURPOSE

    @classmethod
    def build(
        cls,
        rows: Sequence[CounterfactualRiskRowV2],
        *,
        source_commit: str,
        dataset_name: str = "d0-r0-grouped-exact-v2",
    ) -> DatasetManifestV2:
        ordered = tuple(sorted(rows, key=lambda row: row.row_id))
        result = cls(
            rows=ordered,
            source_commit=source_commit,
            dataset_name=dataset_name,
        )
        result.validate()
        return result

    def validate(self) -> None:
        _git(self.source_commit, "dataset source_commit")
        _text(self.dataset_name, "dataset_name")
        if self.purpose != PURPOSE:
            raise CounterfactualRiskDatasetV2Error("unknown dataset purpose")
        if not self.rows:
            raise CounterfactualRiskDatasetV2Error("dataset contains no rows")
        for row in self.rows:
            row.validate()
        row_ids = tuple(row.row_id for row in self.rows)
        if row_ids != tuple(sorted(row_ids)):
            raise CounterfactualRiskDatasetV2Error(
                "dataset rows must be in canonical row-ID order"
            )
        if len(set(row_ids)) != len(row_ids):
            raise CounterfactualRiskDatasetV2Error(
                "dataset row IDs are duplicated"
            )

        first = self.rows[0]
        common = (
            first.candidate_panel.sha256,
            first.metric_names,
            first.metric_units,
            first.metric_directions,
            first.feature_contract_sha256,
            first.metric_contract_sha256,
            first.route_policy_sha256,
            len(_features(first.features)),
        )
        for row in self.rows[1:]:
            row_common = (
                row.candidate_panel.sha256,
                row.metric_names,
                row.metric_units,
                row.metric_directions,
                row.feature_contract_sha256,
                row.metric_contract_sha256,
                row.route_policy_sha256,
                len(_features(row.features)),
            )
            if row_common != common:
                raise CounterfactualRiskDatasetV2Error(
                    "dataset rows disagree on stable panel/contracts/feature axis"
                )

        cells = tuple(
            (row.group_family.sha256, row.cell.sha256) for row in self.rows
        )
        if len(set(cells)) != len(cells):
            raise CounterfactualRiskDatasetV2Error(
                "one group contains a duplicated exact cell"
            )

    @property
    def group_family_sha256s(self) -> tuple[str, ...]:
        self.validate()
        return tuple(sorted({row.group_family.sha256 for row in self.rows}))

    def group_family_map(self) -> dict[str, GroupFamilyIdentity]:
        self.validate()
        return {
            row.group_family.sha256: row.group_family for row in self.rows
        }

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        first = self.rows[0]
        return {
            "schema": DATASET_SCHEMA,
            "purpose": self.purpose,
            "dataset_name": self.dataset_name,
            "source_commit": self.source_commit,
            "ordered_row_ids": [row.row_id for row in self.rows],
            "group_family_sha256s": list(self.group_family_sha256s),
            "candidate_panel": first.candidate_panel.identity_dict(),
            "candidate_panel_sha256": first.candidate_panel.sha256,
            "metric_names": list(first.metric_names),
            "metric_units": list(first.metric_units),
            "metric_directions": list(first.metric_directions),
            "feature_contract_sha256": first.feature_contract_sha256,
            "metric_contract_sha256": first.metric_contract_sha256,
            "route_policy_sha256": first.route_policy_sha256,
            "feature_count": len(_features(first.features)),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())

    def to_document(self) -> dict[str, Any]:
        result = self.identity_dict()
        result["dataset_sha256"] = self.sha256
        result["rows"] = [row.to_record() for row in self.rows]
        return result


@dataclass(frozen=True)
class SplitManifestV2:
    dataset_sha256: str
    train_group_family_sha256s: tuple[str, ...]
    calibration_group_family_sha256s: tuple[str, ...]
    test_group_family_sha256s: tuple[str, ...]
    split_algorithm: str
    split_seed_sha256: str
    selection_manifest_sha256: str
    source_commit: str

    def validate(self, dataset: DatasetManifestV2) -> None:
        dataset.validate()
        _sha(self.dataset_sha256, "split dataset_sha256")
        if self.dataset_sha256 != dataset.sha256:
            raise CounterfactualRiskDatasetV2Error(
                "split manifest names a different dataset"
            )
        train = _canonical_hash_tuple(
            self.train_group_family_sha256s,
            "train_group_family_sha256s",
        )
        calibration = _canonical_hash_tuple(
            self.calibration_group_family_sha256s,
            "calibration_group_family_sha256s",
        )
        test = _canonical_hash_tuple(
            self.test_group_family_sha256s,
            "test_group_family_sha256s",
        )
        if not train or not calibration or not test:
            raise CounterfactualRiskDatasetV2Error(
                "train, calibration and test groups must all be non-empty"
            )
        sets = (set(train), set(calibration), set(test))
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise CounterfactualRiskDatasetV2Error(
                "group-family partitions overlap"
            )
        expected = set(dataset.group_family_sha256s)
        if set().union(*sets) != expected:
            raise CounterfactualRiskDatasetV2Error(
                "split groups do not exactly partition the dataset"
            )
        group_map = dataset.group_family_map()
        source_sets = [
            {
                group_map[group_id].source_family_sha256
                for group_id in group_set
            }
            for group_set in sets
        ]
        if (
            source_sets[0] & source_sets[1]
            or source_sets[0] & source_sets[2]
            or source_sets[1] & source_sets[2]
        ):
            raise CounterfactualRiskDatasetV2Error(
                "one source family has derivatives in multiple partitions"
            )
        _text(self.split_algorithm, "split_algorithm")
        _sha(self.split_seed_sha256, "split_seed_sha256")
        _sha(self.selection_manifest_sha256, "selection_manifest_sha256")
        _git(self.source_commit, "split source_commit")

    def identity_dict(self, dataset: DatasetManifestV2) -> dict[str, Any]:
        self.validate(dataset)
        return {
            "schema": SPLIT_SCHEMA,
            "dataset_sha256": self.dataset_sha256,
            "train_group_family_sha256s": list(
                self.train_group_family_sha256s
            ),
            "calibration_group_family_sha256s": list(
                self.calibration_group_family_sha256s
            ),
            "test_group_family_sha256s": list(
                self.test_group_family_sha256s
            ),
            "split_algorithm": self.split_algorithm,
            "split_seed_sha256": self.split_seed_sha256,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "source_commit": self.source_commit,
        }

    def sha256(self, dataset: DatasetManifestV2) -> str:
        return _mapping_sha(self.identity_dict(dataset))


def critical_candidate_feasibility_v2(
    row: CounterfactualRiskRowV2,
    critical_thresholds: Mapping[str, float],
) -> np.ndarray:
    """Return one feasibility Boolean per stable candidate slot."""

    row.validate()
    if not critical_thresholds:
        raise CounterfactualRiskDatasetV2Error(
            "at least one critical threshold is required"
        )
    indices = {name: index for index, name in enumerate(row.metric_names)}
    feasible = np.ones(len(row.candidate_panel.slots), dtype=bool)
    risks = np.asarray(row.exact_risks, dtype=np.float64)
    available = np.asarray(row.available, dtype=bool)
    for name, threshold in critical_thresholds.items():
        if name not in indices:
            raise CounterfactualRiskDatasetV2Error(
                f"unknown critical metric {name!r}"
            )
        value = float(threshold)
        if not math.isfinite(value) or value < 0:
            raise CounterfactualRiskDatasetV2Error(
                f"critical threshold {name!r} must be finite/non-negative"
            )
        metric = indices[name]
        feasible &= available[:, metric]
        feasible &= risks[:, metric] <= value
    return feasible


def inference_records_v2(
    dataset: DatasetManifestV2,
    *,
    group_family_sha256s: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return canonical feature-only records for selected groups."""

    dataset.validate()
    selected = (
        set(dataset.group_family_sha256s)
        if group_family_sha256s is None
        else set(
            _canonical_hash_tuple(group_family_sha256s, "selected groups")
        )
    )
    unknown = selected - set(dataset.group_family_sha256s)
    if unknown:
        raise CounterfactualRiskDatasetV2Error(
            f"selected groups are absent from the dataset: {sorted(unknown)}"
        )
    records = tuple(
        row.to_inference_record()
        for row in dataset.rows
        if row.group_family.sha256 in selected
    )
    forbidden = (
        "accompaniment_truth",
        "vocal_truth",
        "exact_risk",
        "availability",
        "teacher_route",
        "oracle_margin",
    )
    for record in records:
        lowered = " ".join(record).lower()
        if any(token in lowered for token in forbidden):
            raise AssertionError("truth leaked into the inference projection")
    return records
