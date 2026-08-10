"""Identity-complete grouped exact dataset contract for the D0/R0 comparison.

This module is deliberately offline-only.  It binds exact counterfactual labels,
anti-leakage group families, inference features, and the frozen split used to
compare structured-route imitation (D0) with counterfactual-risk prediction
(R0).  It does not fit a model, render audio, or expose clean truth through the
inference projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np

DATASET_SCHEMA = "audio-extract/counterfactual-risk-dataset/v1"
ROW_SCHEMA = "audio-extract/counterfactual-risk-row/v1"
SPLIT_SCHEMA = "audio-extract/counterfactual-risk-split/v1"
INFERENCE_SCHEMA = "audio-extract/counterfactual-risk-inference-row/v1"
PURPOSE = "D0_R0_EQUAL_COMPUTE"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class CounterfactualRiskDatasetError(ValueError):
    """The exact dataset, split, or inference projection is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise CounterfactualRiskDatasetError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise CounterfactualRiskDatasetError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CounterfactualRiskDatasetError(
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
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 1 or not np.all(np.isfinite(result)):
        raise CounterfactualRiskDatasetError(
            "features must be a non-empty finite one-dimensional array"
        )
    return result


def _risks(value: Any, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise CounterfactualRiskDatasetError(
            f"exact risk shape differs: {result.shape} != {shape}"
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise CounterfactualRiskDatasetError(
            "exact risks must be finite and non-negative"
        )
    return result


def _availability(value: Any, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype != np.bool_:
        raise CounterfactualRiskDatasetError(
            "availability must be a boolean array matching exact risks"
        )
    return np.asarray(raw, dtype=bool)


def _canonical_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    for index, value in enumerate(result):
        _sha(value, f"{name}[{index}]")
    if len(set(result)) != len(result):
        raise CounterfactualRiskDatasetError(f"{name} contains duplicates")
    if result != tuple(sorted(result)):
        raise CounterfactualRiskDatasetError(
            f"{name} must be in canonical lexical order"
        )
    return result


@dataclass(frozen=True)
class CandidateIdentity:
    recipe_id: str
    artifact_pcm_sha256: str

    def validate(self) -> None:
        _sha(self.recipe_id, "candidate recipe_id")
        _sha(self.artifact_pcm_sha256, "candidate artifact_pcm_sha256")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "recipe_id": self.recipe_id,
            "artifact_pcm_sha256": self.artifact_pcm_sha256,
        }


@dataclass(frozen=True)
class GroupFamilyIdentity:
    """One indivisible train/calibration/test family."""

    work_id: str
    recording_session_id: str
    target_singer_id: str
    source_family_sha256: str
    query_condition_sha256: str

    def validate(self) -> None:
        _text(self.work_id, "work_id")
        _text(self.recording_session_id, "recording_session_id")
        _text(self.target_singer_id, "target_singer_id")
        _sha(self.source_family_sha256, "source_family_sha256")
        _sha(self.query_condition_sha256, "query_condition_sha256")

    def identity_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "work_id": self.work_id,
            "recording_session_id": self.recording_session_id,
            "target_singer_id": self.target_singer_id,
            "source_family_sha256": self.source_family_sha256,
            "query_condition_sha256": self.query_condition_sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class CellGeometry:
    sample_rate_hz: int
    resolution_ms: int
    start_frame: int
    end_frame: int
    band_low_hz: int
    band_high_hz: int
    spectral_grid_sha256: str

    def validate(self) -> None:
        for name in (
            "sample_rate_hz",
            "resolution_ms",
            "start_frame",
            "end_frame",
            "band_low_hz",
            "band_high_hz",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CounterfactualRiskDatasetError(f"{name} must be an integer")
        if self.sample_rate_hz < 2 or self.resolution_ms < 1:
            raise CounterfactualRiskDatasetError(
                "sample rate and resolution must be positive"
            )
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise CounterfactualRiskDatasetError("cell time range is invalid")
        if (
            self.band_low_hz < 0
            or self.band_high_hz <= self.band_low_hz
            or self.band_high_hz > self.sample_rate_hz // 2
        ):
            raise CounterfactualRiskDatasetError("cell band range is invalid")
        _sha(self.spectral_grid_sha256, "spectral_grid_sha256")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "resolution_ms": self.resolution_ms,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "band_low_hz": self.band_low_hz,
            "band_high_hz": self.band_high_hz,
            "spectral_grid_sha256": self.spectral_grid_sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class CounterfactualRiskRow:
    group_family: GroupFamilyIdentity
    mixture_pcm_sha256: str
    accompaniment_truth_pcm_sha256: str
    vocal_truth_pcm_sha256: str
    cell: CellGeometry
    candidates: tuple[CandidateIdentity, ...]
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
        _sha(self.mixture_pcm_sha256, "mixture_pcm_sha256")
        _sha(
            self.accompaniment_truth_pcm_sha256,
            "accompaniment_truth_pcm_sha256",
        )
        _sha(self.vocal_truth_pcm_sha256, "vocal_truth_pcm_sha256")
        for name, value in (
            ("feature_contract_sha256", self.feature_contract_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
            ("route_policy_sha256", self.route_policy_sha256),
        ):
            _sha(value, name)

        if len(self.candidates) < 2:
            raise CounterfactualRiskDatasetError(
                "at least two ordered candidate identities are required"
            )
        for candidate in self.candidates:
            candidate.validate()
        candidate_pairs = tuple(
            (candidate.recipe_id, candidate.artifact_pcm_sha256)
            for candidate in self.candidates
        )
        if len(set(candidate_pairs)) != len(candidate_pairs):
            raise CounterfactualRiskDatasetError(
                "candidate recipe/PCM pairs must be unique"
            )

        metric_count = len(self.metric_names)
        if metric_count < 1:
            raise CounterfactualRiskDatasetError("metric axis is empty")
        if not (
            len(self.metric_units) == metric_count
            and len(self.metric_directions) == metric_count
        ):
            raise CounterfactualRiskDatasetError(
                "metric names, units and directions differ in length"
            )
        if len(set(self.metric_names)) != metric_count:
            raise CounterfactualRiskDatasetError("metric names must be unique")
        for index, (name, unit, direction) in enumerate(
            zip(self.metric_names, self.metric_units, self.metric_directions)
        ):
            _text(name, f"metric_names[{index}]")
            _text(unit, f"metric_units[{index}]")
            if direction != "lower_is_better":
                raise CounterfactualRiskDatasetError(
                    "every exact defect metric must declare lower_is_better"
                )

        feature_values = _features(self.features)
        shape = (len(self.candidates), metric_count)
        risks = _risks(self.exact_risks, shape)
        available = _availability(self.available, shape)
        if np.any(risks[~available] != 0.0):
            raise CounterfactualRiskDatasetError(
                "unavailable exact-risk entries must be stored as zero"
            )
        if feature_values.shape != np.asarray(self.features).shape:
            raise CounterfactualRiskDatasetError("feature representation is invalid")

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
            "candidates": [candidate.to_dict() for candidate in self.candidates],
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
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "feature_contract_sha256": self.feature_contract_sha256,
            "feature_sha256": self.feature_sha256,
            "features": _features(self.features).tolist(),
        }


@dataclass(frozen=True)
class DatasetManifest:
    rows: tuple[CounterfactualRiskRow, ...]
    source_commit: str
    dataset_name: str = "d0-r0-grouped-exact-v1"
    purpose: str = PURPOSE

    @classmethod
    def build(
        cls,
        rows: Sequence[CounterfactualRiskRow],
        *,
        source_commit: str,
        dataset_name: str = "d0-r0-grouped-exact-v1",
    ) -> "DatasetManifest":
        ordered = tuple(sorted(tuple(rows), key=lambda row: row.row_id))
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
            raise CounterfactualRiskDatasetError("unknown dataset purpose")
        if not self.rows:
            raise CounterfactualRiskDatasetError("dataset contains no rows")
        for row in self.rows:
            row.validate()
        row_ids = tuple(row.row_id for row in self.rows)
        if row_ids != tuple(sorted(row_ids)):
            raise CounterfactualRiskDatasetError(
                "dataset rows must be in canonical row-ID order"
            )
        if len(set(row_ids)) != len(row_ids):
            raise CounterfactualRiskDatasetError("dataset row IDs are duplicated")

        first = self.rows[0]
        common = (
            first.candidates,
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
                row.candidates,
                row.metric_names,
                row.metric_units,
                row.metric_directions,
                row.feature_contract_sha256,
                row.metric_contract_sha256,
                row.route_policy_sha256,
                len(_features(row.features)),
            )
            if row_common != common:
                raise CounterfactualRiskDatasetError(
                    "dataset rows disagree on frozen panel/contracts/feature axis"
                )

        cells = tuple(
            (row.group_family.sha256, row.cell.sha256) for row in self.rows
        )
        if len(set(cells)) != len(cells):
            raise CounterfactualRiskDatasetError(
                "one group contains a duplicated exact cell"
            )

    @property
    def group_family_sha256s(self) -> tuple[str, ...]:
        self.validate()
        return tuple(sorted({row.group_family.sha256 for row in self.rows}))

    def group_family_map(self) -> dict[str, GroupFamilyIdentity]:
        self.validate()
        result: dict[str, GroupFamilyIdentity] = {}
        for row in self.rows:
            group_id = row.group_family.sha256
            current = result.get(group_id)
            if current is not None and current != row.group_family:
                raise CounterfactualRiskDatasetError(
                    "group-family hash maps to inconsistent content"
                )
            result[group_id] = row.group_family
        return result

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
            "candidate_panel": [
                candidate.to_dict() for candidate in first.candidates
            ],
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
class SplitManifest:
    dataset_sha256: str
    train_group_family_sha256s: tuple[str, ...]
    calibration_group_family_sha256s: tuple[str, ...]
    test_group_family_sha256s: tuple[str, ...]
    split_algorithm: str
    split_seed_sha256: str
    selection_manifest_sha256: str
    source_commit: str

    def validate(self, dataset: DatasetManifest) -> None:
        dataset.validate()
        _sha(self.dataset_sha256, "split dataset_sha256")
        if self.dataset_sha256 != dataset.sha256:
            raise CounterfactualRiskDatasetError(
                "split manifest names a different dataset"
            )
        train = _canonical_tuple(
            self.train_group_family_sha256s,
            "train_group_family_sha256s",
        )
        calibration = _canonical_tuple(
            self.calibration_group_family_sha256s,
            "calibration_group_family_sha256s",
        )
        test = _canonical_tuple(
            self.test_group_family_sha256s,
            "test_group_family_sha256s",
        )
        if not train or not calibration or not test:
            raise CounterfactualRiskDatasetError(
                "train, calibration and test groups must all be non-empty"
            )
        sets = (set(train), set(calibration), set(test))
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise CounterfactualRiskDatasetError(
                "group-family partitions overlap"
            )
        expected = set(dataset.group_family_sha256s)
        if set().union(*sets) != expected:
            raise CounterfactualRiskDatasetError(
                "split groups do not exactly partition the dataset"
            )

        group_map = dataset.group_family_map()
        source_sets = []
        for group_set in sets:
            source_sets.append(
                {
                    group_map[group_id].source_family_sha256
                    for group_id in group_set
                }
            )
        if (
            source_sets[0] & source_sets[1]
            or source_sets[0] & source_sets[2]
            or source_sets[1] & source_sets[2]
        ):
            raise CounterfactualRiskDatasetError(
                "one source family has derivatives in multiple partitions"
            )

        _text(self.split_algorithm, "split_algorithm")
        _sha(self.split_seed_sha256, "split_seed_sha256")
        _sha(self.selection_manifest_sha256, "selection_manifest_sha256")
        _git(self.source_commit, "split source_commit")

    def identity_dict(self, dataset: DatasetManifest) -> dict[str, Any]:
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

    def sha256(self, dataset: DatasetManifest) -> str:
        return _mapping_sha(self.identity_dict(dataset))


def critical_candidate_feasibility(
    row: CounterfactualRiskRow,
    critical_thresholds: Mapping[str, float],
) -> np.ndarray:
    """Return one feasibility Boolean per candidate; missing evidence fails."""

    row.validate()
    if not critical_thresholds:
        raise CounterfactualRiskDatasetError(
            "at least one critical threshold is required"
        )
    indices = {name: index for index, name in enumerate(row.metric_names)}
    feasible = np.ones(len(row.candidates), dtype=bool)
    risks = np.asarray(row.exact_risks, dtype=np.float64)
    available = np.asarray(row.available, dtype=bool)
    for name, threshold in critical_thresholds.items():
        if name not in indices:
            raise CounterfactualRiskDatasetError(
                f"unknown critical metric {name!r}"
            )
        value = float(threshold)
        if not math.isfinite(value) or value < 0:
            raise CounterfactualRiskDatasetError(
                f"critical threshold {name!r} must be finite/non-negative"
            )
        metric = indices[name]
        feasible &= available[:, metric]
        feasible &= risks[:, metric] <= value
    return feasible


def inference_records(
    dataset: DatasetManifest,
    *,
    group_family_sha256s: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return canonical feature-only inference records for selected groups."""

    dataset.validate()
    selected = (
        set(dataset.group_family_sha256s)
        if group_family_sha256s is None
        else set(_canonical_tuple(group_family_sha256s, "selected groups"))
    )
    unknown = selected - set(dataset.group_family_sha256s)
    if unknown:
        raise CounterfactualRiskDatasetError(
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
