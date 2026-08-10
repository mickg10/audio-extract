"""Assemble truth-free flat predictions into exact certified route grids.

The student predicts one row at a time in canonical inference-SHA order, while
the structured decoder consumes ``(time, band, candidate, metric)`` tensors.
This module performs only the identity-preserving reshape.  It joins rows to a
certified complete cell partition by group-scoped cell identity, carries exact
positive rational physical measure, preserves feature-blocked availability, and
refuses missing, extra, duplicated, or cross-group rows.

No exact risks or clean truth are arguments to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np

from .counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
    CellPartitionRegistry,
    RationalMeasure,
)
from .counterfactual_risk_dataset_contract_v1 import GroupFamilyIdentity
from .counterfactual_risk_inference_contract_v1 import (
    InferenceManifestV1,
    InferenceRowV1,
)
from .counterfactual_risk_student_inference_v1 import (
    PredictedUpperRiskManifestV1,
)

PANEL_SCHEMA = "audio-extract/partitioned-upper-risk-panel/v1"
REGISTRY_SCHEMA = "audio-extract/partitioned-upper-risk-registry/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class PartitionedPredictionError(ValueError):
    """A flat prediction cannot be bound to the certified route grid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise PartitionedPredictionError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise PartitionedPredictionError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PartitionedPredictionError(
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
        "schema": PANEL_SCHEMA,
        "component": component,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PartitionedPredictionError(
            f"{name} must be a positive integer, not boolean"
        )
    return value


def _group_family(row: InferenceRowV1) -> GroupFamilyIdentity:
    result = GroupFamilyIdentity(
        work_id=row.work_id,
        recording_session_id=row.recording_session_id,
        target_singer_id=row.target_singer_id,
        source_family_sha256=row.source_family_sha256,
        query_condition_sha256=row.query_condition_sha256,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class PartitionedUpperRiskPanelV1:
    group_family_sha256: str
    work_id: str
    recording_session_id: str
    target_singer_id: str
    source_family_sha256: str
    query_condition_sha256: str
    mixture_pcm_sha256: str
    spectral_grid_sha256: str
    resolution_ms: int
    time_cell_count: int
    band_count: int
    partition_certificate_sha256: str
    prediction_manifest_sha256: str
    model_sha256: str
    model_input_sha256: str
    prediction_policy_sha256: str
    source_commit: str
    candidate_panel_sha256: str
    candidate_slot_sha256s: tuple[str, ...]
    feature_contract_sha256: str
    metric_contract_sha256: str
    route_policy_sha256: str
    metric_names: tuple[str, ...]
    metric_units: tuple[str, ...]
    metric_directions: tuple[str, ...]
    inference_row_sha256s: tuple[str, ...]
    cell_sha256s: tuple[str, ...]
    row_prediction_allowed: tuple[bool, ...]
    measures: tuple[RationalMeasure, ...]
    upper: np.ndarray
    available: np.ndarray

    @property
    def partition_key(self) -> tuple[str, str, int]:
        return (
            self.group_family_sha256,
            self.spectral_grid_sha256,
            self.resolution_ms,
        )

    def validate(self) -> None:
        for name, value in (
            ("group_family_sha256", self.group_family_sha256),
            ("source_family_sha256", self.source_family_sha256),
            ("query_condition_sha256", self.query_condition_sha256),
            ("mixture_pcm_sha256", self.mixture_pcm_sha256),
            ("spectral_grid_sha256", self.spectral_grid_sha256),
            (
                "partition_certificate_sha256",
                self.partition_certificate_sha256,
            ),
            (
                "prediction_manifest_sha256",
                self.prediction_manifest_sha256,
            ),
            ("model_sha256", self.model_sha256),
            ("model_input_sha256", self.model_input_sha256),
            ("prediction_policy_sha256", self.prediction_policy_sha256),
            ("candidate_panel_sha256", self.candidate_panel_sha256),
            ("feature_contract_sha256", self.feature_contract_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
            ("route_policy_sha256", self.route_policy_sha256),
        ):
            _sha(value, name)
        _git(self.source_commit, "source_commit")
        for name, value in (
            ("work_id", self.work_id),
            ("recording_session_id", self.recording_session_id),
            ("target_singer_id", self.target_singer_id),
        ):
            _text(value, name)
        resolution = _positive_int(self.resolution_ms, "resolution_ms")
        time_count = _positive_int(self.time_cell_count, "time_cell_count")
        band_count = _positive_int(self.band_count, "band_count")
        cell_count = time_count * band_count
        if resolution != self.resolution_ms:
            raise AssertionError("resolution validation changed the value")

        group = GroupFamilyIdentity(
            work_id=self.work_id,
            recording_session_id=self.recording_session_id,
            target_singer_id=self.target_singer_id,
            source_family_sha256=self.source_family_sha256,
            query_condition_sha256=self.query_condition_sha256,
        )
        group.validate()
        if group.sha256 != self.group_family_sha256:
            raise PartitionedPredictionError(
                "panel group-family SHA differs from reconstructed identity"
            )

        if len(self.candidate_slot_sha256s) < 2:
            raise PartitionedPredictionError(
                "partitioned panel requires at least two candidate slots"
            )
        for index, value in enumerate(self.candidate_slot_sha256s):
            _sha(value, f"candidate_slot_sha256s[{index}]")
        if len(set(self.candidate_slot_sha256s)) != len(
            self.candidate_slot_sha256s
        ):
            raise PartitionedPredictionError(
                "candidate slot identities are duplicated"
            )

        metric_count = len(self.metric_names)
        if metric_count < 1 or not (
            len(self.metric_units) == metric_count
            and len(self.metric_directions) == metric_count
        ):
            raise PartitionedPredictionError(
                "metric names, units and directions are empty or misaligned"
            )
        if len(set(self.metric_names)) != metric_count:
            raise PartitionedPredictionError("metric names must be unique")
        for name, unit, direction in zip(
            self.metric_names, self.metric_units, self.metric_directions
        ):
            _text(name, "metric name")
            _text(unit, "metric unit")
            if direction != "lower_is_better":
                raise PartitionedPredictionError(
                    "partitioned defect metrics must declare lower_is_better"
                )

        if not (
            len(self.inference_row_sha256s)
            == len(self.cell_sha256s)
            == len(self.row_prediction_allowed)
            == len(self.measures)
            == cell_count
        ):
            raise PartitionedPredictionError(
                "row, cell, prediction and measure axes differ from the grid"
            )
        for index, value in enumerate(self.inference_row_sha256s):
            _sha(value, f"inference_row_sha256s[{index}]")
        for index, value in enumerate(self.cell_sha256s):
            _sha(value, f"cell_sha256s[{index}]")
        if len(set(self.inference_row_sha256s)) != cell_count:
            raise PartitionedPredictionError(
                "partitioned panel repeats an inference row"
            )
        if len(set(self.cell_sha256s)) != cell_count:
            raise PartitionedPredictionError(
                "partitioned panel repeats a cell identity"
            )
        if any(
            not isinstance(value, (bool, np.bool_))
            for value in self.row_prediction_allowed
        ):
            raise PartitionedPredictionError(
                "row_prediction_allowed must contain booleans"
            )
        for measure in self.measures:
            measure.validate()
        if self.total_measure <= 0:
            raise PartitionedPredictionError(
                "partitioned panel total measure is not positive"
            )

        upper = np.asarray(self.upper, dtype=np.float64)
        available = np.asarray(self.available)
        expected_shape = (
            time_count,
            band_count,
            len(self.candidate_slot_sha256s),
            metric_count,
        )
        if upper.shape != expected_shape:
            raise PartitionedPredictionError(
                f"upper-risk grid shape differs: {upper.shape} != {expected_shape}"
            )
        if available.shape != expected_shape or available.dtype != np.bool_:
            raise PartitionedPredictionError(
                "availability must be a boolean tensor matching upper risks"
            )
        if np.any(~np.isfinite(upper[available])) or np.any(
            upper[available] < 0
        ):
            raise PartitionedPredictionError(
                "available upper risks must be finite and non-negative"
            )
        if np.any(upper[~available] != 0.0):
            raise PartitionedPredictionError(
                "unavailable upper risks must be canonical zero"
            )
        flattened = available.reshape(cell_count, *available.shape[2:])
        for index, allowed in enumerate(self.row_prediction_allowed):
            if bool(allowed):
                if not np.all(flattened[index]):
                    raise PartitionedPredictionError(
                        "allowed route cell has unavailable risk heads"
                    )
            elif np.any(flattened[index]):
                raise PartitionedPredictionError(
                    "feature-blocked route cell exposes available risk heads"
                )

    @property
    def total_measure(self) -> Fraction:
        return sum(
            (measure.fraction for measure in self.measures),
            start=Fraction(0, 1),
        )

    def normalized_measure_matrix(self) -> tuple[tuple[Fraction, ...], ...]:
        self.validate()
        total = self.total_measure
        values = tuple(measure.fraction / total for measure in self.measures)
        return tuple(
            tuple(values[index * self.band_count:(index + 1) * self.band_count])
            for index in range(self.time_cell_count)
        )

    @property
    def upper_sha256(self) -> str:
        self.validate()
        return _array_sha(
            np.asarray(self.upper, dtype=np.float64),
            component="upper",
            dtype="<f8",
        )

    @property
    def availability_sha256(self) -> str:
        self.validate()
        return _array_sha(
            np.asarray(self.available, dtype=np.uint8),
            component="available",
            dtype="u1",
        )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        total = self.total_measure
        return {
            "schema": PANEL_SCHEMA,
            "group_family_sha256": self.group_family_sha256,
            "work_id": self.work_id,
            "recording_session_id": self.recording_session_id,
            "target_singer_id": self.target_singer_id,
            "source_family_sha256": self.source_family_sha256,
            "query_condition_sha256": self.query_condition_sha256,
            "mixture_pcm_sha256": self.mixture_pcm_sha256,
            "spectral_grid_sha256": self.spectral_grid_sha256,
            "resolution_ms": self.resolution_ms,
            "time_cell_count": self.time_cell_count,
            "band_count": self.band_count,
            "partition_certificate_sha256": (
                self.partition_certificate_sha256
            ),
            "prediction_manifest_sha256": self.prediction_manifest_sha256,
            "model_sha256": self.model_sha256,
            "model_input_sha256": self.model_input_sha256,
            "prediction_policy_sha256": self.prediction_policy_sha256,
            "source_commit": self.source_commit,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "candidate_slot_sha256s": list(self.candidate_slot_sha256s),
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "route_policy_sha256": self.route_policy_sha256,
            "metric_names": list(self.metric_names),
            "metric_units": list(self.metric_units),
            "metric_directions": list(self.metric_directions),
            "inference_row_sha256s": list(self.inference_row_sha256s),
            "cell_sha256s": list(self.cell_sha256s),
            "row_prediction_allowed": [
                bool(value) for value in self.row_prediction_allowed
            ],
            "measures": [measure.to_dict() for measure in self.measures],
            "total_measure": {
                "numerator": total.numerator,
                "denominator": total.denominator,
            },
            "upper_shape": list(np.asarray(self.upper).shape),
            "upper_sha256": self.upper_sha256,
            "availability_sha256": self.availability_sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class PartitionedUpperRiskRegistryV1:
    panels: tuple[PartitionedUpperRiskPanelV1, ...]
    prediction_manifest_sha256: str
    model_input_sha256: str
    source_commit: str

    def validate(self) -> None:
        _sha(
            self.prediction_manifest_sha256,
            "registry prediction_manifest_sha256",
        )
        _sha(self.model_input_sha256, "registry model_input_sha256")
        _git(self.source_commit, "registry source_commit")
        if not self.panels:
            raise PartitionedPredictionError(
                "partitioned upper-risk registry is empty"
            )
        for panel in self.panels:
            panel.validate()
        keys = tuple(panel.partition_key for panel in self.panels)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise PartitionedPredictionError(
                "partitioned panels must be unique and canonically ordered"
            )
        for panel in self.panels:
            if panel.prediction_manifest_sha256 != (
                self.prediction_manifest_sha256
            ):
                raise PartitionedPredictionError(
                    "panel and registry name different prediction manifests"
                )
            if panel.model_input_sha256 != self.model_input_sha256:
                raise PartitionedPredictionError(
                    "panel and registry name different model inputs"
                )
            if panel.source_commit != self.source_commit:
                raise PartitionedPredictionError(
                    "panel and registry use different source commits"
                )
        row_ids = tuple(
            row_id for panel in self.panels
            for row_id in panel.inference_row_sha256s
        )
        if len(set(row_ids)) != len(row_ids):
            raise PartitionedPredictionError(
                "one inference row appears in multiple partitioned panels"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": REGISTRY_SCHEMA,
            "prediction_manifest_sha256": (
                self.prediction_manifest_sha256
            ),
            "model_input_sha256": self.model_input_sha256,
            "source_commit": self.source_commit,
            "partition_keys": [list(panel.partition_key) for panel in self.panels],
            "panel_sha256s": [panel.sha256 for panel in self.panels],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def _validate_prediction_join(
    inference: InferenceManifestV1,
    prediction: PredictedUpperRiskManifestV1,
) -> tuple[str, ...]:
    inference.validate()
    prediction.validate()
    expected_rows = tuple(
        row.sha256(candidate_panel=inference.candidate_panel)
        for row in inference.rows
    )
    comparisons = {
        "model input": (
            prediction.model_input_sha256,
            inference.model_input_sha256,
        ),
        "source commit": (
            prediction.source_commit,
            inference.source_commit,
        ),
        "inference rows": (
            prediction.inference_row_sha256s,
            expected_rows,
        ),
        "candidate panel": (
            prediction.candidate_panel_sha256,
            inference.candidate_panel.sha256,
        ),
        "feature contract": (
            prediction.feature_contract_sha256,
            inference.rows[0].feature_contract_sha256,
        ),
        "metric contract": (
            prediction.metric_contract_sha256,
            inference.rows[0].metric_contract_sha256,
        ),
        "route policy": (
            prediction.route_policy_sha256,
            inference.rows[0].route_policy_sha256,
        ),
        "metric names": (
            prediction.metric_names,
            inference.rows[0].metric_names,
        ),
        "prediction allowed": (
            prediction.row_prediction_allowed,
            tuple(row.prediction_allowed for row in inference.rows),
        ),
    }
    different = sorted(
        name for name, (actual, expected) in comparisons.items()
        if actual != expected
    )
    if different:
        raise PartitionedPredictionError(
            f"prediction differs from truth-free inference input: {different}"
        )
    return expected_rows


def assemble_partitioned_upper_risks(
    inference: InferenceManifestV1,
    prediction: PredictedUpperRiskManifestV1,
    partitions: CellPartitionRegistry,
) -> PartitionedUpperRiskRegistryV1:
    """Join flat predictions to complete certified partitions without truth."""

    expected_rows = _validate_prediction_join(inference, prediction)
    partitions.validate()
    if partitions.source_commit != inference.source_commit:
        raise PartitionedPredictionError(
            "cell partitions and inference input use different source commits"
        )

    row_by_sha: dict[str, tuple[int, InferenceRowV1, str]] = {}
    for index, (row_sha, row) in enumerate(zip(expected_rows, inference.rows)):
        group_sha = _group_family(row).sha256
        row_by_sha[row_sha] = (index, row, group_sha)

    cells_by_key: dict[
        tuple[str, str, int], dict[str, tuple[str, int, InferenceRowV1]]
    ] = {}
    for row_sha, (index, row, group_sha) in row_by_sha.items():
        key = (
            group_sha,
            row.cell.spectral_grid_sha256,
            row.cell.resolution_ms,
        )
        cell_map = cells_by_key.setdefault(key, {})
        if row.cell.sha256 in cell_map:
            raise PartitionedPredictionError(
                "one group partition contains duplicate inference cell identities"
            )
        cell_map[row.cell.sha256] = (row_sha, index, row)

    certificate_by_key = partitions.by_key()
    if set(cells_by_key) != set(certificate_by_key):
        missing = sorted(set(cells_by_key) - set(certificate_by_key))
        unused = sorted(set(certificate_by_key) - set(cells_by_key))
        raise PartitionedPredictionError(
            "inference rows and partition registry differ; "
            f"missing_certificates={missing}, unused_certificates={unused}"
        )

    prediction_upper = np.asarray(prediction.upper, dtype=np.float64)
    prediction_available = np.asarray(prediction.available, dtype=bool)
    panels: list[PartitionedUpperRiskPanelV1] = []
    consumed: set[str] = set()
    slot_shas = tuple(slot.sha256 for slot in inference.candidate_panel.slots)
    first_inference = inference.rows[0]

    for key in sorted(certificate_by_key):
        certificate: CellPartitionCertificate = certificate_by_key[key]
        certificate.validate()
        cell_map = cells_by_key[key]
        expected_cell_ids = tuple(
            entry.cell_sha256 for entry in certificate.entries
        )
        if set(cell_map) != set(expected_cell_ids):
            missing = sorted(set(expected_cell_ids) - set(cell_map))
            extra = sorted(set(cell_map) - set(expected_cell_ids))
            raise PartitionedPredictionError(
                "inference rows do not exactly cover the certified partition; "
                f"missing={missing}, extra={extra}"
            )

        ordered_rows = [cell_map[cell_id] for cell_id in expected_cell_ids]
        row_shas = tuple(value[0] for value in ordered_rows)
        indices = np.asarray([value[1] for value in ordered_rows], dtype=np.int64)
        rows = tuple(value[2] for value in ordered_rows)
        if consumed & set(row_shas):
            raise PartitionedPredictionError(
                "one inference row is consumed by multiple partitions"
            )
        consumed.update(row_shas)

        first = rows[0]
        common = (
            first.work_id,
            first.recording_session_id,
            first.target_singer_id,
            first.source_family_sha256,
            first.query_condition_sha256,
            first.mixture_pcm_sha256,
        )
        for row in rows[1:]:
            current = (
                row.work_id,
                row.recording_session_id,
                row.target_singer_id,
                row.source_family_sha256,
                row.query_condition_sha256,
                row.mixture_pcm_sha256,
            )
            if current != common:
                raise PartitionedPredictionError(
                    "one certified partition mixes incompatible inference identities"
                )

        upper = prediction_upper[indices].reshape(
            certificate.time_cell_count,
            certificate.band_count,
            prediction_upper.shape[1],
            prediction_upper.shape[2],
        ).copy()
        available = prediction_available[indices].reshape(
            certificate.time_cell_count,
            certificate.band_count,
            prediction_available.shape[1],
            prediction_available.shape[2],
        ).copy()
        upper.setflags(write=False)
        available.setflags(write=False)

        panel = PartitionedUpperRiskPanelV1(
            group_family_sha256=key[0],
            work_id=first.work_id,
            recording_session_id=first.recording_session_id,
            target_singer_id=first.target_singer_id,
            source_family_sha256=first.source_family_sha256,
            query_condition_sha256=first.query_condition_sha256,
            mixture_pcm_sha256=first.mixture_pcm_sha256,
            spectral_grid_sha256=certificate.spectral_grid_sha256,
            resolution_ms=certificate.resolution_ms,
            time_cell_count=certificate.time_cell_count,
            band_count=certificate.band_count,
            partition_certificate_sha256=certificate.sha256,
            prediction_manifest_sha256=prediction.sha256,
            model_sha256=prediction.model_sha256,
            model_input_sha256=prediction.model_input_sha256,
            prediction_policy_sha256=prediction.prediction_policy_sha256,
            source_commit=prediction.source_commit,
            candidate_panel_sha256=prediction.candidate_panel_sha256,
            candidate_slot_sha256s=slot_shas,
            feature_contract_sha256=prediction.feature_contract_sha256,
            metric_contract_sha256=prediction.metric_contract_sha256,
            route_policy_sha256=prediction.route_policy_sha256,
            metric_names=prediction.metric_names,
            metric_units=first_inference.metric_units,
            metric_directions=first_inference.metric_directions,
            inference_row_sha256s=row_shas,
            cell_sha256s=expected_cell_ids,
            row_prediction_allowed=tuple(
                bool(prediction.row_prediction_allowed[index])
                for index in indices
            ),
            measures=tuple(entry.measure for entry in certificate.entries),
            upper=upper,
            available=available,
        )
        panel.validate()
        panels.append(panel)

    if consumed != set(expected_rows):
        missing = sorted(set(expected_rows) - consumed)
        raise PartitionedPredictionError(
            f"some inference rows were not consumed by partitions: {missing}"
        )

    result = PartitionedUpperRiskRegistryV1(
        panels=tuple(panels),
        prediction_manifest_sha256=prediction.sha256,
        model_input_sha256=prediction.model_input_sha256,
        source_commit=prediction.source_commit,
    )
    result.validate()
    return result
