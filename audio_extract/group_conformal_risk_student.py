"""Grouped linear counterfactual-risk student with simultaneous conformal bounds.

This dormant CPU research component fits every candidate x defect head from exact
offline labels, then calibrates one simultaneous upper-risk offset over independent
work/singer/session groups.  Cells within a group are never treated as independent
calibration examples.

For calibration group ``g`` the score is

    max_{i,k,d in g} (risk[i,k,d] - point[i,k,d]) / scale[d].

The split-conformal order statistic is taken across groups.  Consequently a new
exchangeable group is covered simultaneously over all of its cells, candidates,
and calibrated defects, subject to the usual split-conformal assumptions.

Inference accepts only feature tensors.  Exact risks and group IDs are available
only to the offline fit/calibration function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np

MODEL_SCHEMA = "audio-extract/group-conformal-risk-student/v1"
CALIBRATION_SCHEMA = "audio-extract/group-conformal-risk-calibration/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupConformalRiskError(ValueError):
    """The grouped training/calibration contract is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupConformalRiskError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupConformalRiskError(
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


def _array_sha(value: np.ndarray, header: Mapping[str, Any]) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    facts = {
        **dict(header),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(facts) + array.tobytes()
    ).hexdigest()


def _features(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1:
        raise GroupConformalRiskError(
            f"{name} must be a non-empty (examples,features) array"
        )
    if not np.all(np.isfinite(result)):
        raise GroupConformalRiskError(f"{name} contains non-finite values")
    return result


def _risks(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or min(result.shape) < 1:
        raise GroupConformalRiskError(
            f"{name} must be a non-empty (examples,candidates,metrics) array"
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise GroupConformalRiskError(
            f"{name} must contain finite non-negative risks"
        )
    return result


def _groups(value: Sequence[Any], examples: int, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes, bytearray)):
        raise GroupConformalRiskError(f"{name} must be an array of group IDs")
    result = np.asarray(tuple(value), dtype=object)
    if result.shape != (examples,):
        raise GroupConformalRiskError(
            f"{name} length differs from examples: {result.shape} != {(examples,)}"
        )
    normalized = []
    for index, item in enumerate(result):
        if not isinstance(item, str) or not item:
            raise GroupConformalRiskError(
                f"{name}[{index}] must be a non-empty string"
            )
        normalized.append(item)
    return np.asarray(normalized, dtype=object)


def _ordered_unique(groups: np.ndarray) -> tuple[str, ...]:
    return tuple(sorted(set(str(value) for value in groups.tolist())))


@dataclass(frozen=True)
class GroupCalibrationCertificate:
    """Identity-bearing grouped split-conformal calibration witness."""

    target_coverage: float
    guaranteed_group_coverage: float
    calibration_group_count: int
    conformal_rank: int
    simultaneous_offset: float
    metric_scales: tuple[float, ...]
    group_score_mode: str
    training_manifest_sha256: str
    calibration_manifest_sha256: str
    split_manifest_sha256: str
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    coefficients_sha256: str
    code_commit: str
    status: str = "passed"

    def validate(self, *, metric_count: int) -> None:
        if self.status != "passed":
            raise GroupConformalRiskError(
                "group calibration certificate status must be passed"
            )
        for name in ("target_coverage", "guaranteed_group_coverage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise GroupConformalRiskError(f"{name} must lie in (0,1]")
        if self.guaranteed_group_coverage < self.target_coverage:
            raise GroupConformalRiskError(
                "guaranteed group coverage is below the target"
            )
        if (
            isinstance(self.calibration_group_count, bool)
            or not isinstance(self.calibration_group_count, int)
            or self.calibration_group_count < 1
        ):
            raise GroupConformalRiskError(
                "calibration_group_count must be a positive integer"
            )
        if (
            isinstance(self.conformal_rank, bool)
            or not isinstance(self.conformal_rank, int)
            or not 1 <= self.conformal_rank <= self.calibration_group_count
        ):
            raise GroupConformalRiskError("conformal rank is invalid")
        if not math.isfinite(float(self.simultaneous_offset)) or (
            self.simultaneous_offset < 0
        ):
            raise GroupConformalRiskError(
                "simultaneous_offset must be finite and non-negative"
            )
        if len(self.metric_scales) != metric_count:
            raise GroupConformalRiskError(
                "metric scale count differs from the metric axis"
            )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in self.metric_scales
        ):
            raise GroupConformalRiskError(
                "metric scales must be finite and strictly positive"
            )
        if self.group_score_mode != "max_cells_candidates_metrics/v1":
            raise GroupConformalRiskError("unknown group score mode")
        for name in (
            "training_manifest_sha256",
            "calibration_manifest_sha256",
            "split_manifest_sha256",
            "candidate_panel_sha256",
            "feature_contract_sha256",
            "metric_contract_sha256",
            "coefficients_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.code_commit, "calibration code commit")

    def identity_dict(self) -> dict[str, Any]:
        self.validate(metric_count=len(self.metric_scales))
        return {
            "schema": CALIBRATION_SCHEMA,
            "status": self.status,
            "target_coverage": float(self.target_coverage),
            "guaranteed_group_coverage": float(
                self.guaranteed_group_coverage
            ),
            "calibration_group_count": int(self.calibration_group_count),
            "conformal_rank": int(self.conformal_rank),
            "simultaneous_offset": float(self.simultaneous_offset),
            "metric_scales": [float(value) for value in self.metric_scales],
            "group_score_mode": self.group_score_mode,
            "training_manifest_sha256": self.training_manifest_sha256,
            "calibration_manifest_sha256": (
                self.calibration_manifest_sha256
            ),
            "split_manifest_sha256": self.split_manifest_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "coefficients_sha256": self.coefficients_sha256,
            "code_commit": self.code_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class GroupedLinearRiskStudent:
    """Linear K x D risk heads plus one simultaneous group offset."""

    coefficients: np.ndarray
    candidate_count: int
    metric_count: int
    feature_count: int
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    calibration: GroupCalibrationCertificate

    def validate(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = (
            self.feature_count + 1,
            self.candidate_count,
            self.metric_count,
        )
        if coefficients.shape != expected or not np.all(
            np.isfinite(coefficients)
        ):
            raise GroupConformalRiskError(
                f"coefficient shape/value mismatch: {coefficients.shape} != {expected}"
            )
        if min(self.feature_count, self.candidate_count, self.metric_count) < 1:
            raise GroupConformalRiskError("student axis sizes must be positive")
        _sha(self.candidate_panel_sha256, "candidate panel SHA")
        _sha(self.feature_contract_sha256, "feature contract SHA")
        _sha(self.metric_contract_sha256, "metric contract SHA")
        coefficient_sha = _array_sha(
            coefficients,
            {"schema": MODEL_SCHEMA, "component": "coefficients"},
        )
        if coefficient_sha != self.calibration.coefficients_sha256:
            raise GroupConformalRiskError(
                "calibration certificate names different coefficients"
            )
        if self.calibration.candidate_panel_sha256 != (
            self.candidate_panel_sha256
        ):
            raise GroupConformalRiskError(
                "calibration certificate names a different candidate panel"
            )
        if self.calibration.feature_contract_sha256 != (
            self.feature_contract_sha256
        ):
            raise GroupConformalRiskError(
                "calibration certificate names a different feature contract"
            )
        if self.calibration.metric_contract_sha256 != (
            self.metric_contract_sha256
        ):
            raise GroupConformalRiskError(
                "calibration certificate names a different metric contract"
            )
        self.calibration.validate(metric_count=self.metric_count)

    @property
    def coefficients_sha256(self) -> str:
        return _array_sha(
            np.asarray(self.coefficients, dtype=np.float64),
            {"schema": MODEL_SCHEMA, "component": "coefficients"},
        )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": MODEL_SCHEMA,
            "candidate_count": int(self.candidate_count),
            "metric_count": int(self.metric_count),
            "feature_count": int(self.feature_count),
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "coefficients_sha256": self.coefficients_sha256,
            "calibration": self.calibration.identity_dict(),
            "calibration_sha256": self.calibration.sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())

    def predict_point(self, features: Any) -> np.ndarray:
        """Predict non-negative point risks from inference features only."""

        self.validate()
        values = _features(features, "features")
        if values.shape[1] != self.feature_count:
            raise GroupConformalRiskError(
                f"feature count differs: {values.shape[1]} != {self.feature_count}"
            )
        design = np.concatenate(
            (values, np.ones((len(values), 1), dtype=np.float64)), axis=1
        )
        point = np.einsum(
            "nf,fkd->nkd",
            design,
            np.asarray(self.coefficients, dtype=np.float64),
        )
        return np.maximum(point, 0.0)

    def predict_upper(self, features: Any) -> np.ndarray:
        """Predict simultaneous calibrated upper risks without exact truth."""

        point = self.predict_point(features)
        scales = np.asarray(
            self.calibration.metric_scales, dtype=np.float64
        )
        return point + (
            float(self.calibration.simultaneous_offset)
            * scales[None, None, :]
        )


def fit_grouped_linear_risk_student(
    train_features: Any,
    train_risks: Any,
    train_groups: Sequence[Any],
    calibration_features: Any,
    calibration_risks: Any,
    calibration_groups: Sequence[Any],
    *,
    metric_scales: Sequence[float],
    target_coverage: float,
    ridge: float,
    candidate_panel_sha256: str,
    feature_contract_sha256: str,
    metric_contract_sha256: str,
    training_manifest_sha256: str,
    calibration_manifest_sha256: str,
    split_manifest_sha256: str,
    code_commit: str,
) -> GroupedLinearRiskStudent:
    """Fit linear heads and calibrate one group-simultaneous upper offset.

    The train/calibration group sets must be disjoint.  A finite certificate is
    refused when the number of independent calibration groups cannot support the
    requested split-conformal rank.
    """

    x = _features(train_features, "train_features")
    y = _risks(train_risks, "train_risks")
    cx = _features(calibration_features, "calibration_features")
    cy = _risks(calibration_risks, "calibration_risks")
    if x.shape[0] != y.shape[0] or cx.shape[0] != cy.shape[0]:
        raise GroupConformalRiskError(
            "feature/risk example counts differ"
        )
    if x.shape[1] != cx.shape[1] or y.shape[1:] != cy.shape[1:]:
        raise GroupConformalRiskError(
            "training/calibration feature or head shapes differ"
        )
    train_group_values = _groups(train_groups, len(x), "train_groups")
    calibration_group_values = _groups(
        calibration_groups, len(cx), "calibration_groups"
    )
    train_unique = set(_ordered_unique(train_group_values))
    calibration_unique = _ordered_unique(calibration_group_values)
    overlap = train_unique & set(calibration_unique)
    if overlap:
        raise GroupConformalRiskError(
            f"training/calibration groups overlap: {sorted(overlap)}"
        )

    scales = np.asarray(tuple(metric_scales), dtype=np.float64)
    if scales.shape != (y.shape[2],) or not np.all(np.isfinite(scales)) or (
        np.any(scales <= 0)
    ):
        raise GroupConformalRiskError(
            "metric_scales must be one finite positive value per metric"
        )
    coverage = float(target_coverage)
    if not math.isfinite(coverage) or not 0.5 < coverage < 1.0:
        raise GroupConformalRiskError(
            "target_coverage must lie strictly between 0.5 and 1"
        )
    ridge_value = float(ridge)
    if not math.isfinite(ridge_value) or ridge_value <= 0:
        raise GroupConformalRiskError("ridge must be finite and positive")
    for name, value in (
        ("candidate_panel_sha256", candidate_panel_sha256),
        ("feature_contract_sha256", feature_contract_sha256),
        ("metric_contract_sha256", metric_contract_sha256),
        ("training_manifest_sha256", training_manifest_sha256),
        ("calibration_manifest_sha256", calibration_manifest_sha256),
        ("split_manifest_sha256", split_manifest_sha256),
    ):
        _sha(value, name)
    _git(code_commit, "code_commit")

    design = np.concatenate(
        (x, np.ones((len(x), 1), dtype=np.float64)), axis=1
    )
    gram = design.T @ design + ridge_value * np.eye(design.shape[1])
    coefficients = np.linalg.solve(
        gram, design.T @ y.reshape(len(y), -1)
    ).reshape(design.shape[1], *y.shape[1:])
    coefficient_sha = _array_sha(
        coefficients,
        {"schema": MODEL_SCHEMA, "component": "coefficients"},
    )

    calibration_design = np.concatenate(
        (cx, np.ones((len(cx), 1), dtype=np.float64)), axis=1
    )
    point = np.maximum(
        np.einsum("nf,fkd->nkd", calibration_design, coefficients),
        0.0,
    )
    normalized = (cy - point) / scales[None, None, :]
    group_scores = []
    for group in calibration_unique:
        selected = normalized[calibration_group_values == group]
        if selected.size == 0:
            raise AssertionError("calibration group disappeared")
        group_scores.append(float(np.max(selected)))
    score_array = np.asarray(group_scores, dtype=np.float64)
    if not np.all(np.isfinite(score_array)):
        raise GroupConformalRiskError(
            "group nonconformity scores are non-finite"
        )

    group_count = len(calibration_unique)
    rank = math.ceil((group_count + 1) * coverage)
    if rank > group_count:
        raise GroupConformalRiskError(
            "insufficient independent calibration groups for a finite "
            f"{coverage:.6g} certificate: G={group_count}, rank={rank}"
        )
    offset = max(0.0, float(np.sort(score_array, kind="stable")[rank - 1]))
    guaranteed = rank / float(group_count + 1)
    calibration = GroupCalibrationCertificate(
        target_coverage=coverage,
        guaranteed_group_coverage=guaranteed,
        calibration_group_count=group_count,
        conformal_rank=rank,
        simultaneous_offset=offset,
        metric_scales=tuple(float(value) for value in scales),
        group_score_mode="max_cells_candidates_metrics/v1",
        training_manifest_sha256=training_manifest_sha256,
        calibration_manifest_sha256=calibration_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        candidate_panel_sha256=candidate_panel_sha256,
        feature_contract_sha256=feature_contract_sha256,
        metric_contract_sha256=metric_contract_sha256,
        coefficients_sha256=coefficient_sha,
        code_commit=code_commit,
    )
    student = GroupedLinearRiskStudent(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        candidate_count=y.shape[1],
        metric_count=y.shape[2],
        feature_count=x.shape[1],
        candidate_panel_sha256=candidate_panel_sha256,
        feature_contract_sha256=feature_contract_sha256,
        metric_contract_sha256=metric_contract_sha256,
        calibration=calibration,
    )
    student.validate()
    return student
