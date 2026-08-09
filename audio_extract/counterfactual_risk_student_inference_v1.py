"""Fail-closed feature-masked inference for grouped counterfactual-risk students.

This bridge accepts only a truth-free ``InferenceManifestV1``.  Rows missing any
required inference feature are never passed to the student; their risk outputs
remain unavailable with canonical zero storage.  Every model-facing contract,
model identity, input identity, output tensor, mask, and prediction policy is
bound into the resulting manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol
import hashlib
import json
import math
import re

import numpy as np

from .counterfactual_risk_inference_contract_v1 import (
    InferenceContractError,
    InferenceManifestV1,
)

PREDICTION_SCHEMA = "audio-extract/counterfactual-risk-prediction/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class StudentInferenceError(InferenceContractError):
    """A student contract or predicted upper-risk tensor is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise StudentInferenceError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise StudentInferenceError(
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


def _array_sha(value: np.ndarray, *, component: str, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    header = {
        "schema": PREDICTION_SCHEMA,
        "component": component,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


class RiskStudentProtocol(Protocol):
    candidate_count: int
    metric_count: int
    feature_count: int
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str

    @property
    def sha256(self) -> str: ...

    def validate(self) -> None: ...

    def predict_upper(self, features: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class PredictedUpperRiskManifestV1:
    model_sha256: str
    model_input_sha256: str
    prediction_policy_sha256: str
    source_commit: str
    inference_row_sha256s: tuple[str, ...]
    row_prediction_allowed: tuple[bool, ...]
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    route_policy_sha256: str
    metric_names: tuple[str, ...]
    upper: np.ndarray
    available: np.ndarray

    def validate(self) -> None:
        for name, value in (
            ("model_sha256", self.model_sha256),
            ("model_input_sha256", self.model_input_sha256),
            ("prediction_policy_sha256", self.prediction_policy_sha256),
            ("candidate_panel_sha256", self.candidate_panel_sha256),
            ("feature_contract_sha256", self.feature_contract_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
            ("route_policy_sha256", self.route_policy_sha256),
        ):
            _sha(value, name)
        _git(self.source_commit, "prediction source_commit")
        if not self.inference_row_sha256s:
            raise StudentInferenceError("prediction manifest contains no rows")
        for index, value in enumerate(self.inference_row_sha256s):
            _sha(value, f"inference_row_sha256s[{index}]")
        if len(set(self.inference_row_sha256s)) != len(
            self.inference_row_sha256s
        ):
            raise StudentInferenceError(
                "prediction manifest contains duplicate inference rows"
            )
        if self.inference_row_sha256s != tuple(
            sorted(self.inference_row_sha256s)
        ):
            raise StudentInferenceError(
                "prediction rows must be in canonical inference-SHA order"
            )
        if len(self.row_prediction_allowed) != len(
            self.inference_row_sha256s
        ) or any(
            not isinstance(value, (bool, np.bool_))
            for value in self.row_prediction_allowed
        ):
            raise StudentInferenceError(
                "row_prediction_allowed must be one boolean per row"
            )
        if not self.metric_names or len(set(self.metric_names)) != len(
            self.metric_names
        ):
            raise StudentInferenceError(
                "metric names must be non-empty and unique"
            )
        upper = np.asarray(self.upper, dtype=np.float64)
        available = np.asarray(self.available)
        expected_prefix = (
            len(self.inference_row_sha256s),
            len(self.metric_names),
        )
        if upper.ndim != 3 or (
            upper.shape[0] != expected_prefix[0]
            or upper.shape[2] != expected_prefix[1]
        ):
            raise StudentInferenceError(
                "upper risks must have shape (rows,candidates,metrics)"
            )
        if available.shape != upper.shape or available.dtype != np.bool_:
            raise StudentInferenceError(
                "prediction availability must be a boolean tensor matching upper risks"
            )
        if np.any(~np.isfinite(upper[available])) or np.any(
            upper[available] < 0
        ):
            raise StudentInferenceError(
                "available upper risks must be finite and non-negative"
            )
        if np.any(upper[~available] != 0.0):
            raise StudentInferenceError(
                "unavailable upper risks must be stored as canonical zero"
            )
        for index, allowed in enumerate(self.row_prediction_allowed):
            row_mask = available[index]
            if bool(allowed):
                if not np.all(row_mask):
                    raise StudentInferenceError(
                        "allowed prediction row has unavailable risk heads"
                    )
            elif np.any(row_mask):
                raise StudentInferenceError(
                    "feature-blocked prediction row exposes available risk heads"
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
        return {
            "schema": PREDICTION_SCHEMA,
            "model_sha256": self.model_sha256,
            "model_input_sha256": self.model_input_sha256,
            "prediction_policy_sha256": self.prediction_policy_sha256,
            "source_commit": self.source_commit,
            "inference_row_sha256s": list(self.inference_row_sha256s),
            "row_prediction_allowed": [
                bool(value) for value in self.row_prediction_allowed
            ],
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "route_policy_sha256": self.route_policy_sha256,
            "metric_names": list(self.metric_names),
            "upper_shape": list(np.asarray(self.upper).shape),
            "upper_sha256": self.upper_sha256,
            "availability_sha256": self.availability_sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def predict_upper_fail_closed(
    student: RiskStudentProtocol,
    inference: InferenceManifestV1,
    *,
    prediction_policy_sha256: str,
) -> PredictedUpperRiskManifestV1:
    """Predict only rows with complete required features; mask all others."""

    inference.validate()
    student.validate()
    _sha(prediction_policy_sha256, "prediction_policy_sha256")
    first = inference.rows[0]
    comparisons = {
        "candidate panel": (
            student.candidate_panel_sha256,
            inference.candidate_panel.sha256,
        ),
        "feature contract": (
            student.feature_contract_sha256,
            first.feature_contract_sha256,
        ),
        "metric contract": (
            student.metric_contract_sha256,
            first.metric_contract_sha256,
        ),
        "candidate count": (
            int(student.candidate_count),
            len(inference.candidate_panel.slots),
        ),
        "metric count": (
            int(student.metric_count),
            len(first.metric_names),
        ),
        "feature count": (
            int(student.feature_count),
            len(first.features),
        ),
    }
    different = sorted(
        name
        for name, (actual, expected) in comparisons.items()
        if actual != expected
    )
    if different:
        raise StudentInferenceError(
            f"student differs from inference contracts: {different}"
        )
    model_sha256 = student.sha256
    _sha(model_sha256, "student model_sha256")

    row_count = len(inference.rows)
    candidate_count = len(inference.candidate_panel.slots)
    metric_count = len(first.metric_names)
    upper = np.zeros(
        (row_count, candidate_count, metric_count), dtype=np.float64
    )
    available = np.zeros_like(upper, dtype=bool)
    allowed = tuple(row.prediction_allowed for row in inference.rows)
    allowed_indices = [
        index for index, is_allowed in enumerate(allowed) if is_allowed
    ]
    if allowed_indices:
        features = np.asarray(
            [inference.rows[index].features for index in allowed_indices],
            dtype=np.float64,
        )
        predicted = np.asarray(student.predict_upper(features), dtype=np.float64)
        expected_shape = (
            len(allowed_indices),
            candidate_count,
            metric_count,
        )
        if predicted.shape != expected_shape:
            raise StudentInferenceError(
                f"student prediction shape differs: {predicted.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
            raise StudentInferenceError(
                "student predictions must be finite and non-negative"
            )
        upper[np.asarray(allowed_indices, dtype=np.int64)] = predicted
        available[np.asarray(allowed_indices, dtype=np.int64)] = True

    upper.setflags(write=False)
    available.setflags(write=False)
    row_shas = tuple(
        row.sha256(candidate_panel=inference.candidate_panel)
        for row in inference.rows
    )
    result = PredictedUpperRiskManifestV1(
        model_sha256=model_sha256,
        model_input_sha256=inference.model_input_sha256,
        prediction_policy_sha256=prediction_policy_sha256,
        source_commit=inference.source_commit,
        inference_row_sha256s=row_shas,
        row_prediction_allowed=allowed,
        candidate_panel_sha256=inference.candidate_panel.sha256,
        feature_contract_sha256=first.feature_contract_sha256,
        metric_contract_sha256=first.metric_contract_sha256,
        route_policy_sha256=first.route_policy_sha256,
        metric_names=first.metric_names,
        upper=upper,
        available=available,
    )
    result.validate()
    return result
