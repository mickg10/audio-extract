"""Independent-family split-conformal counterfactual-risk student v2.

Calibration units are canonical source-family identities, never cells or
caller-chosen work strings. Model/hyperparameter selection is frozen before the
calibration families are opened. Inference accepts feature tensors only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib, json, math, re
import numpy as np

MODEL_SCHEMA = "audio-extract/group-conformal-risk-student/v2"
CALIBRATION_SCHEMA = "audio-extract/group-conformal-risk-calibration/v2"
SCOPE_SCHEMA = "audio-extract/group-conformal-exchangeability-scope/v1"
SELECTION_SCHEMA = "audio-extract/risk-model-selection-protocol/v1"
ALGORITHM_REVISION = "group-max-split-conformal/v2"
FAMILY_AXIS = "source_family_sha256"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupConformalRiskV2Error(ValueError):
    pass


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupConformalRiskV2Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupConformalRiskV2Error(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise GroupConformalRiskV2Error(f"value is not canonical JSON: {exc}") from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _array_sha(value: np.ndarray, component: str) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    header = {"schema": MODEL_SCHEMA, "component": component,
              "shape": list(array.shape), "dtype": array.dtype.str}
    return "sha256:" + hashlib.sha256(_canonical(header) + array.tobytes()).hexdigest()


def _features(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or not np.all(np.isfinite(result)):
        raise GroupConformalRiskV2Error(
            f"{name} must be a non-empty finite (examples,features) array"
        )
    return result


def _risks(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 3 or min(result.shape) < 1
            or not np.all(np.isfinite(result)) or np.any(result < 0)):
        raise GroupConformalRiskV2Error(
            f"{name} must be a non-empty finite non-negative "
            "(examples,candidates,metrics) array"
        )
    return result


def _families(value: Sequence[Any], examples: int, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes, bytearray)):
        raise GroupConformalRiskV2Error(f"{name} must be an array")
    rows = tuple(value)
    if len(rows) != examples:
        raise GroupConformalRiskV2Error(f"{name} length differs from examples")
    return np.asarray([_sha(item, f"{name}[{i}]") for i, item in enumerate(rows)],
                      dtype=object)


def _family_set(value: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise GroupConformalRiskV2Error(f"{name} must be a sequence")
    rows = tuple(_sha(item, f"{name} item") for item in value)
    if len(set(rows)) != len(rows):
        raise GroupConformalRiskV2Error(f"{name} contains duplicate families")
    return tuple(sorted(rows))


def _require_disjoint(named: Mapping[str, Sequence[str]]) -> None:
    names = tuple(named)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(named[left]) & set(named[right])
            if overlap:
                raise GroupConformalRiskV2Error(
                    f"{left}/{right} source families overlap: {sorted(overlap)}"
                )


@dataclass(frozen=True)
class ExchangeabilityScopeV2:
    exchangeability_scope_sha256: str
    query_quality_stratum_sha256: str
    family_axis: str = FAMILY_AXIS
    calibration_algorithm_revision: str = ALGORITHM_REVISION

    def validate(self) -> None:
        _sha(self.exchangeability_scope_sha256, "exchangeability_scope_sha256")
        _sha(self.query_quality_stratum_sha256, "query_quality_stratum_sha256")
        if self.family_axis != FAMILY_AXIS:
            raise GroupConformalRiskV2Error(f"family_axis must be {FAMILY_AXIS!r}")
        if self.calibration_algorithm_revision != ALGORITHM_REVISION:
            raise GroupConformalRiskV2Error("unknown calibration algorithm revision")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": SCOPE_SCHEMA,
                "exchangeability_scope_sha256": self.exchangeability_scope_sha256,
                "query_quality_stratum_sha256": self.query_quality_stratum_sha256,
                "family_axis": self.family_axis,
                "calibration_algorithm_revision": self.calibration_algorithm_revision}

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class FrozenSelectionProtocolV2:
    selected_model_spec_sha256: str
    selected_hyperparameters_sha256: str
    selection_protocol_sha256: str
    training_manifest_sha256: str
    tuning_manifest_sha256: str
    calibration_data_consulted: bool = False
    status: str = "frozen_before_calibration"

    def validate(self) -> None:
        if self.status != "frozen_before_calibration":
            raise GroupConformalRiskV2Error("selection protocol is not frozen")
        if self.calibration_data_consulted is not False:
            raise GroupConformalRiskV2Error(
                "calibration families may not influence model/hyperparameter selection"
            )
        for name in ("selected_model_spec_sha256", "selected_hyperparameters_sha256",
                     "selection_protocol_sha256", "training_manifest_sha256",
                     "tuning_manifest_sha256"):
            _sha(getattr(self, name), name)
        if self.training_manifest_sha256 == self.tuning_manifest_sha256:
            raise GroupConformalRiskV2Error("training and tuning manifests must differ")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": SELECTION_SCHEMA, "status": self.status,
                "selected_model_spec_sha256": self.selected_model_spec_sha256,
                "selected_hyperparameters_sha256": self.selected_hyperparameters_sha256,
                "selection_protocol_sha256": self.selection_protocol_sha256,
                "training_manifest_sha256": self.training_manifest_sha256,
                "tuning_manifest_sha256": self.tuning_manifest_sha256,
                "calibration_data_consulted": self.calibration_data_consulted}

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class GroupCalibrationCertificateV2:
    target_coverage: float
    guaranteed_group_coverage: float
    calibration_group_count: int
    conformal_rank: int
    raw_order_statistic: float
    simultaneous_offset: float
    metric_scales: tuple[float, ...]
    augmented_design_condition: float
    ridge: float
    svd_rcond: float
    max_augmented_condition: float
    scope: ExchangeabilityScopeV2
    selection_protocol: FrozenSelectionProtocolV2
    calibration_manifest_sha256: str
    split_manifest_sha256: str
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    coefficients_sha256: str
    code_commit: str
    group_score_mode: str = "max_source_family_cells_candidates_metrics/v2"
    status: str = "passed"

    def validate(self, metric_count: int) -> None:
        if self.status != "passed":
            raise GroupConformalRiskV2Error("calibration status must be passed")
        q, groups = float(self.target_coverage), self.calibration_group_count
        if not math.isfinite(q) or not 0.5 < q < 1.0:
            raise GroupConformalRiskV2Error("target_coverage must lie in (0.5,1)")
        if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
            raise GroupConformalRiskV2Error("calibration_group_count is invalid")
        required_rank = math.ceil((groups + 1) * q)
        if required_rank > groups:
            raise GroupConformalRiskV2Error(
                "insufficient independent calibration source families"
            )
        if self.conformal_rank != required_rank:
            raise GroupConformalRiskV2Error("conformal rank is inconsistent")
        guaranteed = self.conformal_rank / (groups + 1)
        if not math.isclose(float(self.guaranteed_group_coverage), guaranteed,
                            rel_tol=0.0, abs_tol=8 * np.finfo(float).eps):
            raise GroupConformalRiskV2Error(
                "guaranteed_group_coverage is inconsistent with conformal rank"
            )
        if not math.isfinite(float(self.raw_order_statistic)):
            raise GroupConformalRiskV2Error("raw_order_statistic must be finite")
        expected_offset = max(0.0, float(self.raw_order_statistic))
        if not math.isclose(float(self.simultaneous_offset), expected_offset,
                            rel_tol=0.0, abs_tol=8 * np.finfo(float).eps
                            * max(1.0, abs(expected_offset))):
            raise GroupConformalRiskV2Error("simultaneous_offset is inconsistent")
        if len(self.metric_scales) != metric_count or any(
                not math.isfinite(float(x)) or float(x) <= 0
                for x in self.metric_scales):
            raise GroupConformalRiskV2Error("metric scales are invalid")
        if self.group_score_mode != "max_source_family_cells_candidates_metrics/v2":
            raise GroupConformalRiskV2Error("unknown group score mode")
        for name in ("ridge", "svd_rcond", "max_augmented_condition"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise GroupConformalRiskV2Error(f"{name} must be positive")
        condition = float(self.augmented_design_condition)
        if not math.isfinite(condition) or not 1 <= condition <= self.max_augmented_condition:
            raise GroupConformalRiskV2Error("augmented design condition is invalid")
        self.scope.validate(); self.selection_protocol.validate()
        for name in ("calibration_manifest_sha256", "split_manifest_sha256",
                     "candidate_panel_sha256", "feature_contract_sha256",
                     "metric_contract_sha256", "coefficients_sha256"):
            _sha(getattr(self, name), name)
        if self.calibration_manifest_sha256 in {
            self.selection_protocol.training_manifest_sha256,
            self.selection_protocol.tuning_manifest_sha256,
        }:
            raise GroupConformalRiskV2Error("calibration data reuses selection data")
        _git(self.code_commit, "code_commit")

    def identity_dict(self) -> dict[str, Any]:
        self.validate(len(self.metric_scales))
        return {"schema": CALIBRATION_SCHEMA, "status": self.status,
                "target_coverage": self.target_coverage,
                "guaranteed_group_coverage": self.guaranteed_group_coverage,
                "calibration_group_count": self.calibration_group_count,
                "conformal_rank": self.conformal_rank,
                "raw_order_statistic": self.raw_order_statistic,
                "simultaneous_offset": self.simultaneous_offset,
                "metric_scales": list(self.metric_scales),
                "group_score_mode": self.group_score_mode,
                "augmented_design_condition": self.augmented_design_condition,
                "ridge": self.ridge, "svd_rcond": self.svd_rcond,
                "max_augmented_condition": self.max_augmented_condition,
                "scope": self.scope.identity_dict(), "scope_sha256": self.scope.sha256,
                "selection_protocol": self.selection_protocol.identity_dict(),
                "selection_protocol_sha256": self.selection_protocol.sha256,
                "calibration_manifest_sha256": self.calibration_manifest_sha256,
                "split_manifest_sha256": self.split_manifest_sha256,
                "candidate_panel_sha256": self.candidate_panel_sha256,
                "feature_contract_sha256": self.feature_contract_sha256,
                "metric_contract_sha256": self.metric_contract_sha256,
                "coefficients_sha256": self.coefficients_sha256,
                "code_commit": self.code_commit}

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class GroupedLinearRiskStudentV2:
    coefficients: np.ndarray
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    calibration: GroupCalibrationCertificateV2

    def validate(self) -> tuple[int, int, int]:
        coef = np.asarray(self.coefficients, dtype=np.float64)
        if coef.ndim != 3 or min(coef.shape) < 1 or not np.all(np.isfinite(coef)):
            raise GroupConformalRiskV2Error("coefficients are invalid")
        for name in ("candidate_panel_sha256", "feature_contract_sha256",
                     "metric_contract_sha256"):
            _sha(getattr(self, name), name)
            if getattr(self.calibration, name) != getattr(self, name):
                raise GroupConformalRiskV2Error(f"calibration names different {name}")
        if _array_sha(coef, "coefficients") != self.calibration.coefficients_sha256:
            raise GroupConformalRiskV2Error("calibration names different coefficients")
        self.calibration.validate(coef.shape[2])
        return coef.shape

    @property
    def sha256(self) -> str:
        shape = self.validate()
        return _mapping_sha({"schema": MODEL_SCHEMA, "shape": list(shape),
                             "candidate_panel_sha256": self.candidate_panel_sha256,
                             "feature_contract_sha256": self.feature_contract_sha256,
                             "metric_contract_sha256": self.metric_contract_sha256,
                             "coefficients_sha256": self.calibration.coefficients_sha256,
                             "calibration_sha256": self.calibration.sha256})

    def predict_point(self, features: Any) -> np.ndarray:
        coef = np.asarray(self.coefficients, dtype=np.float64); self.validate()
        x = _features(features, "features")
        if x.shape[1] + 1 != coef.shape[0]:
            raise GroupConformalRiskV2Error("feature count differs")
        design = np.c_[x, np.ones(len(x))]
        return np.maximum(np.einsum("nf,fkd->nkd", design, coef), 0.0)

    def predict_upper(self, features: Any) -> np.ndarray:
        point = self.predict_point(features)
        scale = np.asarray(self.calibration.metric_scales)
        return point + self.calibration.simultaneous_offset * scale[None, None]


def fit_grouped_linear_risk_student_v2(
    train_features: Any, train_risks: Any, train_source_families: Sequence[Any],
    calibration_features: Any, calibration_risks: Any,
    calibration_source_families: Sequence[Any], *,
    tuning_source_families: Sequence[Any], metric_scales: Sequence[float],
    target_coverage: float, ridge: float, svd_rcond: float,
    max_augmented_condition: float, scope: ExchangeabilityScopeV2,
    selection_protocol: FrozenSelectionProtocolV2,
    candidate_panel_sha256: str, feature_contract_sha256: str,
    metric_contract_sha256: str, calibration_manifest_sha256: str,
    split_manifest_sha256: str, code_commit: str,
) -> GroupedLinearRiskStudentV2:
    x, y = _features(train_features, "train_features"), _risks(train_risks, "train_risks")
    cx, cy = (_features(calibration_features, "calibration_features"),
              _risks(calibration_risks, "calibration_risks"))
    if x.shape[0] != y.shape[0] or cx.shape[0] != cy.shape[0]:
        raise GroupConformalRiskV2Error("feature/risk example counts differ")
    if x.shape[1] != cx.shape[1] or y.shape[1:] != cy.shape[1:]:
        raise GroupConformalRiskV2Error("training/calibration shapes differ")
    train = _families(train_source_families, len(x), "train_source_families")
    cal = _families(calibration_source_families, len(cx),
                    "calibration_source_families")
    tuning = _family_set(tuning_source_families, "tuning_source_families")
    if not tuning:
        raise GroupConformalRiskV2Error("at least one tuning source family is required")
    _require_disjoint({"training": tuple(set(train)), "tuning": tuning,
                       "calibration": tuple(set(cal))})
    scope.validate(); selection_protocol.validate()
    for name, value in (("candidate_panel_sha256", candidate_panel_sha256),
                        ("feature_contract_sha256", feature_contract_sha256),
                        ("metric_contract_sha256", metric_contract_sha256),
                        ("calibration_manifest_sha256", calibration_manifest_sha256),
                        ("split_manifest_sha256", split_manifest_sha256)):
        _sha(value, name)
    _git(code_commit, "code_commit")
    if calibration_manifest_sha256 in {selection_protocol.training_manifest_sha256,
                                       selection_protocol.tuning_manifest_sha256}:
        raise GroupConformalRiskV2Error("calibration manifest reuses selection data")
    scales = np.asarray(tuple(metric_scales), dtype=np.float64)
    if scales.shape != (y.shape[2],) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise GroupConformalRiskV2Error("metric_scales are invalid")
    q, ridge, rcond, limit = map(float, (target_coverage, ridge, svd_rcond,
                                         max_augmented_condition))
    if not 0.5 < q < 1 or any(not math.isfinite(v) or v <= 0
                              for v in (ridge, rcond, limit)):
        raise GroupConformalRiskV2Error("coverage/numerical policy is invalid")

    design = np.c_[x, np.ones(len(x))]
    regularizer = np.eye(design.shape[1]); regularizer[-1, -1] = 0
    aug_x = np.vstack((design, math.sqrt(ridge) * regularizer))
    target = y.reshape(len(y), -1)
    aug_y = np.vstack((target, np.zeros((design.shape[1], target.shape[1]))))
    flat, _, rank, singular = np.linalg.lstsq(aug_x, aug_y, rcond=rcond)
    if rank != design.shape[1] or singular.size != design.shape[1]:
        raise GroupConformalRiskV2Error("augmented ridge design is rank deficient")
    condition = float(singular[0] / singular[-1])
    if not math.isfinite(condition) or condition > limit:
        raise GroupConformalRiskV2Error("augmented ridge design exceeds max_augmented_condition")
    coef = flat.reshape(design.shape[1], y.shape[1], y.shape[2])
    point = np.einsum("nf,fkd->nkd", np.c_[cx, np.ones(len(cx))], coef)
    residual = (cy - point) / scales[None, None]
    families = tuple(sorted(set(cal.tolist())))
    scores = np.asarray([np.max(residual[cal == family]) for family in families])
    groups, conformal_rank = len(families), math.ceil((len(families) + 1) * q)
    if conformal_rank > groups:
        raise GroupConformalRiskV2Error("insufficient independent calibration source families")
    raw = float(np.sort(scores)[conformal_rank - 1])
    certificate = GroupCalibrationCertificateV2(
        target_coverage=q, guaranteed_group_coverage=conformal_rank / (groups + 1),
        calibration_group_count=groups, conformal_rank=conformal_rank,
        raw_order_statistic=raw, simultaneous_offset=max(0.0, raw),
        metric_scales=tuple(map(float, scales)), augmented_design_condition=condition,
        ridge=ridge, svd_rcond=rcond, max_augmented_condition=limit, scope=scope,
        selection_protocol=selection_protocol,
        calibration_manifest_sha256=calibration_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        candidate_panel_sha256=candidate_panel_sha256,
        feature_contract_sha256=feature_contract_sha256,
        metric_contract_sha256=metric_contract_sha256,
        coefficients_sha256=_array_sha(coef, "coefficients"), code_commit=code_commit)
    result = GroupedLinearRiskStudentV2(coef, candidate_panel_sha256,
                                         feature_contract_sha256,
                                         metric_contract_sha256, certificate)
    result.validate(); return result
