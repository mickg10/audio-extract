"""Fail-closed non-human decision for the certified oracle-routing v2 report.

The decision compares reconstructed immutable FLOAT routes against the raw
current median champion and the MDX fallback on complete exact works. Missing
metrics, incomplete optimizer certificates, absent no-vocal controls, or an
unverified artifact produce `INCOMPLETE_EVIDENCE`, never an implicit pass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .oracle_routing_work_contract_v3 import (
    NO_VOCAL_CONTROL_WORK_ID,
    REQUIRED_WORKS,
    WORK_CONTRACT_SHA256,
)
from .oracle_routing_work_contract_v3 import (
    identity_dict as work_contract_identity,
)

DECISION_SCHEMA = "audio-extract/oracle-routing-decision/v2"
REPORT_SCHEMA = "audio-extract/oracle-routing-envelope/v2"

REQUIRED_RESOLUTIONS = ("2.0", "1.0", "0.5")
BASELINE = "median_raw"
FALLBACK = "residual_mdx23c"
ROUTED_METHODS = ("O2_global_medoid", "O3_certified_convex")

VOICE = "retained_voice_db_p90"  # lower/more negative is better
HOLE = "event_hole_db_p90"  # lower is better
ARTIFACT = "artifact_ratio_p90"  # lower is better
WIDTH = "stereo_width_dev_db/v2"  # lower is better
COHERENCE = "interchannel_coherence_dev/v2"  # lower is better
HALL = "hall_tail_dev_db/v2"  # lower is better
TRANSIENT_LOSS = "transient_loss_db/v2"  # lower is better
TRANSIENT_EXCESS = "transient_excess_db/v2"  # lower is better

REQUIRED_METRICS = (
    VOICE,
    HOLE,
    ARTIFACT,
    WIDTH,
    COHERENCE,
    HALL,
    TRANSIENT_LOSS,
    TRANSIENT_EXCESS,
)


class DecisionEvidenceError(ValueError):
    """The report lacks complete or coherent evidence for a binding decision."""


@dataclass(frozen=True)
class RoutingGateConfig:
    critical_improvement_db: float = 1.5
    opposing_regression_db: float = 0.5
    broad_critical_regression_db: float = 0.5
    artifact_ratio_limit: float = 1.10
    no_vocal_ratio_limit: float = 1.05
    stereo_width_regression_db: float = 0.5
    coherence_regression: float = 0.05
    hall_regression_db: float = 0.5
    transient_regression_db: float = 0.5
    worst_event_regression: float = 0.10
    max_time_boundary_ratio: float = 3.0
    max_frequency_ringing_ratio: float = 1.5
    identity_roundtrip_max_abs: float = 2e-5
    identity_roundtrip_rms: float = 2e-6
    o2_mip_gap: float = 0.0
    o3_projected_gradient_tolerance: float = 1e-7
    o3_simplex_tolerance: float = 1e-8
    near_zero: float = 1e-15

    def validate(self) -> None:
        for name, value in asdict(self).items():
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.artifact_ratio_limit < 1.0 or self.no_vocal_ratio_limit < 1.0:
            raise ValueError("ratio limits must be at least one")


@dataclass(frozen=True)
class MethodDecision:
    resolution_seconds: str
    method: str
    evidence_valid: bool
    actionable_oracle_gap: bool
    critical_gains_db: dict[str, float]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["failures"] = list(self.failures)
        return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionEvidenceError(f"{label} must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DecisionEvidenceError(f"{label} is missing or non-finite: {value!r}")
    return float(value)


def _method(work: Mapping[str, Any], name: str, label: str) -> Mapping[str, Any]:
    methods = _mapping(work.get("methods"), f"{label}.methods")
    if name not in methods:
        raise DecisionEvidenceError(f"{label} lacks method {name}")
    return _mapping(methods[name], f"{label}.methods.{name}")


def _metrics(method: Mapping[str, Any], label: str) -> dict[str, float]:
    metrics = _mapping(method.get("metrics"), f"{label}.metrics")
    return {
        name: _finite(metrics.get(name), f"{label}.metrics.{name}")
        for name in REQUIRED_METRICS
    }


def _artifact(method: Mapping[str, Any], label: str) -> None:
    artifact = _mapping(method.get("artifact"), f"{label}.artifact")
    if artifact.get("subtype") != "FLOAT":
        raise DecisionEvidenceError(f"{label} artifact is not FLOAT")
    if artifact.get("reopen_verified") is not True:
        raise DecisionEvidenceError(f"{label} artifact lacks reopen verification")
    if int(artifact.get("sample_rate_hz", 0)) != 44_100:
        raise DecisionEvidenceError(f"{label} artifact sample rate is not 44100")
    if tuple(artifact.get("channels") or ()) != ("FL", "FR"):
        raise DecisionEvidenceError(f"{label} artifact channels are not FL/FR")
    if int(artifact.get("frames", 0)) <= 0:
        raise DecisionEvidenceError(f"{label} artifact frame count is invalid")
    for identity_name in ("artifact_pcm_sha256", "container_sha256"):
        value = str(artifact.get(identity_name) or "")
        if not value.startswith("sha256:") or len(value) != 71:
            raise DecisionEvidenceError(f"{label} artifact lacks valid {identity_name}")


def _identity_control(
    method: Mapping[str, Any], label: str, config: RoutingGateConfig
) -> None:
    control = _mapping(method.get("identity_roundtrip"), f"{label}.identity_roundtrip")
    maximum = _finite(control.get("max_abs"), f"{label}.identity_roundtrip.max_abs")
    rms = _finite(control.get("rms"), f"{label}.identity_roundtrip.rms")
    if maximum > config.identity_roundtrip_max_abs:
        raise DecisionEvidenceError(
            f"{label} identity max_abs {maximum} exceeds "
            f"{config.identity_roundtrip_max_abs}"
        )
    if rms > config.identity_roundtrip_rms:
        raise DecisionEvidenceError(
            f"{label} identity rms {rms} exceeds {config.identity_roundtrip_rms}"
        )


def _certificate(
    method: Mapping[str, Any], name: str, label: str, config: RoutingGateConfig
) -> None:
    certificate = _mapping(
        method.get("optimizer_certificate"), f"{label}.optimizer_certificate"
    )
    if certificate.get("objective_recomputed") is not True:
        raise DecisionEvidenceError(
            f"{label} objective was not independently recomputed"
        )
    if certificate.get("complete_grid") is not True:
        raise DecisionEvidenceError(f"{label} does not cover the complete cell grid")
    if name == "O2_global_medoid":
        if certificate.get("kind") != "global_potts_milp":
            raise DecisionEvidenceError(f"{label} has the wrong O2 certificate kind")
        if certificate.get("success") is not True:
            raise DecisionEvidenceError(f"{label} O2 solver did not certify success")
        gap = _finite(certificate.get("mip_gap"), f"{label}.mip_gap")
        if gap > config.o2_mip_gap + 1e-12:
            raise DecisionEvidenceError(
                f"{label} O2 gap {gap} exceeds {config.o2_mip_gap}"
            )
    elif name == "O3_certified_convex":
        if certificate.get("kind") != "convex_projected_gradient":
            raise DecisionEvidenceError(f"{label} has the wrong O3 certificate kind")
        for fact in ("converged", "monotone_objective", "starts_agree"):
            if certificate.get(fact) is not True:
                raise DecisionEvidenceError(f"{label} O3 lacks {fact}")
        projected = _finite(
            certificate.get("projected_gradient_norm"),
            f"{label}.projected_gradient_norm",
        )
        simplex = _finite(
            certificate.get("max_simplex_error"),
            f"{label}.max_simplex_error",
        )
        if projected > config.o3_projected_gradient_tolerance:
            raise DecisionEvidenceError(
                f"{label} O3 projected-gradient norm {projected} exceeds "
                f"{config.o3_projected_gradient_tolerance}"
            )
        if simplex > config.o3_simplex_tolerance:
            raise DecisionEvidenceError(
                f"{label} O3 simplex error {simplex} exceeds "
                f"{config.o3_simplex_tolerance}"
            )
    else:  # pragma: no cover - caller owns routed-method set
        raise DecisionEvidenceError(f"unknown routed method {name}")


def _route_artifacts(
    method: Mapping[str, Any], label: str, config: RoutingGateConfig
) -> tuple[str, ...]:
    failures = []
    seam = _mapping(method.get("seam_check"), f"{label}.seam_check")
    time_ratio = _finite(
        seam.get("max_time_boundary_jump_over_p99_derivative"),
        f"{label}.time_boundary_ratio",
    )
    frequency_ratio = _finite(
        seam.get("max_frequency_ringing_ratio"),
        f"{label}.frequency_ringing_ratio",
    )
    if time_ratio > config.max_time_boundary_ratio:
        failures.append(
            f"{label} time-boundary ratio {time_ratio} exceeds "
            f"{config.max_time_boundary_ratio}"
        )
    if frequency_ratio > config.max_frequency_ringing_ratio:
        failures.append(
            f"{label} frequency-ringing ratio {frequency_ratio} exceeds "
            f"{config.max_frequency_ringing_ratio}"
        )
    event = _mapping(
        method.get("worst_identifiable_event"),
        f"{label}.worst_identifiable_event",
    )
    if event.get("available") is not True:
        raise DecisionEvidenceError(f"{label} lacks a worst identifiable event")
    _finite(event.get("composite_risk"), f"{label}.worst_event.composite_risk")
    return tuple(failures)


def _ratio(candidate: float, baseline: float, near_zero: float) -> float:
    if baseline <= near_zero:
        return 1.0 if candidate <= near_zero else math.inf
    return candidate / baseline


def _regression(candidate: float, baseline: float) -> float:
    """Positive means the lower-is-better metric got worse."""
    return candidate - baseline


def _no_vocal(work: Mapping[str, Any], method: str, label: str) -> float:
    controls = _mapping(work.get("no_vocal"), f"{label}.no_vocal")
    row = _mapping(controls.get(method), f"{label}.no_vocal.{method}")
    return _finite(
        row.get("false_positive_energy_ratio"),
        f"{label}.no_vocal.{method}.false_positive_energy_ratio",
    )


def evaluate_method(
    resolution: str,
    works: Mapping[str, Any],
    method_name: str,
    config: RoutingGateConfig,
) -> MethodDecision:
    failures: list[str] = []
    gains: dict[str, float] = {}
    try:
        if set(works) != set(REQUIRED_WORKS):
            raise DecisionEvidenceError(
                "work set differs from the frozen hall-bearing contract: "
                f"missing={sorted(set(REQUIRED_WORKS) - set(works))}, "
                f"extra={sorted(set(works) - set(REQUIRED_WORKS))}"
            )
        rows = {
            work: _mapping(works.get(work), f"{resolution}.{work}")
            for work in REQUIRED_WORKS
        }
        values: dict[str, dict[str, dict[str, float]]] = {}
        methods: dict[str, dict[str, Mapping[str, Any]]] = {}
        for work, row in rows.items():
            methods[work] = {}
            values[work] = {}
            for name in (BASELINE, FALLBACK, method_name):
                evidence = _method(row, name, f"{resolution}.{work}")
                methods[work][name] = evidence
                values[work][name] = _metrics(evidence, f"{resolution}.{work}.{name}")
                _artifact(evidence, f"{resolution}.{work}.{name}")
            # Baseline identity controls make STFT damage explicit.
            identity_name = "median_stft_identity"
            identity = _method(row, identity_name, f"{resolution}.{work}")
            _artifact(identity, f"{resolution}.{work}.{identity_name}")
            _identity_control(identity, f"{resolution}.{work}.{identity_name}", config)
            _certificate(
                methods[work][method_name],
                method_name,
                f"{resolution}.{work}.{method_name}",
                config,
            )
            failures.extend(
                _route_artifacts(
                    methods[work][method_name],
                    f"{resolution}.{work}.{method_name}",
                    config,
                )
            )

        verdi_base = values["bologna_verdi"][BASELINE]
        verdi = values["bologna_verdi"][method_name]
        don_base = values["bologna_donizetti"][BASELINE]
        don = values["bologna_donizetti"][method_name]
        gains = {
            "verdi_retained_voice_db": verdi_base[VOICE] - verdi[VOICE],
            "donizetti_event_hole_db": don_base[HOLE] - don[HOLE],
        }
        if max(gains.values()) < config.critical_improvement_db:
            failures.append(
                "neither critical target axis improves by the required amount"
            )
        if _regression(verdi[HOLE], verdi_base[HOLE]) > config.opposing_regression_db:
            failures.append("Verdi event-hole axis regresses beyond the limit")
        if _regression(don[VOICE], don_base[VOICE]) > config.opposing_regression_db:
            failures.append("Donizetti retained-voice axis regresses beyond the limit")

        # All complete works remain bounded on both critical axes against both
        # the product champion and the conservative MDX fallback.
        for work in REQUIRED_WORKS:
            candidate = values[work][method_name]
            baseline = values[work][BASELINE]
            fallback = values[work][FALLBACK]
            for metric in (VOICE, HOLE):
                if _regression(candidate[metric], baseline[metric]) > (
                    config.broad_critical_regression_db
                ):
                    failures.append(
                        f"{work} {metric} regresses against the median champion"
                    )
                if _regression(candidate[metric], fallback[metric]) > (
                    config.broad_critical_regression_db
                ):
                    failures.append(
                        f"{work} {metric} regresses against the MDX fallback"
                    )
            artifact_ratio = _ratio(
                candidate[ARTIFACT], baseline[ARTIFACT], config.near_zero
            )
            if artifact_ratio > config.artifact_ratio_limit:
                failures.append(
                    f"{work} orthogonal-artifact ratio regresses beyond the limit"
                )
            limits = (
                (WIDTH, config.stereo_width_regression_db),
                (COHERENCE, config.coherence_regression),
                (HALL, config.hall_regression_db),
                (TRANSIENT_LOSS, config.transient_regression_db),
                (TRANSIENT_EXCESS, config.transient_regression_db),
            )
            for metric, limit in limits:
                if _regression(candidate[metric], baseline[metric]) > limit:
                    failures.append(f"{work} {metric} regresses beyond the limit")
            candidate_event = _mapping(
                methods[work][method_name]["worst_identifiable_event"],
                f"{work}.{method_name}.worst_identifiable_event",
            )
            baseline_event = _mapping(
                methods[work][BASELINE].get("worst_identifiable_event"),
                f"{work}.{BASELINE}.worst_identifiable_event",
            )
            if baseline_event.get("available") is not True:
                raise DecisionEvidenceError(
                    f"{work} median lacks a worst identifiable event"
                )
            event_regression = _finite(
                candidate_event.get("composite_risk"), "candidate event"
            ) - _finite(baseline_event.get("composite_risk"), "baseline event")
            if event_regression > config.worst_event_regression:
                failures.append(f"{work} worst identifiable event regresses")

        no_vocal = rows[NO_VOCAL_CONTROL_WORK_ID]
        control_label = f"{resolution}.{NO_VOCAL_CONTROL_WORK_ID}"
        candidate_fp = _no_vocal(no_vocal, method_name, control_label)
        baseline_fp = _no_vocal(no_vocal, BASELINE, control_label)
        if _ratio(candidate_fp, baseline_fp, config.near_zero) > (
            config.no_vocal_ratio_limit
        ):
            failures.append("Aalto no-vocal false-positive energy regresses")
    except (DecisionEvidenceError, KeyError, TypeError, ValueError) as exc:
        return MethodDecision(
            resolution_seconds=resolution,
            method=method_name,
            evidence_valid=False,
            actionable_oracle_gap=False,
            critical_gains_db=gains,
            failures=(str(exc),),
        )

    return MethodDecision(
        resolution_seconds=resolution,
        method=method_name,
        evidence_valid=True,
        actionable_oracle_gap=not failures,
        critical_gains_db=gains,
        failures=tuple(dict.fromkeys(failures)),
    )


def evaluate_report(
    report: Mapping[str, Any],
    *,
    config: RoutingGateConfig | None = None,
    required_resolutions: Sequence[str] = REQUIRED_RESOLUTIONS,
) -> dict[str, Any]:
    """Return a binding fail-closed architecture decision."""

    cfg = config or RoutingGateConfig()
    cfg.validate()
    if report.get("schema") != REPORT_SCHEMA:
        return {
            "schema": DECISION_SCHEMA,
            "decision": "INCOMPLETE_EVIDENCE",
            "reason": f"wrong report schema: {report.get('schema')!r}",
            "config": asdict(cfg),
            "method_decisions": [],
        }
    if (
        report.get("work_contract") != work_contract_identity()
        or report.get("work_contract_sha256") != WORK_CONTRACT_SHA256
    ):
        return {
            "schema": DECISION_SCHEMA,
            "decision": "INCOMPLETE_EVIDENCE",
            "reason": "report work contract differs from the frozen hall-bearing identity",
            "config": asdict(cfg),
            "recommendation": "COMPLETE_CERTIFICATES_AND_CONTROLS",
            "selected": None,
            "method_decisions": [],
        }
    try:
        resolutions = _mapping(report.get("resolutions"), "resolutions")
    except DecisionEvidenceError as exc:
        return {
            "schema": DECISION_SCHEMA,
            "decision": "INCOMPLETE_EVIDENCE",
            "reason": str(exc),
            "config": asdict(cfg),
            "method_decisions": [],
        }

    decisions: list[MethodDecision] = []
    for resolution in required_resolutions:
        if resolution not in resolutions:
            decisions.extend(
                MethodDecision(
                    resolution,
                    method,
                    False,
                    False,
                    {},
                    (f"missing required resolution {resolution}",),
                )
                for method in ROUTED_METHODS
            )
            continue
        row = resolutions[resolution]
        try:
            works = _mapping(row.get("works"), f"resolutions.{resolution}.works")
        except DecisionEvidenceError as exc:
            decisions.extend(
                MethodDecision(resolution, method, False, False, {}, (str(exc),))
                for method in ROUTED_METHODS
            )
            continue
        decisions.extend(
            evaluate_method(resolution, works, method, cfg) for method in ROUTED_METHODS
        )

    all_valid = bool(decisions) and all(item.evidence_valid for item in decisions)
    selected = next(
        (
            item
            for item in decisions
            if item.method == "O2_global_medoid" and item.resolution_seconds == "1.0"
        ),
        None,
    )
    sensitivity_hits = [
        item
        for item in decisions
        if item.method == "O2_global_medoid"
        and item.resolution_seconds in {"2.0", "0.5"}
        and item.actionable_oracle_gap
    ]
    if not all_valid or selected is None:
        decision = "INCOMPLETE_EVIDENCE"
        recommendation = "COMPLETE_CERTIFICATES_AND_CONTROLS"
        reason = "one or more required method/resolution results are invalid or absent"
        selected_payload = None
    elif selected.actionable_oracle_gap:
        decision = "ACTIONABLE_ROUTING_GAP"
        recommendation = "TRAIN_TARGET_SINGER_FROZEN_MEMBER_GATE"
        reason = (
            "preregistered O2_global_medoid passes the complete-work certified "
            "gate at the exact primary 1.0s resolution"
        )
        selected_payload = selected.to_dict()
    elif sensitivity_hits:
        decision = "RESOLUTION_SENSITIVE_GAP"
        recommendation = "DO_NOT_PROMOTE__REVIEW_FROZEN_SCALE_SENSITIVITY"
        reason = (
            "preregistered O2_global_medoid fails at the primary resolution "
            "but passes at one or more frozen sensitivity resolutions"
        )
        selected_payload = None
    else:
        decision = "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"
        recommendation = "CHANGE_BASIS_OR_BUILD_TARGET_SINGER_CORRECTION"
        reason = (
            "all required certified resolutions completed without a route that "
            "beats the raw median champion under the frozen limits"
        )
        selected_payload = None

    return {
        "schema": DECISION_SCHEMA,
        "decision": decision,
        "recommendation": recommendation,
        "reason": reason,
        "config": asdict(cfg),
        "required_resolutions": list(required_resolutions),
        "selected": selected_payload,
        "method_decisions": [item.to_dict() for item in decisions],
    }
