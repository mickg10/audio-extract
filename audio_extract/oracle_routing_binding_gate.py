"""Binding decision gate for the certified opera routing envelope.

This module evaluates immutable complete-work reports.  It renders no audio and
optimizes no route.  Missing, non-finite, differently scoped, or transform-
confounded evidence is invalid rather than a pass.

A routed method is compared against the metric-wise frontier formed by every
declared production baseline.  Consequently, beating a weak single separator
while losing to the current median champion cannot produce an actionable result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import math


REPORT_SCHEMA = "audio-extract/oracle-routing-binding-report/v2"
DECISION_SCHEMA = "audio-extract/oracle-routing-binding-decision/v2"

LOWER_IS_BETTER = "lower"
HIGHER_IS_BETTER = "higher"
ABSOLUTE_REGRESSION = "absolute"
RELATIVE_REGRESSION = "relative"

_REQUIRED_REPORT_HASHES = (
    "candidate_manifest_sha256",
    "truth_manifest_sha256",
    "routing_config_sha256",
)


class BindingGateError(RuntimeError):
    """The supplied report cannot support a binding architecture decision."""


@dataclass(frozen=True)
class MetricRule:
    work_id: str
    name: str
    direction: str = LOWER_IS_BETTER
    max_regression: float = 0.0
    regression_kind: str = ABSOLUTE_REGRESSION
    relative_floor: float = 1e-12

    def validate(self) -> None:
        if not self.work_id or not self.name:
            raise ValueError("metric work_id/name must be non-empty")
        if self.direction not in (LOWER_IS_BETTER, HIGHER_IS_BETTER):
            raise ValueError(f"invalid metric direction {self.direction!r}")
        if self.regression_kind not in (
            ABSOLUTE_REGRESSION,
            RELATIVE_REGRESSION,
        ):
            raise ValueError(
                f"invalid regression_kind {self.regression_kind!r}"
            )
        if (
            not math.isfinite(float(self.max_regression))
            or self.max_regression < 0
        ):
            raise ValueError("max_regression must be finite and non-negative")
        if (
            not math.isfinite(float(self.relative_floor))
            or self.relative_floor <= 0
        ):
            raise ValueError("relative_floor must be finite and positive")


@dataclass(frozen=True)
class CriticalAxis(MetricRule):
    minimum_improvement: float = 1.5

    def validate(self) -> None:
        super().validate()
        if (
            not math.isfinite(float(self.minimum_improvement))
            or self.minimum_improvement < 0
        ):
            raise ValueError(
                "minimum_improvement must be finite and non-negative"
            )


@dataclass(frozen=True)
class AbsoluteGate:
    work_id: str
    metric: str
    maximum: float | None = None
    minimum: float | None = None

    def validate(self) -> None:
        if not self.work_id or not self.metric:
            raise ValueError("absolute gate work_id/metric must be non-empty")
        if self.maximum is None and self.minimum is None:
            raise ValueError("absolute gate needs maximum or minimum")
        for value in (self.maximum, self.minimum):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("absolute gate bounds must be finite")
        if (
            self.maximum is not None
            and self.minimum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("absolute gate minimum exceeds maximum")


@dataclass(frozen=True)
class TransformTolerance:
    max_abs: float = 2e-5
    rms: float = 2e-6
    spectral_error: float = 1e-5
    stereo_error: float = 1e-5

    def validate(self) -> None:
        for name in ("max_abs", "rms", "spectral_error", "stereo_error"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class BindingGateConfig:
    primary_resolution_seconds: float
    baseline_methods: tuple[str, ...]
    routed_methods: tuple[str, ...] = ("O2", "O3")
    required_basis_roles: tuple[str, ...] = (
        "median_mdx_mel_bs",
        "geomedian_mdx_mel_bs",
        "convex_fusion_uniform",
        "mdx23c",
        "melband",
        "bs_roformer",
        "htdemucs_04573f0d",
        "htdemucs_955717e8",
    )
    critical_axes: tuple[CriticalAxis, ...] = ()
    minimum_critical_improvements: int = 1
    stability_axes: tuple[MetricRule, ...] = ()
    absolute_gates: tuple[AbsoluteGate, ...] = ()
    no_vocal_work_id: str = "aalto_mozart_dry"
    no_vocal_max_ratio: float = 1.05
    no_vocal_floor: float = 1e-15
    transform_tolerance: TransformTolerance = field(
        default_factory=TransformTolerance
    )
    sensitivity_resolutions_seconds: tuple[float, ...] = (0.5, 2.0)

    def validate(self) -> None:
        if (
            not math.isfinite(float(self.primary_resolution_seconds))
            or self.primary_resolution_seconds <= 0
        ):
            raise ValueError("primary resolution must be finite and positive")
        if not self.baseline_methods:
            raise ValueError("at least one baseline method is required")
        if len(set(self.baseline_methods)) != len(self.baseline_methods):
            raise ValueError("baseline methods must be unique")
        if not self.routed_methods:
            raise ValueError("at least one routed method is required")
        if len(set(self.routed_methods)) != len(self.routed_methods):
            raise ValueError("routed methods must be unique")
        if set(self.baseline_methods) & set(self.routed_methods):
            raise ValueError("baseline/routed method sets must be disjoint")
        if not self.critical_axes:
            raise ValueError("at least one critical axis is required")
        if not 1 <= self.minimum_critical_improvements <= len(
            self.critical_axes
        ):
            raise ValueError("invalid minimum_critical_improvements")
        if (
            not math.isfinite(float(self.no_vocal_max_ratio))
            or self.no_vocal_max_ratio < 1.0
        ):
            raise ValueError("no_vocal_max_ratio must be finite and >= 1")
        if (
            not math.isfinite(float(self.no_vocal_floor))
            or self.no_vocal_floor <= 0
        ):
            raise ValueError("no_vocal_floor must be finite and positive")
        for rule in self.critical_axes:
            rule.validate()
        for rule in self.stability_axes:
            rule.validate()
        for gate in self.absolute_gates:
            gate.validate()
        self.transform_tolerance.validate()
        seen = set()
        for resolution in self.sensitivity_resolutions_seconds:
            value = float(resolution)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("sensitivity resolutions must be positive")
            key = f"{value:g}"
            if key in seen:
                raise ValueError("sensitivity resolutions must be unique")
            seen.add(key)
            if math.isclose(value, float(self.primary_resolution_seconds)):
                raise ValueError(
                    "primary resolution must not be a sensitivity resolution"
                )


def default_opera_binding_config(
    *,
    baseline_methods: Sequence[str],
    primary_resolution_seconds: float = 1.0,
) -> BindingGateConfig:
    """Return the frozen first binding gate for the current exact corpus.

    The runner must flatten worst-event, transient, hall, seam, and frequency-
    boundary observations into the metric names used here.
    """

    artifact_rules = tuple(
        MetricRule(
            work_id=work,
            name="artifact_ratio_p90",
            max_regression=0.10,
            regression_kind=RELATIVE_REGRESSION,
        )
        for work in (
            "bologna_verdi",
            "bologna_donizetti",
            "bologna_puccini",
            "aalto_mozart_dry",
        )
    )
    transient_rules = tuple(
        MetricRule(
            work_id=work,
            name=name,
            max_regression=0.10,
            regression_kind=RELATIVE_REGRESSION,
        )
        for work in ("bologna_verdi", "bologna_donizetti")
        for name in (
            "transient_loss_ratio/v2",
            "transient_excess_ratio/v2",
            "worst_event_composite_risk",
        )
    )
    return BindingGateConfig(
        primary_resolution_seconds=primary_resolution_seconds,
        baseline_methods=tuple(baseline_methods),
        critical_axes=(
            CriticalAxis(
                work_id="bologna_verdi",
                name="retained_voice_db_p90",
                minimum_improvement=1.5,
                max_regression=0.5,
            ),
            CriticalAxis(
                work_id="bologna_donizetti",
                name="event_hole_db_p90",
                minimum_improvement=1.5,
                max_regression=0.5,
            ),
        ),
        minimum_critical_improvements=1,
        stability_axes=(
            MetricRule(
                work_id="bologna_verdi",
                name="event_hole_db_p90",
                max_regression=0.5,
            ),
            MetricRule(
                work_id="bologna_donizetti",
                name="retained_voice_db_p90",
                max_regression=0.5,
            ),
            MetricRule(
                work_id="aalto_mozart_dry",
                name="retained_voice_db_p90",
                max_regression=0.5,
            ),
            MetricRule(
                work_id="aalto_mozart_dry",
                name="event_hole_db_p90",
                max_regression=0.5,
            ),
            MetricRule(
                work_id="aalto_mozart_dry",
                name="stereo_width_dev_db/v2",
                max_regression=0.5,
            ),
            MetricRule(
                work_id="aalto_mozart_dry",
                name="interchannel_coherence_dev/v2",
                max_regression=0.05,
            ),
            MetricRule(
                work_id="aalto_mozart_dry",
                name="hall_tail_error_db/v1",
                max_regression=0.5,
            ),
        )
        + artifact_rules
        + transient_rules,
        absolute_gates=tuple(
            AbsoluteGate(
                work_id=work,
                metric="seam_max_boundary_jump_over_p99_derivative",
                maximum=2.0,
            )
            for work in ("bologna_verdi", "bologna_donizetti")
        )
        + tuple(
            AbsoluteGate(
                work_id=work,
                metric="frequency_boundary_ringing_ratio",
                maximum=1.25,
            )
            for work in ("bologna_verdi", "bologna_donizetti")
        ),
    )


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BindingGateError(f"missing/non-finite {label}: {value!r}")
    return float(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise BindingGateError(f"{label} is not a SHA-256 identity")
    suffix = value[7:]
    if len(suffix) != 64:
        raise BindingGateError(f"{label} is not a SHA-256 identity")
    try:
        int(suffix, 16)
    except ValueError as exc:
        raise BindingGateError(f"{label} is not a SHA-256 identity") from exc
    return value.lower()


def _git_commit(value: Any, label: str = "code_commit") -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise BindingGateError(f"{label} is not a full Git commit identity")
    try:
        int(value, 16)
    except ValueError as exc:
        raise BindingGateError(
            f"{label} is not a full Git commit identity"
        ) from exc
    return value.lower()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BindingGateError(f"{label} is not a positive integer")
    return value


def _resolution_key(value: float) -> str:
    return f"{float(value):g}"


def _resolution(report: Mapping[str, Any], seconds: float) -> Mapping[str, Any]:
    resolutions = report.get("resolutions")
    if not isinstance(resolutions, Mapping):
        raise BindingGateError("report lacks resolutions")
    key = _resolution_key(seconds)
    result = resolutions.get(key)
    if not isinstance(result, Mapping):
        raise BindingGateError(f"report lacks resolution {key}")
    return result


def _works(resolution: Mapping[str, Any]) -> Mapping[str, Any]:
    works = resolution.get("works")
    if not isinstance(works, Mapping):
        raise BindingGateError("resolution lacks works")
    return works


def _artifact(
    output: Mapping[str, Any],
    *,
    label: str,
    routed: bool,
) -> tuple[int, int, tuple[str, ...]]:
    artifact = output.get("artifact")
    if not isinstance(artifact, Mapping):
        raise BindingGateError(f"{label} lacks artifact facts")
    _sha(artifact.get("artifact_pcm_sha256"), f"{label}.artifact_pcm_sha256")
    _sha(artifact.get("container_sha256"), f"{label}.container_sha256")
    if artifact.get("subtype") != "FLOAT":
        raise BindingGateError(f"{label} artifact is not FLOAT")
    frames = _positive_int(artifact.get("frames"), f"{label}.frames")
    sample_rate = _positive_int(
        artifact.get("sample_rate_hz"), f"{label}.sample_rate_hz"
    )
    channels = artifact.get("channels")
    if not isinstance(channels, Sequence) or isinstance(
        channels, (str, bytes)
    ):
        raise BindingGateError(f"{label}.channels is invalid")
    channel_tuple = tuple(str(value) for value in channels)
    if not channel_tuple or any(not channel for channel in channel_tuple):
        raise BindingGateError(f"{label} has an invalid sample grid")
    if routed:
        recipe_id = output.get("recipe_id")
        if not isinstance(recipe_id, str) or not recipe_id:
            raise BindingGateError(f"{label} lacks route recipe_id")
        _sha(
            output.get("routing_plan_sha256"),
            f"{label}.routing_plan_sha256",
        )
    return frames, sample_rate, channel_tuple


def _output(
    resolution: Mapping[str, Any],
    work_id: str,
    method: str,
    routed_methods: Sequence[str],
) -> Mapping[str, Any]:
    work = _works(resolution).get(work_id)
    if not isinstance(work, Mapping):
        raise BindingGateError(f"missing work {work_id}")
    outputs = work.get("outputs")
    if not isinstance(outputs, Mapping):
        raise BindingGateError(f"{work_id} lacks outputs")
    output = outputs.get(method)
    if not isinstance(output, Mapping):
        raise BindingGateError(f"{work_id} lacks output {method}")
    if output.get("status") == "rejected":
        reason = output.get("certificate_error")
        if method not in routed_methods or not isinstance(reason, str) or not reason:
            raise BindingGateError(
                f"{work_id}/{method} has invalid rejection evidence"
            )
        return output
    _artifact(
        output,
        label=f"{work_id}/{method}",
        routed=method in routed_methods,
    )
    if not isinstance(output.get("metrics"), Mapping):
        raise BindingGateError(f"{work_id}/{method} lacks metrics")
    return output


def _is_rejected(output: Mapping[str, Any]) -> bool:
    return output.get("status") == "rejected"


def _validate_work_grids(
    resolution: Mapping[str, Any],
    methods: Sequence[str],
    routed_methods: Sequence[str],
) -> dict[str, Any]:
    result = {}
    for work_id, work in _works(resolution).items():
        if not isinstance(work, Mapping):
            raise BindingGateError(f"work {work_id} is not an object")
        grids = {}
        for method in methods:
            output = _output(
                resolution, work_id, method, routed_methods
            )
            if _is_rejected(output):
                continue
            grids[method] = _artifact(
                output,
                label=f"{work_id}/{method}",
                routed=method in routed_methods,
            )
        unique = set(grids.values())
        if len(unique) != 1:
            raise BindingGateError(
                f"{work_id} outputs occupy different sample grids: {grids}"
            )
        result[work_id] = {
            "frames": next(iter(unique))[0],
            "sample_rate_hz": next(iter(unique))[1],
            "channels": list(next(iter(unique))[2]),
        }
    return result


def _metric(
    resolution: Mapping[str, Any],
    work_id: str,
    method: str,
    name: str,
    routed_methods: Sequence[str],
) -> float:
    output = _output(resolution, work_id, method, routed_methods)
    if _is_rejected(output):
        raise BindingGateError(
            f"{work_id}/{method} was rejected and has no metrics"
        )
    return _finite(
        output["metrics"].get(name),
        f"{work_id}/{method}.{name}",
    )


def _frontier(
    resolution: Mapping[str, Any],
    *,
    work_id: str,
    metric: str,
    baselines: Sequence[str],
    routed_methods: Sequence[str],
    direction: str,
) -> tuple[float, str, dict[str, float]]:
    values = {
        method: _metric(
            resolution, work_id, method, metric, routed_methods
        )
        for method in baselines
    }
    selector = min if direction == LOWER_IS_BETTER else max
    value = selector(values.values())
    method = next(
        candidate
        for candidate in baselines
        if math.isclose(values[candidate], value, rel_tol=0, abs_tol=1e-15)
    )
    return value, method, values


def _improvement(candidate: float, frontier: float, direction: str) -> float:
    return (
        frontier - candidate
        if direction == LOWER_IS_BETTER
        else candidate - frontier
    )


def _regression(
    candidate: float,
    frontier: float,
    rule: MetricRule,
) -> float:
    deterioration = max(
        0.0, -_improvement(candidate, frontier, rule.direction)
    )
    if rule.regression_kind == ABSOLUTE_REGRESSION:
        return deterioration
    return deterioration / max(abs(frontier), rule.relative_floor)


def _basis(
    report: Mapping[str, Any],
    config: BindingGateConfig,
    work_ids: Sequence[str],
) -> dict:
    rows = report.get("basis")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise BindingGateError("report lacks basis rows")
    panels: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BindingGateError(f"basis[{index}] is not an object")
        role = row.get("role")
        work_id = row.get("work_id")
        scope = row.get("scope")
        if not isinstance(role, str) or not role:
            raise BindingGateError(f"basis[{index}] lacks role")
        if work_id not in work_ids:
            raise BindingGateError(f"basis[{index}] has unknown work_id")
        if scope not in ("full", "no_vocal"):
            raise BindingGateError(f"basis[{index}] has invalid scope")
        candidate_id = _sha(
            row.get("candidate_id"), f"basis[{index}].candidate_id"
        )
        pcm = _sha(
            row.get("artifact_pcm_sha256"),
            f"basis[{index}].artifact_pcm_sha256",
        )
        panels.setdefault((str(work_id), str(scope)), []).append(
            {"role": role, "candidate_id": candidate_id, "pcm": pcm}
        )

    required = set(config.required_basis_roles)
    summaries = {}
    total_roles = 0
    total_unique = 0
    for work_id in work_ids:
        for scope in ("full", "no_vocal"):
            key = (work_id, scope)
            panel_rows = panels.get(key, [])
            roles = [str(row["role"]) for row in panel_rows]
            if len(roles) != len(set(roles)):
                raise BindingGateError(
                    f"basis repeats a role in {work_id}/{scope}"
                )
            missing = sorted(required - set(roles))
            extra = sorted(set(roles) - required)
            if missing or extra:
                raise BindingGateError(
                    f"basis roles differ in {work_id}/{scope}: "
                    f"missing={missing}, extra={extra}"
                )
            pcm_to_canonical: dict[str, str] = {}
            aliases = {
                str(row["role"]): pcm_to_canonical.setdefault(
                    str(row["pcm"]), str(row["candidate_id"])
                )
                for row in panel_rows
            }
            unique = len(pcm_to_canonical)
            total_roles += len(aliases)
            total_unique += unique
            summaries[f"{work_id}/{scope}"] = {
                "role_aliases": aliases,
                "unique_pcm_candidates": unique,
                "deduplicated_roles": len(aliases) - unique,
            }
    return {
        "panels": summaries,
        "unique_pcm_candidates": total_unique,
        "deduplicated_roles": total_roles - total_unique,
    }


def _transform_controls(
    resolution: Mapping[str, Any],
    config: BindingGateConfig,
) -> dict[str, Any]:
    tolerance = config.transform_tolerance
    result = {}
    for work_id, work in _works(resolution).items():
        controls = work.get("transform_controls") if isinstance(work, Mapping) else None
        if not isinstance(controls, Mapping):
            raise BindingGateError(
                f"primary resolution work {work_id} lacks transform controls"
            )
        work_result = {}
        for name in ("O1", "median_mdx_mel_bs"):
            facts = controls.get(name)
            if not isinstance(facts, Mapping):
                raise BindingGateError(
                    f"missing transform control {work_id}/{name}"
                )
            _sha(
                facts.get("raw_artifact_pcm_sha256"),
                f"{work_id}/{name}.raw_artifact_pcm_sha256",
            )
            _sha(
                facts.get("stft_artifact_pcm_sha256"),
                f"{work_id}/{name}.stft_artifact_pcm_sha256",
            )
            values = {
                metric: _finite(
                    facts.get(metric), f"{work_id}/{name}.{metric}"
                )
                for metric in (
                    "max_abs",
                    "rms",
                    "spectral_error",
                    "stereo_error",
                )
            }
            failures = [
                metric
                for metric, value in values.items()
                if value > float(getattr(tolerance, metric))
            ]
            if failures:
                raise BindingGateError(
                    f"transform control {work_id}/{name} exceeds tolerance: "
                    f"{failures}"
                )
            work_result[name] = {"values": values, "failures": []}
        result[str(work_id)] = work_result
    return result


def _no_vocal_entry(
    resolution: Mapping[str, Any],
    work_id: str,
    method: str,
    routed_methods: Sequence[str],
) -> Mapping[str, Any]:
    work = _works(resolution).get(work_id)
    if not isinstance(work, Mapping):
        raise BindingGateError(f"missing no-vocal work {work_id}")
    no_vocal = work.get("no_vocal")
    if not isinstance(no_vocal, Mapping):
        raise BindingGateError(f"{work_id} lacks no-vocal facts")
    entry = no_vocal.get(method)
    if not isinstance(entry, Mapping):
        raise BindingGateError(f"no-vocal facts lack {method}")
    if entry.get("status") == "rejected":
        reason = entry.get("certificate_error")
        if method not in routed_methods or not isinstance(reason, str) or not reason:
            raise BindingGateError(
                f"no_vocal/{work_id}/{method} has invalid rejection evidence"
            )
        return entry
    _artifact(
        {"artifact": entry.get("artifact")},
        label=f"no_vocal/{method}",
        routed=False,
    )
    if method in routed_methods:
        _sha(
            entry.get("routing_plan_sha256"),
            f"no_vocal/{method}.routing_plan_sha256",
        )
    _finite(
        entry.get("false_positive_energy_ratio"),
        f"no_vocal/{method}.false_positive_energy_ratio",
    )
    return entry


def _validate_no_vocal_panels(
    resolution: Mapping[str, Any],
    config: BindingGateConfig,
) -> dict[str, Any]:
    """Validate complete no-vocal artifacts and exact route-plan reuse."""

    methods = tuple(config.baseline_methods) + tuple(config.routed_methods)
    result = {}
    for work_id in _works(resolution):
        full_grid = _artifact(
            _output(
                resolution,
                work_id,
                methods[0],
                config.routed_methods,
            ),
            label=f"{work_id}/{methods[0]}",
            routed=methods[0] in config.routed_methods,
        )
        grids = {}
        plans = {}
        for method in methods:
            full_output = _output(
                resolution, work_id, method, config.routed_methods
            )
            entry = _no_vocal_entry(
                resolution, work_id, method, config.routed_methods
            )
            if _is_rejected(full_output):
                if not _is_rejected(entry) or entry.get(
                    "certificate_error"
                ) != full_output.get("certificate_error"):
                    raise BindingGateError(
                        f"{work_id}/{method} rejection differs in no-vocal scope"
                    )
                continue
            if _is_rejected(entry):
                raise BindingGateError(
                    f"{work_id}/{method} is complete but no-vocal is rejected"
                )
            grid = _artifact(
                {"artifact": entry.get("artifact")},
                label=f"no_vocal/{work_id}/{method}",
                routed=False,
            )
            if grid != full_grid:
                raise BindingGateError(
                    f"{work_id}/{method} full and no-vocal grids differ"
                )
            grids[method] = grid
            if method in config.routed_methods:
                full_plan = _sha(
                    _output(
                        resolution,
                        work_id,
                        method,
                        config.routed_methods,
                    ).get("routing_plan_sha256"),
                    f"{work_id}/{method}.routing_plan_sha256",
                )
                control_plan = _sha(
                    entry.get("routing_plan_sha256"),
                    f"no_vocal/{work_id}/{method}.routing_plan_sha256",
                )
                if control_plan != full_plan:
                    raise BindingGateError(
                        f"{work_id}/{method} full and no-vocal route plans differ"
                    )
                plans[method] = full_plan
        result[str(work_id)] = {
            "sample_grid": {
                "frames": full_grid[0],
                "sample_rate_hz": full_grid[1],
                "channels": list(full_grid[2]),
            },
            "routing_plan_sha256": plans,
        }
    return result


def _no_vocal_ratio(
    resolution: Mapping[str, Any],
    method: str,
    config: BindingGateConfig,
) -> tuple[float, float, dict[str, float]]:
    values = {}
    for baseline in config.baseline_methods:
        entry = _no_vocal_entry(
            resolution,
            config.no_vocal_work_id,
            baseline,
            config.routed_methods,
        )
        values[baseline] = float(entry["false_positive_energy_ratio"])
    candidate_entry = _no_vocal_entry(
        resolution,
        config.no_vocal_work_id,
        method,
        config.routed_methods,
    )
    candidate = float(candidate_entry["false_positive_energy_ratio"])
    frontier = min(values.values())
    return (
        candidate / max(frontier, config.no_vocal_floor),
        frontier,
        values,
    )


def _evaluate_rule(
    resolution: Mapping[str, Any],
    method: str,
    rule: MetricRule,
    config: BindingGateConfig,
) -> dict[str, Any]:
    frontier, source, baselines = _frontier(
        resolution,
        work_id=rule.work_id,
        metric=rule.name,
        baselines=config.baseline_methods,
        routed_methods=config.routed_methods,
        direction=rule.direction,
    )
    candidate = _metric(
        resolution,
        rule.work_id,
        method,
        rule.name,
        config.routed_methods,
    )
    improvement = _improvement(candidate, frontier, rule.direction)
    regression = _regression(candidate, frontier, rule)
    return {
        "work_id": rule.work_id,
        "metric": rule.name,
        "candidate": candidate,
        "frontier": frontier,
        "frontier_method": source,
        "baseline_values": baselines,
        "direction": rule.direction,
        "improvement": improvement,
        "regression": regression,
        "regression_kind": rule.regression_kind,
        "max_regression": rule.max_regression,
        "pass": regression <= rule.max_regression,
    }


def _evaluate_method(
    resolution: Mapping[str, Any],
    method: str,
    config: BindingGateConfig,
) -> dict[str, Any]:
    rejected = []
    for work_id in _works(resolution):
        output = _output(
            resolution, work_id, method, config.routed_methods
        )
        if _is_rejected(output):
            rejected.append(
                f"{work_id}/{method} rejected: "
                f"{output['certificate_error']}"
            )
    if rejected:
        return {
            "method": method,
            "actionable": False,
            "critical_improvements": 0,
            "critical_axes": [],
            "stability_axes": [],
            "absolute_gates": [],
            "no_vocal": None,
            "failures": rejected,
        }
    failures: list[str] = []
    critical = []
    improvement_passes = 0
    for rule in config.critical_axes:
        row = _evaluate_rule(resolution, method, rule, config)
        row["minimum_improvement"] = rule.minimum_improvement
        row["improvement_pass"] = (
            row["improvement"] >= rule.minimum_improvement
        )
        if row["improvement_pass"]:
            improvement_passes += 1
        if not row["pass"]:
            failures.append(
                f"{rule.work_id}/{rule.name} regression "
                f"{row['regression']:.6g} > {rule.max_regression}"
            )
        critical.append(row)
    if improvement_passes < config.minimum_critical_improvements:
        failures.append(
            f"only {improvement_passes} critical improvements; "
            f"need {config.minimum_critical_improvements}"
        )

    stability = []
    for rule in config.stability_axes:
        row = _evaluate_rule(resolution, method, rule, config)
        if not row["pass"]:
            failures.append(
                f"{rule.work_id}/{rule.name} regression "
                f"{row['regression']:.6g} > {rule.max_regression}"
            )
        stability.append(row)

    absolute = []
    for gate in config.absolute_gates:
        value = _metric(
            resolution,
            gate.work_id,
            method,
            gate.metric,
            config.routed_methods,
        )
        passed = (
            (gate.maximum is None or value <= gate.maximum)
            and (gate.minimum is None or value >= gate.minimum)
        )
        if not passed:
            failures.append(
                f"{gate.work_id}/{gate.metric}={value:.6g} "
                f"outside [{gate.minimum},{gate.maximum}]"
            )
        absolute.append(
            {
                "work_id": gate.work_id,
                "metric": gate.metric,
                "value": value,
                "minimum": gate.minimum,
                "maximum": gate.maximum,
                "pass": passed,
            }
        )

    ratio, frontier, baseline_values = _no_vocal_ratio(
        resolution, method, config
    )
    if ratio > config.no_vocal_max_ratio:
        failures.append(
            f"no-vocal false-positive ratio {ratio:.6g} "
            f"> {config.no_vocal_max_ratio}"
        )
    return {
        "method": method,
        "actionable": not failures,
        "critical_improvements": improvement_passes,
        "critical_axes": critical,
        "stability_axes": stability,
        "absolute_gates": absolute,
        "no_vocal": {
            "candidate_over_frontier": ratio,
            "frontier": frontier,
            "baseline_values": baseline_values,
            "maximum_ratio": config.no_vocal_max_ratio,
            "pass": ratio <= config.no_vocal_max_ratio,
        },
        "failures": failures,
    }


def _evaluate_resolution(
    report: Mapping[str, Any],
    seconds: float,
    config: BindingGateConfig,
    *,
    require_transform_controls: bool,
) -> dict[str, Any]:
    resolution = _resolution(report, seconds)
    methods = tuple(config.baseline_methods) + tuple(config.routed_methods)
    grids = _validate_work_grids(
        resolution, methods, config.routed_methods
    )
    no_vocal = _validate_no_vocal_panels(resolution, config)
    transform = (
        _transform_controls(resolution, config)
        if require_transform_controls
        else None
    )
    results = {
        method: _evaluate_method(resolution, method, config)
        for method in config.routed_methods
    }
    return {
        "valid": True,
        "sample_grids": grids,
        "no_vocal_panels": no_vocal,
        "transform_controls": transform,
        "methods": results,
        "actionable_methods": [
            method
            for method, result in results.items()
            if result["actionable"]
        ],
    }


def evaluate_binding_report(
    report: Mapping[str, Any],
    config: BindingGateConfig,
) -> dict[str, Any]:
    """Return a binding architecture decision.

    Only the preregistered primary resolution can be ``ACTIONABLE``.  A finer or
    coarser sensitivity result can make a negative primary result
    ``RESOLUTION_SENSITIVE``, but is never cherry-picked per work.
    """

    config.validate()
    if report.get("schema") != REPORT_SCHEMA:
        return {
            "schema": DECISION_SCHEMA,
            "status": "INVALID_EVIDENCE",
            "failures": [
                f"wrong report schema {report.get('schema')!r}"
            ],
        }
    try:
        provenance = {"code_commit": _git_commit(report.get("code_commit"))}
        provenance.update({
            field: _sha(report.get(field), field)
            for field in _REQUIRED_REPORT_HASHES
        })
        primary_resolution = _resolution(
            report, config.primary_resolution_seconds
        )
        work_ids = tuple(str(work_id) for work_id in _works(primary_resolution))
        basis = _basis(report, config, work_ids)
        primary = _evaluate_resolution(
            report,
            config.primary_resolution_seconds,
            config,
            require_transform_controls=True,
        )
        sensitivity = {}
        for seconds in config.sensitivity_resolutions_seconds:
            key = _resolution_key(seconds)
            sensitivity[key] = _evaluate_resolution(
                report,
                seconds,
                config,
                require_transform_controls=False,
            )
    except BindingGateError as exc:
        return {
            "schema": DECISION_SCHEMA,
            "status": "INVALID_EVIDENCE",
            "failures": [str(exc)],
        }

    actionable = primary["actionable_methods"]
    sensitivity_actionable = sorted(
        {
            method
            for outcome in sensitivity.values()
            for method in outcome.get("actionable_methods", [])
        }
    )
    if actionable:
        status = "ACTIONABLE"
        architecture_action = "TRAIN_FROZEN_QUERY_CONDITIONED_GATE"
    elif sensitivity_actionable:
        status = "RESOLUTION_SENSITIVE"
        architecture_action = (
            "FREEZE_AND_RERUN_FINER_RESOLUTION_BEFORE_ARCHITECTURE_CHOICE"
        )
    else:
        status = "NO_ACTIONABLE_GAP"
        architecture_action = (
            "EXPAND_CANDIDATE_BASIS_OR_BUILD_CORRECTION_STUDENT"
        )
    return {
        "schema": DECISION_SCHEMA,
        "status": status,
        "provenance": provenance,
        "basis": basis,
        "primary_resolution_seconds": config.primary_resolution_seconds,
        "actionable_methods": actionable,
        "sensitivity_actionable_methods": sensitivity_actionable,
        "architecture_action": architecture_action,
        "primary": primary,
        "sensitivity": sensitivity,
        "failures": [],
    }
