"""Overlap-corrected exact surrogate-alignment diagnostic v2.

V2 preserves the replay-only purpose of v1 while making the measure explicit:

* every waveform tile receives a Voronoi/partition-of-unity frame measure, so
  overlapping windows do not double-count the centre of a crop or overweight a
  forced final window;
* the source-energy floor has the same units as each local Gram matrix;
* every unavailable source-coordinate tile declares one exact fallback reason:
  no-vocal, vocal-only, silent, or ill-conditioned;
* direct-target tail risk is reported both globally and by fallback reason;
* source/fallback coverage is reported by physical frame measure, not only tile
  count.

The diagnostic remains NumPy-only and non-differentiable. It must first align on
frozen checkpoint outputs before any loss implementation is authorized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np

SCHEMA = "audio-extract/classical-surrogate-alignment/v2"
MODE_SOURCE = "source_coordinates"
MODE_DIRECT = "direct_fallback"
FALLBACK_REASONS = (
    "no_vocal",
    "vocal_only",
    "silent",
    "ill_conditioned",
)


class SurrogateAlignmentV2Error(RuntimeError):
    """The exact-grid input or diagnostic configuration is invalid."""


@dataclass(frozen=True)
class SurrogateConfigV2:
    sample_rate_hz: int = 44_100
    tile_seconds: float = 0.5
    hop_seconds: float = 0.25
    tail_fraction: float = 0.10
    ridge_relative: float = 1e-8
    max_condition: float = 1e6
    energy_floor_relative: float = 1e-10
    energy_floor_absolute: float = 1e-18
    representation: str = "waveform-joint-stereo-real/v2"

    def validate(self) -> None:
        if not isinstance(self.sample_rate_hz, int) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be a positive integer")
        for name in (
            "tile_seconds",
            "hop_seconds",
            "tail_fraction",
            "ridge_relative",
            "max_condition",
            "energy_floor_relative",
            "energy_floor_absolute",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.tail_fraction > 1:
            raise ValueError("tail_fraction must not exceed one")
        if self.hop_seconds > self.tile_seconds:
            raise ValueError("hop_seconds must not exceed tile_seconds")
        if self.max_condition <= 1:
            raise ValueError("max_condition must exceed one")
        if self.representation != "waveform-joint-stereo-real/v2":
            raise ValueError("unknown surrogate representation")

    @property
    def tile_frames(self) -> int:
        self.validate()
        return max(16, round(self.tile_seconds * self.sample_rate_hz))

    @property
    def hop_frames(self) -> int:
        self.validate()
        return max(1, round(self.hop_seconds * self.sample_rate_hz))


@dataclass(frozen=True)
class TileRiskV2:
    tile_index: int
    start_frame: int
    end_frame: int
    mode: str
    fallback_reason: str | None
    measure_frames: float
    accompaniment_energy: float
    vocal_energy: float
    source_energy_floor: float
    condition_number: float | None
    gamma_v: float | None
    gamma_a: float | None
    recall_risk: float | None
    theft_risk: float | None
    artifact_risk: float | None
    direct_risk: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SurrogateReportV2:
    schema: str
    config: dict[str, Any]
    frames: int
    source_coordinate_tiles: int
    direct_fallback_tiles: int
    total_tiles: int
    source_coordinate_measure_fraction: float
    fallback_tile_counts: dict[str, int]
    fallback_measure_fractions: dict[str, float]
    recall_cvar: float | None
    theft_cvar: float | None
    artifact_cvar: float | None
    direct_fallback_cvar: float | None
    direct_fallback_cvar_by_reason: dict[str, float | None]
    recall_max: float | None
    theft_max: float | None
    artifact_max: float | None
    direct_fallback_max: float | None
    direct_fallback_max_by_reason: dict[str, float | None]
    tiles: tuple[TileRiskV2, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tiles"] = [tile.to_dict() for tile in self.tiles]
        return result


def _audio(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] < 1:
        raise ValueError(f"{name} must have shape (frames,2)")
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError(f"{name} must contain floating samples")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite samples")
    return np.asarray(result, dtype=np.float64)


def tile_ranges(frames: int, tile: int, hop: int) -> tuple[tuple[int, int], ...]:
    if frames <= 0 or tile <= 0 or hop <= 0:
        raise ValueError("frames/tile/hop must be positive")
    if frames <= tile:
        return ((0, frames),)
    starts = list(range(0, frames - tile + 1, hop))
    final = frames - tile
    if starts[-1] != final:
        starts.append(final)
    return tuple((start, min(frames, start + tile)) for start in starts)


def partition_measure_weights(
    frames: int,
    ranges: Sequence[Sequence[int]],
) -> tuple[float, ...]:
    """Assign each physical frame exactly one unit across overlapping tiles."""

    coverage = np.zeros(frames, dtype=np.int64)
    normalized = []
    for value in ranges:
        if len(value) != 2:
            raise ValueError("tile range must contain start/end")
        start, end = int(value[0]), int(value[1])
        if start < 0 or end <= start or end > frames:
            raise ValueError(f"invalid tile range {(start, end)}")
        coverage[start:end] += 1
        normalized.append((start, end))
    if not normalized or np.any(coverage <= 0):
        raise ValueError("tile ranges must cover every frame")
    inverse = 1.0 / coverage.astype(np.float64)
    result = tuple(float(inverse[start:end].sum()) for start, end in normalized)
    if not math.isclose(
        sum(result), float(frames), rel_tol=0.0,
        abs_tol=8.0 * np.finfo(np.float64).eps * max(1.0, frames),
    ):
        raise SurrogateAlignmentV2Error("tile measure does not partition frames")
    if any(weight <= 0 or not math.isfinite(weight) for weight in result):
        raise SurrogateAlignmentV2Error("tile measure is invalid")
    return result


def _eigenvalues(aa: float, vv: float, av: float) -> tuple[float, float]:
    trace = aa + vv
    disc = math.sqrt(max(0.0, (aa - vv) ** 2 + 4.0 * av * av))
    return max(0.0, 0.5 * (trace - disc)), max(0.0, 0.5 * (trace + disc))


def _fallback_reason(
    aa: float,
    vv: float,
    floor: float,
    identifiable: bool,
) -> str | None:
    if identifiable:
        return None
    if aa <= floor and vv <= floor:
        return "silent"
    if vv <= floor:
        return "no_vocal"
    if aa <= floor:
        return "vocal_only"
    return "ill_conditioned"


def _fit_tile(
    vocal_estimate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    tile_index: int,
    start: int,
    stop: int,
    measure_frames: float,
    global_energy_density: float,
    config: SurrogateConfigV2,
) -> TileRiskV2:
    estimate = vocal_estimate[start:stop].reshape(-1)
    a = accompaniment[start:stop].reshape(-1)
    v = vocal[start:stop].reshape(-1)
    aa = float(np.dot(a, a))
    vv = float(np.dot(v, v))
    av = float(np.dot(a, v))
    energy = aa + vv
    floor = max(
        float(config.energy_floor_absolute),
        float(config.energy_floor_relative)
        * global_energy_density
        * float(estimate.size),
    )
    minimum, maximum = _eigenvalues(aa, vv, av)
    condition = math.inf if minimum <= 0 else float(maximum / minimum)
    identifiable = (
        aa > floor
        and vv > floor
        and maximum > floor
        and minimum > maximum / float(config.max_condition)
        and math.isfinite(condition)
    )
    reason = _fallback_reason(aa, vv, floor, identifiable)

    if identifiable:
        ridge = float(config.ridge_relative) * max(0.5 * (aa + vv), floor)
        matrix = np.asarray([[aa + ridge, av], [av, vv + ridge]])
        rhs = np.asarray([
            float(np.dot(a, estimate)),
            float(np.dot(v, estimate)),
        ])
        gamma_a, gamma_v = np.linalg.solve(matrix, rhs)
        residue = estimate - gamma_a * a - gamma_v * v
        denominator = max(energy, floor)
        return TileRiskV2(
            tile_index=tile_index,
            start_frame=start,
            end_frame=stop,
            mode=MODE_SOURCE,
            fallback_reason=None,
            measure_frames=measure_frames,
            accompaniment_energy=aa,
            vocal_energy=vv,
            source_energy_floor=floor,
            condition_number=condition,
            gamma_v=float(gamma_v),
            gamma_a=float(gamma_a),
            recall_risk=float((1.0 - gamma_v) ** 2 * vv / denominator),
            theft_risk=float(gamma_a**2 * aa / denominator),
            artifact_risk=float(np.dot(residue, residue) / denominator),
            direct_risk=None,
        )

    accompaniment_estimate = (
        accompaniment[start:stop]
        + vocal[start:stop]
        - vocal_estimate[start:stop]
    )
    error = accompaniment_estimate.reshape(-1) - a
    direct = float(np.dot(error, error) / max(energy, floor))
    return TileRiskV2(
        tile_index=tile_index,
        start_frame=start,
        end_frame=stop,
        mode=MODE_DIRECT,
        fallback_reason=reason,
        measure_frames=measure_frames,
        accompaniment_energy=aa,
        vocal_energy=vv,
        source_energy_floor=floor,
        condition_number=(condition if math.isfinite(condition) else None),
        gamma_v=None,
        gamma_a=None,
        recall_risk=None,
        theft_risk=None,
        artifact_risk=None,
        direct_risk=direct,
    )


def weighted_cvar(
    values: Sequence[float],
    weights: Sequence[float],
    tail_fraction: float,
) -> float | None:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if values_array.size == 0:
        return None
    if values_array.shape != weights_array.shape:
        raise ValueError("CVaR values and weights differ")
    if (
        not np.all(np.isfinite(values_array))
        or not np.all(np.isfinite(weights_array))
        or np.any(values_array < 0)
        or np.any(weights_array <= 0)
    ):
        raise ValueError("CVaR values/weights must be finite and valid")
    fraction = float(tail_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("tail_fraction must lie in (0,1]")
    target = fraction * float(weights_array.sum())
    order = np.argsort(values_array, kind="stable")[::-1]
    remaining = target
    total = 0.0
    used = 0.0
    for index in order:
        amount = min(float(weights_array[index]), remaining)
        if amount > 0:
            total += amount * float(values_array[index])
            used += amount
            remaining -= amount
        if remaining <= 8.0 * np.finfo(np.float64).eps * max(1.0, target):
            break
    if used <= 0:
        raise SurrogateAlignmentV2Error("CVaR selected no measure")
    return total / used


def _aggregate(
    rows: Sequence[TileRiskV2],
    field: str,
    tail_fraction: float,
) -> tuple[float | None, float | None]:
    values = [float(getattr(tile, field)) for tile in rows]
    weights = [tile.measure_frames for tile in rows]
    return (
        weighted_cvar(values, weights, tail_fraction),
        (max(values) if values else None),
    )


def evaluate_surrogate_v2(
    vocal_estimate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    config: SurrogateConfigV2 | None = None,
) -> SurrogateReportV2:
    cfg = config or SurrogateConfigV2()
    cfg.validate()
    estimate = _audio(vocal_estimate, "vocal_estimate")
    a = _audio(accompaniment, "accompaniment")
    v = _audio(vocal, "vocal")
    if estimate.shape != a.shape or v.shape != a.shape:
        raise ValueError("surrogate inputs occupy different sample grids")
    frames = len(a)
    ranges = tile_ranges(frames, cfg.tile_frames, cfg.hop_frames)
    measures = partition_measure_weights(frames, ranges)
    global_density = float(
        np.dot(a.reshape(-1), a.reshape(-1))
        + np.dot(v.reshape(-1), v.reshape(-1))
    ) / float(a.size)
    tiles = tuple(
        _fit_tile(
            estimate,
            a,
            v,
            tile_index=index,
            start=start,
            stop=stop,
            measure_frames=measure,
            global_energy_density=global_density,
            config=cfg,
        )
        for index, ((start, stop), measure) in enumerate(zip(ranges, measures))
    )
    source = tuple(tile for tile in tiles if tile.mode == MODE_SOURCE)
    direct = tuple(tile for tile in tiles if tile.mode == MODE_DIRECT)
    if len(source) + len(direct) != len(tiles):
        raise AssertionError("a tile disappeared from the exact mode partition")
    if any(
        tile.fallback_reason not in FALLBACK_REASONS for tile in direct
    ):
        raise AssertionError("direct tile lacks a declared fallback reason")
    total_measure = sum(tile.measure_frames for tile in tiles)
    source_measure = sum(tile.measure_frames for tile in source)

    recall_cvar, recall_max = _aggregate(source, "recall_risk", cfg.tail_fraction)
    theft_cvar, theft_max = _aggregate(source, "theft_risk", cfg.tail_fraction)
    artifact_cvar, artifact_max = _aggregate(
        source, "artifact_risk", cfg.tail_fraction
    )
    direct_cvar, direct_max = _aggregate(
        direct, "direct_risk", cfg.tail_fraction
    )
    counts = {
        reason: sum(tile.fallback_reason == reason for tile in direct)
        for reason in FALLBACK_REASONS
    }
    fractions = {
        reason: (
            sum(
                tile.measure_frames for tile in direct
                if tile.fallback_reason == reason
            ) / total_measure
        )
        for reason in FALLBACK_REASONS
    }
    cvar_by_reason = {}
    max_by_reason = {}
    for reason in FALLBACK_REASONS:
        rows = tuple(
            tile for tile in direct if tile.fallback_reason == reason
        )
        cvar_by_reason[reason], max_by_reason[reason] = _aggregate(
            rows, "direct_risk", cfg.tail_fraction
        )

    return SurrogateReportV2(
        schema=SCHEMA,
        config=asdict(cfg),
        frames=frames,
        source_coordinate_tiles=len(source),
        direct_fallback_tiles=len(direct),
        total_tiles=len(tiles),
        source_coordinate_measure_fraction=source_measure / total_measure,
        fallback_tile_counts=counts,
        fallback_measure_fractions=fractions,
        recall_cvar=recall_cvar,
        theft_cvar=theft_cvar,
        artifact_cvar=artifact_cvar,
        direct_fallback_cvar=direct_cvar,
        direct_fallback_cvar_by_reason=cvar_by_reason,
        recall_max=recall_max,
        theft_max=theft_max,
        artifact_max=artifact_max,
        direct_fallback_max=direct_max,
        direct_fallback_max_by_reason=max_by_reason,
        tiles=tiles,
    )


def pairwise_false_safe_v2(
    baseline_surrogate: Mapping[str, float | None],
    candidate_surrogate: Mapping[str, float | None],
    baseline_external: Mapping[str, float],
    candidate_external: Mapping[str, float],
    *,
    external_regression_limit_db: float = 0.5,
) -> tuple[str, ...]:
    limit = float(external_regression_limit_db)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("external_regression_limit_db must be finite/nonnegative")
    failures = []
    for surrogate_name, external_name in (
        ("recall_cvar", "retained_voice_db_p90"),
        ("theft_cvar", "event_hole_db_p90"),
    ):
        base_risk = baseline_surrogate.get(surrogate_name)
        next_risk = candidate_surrogate.get(surrogate_name)
        if base_risk is None or next_risk is None:
            continue
        for value, label in (
            (base_risk, "baseline surrogate"),
            (next_risk, "candidate surrogate"),
            (baseline_external.get(external_name), "baseline external"),
            (candidate_external.get(external_name), "candidate external"),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{label} value is missing/non-finite")
        surrogate_improves = float(next_risk) < float(base_risk)
        external_regression = (
            float(candidate_external[external_name])
            - float(baseline_external[external_name])
        )
        if surrogate_improves and external_regression > limit:
            failures.append(
                f"{surrogate_name} improves while {external_name} regresses "
                f"{external_regression:.6g} dB"
            )
    return tuple(failures)
