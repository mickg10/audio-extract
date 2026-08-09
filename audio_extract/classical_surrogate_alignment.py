"""Exact no-optimizer surrogate-alignment diagnostics for opera separation.

The failed Tier-A continuation showed that a scalar reconstruction/control loss
can improve while the user-facing retained-soloist gate worsens. This module
constructs an exact, local, defect-separated diagnostic on frozen checkpoint
outputs. It is intentionally NumPy-only and non-differentiable: its first job is
to prove that a proposed loss family ranks already-rendered checkpoints in the
same direction as the exact product gates before another optimizer step is
allowed.

For each local tile, fit the vocal estimate as

    V_hat = gamma_v * V + gamma_a * A + R.

The module reports independent risks for target recall, accompaniment theft,
orthogonal artifacts, and a direct exact-target fallback when A/V attribution is
not identifiable. Every nonempty tile has exactly one mode and contributes to a
coverage-weighted tail statistic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np

SCHEMA = "audio-extract/classical-surrogate-alignment/v1"
MODE_SOURCE = "source_coordinates"
MODE_DIRECT = "direct_fallback"


class SurrogateAlignmentError(RuntimeError):
    """The exact-grid input or surrogate configuration is invalid."""


@dataclass(frozen=True)
class SurrogateConfig:
    sample_rate_hz: int = 44_100
    tile_seconds: float = 0.5
    hop_seconds: float = 0.25
    tail_fraction: float = 0.10
    ridge_relative: float = 1e-8
    max_condition: float = 1e6
    energy_floor_relative: float = 1e-10
    energy_floor_absolute: float = 1e-18

    def validate(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
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

    @property
    def tile_frames(self) -> int:
        self.validate()
        return max(16, round(self.tile_seconds * self.sample_rate_hz))

    @property
    def hop_frames(self) -> int:
        self.validate()
        return max(1, round(self.hop_seconds * self.sample_rate_hz))


@dataclass(frozen=True)
class TileRisk:
    tile_index: int
    start_frame: int
    end_frame: int
    mode: str
    weight: float
    accompaniment_energy: float
    vocal_energy: float
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
class SurrogateReport:
    schema: str
    config: dict[str, Any]
    frames: int
    source_coordinate_tiles: int
    direct_fallback_tiles: int
    total_tiles: int
    source_coordinate_fraction: float
    recall_cvar: float | None
    theft_cvar: float | None
    artifact_cvar: float | None
    direct_fallback_cvar: float | None
    recall_max: float | None
    theft_max: float | None
    artifact_max: float | None
    direct_fallback_max: float | None
    tiles: tuple[TileRisk, ...]

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


def _starts(frames: int, tile: int, hop: int) -> tuple[int, ...]:
    if frames <= tile:
        return (0,)
    result = list(range(0, frames - tile + 1, hop))
    final = frames - tile
    if result[-1] != final:
        result.append(final)
    return tuple(result)


def _eigenvalues_2x2(aa: float, vv: float, av: float) -> tuple[float, float]:
    trace = aa + vv
    discriminant = math.sqrt(max(0.0, (aa - vv) ** 2 + 4.0 * av * av))
    return 0.5 * (trace - discriminant), 0.5 * (trace + discriminant)


def _fit_tile(
    vocal_estimate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    tile_index: int,
    start: int,
    stop: int,
    global_energy_scale: float,
    config: SurrogateConfig,
) -> TileRisk:
    estimate = vocal_estimate[start:stop].reshape(-1)
    a = accompaniment[start:stop].reshape(-1)
    v = vocal[start:stop].reshape(-1)
    aa = float(np.dot(a, a))
    vv = float(np.dot(v, v))
    av = float(np.dot(a, v))
    energy = aa + vv
    floor = max(
        float(config.energy_floor_absolute),
        float(config.energy_floor_relative) * global_energy_scale,
    )
    weight = float(stop - start)

    minimum, maximum = _eigenvalues_2x2(aa, vv, av)
    condition = math.inf if minimum <= 0.0 else float(maximum / minimum)
    identifiable = (
        aa > floor
        and vv > floor
        and maximum > floor
        and minimum > maximum / float(config.max_condition)
        and math.isfinite(condition)
    )

    if identifiable:
        ridge = float(config.ridge_relative) * max(0.5 * (aa + vv), floor)
        matrix = np.asarray([[aa + ridge, av], [av, vv + ridge]])
        rhs = np.asarray(
            [float(np.dot(a, estimate)), float(np.dot(v, estimate))]
        )
        try:
            gamma_a, gamma_v = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded above
            raise SurrogateAlignmentError(
                "identifiable tile solve failed"
            ) from exc
        residue = estimate - gamma_a * a - gamma_v * v
        denominator = max(energy, floor)
        return TileRisk(
            tile_index=tile_index,
            start_frame=start,
            end_frame=stop,
            mode=MODE_SOURCE,
            weight=weight,
            accompaniment_energy=aa,
            vocal_energy=vv,
            condition_number=condition,
            gamma_v=float(gamma_v),
            gamma_a=float(gamma_a),
            recall_risk=float(
                (1.0 - gamma_v) ** 2 * vv / denominator
            ),
            theft_risk=float(gamma_a**2 * aa / denominator),
            artifact_risk=float(np.dot(residue, residue) / denominator),
            direct_risk=None,
        )

    accompaniment_estimate = (
        accompaniment[start:stop]
        + vocal[start:stop]
        - vocal_estimate[start:stop]
    )
    direct_error = accompaniment_estimate.reshape(-1) - a
    direct = float(
        np.dot(direct_error, direct_error) / max(energy, floor)
    )
    return TileRisk(
        tile_index=tile_index,
        start_frame=start,
        end_frame=stop,
        mode=MODE_DIRECT,
        weight=weight,
        accompaniment_energy=aa,
        vocal_energy=vv,
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
    """Return the weighted mean of the largest ``tail_fraction`` mass."""

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
        raise ValueError(
            "CVaR inputs must be finite and nonnegative/positive"
        )
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
        if remaining <= 1e-15 * max(1.0, target):
            break
    if used <= 0:
        raise SurrogateAlignmentError("CVaR selected no weight")
    return total / used


def evaluate_surrogate(
    vocal_estimate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    config: SurrogateConfig | None = None,
) -> SurrogateReport:
    cfg = config or SurrogateConfig()
    cfg.validate()
    estimate = _audio(vocal_estimate, "vocal_estimate")
    a = _audio(accompaniment, "accompaniment")
    v = _audio(vocal, "vocal")
    if estimate.shape != a.shape or v.shape != a.shape:
        raise ValueError(
            "surrogate inputs occupy different sample grids"
        )
    frames = len(a)
    global_scale = float(
        np.dot(a.reshape(-1), a.reshape(-1))
        + np.dot(v.reshape(-1), v.reshape(-1))
    ) / max(1, frames)
    tiles = tuple(
        _fit_tile(
            estimate,
            a,
            v,
            tile_index=index,
            start=start,
            stop=min(frames, start + cfg.tile_frames),
            global_energy_scale=global_scale,
            config=cfg,
        )
        for index, start in enumerate(
            _starts(frames, cfg.tile_frames, cfg.hop_frames)
        )
    )
    source = [tile for tile in tiles if tile.mode == MODE_SOURCE]
    direct = [tile for tile in tiles if tile.mode == MODE_DIRECT]
    if len(source) + len(direct) != len(tiles):
        raise AssertionError(
            "a tile disappeared from the exact mode partition"
        )

    def aggregate(name: str, rows: Sequence[TileRisk]):
        values = [float(getattr(tile, name)) for tile in rows]
        weights = [tile.weight for tile in rows]
        return (
            weighted_cvar(values, weights, cfg.tail_fraction),
            (max(values) if values else None),
        )

    recall_cvar, recall_max = aggregate("recall_risk", source)
    theft_cvar, theft_max = aggregate("theft_risk", source)
    artifact_cvar, artifact_max = aggregate("artifact_risk", source)
    direct_cvar, direct_max = aggregate("direct_risk", direct)
    return SurrogateReport(
        schema=SCHEMA,
        config=asdict(cfg),
        frames=frames,
        source_coordinate_tiles=len(source),
        direct_fallback_tiles=len(direct),
        total_tiles=len(tiles),
        source_coordinate_fraction=len(source) / len(tiles),
        recall_cvar=recall_cvar,
        theft_cvar=theft_cvar,
        artifact_cvar=artifact_cvar,
        direct_fallback_cvar=direct_cvar,
        recall_max=recall_max,
        theft_max=theft_max,
        artifact_max=artifact_max,
        direct_fallback_max=direct_max,
        tiles=tiles,
    )


def pairwise_false_safe(
    baseline_surrogate: Mapping[str, float | None],
    candidate_surrogate: Mapping[str, float | None],
    baseline_external: Mapping[str, float],
    candidate_external: Mapping[str, float],
    *,
    external_regression_limit_db: float = 0.5,
) -> tuple[str, ...]:
    """Return cases where a surrogate improves but an exact gate worsens."""

    failures = []
    pairs = (
        ("recall_cvar", "retained_voice_db_p90"),
        ("theft_cvar", "event_hole_db_p90"),
    )
    for surrogate_name, external_name in pairs:
        base_risk = baseline_surrogate.get(surrogate_name)
        candidate_risk = candidate_surrogate.get(surrogate_name)
        if base_risk is None or candidate_risk is None:
            continue
        surrogate_improves = float(candidate_risk) < float(base_risk)
        external_regression = (
            float(candidate_external[external_name])
            - float(baseline_external[external_name])
        )
        if (
            surrogate_improves
            and external_regression > external_regression_limit_db
        ):
            failures.append(
                f"{surrogate_name} improves while {external_name} regresses "
                f"{external_regression:.6g} dB"
            )
    return tuple(failures)
