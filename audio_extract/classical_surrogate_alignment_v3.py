"""Band-local, multiscale surrogate-alignment diagnostic for opera separation.

This is a replay-only, exact-reference diagnostic. It does not train a model.

V3 corrects four limitations of the earlier waveform-wide diagnostic:

* source coordinates are fitted independently in the same broad frequency bands
  used by the event-hole metric, so theft of a low-energy but audible band is
  not diluted by unrelated broadband energy;
* the fit is an unregularized orthogonal projection after a scale-invariant
  identifiability gate, so ``artifact_risk`` is source-orthogonal rather than
  ridge shrinkage error;
* time is evaluated on a frozen multiscale grid and every scale is reported
  separately; no per-work scale selection is permitted;
* weighted CVaR uses the exact fractional tail mass and is invariant to a
  positive rescaling of its measure.

For one identifiable complex STFT cell, the vocal estimate is decomposed as

    V_hat = gamma_a A + gamma_v V + R,

where ``R`` is orthogonal to both exact sources. The independent risks are

    recall   = |1-gamma_v|^2 E_V / (E_A+E_V)
    theft    = |gamma_a|^2 E_A / (E_A+E_V)
    artifact = ||R||^2 / (E_A+E_V).

No-vocal, vocal-only, silent, and ill-conditioned cells use exact direct
accompaniment-target error and remain distinct failure regimes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np

SCHEMA = "audio-extract/classical-surrogate-alignment/v3"
REPRESENTATION = "complex-stft-band-local-joint-stereo/v3"
MODE_SOURCE = "source_coordinates"
FALLBACK_REASONS = (
    "no_vocal",
    "vocal_only",
    "silent",
    "ill_conditioned",
)
DEFAULT_BANDS_HZ = (
    (20.0, 120.0),
    (120.0, 500.0),
    (500.0, 2_000.0),
    (2_000.0, 5_000.0),
    (5_000.0, 10_000.0),
    (10_000.0, 20_000.0),
)
DEFAULT_TIME_SCALES = (
    (0.50, 0.25),
    (0.125, 0.0625),
    (0.040, 0.020),
)
EXTERNAL_AXIS_METRICS = {
    "voice": (
        "retained_voice_db_p90",
        "retained_voice_coef_p90",
    ),
    "hole": (
        "event_hole_db_p90",
        "event_hole_db_max",
        "alpha_error_p90",
    ),
    "artifact": (
        "artifact_ratio_p90",
    ),
}


class SurrogateAlignmentV3Error(RuntimeError):
    """The exact grid, representation, or aggregation is invalid."""


@dataclass(frozen=True)
class SurrogateConfigV3:
    sample_rate_hz: int = 44_100
    window_ms: float = 40.0
    stft_hop_ms: float = 10.0
    n_fft: int = 2_048
    bands_hz: tuple[tuple[float, float], ...] = DEFAULT_BANDS_HZ
    time_scales_seconds: tuple[tuple[float, float], ...] = DEFAULT_TIME_SCALES
    tail_fraction: float = 0.10
    source_fraction_floor: float = 1e-10
    max_condition: float = 1e6
    representation: str = REPRESENTATION

    def validate(self) -> None:
        if not isinstance(self.sample_rate_hz, int) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be a positive integer")
        if not isinstance(self.n_fft, int) or self.n_fft < 8:
            raise ValueError("n_fft must be an integer >= 8")
        for name in (
            "window_ms",
            "stft_hop_ms",
            "tail_fraction",
            "source_fraction_floor",
            "max_condition",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.tail_fraction > 1:
            raise ValueError("tail_fraction must not exceed one")
        if self.source_fraction_floor >= 0.5:
            raise ValueError("source_fraction_floor must be below 0.5")
        if self.max_condition <= 1:
            raise ValueError("max_condition must exceed one")
        if self.window_frames > self.n_fft:
            raise ValueError("n_fft must be at least the analysis window")
        nyquist = self.sample_rate_hz / 2.0
        if not self.bands_hz:
            raise ValueError("at least one frequency band is required")
        previous = -math.inf
        for index, pair in enumerate(self.bands_hz):
            if len(pair) != 2:
                raise ValueError(f"band {index} must contain low/high edges")
            low, high = (float(pair[0]), float(pair[1]))
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or low < 0
                or high <= low
                or high > nyquist
                or low < previous
            ):
                raise ValueError(f"invalid/nonmonotone band {index}: {pair}")
            previous = high
        if not self.time_scales_seconds:
            raise ValueError("at least one time scale is required")
        seen = set()
        for index, pair in enumerate(self.time_scales_seconds):
            if len(pair) != 2:
                raise ValueError(f"time scale {index} must contain window/hop")
            window, hop = (float(pair[0]), float(pair[1]))
            if (
                not math.isfinite(window)
                or not math.isfinite(hop)
                or window <= 0
                or hop <= 0
                or hop > window
            ):
                raise ValueError(f"invalid time scale {index}: {pair}")
            key = (window, hop)
            if key in seen:
                raise ValueError("time scales must be unique")
            seen.add(key)
        if self.representation != REPRESENTATION:
            raise ValueError("unknown surrogate representation")

    @property
    def window_frames(self) -> int:
        return max(8, round(self.sample_rate_hz * self.window_ms / 1000.0))

    @property
    def stft_hop_frames(self) -> int:
        return max(1, round(self.sample_rate_hz * self.stft_hop_ms / 1000.0))


@dataclass(frozen=True)
class CellRiskV3:
    scale_index: int
    time_index: int
    band_index: int
    start_stft_frame: int
    end_stft_frame: int
    band_low_hz: float
    band_high_hz: float
    mode: str
    fallback_reason: str | None
    measure: float
    accompaniment_energy: float
    vocal_energy: float
    condition_number: float | None
    gamma_v_real: float | None
    gamma_v_imag: float | None
    gamma_a_real: float | None
    gamma_a_imag: float | None
    recall_risk: float | None
    theft_risk: float | None
    artifact_risk: float | None
    direct_risk: float | None
    orthogonality_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleReportV3:
    scale_index: int
    window_seconds: float
    hop_seconds: float
    time_tile_frames: int
    time_hop_frames: int
    total_cells: int
    mode_counts: dict[str, int]
    mode_measure_fractions: dict[str, float]
    recall_cvar: float | None
    recall_max: float | None
    theft_cvar: float | None
    theft_max: float | None
    artifact_cvar: float | None
    artifact_max: float | None
    direct_cvar_by_reason: dict[str, float | None]
    direct_max_by_reason: dict[str, float | None]
    voice_axis_cvar: float | None
    voice_axis_max: float | None
    hole_axis_cvar: float | None
    hole_axis_max: float | None
    artifact_axis_cvar: float | None
    artifact_axis_max: float | None
    max_orthogonality_error: float | None
    cells: tuple[CellRiskV3, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cells"] = [cell.to_dict() for cell in self.cells]
        return result


@dataclass(frozen=True)
class SurrogateReportV3:
    schema: str
    config: dict[str, Any]
    frames: int
    stft_frames: int
    frequency_bins: int
    scale_reports: tuple[ScaleReportV3, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scale_reports"] = [
            report.to_dict() for report in self.scale_reports
        ]
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


def tile_ranges(
    frames: int,
    tile: int,
    hop: int,
) -> tuple[tuple[int, int], ...]:
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
    """Assign each analysis frame exactly one unit across overlapping tiles."""

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
        raise ValueError("tile ranges must cover every analysis frame")
    inverse = 1.0 / coverage.astype(np.float64)
    result = tuple(
        float(inverse[start:end].sum()) for start, end in normalized
    )
    tolerance = (
        16.0 * np.finfo(np.float64).eps * max(1.0, float(frames))
    )
    if not math.isclose(
        sum(result), float(frames), rel_tol=0.0, abs_tol=tolerance
    ):
        raise SurrogateAlignmentV3Error(
            "tile measure does not partition analysis frames"
        )
    if any(weight <= 0 or not math.isfinite(weight) for weight in result):
        raise SurrogateAlignmentV3Error("tile measure is invalid")
    return result


def weighted_cvar(
    values: Sequence[float],
    weights: Sequence[float],
    tail_fraction: float,
) -> float | None:
    """Exact weighted mean of the largest ``tail_fraction`` measure."""

    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if value_array.size == 0:
        return None
    if value_array.shape != weight_array.shape:
        raise ValueError("CVaR values and weights differ")
    if (
        not np.all(np.isfinite(value_array))
        or not np.all(np.isfinite(weight_array))
        or np.any(value_array < 0)
        or np.any(weight_array <= 0)
    ):
        raise ValueError("CVaR values/weights must be finite and valid")
    fraction = float(tail_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("tail_fraction must lie in (0,1]")
    target = fraction * float(weight_array.sum())
    if not math.isfinite(target) or target <= 0:
        raise ValueError("CVaR selected measure must be finite and positive")

    order = np.argsort(value_array, kind="stable")[::-1]
    remaining = target
    total = 0.0
    used = 0.0
    for index in order:
        if remaining <= 0.0:
            break
        amount = min(float(weight_array[index]), remaining)
        total += amount * float(value_array[index])
        used += amount
        remaining -= amount
    tolerance = 32.0 * np.finfo(np.float64).eps * target
    if abs(remaining) > tolerance or used <= 0:
        raise SurrogateAlignmentV3Error(
            f"CVaR failed to consume exact tail mass: remaining={remaining}"
        )
    return total / used


def _stft_stereo(
    audio: np.ndarray,
    config: SurrogateConfigV3,
) -> tuple[np.ndarray, np.ndarray]:
    window_frames = config.window_frames
    hop = config.stft_hop_frames
    frames = len(audio)
    if frames <= window_frames:
        count = 1
    else:
        count = 1 + math.ceil((frames - window_frames) / hop)
    padded_frames = (count - 1) * hop + window_frames
    if padded_frames > frames:
        audio = np.pad(audio, ((0, padded_frames - frames), (0, 0)))
    starts = hop * np.arange(count, dtype=np.int64)
    indices = starts[:, None] + np.arange(window_frames)[None, :]
    window = np.hanning(window_frames).astype(np.float64)
    framed = audio[indices] * window[None, :, None]
    spectrum = np.fft.rfft(framed, n=config.n_fft, axis=1)
    frequencies = np.fft.rfftfreq(
        config.n_fft, d=1.0 / config.sample_rate_hz
    )
    if not np.all(np.isfinite(spectrum)):
        raise SurrogateAlignmentV3Error("STFT produced non-finite values")
    return np.asarray(spectrum, dtype=np.complex128), frequencies


def _band_slices(
    frequencies: np.ndarray,
    bands_hz: Sequence[Sequence[float]],
) -> tuple[np.ndarray, ...]:
    result = []
    for index, pair in enumerate(bands_hz):
        low, high = float(pair[0]), float(pair[1])
        selected = np.flatnonzero(
            (frequencies >= low) & (frequencies < high)
        )
        if selected.size == 0:
            raise ValueError(
                f"frequency band {index} contains no FFT bins: {(low, high)}"
            )
        result.append(selected)
    return tuple(result)


def fit_complex_cell_v3(
    vocal_estimate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    source_fraction_floor: float,
    max_condition: float,
    silent_reference_energy: float,
) -> dict[str, Any]:
    """Fit one complex joint-stereo source-coordinate cell.

    Arrays are flattened complex coefficients. The solve is unregularized after
    a scale-invariant identifiability gate, making the residual an actual
    orthogonal projection.
    """

    y = np.asarray(vocal_estimate, dtype=np.complex128).reshape(-1)
    a = np.asarray(accompaniment, dtype=np.complex128).reshape(-1)
    v = np.asarray(vocal, dtype=np.complex128).reshape(-1)
    if y.size < 1 or a.shape != y.shape or v.shape != y.shape:
        raise ValueError("complex cell arrays must be equal and non-empty")
    if not (
        np.all(np.isfinite(y))
        and np.all(np.isfinite(a))
        and np.all(np.isfinite(v))
    ):
        raise ValueError("complex cell contains non-finite values")
    floor = float(source_fraction_floor)
    condition_limit = float(max_condition)
    if (
        not math.isfinite(floor)
        or floor <= 0
        or floor >= 0.5
        or not math.isfinite(condition_limit)
        or condition_limit <= 1
    ):
        raise ValueError("invalid source identifiability settings")
    silent_reference = float(silent_reference_energy)
    if not math.isfinite(silent_reference) or silent_reference <= 0:
        raise ValueError("silent_reference_energy must be finite and positive")

    aa = float(np.vdot(a, a).real)
    vv = float(np.vdot(v, v).real)
    total = aa + vv
    if total == 0.0:
        direct = float(np.vdot(y, y).real / silent_reference)
        return {
            "mode": "silent",
            "condition_number": None,
            "gamma_a": None,
            "gamma_v": None,
            "recall_risk": None,
            "theft_risk": None,
            "artifact_risk": None,
            "direct_risk": direct,
            "orthogonality_error": None,
            "accompaniment_energy": aa,
            "vocal_energy": vv,
        }

    a_fraction = aa / total
    v_fraction = vv / total
    if v_fraction <= floor:
        reason = "no_vocal"
    elif a_fraction <= floor:
        reason = "vocal_only"
    else:
        reason = None

    av = np.vdot(a, v)
    gram = np.asarray(
        [[aa, av], [np.conj(av), vv]], dtype=np.complex128
    )
    normalized_gram = gram / total
    eigenvalues = np.linalg.eigvalsh(normalized_gram)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    condition = (
        math.inf
        if minimum <= 0.0
        else float(maximum / minimum)
    )
    if reason is None and (
        not math.isfinite(condition) or condition > condition_limit
    ):
        reason = "ill_conditioned"

    if reason is not None:
        error = v - y
        direct = float(np.vdot(error, error).real / total)
        return {
            "mode": reason,
            "condition_number": (
                condition if math.isfinite(condition) else None
            ),
            "gamma_a": None,
            "gamma_v": None,
            "recall_risk": None,
            "theft_risk": None,
            "artifact_risk": None,
            "direct_risk": direct,
            "orthogonality_error": None,
            "accompaniment_energy": aa,
            "vocal_energy": vv,
        }

    rhs = np.asarray(
        [np.vdot(a, y), np.vdot(v, y)], dtype=np.complex128
    ) / total
    gamma_a, gamma_v = np.linalg.solve(normalized_gram, rhs)
    residual = y - gamma_a * a - gamma_v * v
    residual_energy = float(np.vdot(residual, residual).real)
    residual_norm = math.sqrt(max(residual_energy, 0.0))
    orthogonality = 0.0
    numerical_zero = 256.0 * np.finfo(np.float64).eps * total
    if residual_energy > numerical_zero:
        orthogonality = max(
            abs(np.vdot(a, residual))
            / (math.sqrt(aa) * residual_norm),
            abs(np.vdot(v, residual))
            / (math.sqrt(vv) * residual_norm),
        )
    return {
        "mode": MODE_SOURCE,
        "condition_number": condition,
        "gamma_a": complex(gamma_a),
        "gamma_v": complex(gamma_v),
        "recall_risk": float(abs(1.0 - gamma_v) ** 2 * vv / total),
        "theft_risk": float(abs(gamma_a) ** 2 * aa / total),
        "artifact_risk": float(residual_energy / total),
        "direct_risk": None,
        "orthogonality_error": float(orthogonality),
        "accompaniment_energy": aa,
        "vocal_energy": vv,
    }


def _aggregate(
    cells: Sequence[CellRiskV3],
    field: str,
    tail_fraction: float,
) -> tuple[float | None, float | None]:
    selected = [
        cell for cell in cells if getattr(cell, field) is not None
    ]
    if not selected:
        return None, None
    values = [float(getattr(cell, field)) for cell in selected]
    weights = [float(cell.measure) for cell in selected]
    return weighted_cvar(values, weights, tail_fraction), max(values)


def _maximum_available(values: Sequence[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return max(selected) if selected else None


def evaluate_surrogate_v3(
    vocal_estimate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    config: SurrogateConfigV3 | None = None,
) -> SurrogateReportV3:
    cfg = config or SurrogateConfigV3()
    cfg.validate()
    estimate = _audio(vocal_estimate, "vocal_estimate")
    a = _audio(accompaniment, "accompaniment")
    v = _audio(vocal, "vocal")
    if estimate.shape != a.shape or v.shape != a.shape:
        raise ValueError("surrogate inputs occupy different sample grids")

    estimate_spectrum, frequencies = _stft_stereo(estimate, cfg)
    accompaniment_spectrum, frequencies_a = _stft_stereo(a, cfg)
    vocal_spectrum, frequencies_v = _stft_stereo(v, cfg)
    if not (
        np.array_equal(frequencies, frequencies_a)
        and np.array_equal(frequencies, frequencies_v)
        and estimate_spectrum.shape == accompaniment_spectrum.shape
        and vocal_spectrum.shape == accompaniment_spectrum.shape
    ):
        raise AssertionError("STFT grids differ")
    bands = _band_slices(frequencies, cfg.bands_hz)

    global_band_density = []
    for selected in bands:
        a_band = accompaniment_spectrum[:, selected, :]
        v_band = vocal_spectrum[:, selected, :]
        energy = float(
            np.vdot(a_band, a_band).real
            + np.vdot(v_band, v_band).real
        )
        global_band_density.append(
            energy / max(1, int(a_band.size))
        )
    total_density = float(
        np.vdot(accompaniment_spectrum, accompaniment_spectrum).real
        + np.vdot(vocal_spectrum, vocal_spectrum).real
    ) / max(1, int(accompaniment_spectrum.size))

    scale_reports = []
    stft_frames = accompaniment_spectrum.shape[0]
    hop_seconds = cfg.stft_hop_frames / cfg.sample_rate_hz
    for scale_index, (window_seconds, scale_hop_seconds) in enumerate(
        cfg.time_scales_seconds
    ):
        time_tile_frames = max(1, round(window_seconds / hop_seconds))
        time_hop_frames = max(1, round(scale_hop_seconds / hop_seconds))
        ranges = tile_ranges(
            stft_frames, time_tile_frames, time_hop_frames
        )
        time_measures = partition_measure_weights(stft_frames, ranges)
        cells = []
        for time_index, ((start, end), time_measure) in enumerate(
            zip(ranges, time_measures)
        ):
            for band_index, selected in enumerate(bands):
                y = estimate_spectrum[start:end, selected, :].reshape(-1)
                a_cell = accompaniment_spectrum[
                    start:end, selected, :
                ].reshape(-1)
                v_cell = vocal_spectrum[
                    start:end, selected, :
                ].reshape(-1)
                reference_density = (
                    global_band_density[band_index]
                    if global_band_density[band_index] > 0
                    else total_density
                )
                silent_reference = max(
                    reference_density * max(1, y.size),
                    np.finfo(np.float64).tiny,
                )
                fit = fit_complex_cell_v3(
                    y,
                    a_cell,
                    v_cell,
                    source_fraction_floor=cfg.source_fraction_floor,
                    max_condition=cfg.max_condition,
                    silent_reference_energy=silent_reference,
                )
                gamma_a = fit["gamma_a"]
                gamma_v = fit["gamma_v"]
                cells.append(CellRiskV3(
                    scale_index=scale_index,
                    time_index=time_index,
                    band_index=band_index,
                    start_stft_frame=start,
                    end_stft_frame=end,
                    band_low_hz=float(cfg.bands_hz[band_index][0]),
                    band_high_hz=float(cfg.bands_hz[band_index][1]),
                    mode=(
                        MODE_SOURCE
                        if fit["mode"] == MODE_SOURCE
                        else "direct_fallback"
                    ),
                    fallback_reason=(
                        None
                        if fit["mode"] == MODE_SOURCE
                        else str(fit["mode"])
                    ),
                    measure=float(time_measure),
                    accompaniment_energy=float(
                        fit["accompaniment_energy"]
                    ),
                    vocal_energy=float(fit["vocal_energy"]),
                    condition_number=fit["condition_number"],
                    gamma_v_real=(
                        None if gamma_v is None else float(gamma_v.real)
                    ),
                    gamma_v_imag=(
                        None if gamma_v is None else float(gamma_v.imag)
                    ),
                    gamma_a_real=(
                        None if gamma_a is None else float(gamma_a.real)
                    ),
                    gamma_a_imag=(
                        None if gamma_a is None else float(gamma_a.imag)
                    ),
                    recall_risk=fit["recall_risk"],
                    theft_risk=fit["theft_risk"],
                    artifact_risk=fit["artifact_risk"],
                    direct_risk=fit["direct_risk"],
                    orthogonality_error=fit["orthogonality_error"],
                ))

        cell_tuple = tuple(cells)
        expected_count = len(ranges) * len(bands)
        if len(cell_tuple) != expected_count:
            raise AssertionError("a time/band cell disappeared")
        total_measure = sum(cell.measure for cell in cell_tuple)
        expected_measure = float(stft_frames * len(bands))
        if not math.isclose(
            total_measure,
            expected_measure,
            rel_tol=0.0,
            abs_tol=(
                32.0
                * np.finfo(np.float64).eps
                * max(1.0, expected_measure)
            ),
        ):
            raise SurrogateAlignmentV3Error(
                "multiband measure does not partition the scale"
            )

        mode_counts = {
            MODE_SOURCE: sum(
                cell.mode == MODE_SOURCE for cell in cell_tuple
            ),
            **{
                reason: sum(
                    cell.fallback_reason == reason for cell in cell_tuple
                )
                for reason in FALLBACK_REASONS
            },
        }
        mode_measures = {
            MODE_SOURCE: sum(
                cell.measure for cell in cell_tuple
                if cell.mode == MODE_SOURCE
            )
            / total_measure,
            **{
                reason: sum(
                    cell.measure for cell in cell_tuple
                    if cell.fallback_reason == reason
                )
                / total_measure
                for reason in FALLBACK_REASONS
            },
        }
        if sum(mode_counts.values()) != len(cell_tuple):
            raise AssertionError("cell mode partition is incomplete")
        if not math.isclose(
            sum(mode_measures.values()), 1.0, abs_tol=1e-12
        ):
            raise AssertionError("cell measure partition is incomplete")

        recall_cvar, recall_max = _aggregate(
            cell_tuple, "recall_risk", cfg.tail_fraction
        )
        theft_cvar, theft_max = _aggregate(
            cell_tuple, "theft_risk", cfg.tail_fraction
        )
        artifact_cvar, artifact_max = _aggregate(
            cell_tuple, "artifact_risk", cfg.tail_fraction
        )
        direct_cvar_by_reason = {}
        direct_max_by_reason = {}
        for reason in FALLBACK_REASONS:
            selected_cells = tuple(
                cell for cell in cell_tuple
                if cell.fallback_reason == reason
            )
            (
                direct_cvar_by_reason[reason],
                direct_max_by_reason[reason],
            ) = _aggregate(
                selected_cells, "direct_risk", cfg.tail_fraction
            )

        voice_axis_cvar = _maximum_available((
            recall_cvar,
            direct_cvar_by_reason["vocal_only"],
            direct_cvar_by_reason["ill_conditioned"],
        ))
        voice_axis_max = _maximum_available((
            recall_max,
            direct_max_by_reason["vocal_only"],
            direct_max_by_reason["ill_conditioned"],
        ))
        hole_axis_cvar = _maximum_available((
            theft_cvar,
            direct_cvar_by_reason["no_vocal"],
            direct_cvar_by_reason["ill_conditioned"],
        ))
        hole_axis_max = _maximum_available((
            theft_max,
            direct_max_by_reason["no_vocal"],
            direct_max_by_reason["ill_conditioned"],
        ))
        artifact_axis_cvar = _maximum_available((
            artifact_cvar,
            direct_cvar_by_reason["silent"],
            direct_cvar_by_reason["ill_conditioned"],
        ))
        artifact_axis_max = _maximum_available((
            artifact_max,
            direct_max_by_reason["silent"],
            direct_max_by_reason["ill_conditioned"],
        ))
        orthogonality = [
            float(cell.orthogonality_error)
            for cell in cell_tuple
            if cell.orthogonality_error is not None
        ]
        scale_reports.append(ScaleReportV3(
            scale_index=scale_index,
            window_seconds=float(window_seconds),
            hop_seconds=float(scale_hop_seconds),
            time_tile_frames=time_tile_frames,
            time_hop_frames=time_hop_frames,
            total_cells=len(cell_tuple),
            mode_counts=mode_counts,
            mode_measure_fractions=mode_measures,
            recall_cvar=recall_cvar,
            recall_max=recall_max,
            theft_cvar=theft_cvar,
            theft_max=theft_max,
            artifact_cvar=artifact_cvar,
            artifact_max=artifact_max,
            direct_cvar_by_reason=direct_cvar_by_reason,
            direct_max_by_reason=direct_max_by_reason,
            voice_axis_cvar=voice_axis_cvar,
            voice_axis_max=voice_axis_max,
            hole_axis_cvar=hole_axis_cvar,
            hole_axis_max=hole_axis_max,
            artifact_axis_cvar=artifact_axis_cvar,
            artifact_axis_max=artifact_axis_max,
            max_orthogonality_error=(
                max(orthogonality) if orthogonality else None
            ),
            cells=cell_tuple,
        ))

    return SurrogateReportV3(
        schema=SCHEMA,
        config=asdict(cfg),
        frames=len(a),
        stft_frames=stft_frames,
        frequency_bins=len(frequencies),
        scale_reports=tuple(scale_reports),
    )


def _report_mapping(
    value: SurrogateReportV3 | Mapping[str, Any],
) -> Mapping[str, Any]:
    return value.to_dict() if isinstance(value, SurrogateReportV3) else value


def _axis_improves(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    axis: str,
    *,
    tolerance: float,
) -> bool:
    fields = (f"{axis}_axis_cvar", f"{axis}_axis_max")
    baseline_scales = baseline.get("scale_reports")
    candidate_scales = candidate.get("scale_reports")
    if (
        not isinstance(baseline_scales, Sequence)
        or not isinstance(candidate_scales, Sequence)
        or len(baseline_scales) != len(candidate_scales)
        or not baseline_scales
    ):
        raise ValueError("surrogate reports have incompatible scale sets")
    strictly_better = False
    compared = 0
    for baseline_scale, candidate_scale in zip(
        baseline_scales, candidate_scales
    ):
        if (
            baseline_scale.get("scale_index")
            != candidate_scale.get("scale_index")
            or baseline_scale.get("window_seconds")
            != candidate_scale.get("window_seconds")
            or baseline_scale.get("hop_seconds")
            != candidate_scale.get("hop_seconds")
        ):
            raise ValueError("surrogate reports use different scale identities")
        for field in fields:
            base_value = baseline_scale.get(field)
            next_value = candidate_scale.get(field)
            if base_value is None and next_value is None:
                continue
            if base_value is None or next_value is None:
                raise ValueError(
                    f"surrogate availability changed for {field}"
                )
            for value in (base_value, next_value):
                if not isinstance(value, (int, float)) or not math.isfinite(
                    float(value)
                ):
                    raise ValueError(
                        f"surrogate field {field} is missing/non-finite"
                    )
            compared += 1
            delta = float(next_value) - float(base_value)
            if delta > tolerance:
                return False
            if delta < -tolerance:
                strictly_better = True
    return compared > 0 and strictly_better


def pairwise_false_safe_v3(
    baseline_surrogate: SurrogateReportV3 | Mapping[str, Any],
    candidate_surrogate: SurrogateReportV3 | Mapping[str, Any],
    baseline_external: Mapping[str, float],
    candidate_external: Mapping[str, float],
    *,
    external_regression_limits: Mapping[str, float],
    surrogate_tolerance: float = 0.0,
) -> tuple[str, ...]:
    """Find a surrogate-improves / exact-product-regresses contradiction.

    A surrogate axis improves only when its CVaR and maximum are no worse at
    every frozen scale and at least one is strictly better. Fallback risks are
    already included in the axis summaries.
    """

    tolerance = float(surrogate_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("surrogate_tolerance must be finite/nonnegative")
    required_external = {
        metric
        for metrics in EXTERNAL_AXIS_METRICS.values()
        for metric in metrics
    }
    if set(external_regression_limits) != required_external:
        raise ValueError(
            "external_regression_limits must cover every exact gate"
        )
    limits = {}
    for metric, value in external_regression_limits.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(
                f"external regression limit is invalid: {metric}"
            )
        limits[metric] = numeric

    baseline = _report_mapping(baseline_surrogate)
    candidate = _report_mapping(candidate_surrogate)
    failures = []
    for axis, metric_names in EXTERNAL_AXIS_METRICS.items():
        if not _axis_improves(
            baseline, candidate, axis, tolerance=tolerance
        ):
            continue
        for metric in metric_names:
            base_value = baseline_external.get(metric)
            next_value = candidate_external.get(metric)
            for value, label in (
                (base_value, f"baseline {metric}"),
                (next_value, f"candidate {metric}"),
            ):
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"{label} is missing/non-finite")
            regression = float(next_value) - float(base_value)
            if regression > limits[metric]:
                failures.append(
                    f"{axis} surrogate improves while {metric} regresses "
                    f"{regression:.6g}"
                )
    return tuple(failures)
