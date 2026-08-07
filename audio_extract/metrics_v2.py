"""Defect metric v2 (v2.1 WP5 / §10).

Contract: every measurement is an *observation* — ``{metric, value, uncertainty,
available, unit, details}`` — and missing evidence is ``available=False``, never a
zero. The centerpiece replaces the global-p75 pumping scalar with an
**event-conditioned hole metric**: deficits are measured only during vocal events,
against an expected envelope built from flanking context and a leave-one-out
family-balanced consensus, with vocal-contaminated cells marked *unknown* instead
of silently "better".
"""

from __future__ import annotations

import numpy as np

from . import dsp

_EPS = 1e-12


def obs(metric: str, value: float | None, *, available: bool = True,
        uncertainty: float | None = None, unit: str = "", details: dict | None = None) -> dict:
    return {"metric": metric, "value": None if value is None else float(value),
            "uncertainty": uncertainty, "available": bool(available),
            "unit": unit, "details": details or {}}


# ---------------------------------------------------------------------------
# global level fit (§10.4): ONE gain for fullness/brightness comparisons
# ---------------------------------------------------------------------------
def fit_global_gain_db(cand: np.ndarray, ref: np.ndarray, sr: int,
                       exclude_frames: np.ndarray | None = None) -> float:
    """Median per-frame RMS difference (dB), optionally excluding (vocal) frames.
    Applied once; never framewise normalization."""
    ec = dsp.frame_rms_db(cand, sr)
    er = dsp.frame_rms_db(ref, sr)
    n = min(len(ec), len(er))
    d = er[:n] - ec[:n]
    if exclude_frames is not None:
        keep = ~exclude_frames[:n]
        if keep.sum() >= 5:
            d = d[keep]
    return float(np.median(d))


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    return dsp.as2d(x) * (10 ** (gain_db / 20.0))


# ---------------------------------------------------------------------------
# leave-one-out family-balanced feature consensus (§10.9)
# ---------------------------------------------------------------------------
def loo_family_consensus(envs: dict[str, np.ndarray], families: dict[str, str],
                         exclude: str) -> tuple[np.ndarray | None, dict]:
    """Expected band envelope for candidate ``exclude``: aggregate the OTHER
    candidates' envelopes within family first, then across families (equal family
    weight). Returns ``(expected, info)``; expected is None when no other
    candidate exists. ``info`` reports family diversity for gating."""
    contrib: dict[str, list[np.ndarray]] = {}
    for cid, env in envs.items():
        if cid == exclude:
            continue
        contrib.setdefault(families.get(cid, "unknown"), []).append(env)
    if not contrib:
        return None, {"families": 0, "members": 0}
    T = min(min(e.shape[1] for e in mem) for mem in contrib.values())
    fam_means = [np.mean([e[:, :T] for e in mem], axis=0) for mem in contrib.values()]
    expected = np.mean(fam_means, axis=0)
    info = {"families": len(contrib),
            "members": sum(len(m) for m in contrib.values()),
            "family_names": sorted(contrib)}
    return expected, info


# ---------------------------------------------------------------------------
# event-conditioned hole metric (§10.2)
# ---------------------------------------------------------------------------
def event_holes(cand_env: np.ndarray, expected_env: np.ndarray, vocal_env: np.ndarray,
                events: list[tuple[int, int]], *, hop_ms: float = 10.0,
                contamination_margin_db: float = -6.0,
                context_frames: int = 30, min_band_db_below_peak: float = 40.0,
                min_hole_ms: float = 20.0) -> list[dict]:
    """Per vocal event × band deficits against the expected envelope.

    * The expected level per (event, band) blends the LOO consensus with the
      flanking pre/post context — so a *written* diminuendo (present in both) is
      not scored as a hole.
    * A cell where the vocal envelope is within ``contamination_margin_db`` of the
      expected accompaniment is dominated by voice: it is marked UNKNOWN and feeds
      ``masked_hole_uncertainty`` instead of the deficit (bleed can no longer make
      a candidate look better).
    * NEAR-SILENT target bands are gated (oracle §8.1): a spectacular dB deficit in
      a band that carries negligible accompaniment energy is inaudible — cells
      whose expected level sits ``min_band_db_below_peak`` below the track's peak
      band energy, or whose hole is shorter than ``min_hole_ms``, are skipped.

    Returns observation dicts: event_hole_depth/area/duration/recovery (dB-based)
    + masked_hole_uncertainty, aggregated over the WORST events, not averaged away.
    """
    B, T = cand_env.shape
    Te = min(T, expected_env.shape[1], vocal_env.shape[1])
    peak_band_db = float(np.max(expected_env[:, :Te])) if Te > 0 else -np.inf
    silent_floor = peak_band_db - min_band_db_below_peak
    depths, areas, durs, recovs = [], [], [], []
    silent_cells = 0
    masked_cells = total_cells = 0
    for (e0, e1) in events:
        e0c, e1c = max(0, min(e0, Te - 1)), max(1, min(e1, Te))
        if e1c <= e0c:
            continue
        pre0 = max(0, e0c - context_frames)
        post1 = min(Te, e1c + context_frames)
        for b in range(B):
            total_cells += 1
            ctx = np.concatenate([expected_env[b, pre0:e0c], expected_env[b, e1c:post1]])
            flank = np.concatenate([cand_env[b, pre0:e0c], cand_env[b, e1c:post1]])
            base = expected_env[b, e0c:e1c]
            # blend consensus with local trend: if the candidate's own flanks sit
            # below the consensus flanks, the difference is candidate level, not a hole
            level_off = float(np.median(flank) - np.median(ctx)) if ctx.size and flank.size else 0.0
            expected_evt = base + level_off
            # §8.1: near-silent target band -> a big dB deficit here is inaudible
            if expected_evt.max(initial=-np.inf) < silent_floor:
                silent_cells += 1
                continue
            if vocal_env[b, e0c:e1c].max(initial=-np.inf) >= expected_evt.max() + contamination_margin_db:
                masked_cells += 1
                continue
            deficit = np.clip(expected_evt - cand_env[b, e0c:e1c], 0, None)
            if deficit.size == 0:
                continue
            depth = float(deficit.max())
            if depth <= 0.5:  # below measurement noise
                continue
            above = deficit > max(1.0, 0.5 * depth)
            if float(above.sum() * hop_ms) < min_hole_ms:  # §8.1: too brief to matter
                continue
            depths.append(depth)
            areas.append(float(deficit.sum() * hop_ms / 1000.0))
            durs.append(float(above.sum() * hop_ms))
            # §6b: recovery is measured AFTER the peak deficit, not from the event
            # start — time from the worst frame back below the half-depth threshold.
            peak_i = int(np.argmax(deficit))
            tail = above[peak_i:]
            rec = int(np.argmax(~tail)) if (tail.size and not tail.all()) else int(tail.size)
            recovs.append(float(rec * hop_ms))

    def _worst(vals: list[float], k: int = 3) -> float:
        return float(np.mean(sorted(vals, reverse=True)[:k])) if vals else 0.0

    masked_frac = masked_cells / total_cells if total_cells else 0.0
    # §6b: availability requires cells that were actually EVALUATED — if every cell
    # was masked or silent-gated, the score is UNKNOWN, never a perfect 0.0.
    evaluated_cells = total_cells - masked_cells - silent_cells
    available = evaluated_cells > 0
    details = {"events": len(events), "cells": total_cells,
               "evaluated_cells": evaluated_cells, "masked_cells": masked_cells,
               "silent_cells": silent_cells}
    return [
        obs("event_hole_depth/v2", _worst(depths) if available else None,
            available=available, uncertainty=masked_frac, unit="dB", details=details),
        obs("event_hole_area/v2", _worst(areas) if available else None,
            available=available, unit="dB*s"),
        obs("event_hole_duration/v2", _worst(durs) if available else None,
            available=available, unit="ms"),
        obs("event_hole_recovery/v2", _worst(recovs) if available else None,
            available=available, unit="ms"),
        obs("masked_hole_uncertainty/v2", masked_frac, available=(total_cells > 0),
            details={"masked_cells": masked_cells, "total_cells": total_cells}),
    ]


# ---------------------------------------------------------------------------
# fullness / brightness v2 (§10.4–10.5) on ONE level-matched copy
# ---------------------------------------------------------------------------
def fullness_v2(cand: np.ndarray, ref: np.ndarray, sr: int,
                vocal_frames: np.ndarray | None = None) -> list[dict]:
    g = fit_global_gain_db(cand, ref, sr, exclude_frames=vocal_frames)
    cand_m = apply_gain_db(cand, g)
    ec = dsp.band_envelope_db(cand_m, sr, dsp.PUMP_BANDS)
    er = dsp.band_envelope_db(ref, sr, dsp.PUMP_BANDS)
    T = min(ec.shape[1], er.shape[1])
    deficit = np.clip(er[:, :T] - ec[:, :T], 0, None)
    per_band = deficit.mean(axis=1)
    worst_b = int(np.argmax(per_band))
    return [
        obs("band_deficit_db/v2", float(per_band.mean()), unit="dB",
            uncertainty=float(per_band.std() / np.sqrt(len(per_band))),
            details={"applied_gain_db": round(g, 3),
                     "worst_band_hz": list(dsp.PUMP_BANDS[worst_b])}),
        obs("contiguous_hole_db/v2", float(deficit.max()), unit="dB"),
    ]


def brightness_v2(cand: np.ndarray, ref: np.ndarray, sr: int,
                  vocal_frames: np.ndarray | None = None) -> list[dict]:
    g = fit_global_gain_db(cand, ref, sr, exclude_frames=vocal_frames)
    cand_m = apply_gain_db(cand, g)
    ec = dsp.band_envelope_db(cand_m, sr, dsp.BRIGHT_BANDS)
    er = dsp.band_envelope_db(ref, sr, dsp.BRIGHT_BANDS)
    T = min(ec.shape[1], er.shape[1])
    env_dist = float(np.mean(np.abs(ec[:, :T] - er[:, :T])))
    cc = dsp.spectral_centroid_hz(cand_m, sr)
    cr = dsp.spectral_centroid_hz(ref, sr)
    ratios_c = [dsp.band_energy_ratio(cand_m, sr, lo, hi) for lo, hi in dsp.BRIGHT_BANDS]
    ratios_r = [dsp.band_energy_ratio(ref, sr, lo, hi) for lo, hi in dsp.BRIGHT_BANDS]
    ratio_dev = float(np.mean(np.abs(np.array(ratios_c) - np.array(ratios_r))))
    return [
        obs("erb_envelope_dist_db/v2", env_dist, unit="dB",
            details={"applied_gain_db": round(g, 3)}),
        obs("centroid_deviation/v2", abs(cc - cr) / (cr + _EPS),
            details={"centroid_hz": round(cc, 1), "ref_hz": round(cr, 1)}),
        obs("hf_ratio_deviation/v2", ratio_dev),
    ]


# ---------------------------------------------------------------------------
# stereo / transient v2 (§10.7–10.8)
# ---------------------------------------------------------------------------
def stereo_v2(cand: np.ndarray, ref: np.ndarray) -> list[dict]:
    def _log_width(x):
        r = dsp.stereo_side_ratio(x)
        return float(np.clip(10 * np.log10(r + 1e-6), -60.0, 20.0))

    wc, wr = _log_width(cand), _log_width(ref)
    c2, r2 = dsp.as2d(cand), dsp.as2d(ref)

    def _coh(x):
        if x.shape[1] < 2 or np.std(x[:, 0]) < _EPS or np.std(x[:, 1]) < _EPS:
            return 1.0
        return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])

    return [
        obs("stereo_width_dev_db/v2", abs(wc - wr), unit="dB",
            details={"cand_db": round(wc, 2), "ref_db": round(wr, 2)}),
        obs("interchannel_coherence_dev/v2", abs(_coh(c2) - _coh(r2))),
    ]


def transient_v2(cand: np.ndarray, ref: np.ndarray, sr: int) -> list[dict]:
    """Both directions (§10.8): loss of valid transients AND artificial excess."""
    def _flux(x):
        p, _ = dsp.stft_power(x, sr)
        m = np.sqrt(p)
        f = np.clip(np.diff(m, axis=0), 0, None).sum(axis=1)
        return f / (m.mean() + _EPS)

    fc, fr = _flux(cand), _flux(ref)
    n = min(len(fc), len(fr))
    d = fc[:n] - fr[:n]
    loss = float(np.clip(-d, 0, None).mean())
    excess = float(np.clip(d, 0, None).mean())
    return [obs("transient_loss/v2", loss), obs("transient_excess/v2", excess)]


# ---------------------------------------------------------------------------
# aggregation (§10.10): never a uniform average
# ---------------------------------------------------------------------------
def aggregate(values: list[float], *, n_boot: int = 200, seed: int = 0) -> dict:
    if not values:
        return {"available": False}
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(v, size=len(v), replace=True))
                      for _ in range(n_boot)])
    return {
        "available": True,
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "worst": float(np.max(v)),
        "mean": float(np.mean(v)),
        "boot_lower": float(np.percentile(boots, 5)),
        "boot_upper": float(np.percentile(boots, 95)),
        "n": len(values),
    }
