"""Objective QA metric bank (docs/v2 §4.3).

Each metric returns a dict with a primary scalar plus details. Metrics are
reference-*optional*: when a reference (synthetic ground truth or robust candidate
consensus) is supplied they use it; otherwise they fall back to self / pre-post
trend. The calibration bar is metamorphic monotonicity on injected-defect ladders
(docs/v2 §benchmark): a metric that fails monotonicity is not ready to rank.

All analysis runs on the *raw* candidate (untouched gain); a single global gain
match is applied only for the audition/judge copy (§4.4, :func:`gain_match`).
"""

from __future__ import annotations

import numpy as np

from . import dsp

_EPS = 1e-12


# --- two-copies rule ---------------------------------------------------------
def gain_match(cand: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """One fixed global gain so ``cand`` matches ``reference`` RMS. Audition copy
    only — never used before dynamics/pumping analysis."""
    c, r = dsp.as2d(cand), dsp.as2d(reference)
    rc = np.sqrt(np.mean(c ** 2) + _EPS)
    rr = np.sqrt(np.mean(r ** 2) + _EPS)
    return c * float(rr / rc)


# --- hard technical checks ---------------------------------------------------
def hard_checks(cand: np.ndarray, sr: int, *, expected_sr: int | None = None,
                expected_frames: int | None = None, expected_channels: int | None = None,
                source_peak: float | None = None) -> dict:
    """``source_peak``: peak of the source mixture. Float audio legitimately
    exceeds 1.0 (e.g. mp3 decode overshoot); a candidate inheriting the source's
    own headroom is NOT clipping (GPU-validation finding: 4/5 residuals were
    silently rejected because the mp3 decoded to peak 1.296)."""
    a = dsp.as2d(cand)
    problems: list[str] = []
    if not np.all(np.isfinite(a)):
        problems.append("nan_or_inf")
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    clip_limit = max(1.0 - 1e-6, (source_peak or 0.0) * 1.05)
    if peak >= clip_limit:
        problems.append("clipping")
    if peak <= 1e-5:
        problems.append("unexpected_silence")
    dc = float(np.mean(a))
    if abs(dc) > 1e-2:
        problems.append("excessive_dc")
    if expected_sr is not None and sr != expected_sr:
        problems.append("wrong_sample_rate")
    if expected_frames is not None and a.shape[0] != expected_frames:
        problems.append("wrong_duration")
    if expected_channels is not None and a.shape[1] != expected_channels:
        problems.append("wrong_channel_count")
    return {"ok": not problems, "problems": problems, "peak": peak, "dc": dc}


# --- leakage (higher = more vocal bleed) -------------------------------------
def leakage(inst: np.ndarray, vocal_ref: np.ndarray, sr: int) -> dict:
    """Vocal energy leaking into the instrumental, referenced to the vocal
    *estimate* (not ground truth): envelope correlation + synchronized excess."""
    corr = dsp.envelope_correlation(inst, vocal_ref, sr)
    # Energy of the instrumental during voiced frames vs unvoiced, in dB.
    ev_inst = dsp.frame_rms_db(inst, sr)
    ev_voc = dsp.frame_rms_db(vocal_ref, sr)
    n = min(len(ev_inst), len(ev_voc))
    ev_inst, ev_voc = ev_inst[:n], ev_voc[:n]
    voiced = ev_voc > (np.max(ev_voc) - 20.0)
    if voiced.any() and (~voiced).any():
        sync_excess = float(np.mean(ev_inst[voiced]) - np.mean(ev_inst[~voiced]))
    else:
        sync_excess = 0.0
    score = corr + max(0.0, sync_excess) / 20.0
    return {"leakage": score, "envelope_corr": corr, "sync_excess_db": sync_excess}


# --- fullness / spectral holes (higher = more holes) -------------------------
def fullness(cand: np.ndarray, reference: np.ndarray, sr: int) -> dict:
    """Multiband energy deficit of ``cand`` relative to ``reference`` (ground
    truth or consensus). Reports the mean positive band deficit in dB."""
    ec = dsp.band_envelope_db(cand, sr, dsp.PUMP_BANDS)
    er = dsp.band_envelope_db(reference, sr, dsp.PUMP_BANDS)
    t = min(ec.shape[1], er.shape[1])
    deficit = np.clip(er[:, :t] - ec[:, :t], 0, None)
    worst_band = int(np.argmax(deficit.mean(axis=1)))
    return {
        "holes_db": float(deficit.mean()),
        "worst_band_hz": list(dsp.PUMP_BANDS[worst_band]),
        "max_deficit_db": float(deficit.max()),
    }


# --- pumping / dynamic holes (reference-free; higher = more pumping) ---------
def pumping(cand: np.ndarray, sr: int) -> dict:
    """Vocal-synchronized dynamic dips, reference-free. Per band, the expected
    (undipped) level is a robust high percentile of the band envelope; the dip is
    the positive shortfall below it (docs/v2 §4.3 pumping formula)."""
    env = dsp.band_envelope_db(cand, sr, dsp.PUMP_BANDS)  # (bands, T)
    per_band = []
    for b in range(env.shape[0]):
        e = env[b]
        expected = np.percentile(e, 75)  # undipped level dominates the upper quartile
        dip = np.clip(expected - e, 0, None)
        per_band.append(float(dip[dip > 0].mean()) if np.any(dip > 0) else 0.0)
    worst = int(np.argmax(per_band))
    return {
        "pump_depth_db": float(np.mean(per_band)),
        "worst_band_hz": list(dsp.PUMP_BANDS[worst]),
        "per_band_db": per_band,
    }


# --- brightness / timbre (higher deviation = worse) --------------------------
def brightness_deviation(cand: np.ndarray, reference: np.ndarray, sr: int) -> dict:
    """Faithful-brightness deviation from a reference (never rewards higher
    centroid per se). Combines centroid drift and high-band energy-ratio drift."""
    cc = dsp.spectral_centroid_hz(cand, sr)
    cr = dsp.spectral_centroid_hz(reference, sr)
    centroid_dev = abs(cc - cr) / (cr + _EPS)
    hi_c = dsp.band_energy_ratio(cand, sr, 4000, 20000)
    hi_r = dsp.band_energy_ratio(reference, sr, 4000, 20000)
    highband_dev = abs(hi_c - hi_r)
    return {
        "brightness_deviation": centroid_dev + highband_dev,
        "centroid_hz": cc,
        "centroid_ref_hz": cr,
        "highband_ratio": hi_c,
        "highband_ref_ratio": hi_r,
    }


# --- transients (lower = more smearing) --------------------------------------
def transients(cand: np.ndarray, sr: int) -> dict:
    power, _ = dsp.stft_power(cand, sr)
    mag = np.sqrt(power)
    flux = np.clip(np.diff(mag, axis=0), 0, None).sum(axis=1)
    transient_flux = float(flux.mean() / (mag.mean() + _EPS))
    sig = dsp.mono(cand)
    crest = float(np.max(np.abs(sig)) / (np.sqrt(np.mean(sig ** 2)) + _EPS))
    return {"transient_flux": transient_flux, "crest_factor": crest}


# --- hall / reverberation (higher damage = more truncation) ------------------
def hall(cand: np.ndarray, reference: np.ndarray, sr: int, activity: np.ndarray) -> dict:
    """Post-offset tail-energy deficit relative to the reference. Detects
    candidate-specific tail truncation after vocal offsets."""
    a = np.clip(np.asarray(activity, dtype=np.float64), 0, 1) > 0.1
    if a.size < 2:
        return {"hall_damage_db": 0.0, "offsets": 0}
    offsets = np.where((~a[1:]) & (a[:-1]))[0] + 1
    c, r = dsp.mono(cand), dsp.mono(reference)
    win = int(0.30 * sr)
    gap = int(0.02 * sr)
    deficits = []
    for off in offsets:
        s, e = off + gap, min(len(c), len(r), off + gap + win)
        if e - s < gap:
            continue
        ec = 10 * np.log10(np.mean(c[s:e] ** 2) + _EPS)
        er = 10 * np.log10(np.mean(r[s:e] ** 2) + _EPS)
        deficits.append(max(0.0, er - ec))
    return {
        "hall_damage_db": float(np.mean(deficits)) if deficits else 0.0,
        "offsets": int(len(offsets)),
    }


# --- stereo / phase (lower width = more collapse) ----------------------------
def stereo(cand: np.ndarray) -> dict:
    ratio = dsp.stereo_side_ratio(cand)
    a = dsp.as2d(cand)
    if a.shape[1] >= 2 and np.std(a[:, 0]) > _EPS and np.std(a[:, 1]) > _EPS:
        coh = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    else:
        coh = 1.0
    return {"stereo_width": ratio, "interchannel_coherence": coh}


# --- aggregate ---------------------------------------------------------------
def measure_all(cand: np.ndarray, sr: int, *, reference: np.ndarray | None = None,
                vocal_ref: np.ndarray | None = None, activity: np.ndarray | None = None) -> dict:
    """Run every applicable metric. Reference-dependent metrics are skipped when
    their reference is absent."""
    out: dict = {
        "hard_checks": hard_checks(cand, sr),
        "pumping": pumping(cand, sr),
        "transients": transients(cand, sr),
        "stereo": stereo(cand),
    }
    if vocal_ref is not None:
        out["leakage"] = leakage(cand, vocal_ref, sr)
    if reference is not None:
        out["fullness"] = fullness(cand, reference, sr)
        out["brightness"] = brightness_deviation(cand, reference, sr)
        if activity is not None:
            out["hall"] = hall(cand, reference, sr, activity)
    return out
