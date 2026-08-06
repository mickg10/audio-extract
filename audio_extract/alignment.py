"""Delay + polarity + channel alignment (v2.1 WP0 / §5.4).

Residual (``mixture - vocal``) and ensembles are only valid once candidates share a
grid AND are delay/polarity/channel aligned — length truncation is not alignment.
This estimates and applies the correction and reports confidence + residual error so
the gate can refuse a low-confidence alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import as2d, mono

_EPS = 1e-12


@dataclass
class Alignment:
    delay_samples: int      # integer delay of cand vs ref (cand[i] ~ ref[i - delay])
    fractional: float       # residual sub-sample delay in [-0.5, 0.5]
    polarity: int           # +1 or -1
    channel_swap: bool
    confidence: float       # normalized correlation peak in [0, 1]
    residual_db: float      # RMS error after alignment, dB relative to ref


def _rfft_xcorr(ref: np.ndarray, cand: np.ndarray, phat: bool = False) -> np.ndarray:
    n = len(ref) + len(cand) - 1
    nf = 1 << (n - 1).bit_length()
    R = np.fft.rfft(ref, nf)
    C = np.fft.rfft(cand, nf)
    cross = R * np.conj(C)
    if phat:
        cross = cross / (np.abs(cross) + _EPS)
    return np.fft.irfft(cross, nf)


def _peak_delay(ref: np.ndarray, cand: np.ndarray, max_shift: int, phat: bool = False):
    xc = _rfft_xcorr(ref, cand, phat=phat)
    # lags 0..max_shift live at the front; negative lags wrap to the tail.
    pos = xc[: max_shift + 1]
    neg = xc[-max_shift:] if max_shift > 0 else xc[:0]
    vals = np.concatenate([neg, pos])
    lags = np.arange(-max_shift, max_shift + 1)
    absvals = np.abs(vals)
    k = int(np.argmax(absvals))
    delay = int(lags[k])
    polarity = 1 if vals[k] >= 0 else -1
    denom = np.linalg.norm(ref) * np.linalg.norm(cand) + _EPS
    conf = float(absvals[k] / denom)
    # parabolic sub-sample refinement around the peak
    frac = 0.0
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = absvals[k - 1], absvals[k], absvals[k + 1]
        denom2 = (y0 - 2 * y1 + y2)
        if abs(denom2) > _EPS:
            frac = float(0.5 * (y0 - y2) / denom2)
    return delay, frac, polarity, conf


def _frac_shift(x: np.ndarray, frac: float) -> np.ndarray:
    """Sub-sample shift via FFT phase ramp (per channel)."""
    if abs(frac) < 1e-6:
        return x
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n)
    ramp = np.exp(-2j * np.pi * freqs * frac)
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = np.fft.irfft(np.fft.rfft(x[:, ch]) * ramp, n)
    return out


def detect_channel_swap(ref: np.ndarray, cand: np.ndarray) -> bool:
    r, c = as2d(ref), as2d(cand)
    if r.shape[1] < 2 or c.shape[1] < 2:
        return False
    n = min(len(r), len(c))

    def corr(a, b):
        return abs(float(np.dot(a[:n], b[:n])))
    straight = corr(r[:, 0], c[:, 0]) + corr(r[:, 1], c[:, 1])
    swapped = corr(r[:, 0], c[:, 1]) + corr(r[:, 1], c[:, 0])
    return swapped > straight * 1.05


def estimate_alignment(ref: np.ndarray, cand: np.ndarray, *, max_shift: int = 4096,
                       use_phat: bool = True) -> Alignment:
    ref2, cand2 = as2d(ref), as2d(cand)
    swap = detect_channel_swap(ref2, cand2)
    cand_probe = cand2[:, ::-1] if swap else cand2
    rm, cm = mono(ref2), mono(cand_probe)
    ms = min(max_shift, len(rm) - 1, len(cm) - 1)
    delay, frac, polarity, conf = _peak_delay(rm, cm, ms, phat=use_phat)
    aligned = apply_alignment(cand2, Alignment(delay, frac, polarity, swap, conf, 0.0),
                              target_len=ref2.shape[0])
    n = min(len(ref2), len(aligned))
    err = np.sqrt(np.mean((ref2[:n] - aligned[:n]) ** 2) + _EPS)
    ref_rms = np.sqrt(np.mean(ref2[:n] ** 2) + _EPS)
    residual_db = 20 * np.log10(err / ref_rms + _EPS)
    return Alignment(delay, frac, polarity, swap, conf, float(residual_db))


def apply_alignment(cand: np.ndarray, al: Alignment, target_len: int) -> np.ndarray:
    x = as2d(cand).astype(np.float64)
    if al.channel_swap and x.shape[1] >= 2:
        x = x[:, ::-1].copy()
    x = x * al.polarity
    # aligned[i] = cand[i - (delay+frac)]; remove the estimated lag to land on ref
    d = -al.delay_samples
    if d > 0:
        x = x[d:]
    elif d < 0:
        x = np.vstack([np.zeros((-d, x.shape[1])), x])
    if abs(al.fractional) > 1e-6:
        x = _frac_shift(x, al.fractional)
    if x.shape[0] < target_len:
        x = np.vstack([x, np.zeros((target_len - x.shape[0], x.shape[1]))])
    return x[:target_len]
