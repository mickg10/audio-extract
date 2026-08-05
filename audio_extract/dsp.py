"""Shared DSP primitives for the metric bank and passage miner.

Everything accumulates in float64 (docs/v2 §7.1). Audio arrays are shaped
``(samples, channels)``; helpers downmix to mono where a scalar-per-frame view is
needed. STFT is a plain framed rFFT so results are deterministic and independent
of any library's windowing defaults.
"""

from __future__ import annotations

import numpy as np

# Multiband edges for pumping / dynamic-hole analysis (docs/v2 §4.3).
PUMP_BANDS: list[tuple[float, float]] = [
    (20, 120), (120, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)
]
# Brightness ratio bands (docs/v2 §4.3 "Brightness / timbre").
BRIGHT_BANDS: list[tuple[float, float]] = [(2000, 4000), (4000, 8000), (8000, 12000), (12000, 20000)]

_EPS = 1e-12


def as2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def mono(x: np.ndarray) -> np.ndarray:
    """Downmix to a 1-D float64 signal."""
    a = as2d(x)
    return a.mean(axis=1)


def _next_pow2(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def stft_power(x: np.ndarray, sr: int, win_ms: float = 40.0, hop_ms: float = 10.0):
    """Return ``(power[time, freq], freqs)`` — magnitude-squared, Hann-windowed.

    ``x`` may be multichannel; it is downmixed to mono first.
    """
    sig = mono(x)
    win = max(8, int(round(sr * win_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    n_fft = _next_pow2(win)
    window = np.hanning(win).astype(np.float64)

    if len(sig) < win:
        sig = np.pad(sig, (0, win - len(sig)))
    n_frames = 1 + (len(sig) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = sig[idx] * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return power, freqs


def band_envelope_db(x: np.ndarray, sr: int, bands: list[tuple[float, float]],
                     win_ms: float = 40.0, hop_ms: float = 10.0) -> np.ndarray:
    """Per-band energy envelope over time, in dB. Shape ``(n_bands, time)``."""
    power, freqs = stft_power(x, sr, win_ms, hop_ms)
    out = np.empty((len(bands), power.shape[0]), dtype=np.float64)
    for i, (lo, hi) in enumerate(bands):
        sel = (freqs >= lo) & (freqs < hi)
        e = power[:, sel].sum(axis=1) if sel.any() else np.full(power.shape[0], _EPS)
        out[i] = 10.0 * np.log10(e + _EPS)
    return out


def frame_rms_db(x: np.ndarray, sr: int, win_ms: float = 40.0, hop_ms: float = 10.0) -> np.ndarray:
    """Per-frame broadband RMS in dBFS (mono)."""
    sig = mono(x)
    win = max(8, int(round(sr * win_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if len(sig) < win:
        sig = np.pad(sig, (0, win - len(sig)))
    n_frames = 1 + (len(sig) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = sig[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + _EPS)
    return 20.0 * np.log10(rms + _EPS)


def median_filter1d(v: np.ndarray, k: int) -> np.ndarray:
    """Odd-length sliding median (edge-replicated). Used for robust envelopes."""
    if k <= 1:
        return v.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(vp, k)
    return np.median(win, axis=1)


def spectral_centroid_hz(x: np.ndarray, sr: int) -> float:
    power, freqs = stft_power(x, sr)
    p = power.sum(axis=0)
    total = p.sum()
    if total <= _EPS:
        return 0.0
    return float((freqs * p).sum() / total)


def band_energy_ratio(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Fraction of total spectral energy in ``[lo, hi)``."""
    power, freqs = stft_power(x, sr)
    p = power.sum(axis=0)
    total = p.sum()
    if total <= _EPS:
        return 0.0
    sel = (freqs >= lo) & (freqs < hi)
    return float(p[sel].sum() / total)


def envelope_correlation(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Pearson correlation of two broadband energy envelopes (0..1 clamped)."""
    ea = frame_rms_db(a, sr)
    eb = frame_rms_db(b, sr)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n], eb[:n]
    if n < 2 or np.std(ea) < _EPS or np.std(eb) < _EPS:
        return 0.0
    c = float(np.corrcoef(ea, eb)[0, 1])
    return max(0.0, c)


def stereo_side_ratio(x: np.ndarray) -> float:
    """Side/mid energy ratio. 0 for mono, grows with width."""
    a = as2d(x)
    if a.shape[1] < 2:
        return 0.0
    mid = 0.5 * (a[:, 0] + a[:, 1])
    side = 0.5 * (a[:, 0] - a[:, 1])
    em = float(np.mean(mid ** 2)) + _EPS
    es = float(np.mean(side ** 2))
    return es / em
