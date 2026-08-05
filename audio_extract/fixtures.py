"""Synthetic opera-like benchmark (docs/v2 §"Synthetic opera-like benchmark").

Real mixes lack ground-truth accompaniment, so metric calibration runs on
synthetic stems with known defects. ``linear_mix`` gives ``M = V + A`` where the
ground-truth accompaniment is exactly ``A``. The injected-defect helpers build
severity ladders whose monotonicity the metric bank must respect.

Deterministic: all randomness comes from a seeded ``np.random.Generator``.
"""

from __future__ import annotations

import numpy as np

from .dsp import as2d


def _fft_convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Linear convolution via rFFT, truncated to ``len(x)`` (fast reverb)."""
    n = len(x) + len(ir) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)
    return y[: len(x)]


def _stereo(mono_sig: np.ndarray, spread: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Make a 2-channel signal; ``spread`` in [0,1] decorrelates the channels."""
    if spread <= 0 or rng is None:
        return np.column_stack([mono_sig, mono_sig])
    side = rng.standard_normal(len(mono_sig))
    side = side / (np.max(np.abs(side)) + 1e-9)
    return np.column_stack([mono_sig + spread * 0.5 * side, mono_sig - spread * 0.5 * side])


def synth_vocal(sr: int = 44100, dur_s: float = 6.0, f0: float = 523.25,
                phrases=None, seed: int = 0):
    """Return ``(audio[n,2], activity[n])``. A vibrato'd harmonic 'voice' gated
    into phrases; ``activity`` is a 0..1 per-sample gate. Default phrases scale to
    ``dur_s`` so there are always two well-separated vocal events."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    if phrases is None:
        phrases = [(0.17 * dur_s, 0.42 * dur_s), (0.58 * dur_s, 0.87 * dur_s)]
    t = np.arange(n) / sr
    vib = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * t)  # 5.5 Hz vibrato
    sig = np.zeros(n)
    for k, amp in enumerate([1.0, 0.5, 0.33, 0.22, 0.14], start=1):
        sig += amp * np.sin(2 * np.pi * f0 * k * vib * t)
    sig /= np.max(np.abs(sig)) + 1e-9

    activity = np.zeros(n)
    fade = int(0.03 * sr)
    for (a, b) in phrases:
        i0, i1 = max(0, int(a * sr)), min(n, int(b * sr))
        if i1 - i0 <= 2 * fade:
            continue
        gate = np.ones(i1 - i0)
        gate[:fade] = np.linspace(0, 1, fade)
        gate[-fade:] = np.linspace(1, 0, fade)
        activity[i0:i1] = gate
    sig = sig * activity * 0.6
    sig = sig + 0.01 * rng.standard_normal(n) * activity  # faint breath, seed-dependent
    return _stereo(sig, spread=0.0), activity


def synth_orchestra(sr: int = 44100, dur_s: float = 6.0, seed: int = 1) -> np.ndarray:
    """A continuous, wide, reverberant 'orchestra' bed with harmonic partials +
    noise + an exponential hall tail."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for base in (110.0, 146.83, 220.0, 329.63):  # A2, D3, A3, E4-ish
        detune = 1.0 + rng.uniform(-0.002, 0.002)
        for k, amp in enumerate([1.0, 0.5, 0.3, 0.2], start=1):
            sig += amp * np.sin(2 * np.pi * base * k * detune * t + rng.uniform(0, 2 * np.pi))
    sig += 0.15 * rng.standard_normal(n)  # bowed-noise bed
    sig /= np.max(np.abs(sig)) + 1e-9

    wide = _stereo(sig, spread=0.6, rng=rng)

    # Exponential-decay reverb tail so hall metrics have something to truncate.
    ir_len = int(0.5 * sr)
    tau = 0.12 * sr
    ir = np.exp(-np.arange(ir_len) / tau) * rng.standard_normal(ir_len)
    ir[0] = 1.0
    out = np.empty_like(wide)
    for ch in range(2):
        rev = _fft_convolve(wide[:, ch], ir)
        out[:, ch] = 0.8 * wide[:, ch] + 0.2 * rev / (np.max(np.abs(rev)) + 1e-9)
    return out * 0.5


def linear_mix(vocal: np.ndarray, accompaniment: np.ndarray) -> np.ndarray:
    """M = V + A."""
    v, a = as2d(vocal), as2d(accompaniment)
    n = min(len(v), len(a))
    return v[:n] + a[:n]


# --- injected defect ladders -------------------------------------------------
def add_vocal_bleed(inst: np.ndarray, vocal: np.ndarray, gain_db: float) -> np.ndarray:
    g = 10 ** (gain_db / 20.0)
    a, v = as2d(inst).copy(), as2d(vocal)
    n = min(len(a), len(v))
    a[:n] += g * v[:n]
    return a


def inject_pumping(inst: np.ndarray, activity: np.ndarray, depth_db: float) -> np.ndarray:
    """Multiply the accompaniment down by ``depth_db`` while the voice is active."""
    a = as2d(inst).copy()
    n = min(len(a), len(activity))
    floor = 10 ** (-abs(depth_db) / 20.0)
    gain = 1.0 - (1.0 - floor) * np.clip(activity[:n], 0, 1)
    a[:n] *= gain[:, None]
    return a


def high_shelf(x: np.ndarray, sr: int, gain_db: float, fc: float = 4000.0) -> np.ndarray:
    """FFT-domain high shelf; negative ``gain_db`` cuts brightness above ``fc``."""
    a = as2d(x).copy()
    g = 10 ** (gain_db / 20.0)
    n = a.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    ramp = np.ones_like(freqs)
    above = freqs >= fc
    ramp[above] = g
    for ch in range(a.shape[1]):
        X = np.fft.rfft(a[:, ch])
        a[:, ch] = np.fft.irfft(X * ramp, n=n)
    return a


def spectral_hole(x: np.ndarray, sr: int, lo: float, hi: float, depth_db: float) -> np.ndarray:
    a = as2d(x).copy()
    g = 10 ** (-abs(depth_db) / 20.0)
    n = a.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = np.ones_like(freqs)
    mask[(freqs >= lo) & (freqs < hi)] = g
    for ch in range(a.shape[1]):
        X = np.fft.rfft(a[:, ch])
        a[:, ch] = np.fft.irfft(X * mask, n=n)
    return a


def truncate_hall(x: np.ndarray, activity: np.ndarray, sr: int, cut_ms: float) -> np.ndarray:
    """Zero the reverb tail for ``cut_ms`` after each vocal offset."""
    a = as2d(x).copy()
    n = min(len(a), len(activity))
    act = np.clip(activity[:n], 0, 1) > 0.1
    offsets = np.where((~act[1:]) & (act[:-1]))[0] + 1
    cut = int(sr * cut_ms / 1000.0)
    for off in offsets:
        end = min(n, off + cut)
        ramp = np.linspace(1, 0, end - off)
        a[off:end] *= ramp[:, None]
    return a


def narrow_stereo(x: np.ndarray, amount: float) -> np.ndarray:
    """Collapse toward mono; ``amount`` in [0,1] (1 == full mono)."""
    a = as2d(x).copy()
    if a.shape[1] < 2:
        return a
    mid = 0.5 * (a[:, 0] + a[:, 1])
    side = 0.5 * (a[:, 0] - a[:, 1]) * (1.0 - np.clip(amount, 0, 1))
    return np.column_stack([mid + side, mid - side])


def smear_transients(x: np.ndarray, sr: int, blur_ms: float) -> np.ndarray:
    """Moving-average temporal blur — reduces spectral flux and crest factor."""
    a = as2d(x).copy()
    k = max(1, int(sr * blur_ms / 1000.0))
    if k <= 1:
        return a
    kernel = np.ones(k) / k
    for ch in range(a.shape[1]):
        a[:, ch] = np.convolve(a[:, ch], kernel, mode="same")
    return a
