"""Hole-filler common utilities: paths, alignment (lifted from vad_train/code,
adapted), alpha fit, resample, band energies, voiced-region labels.

Data semantics (verified 2026-08-18 against vad_train inspection + manifest v4):
  <pid>_orchestra.wav = true orchestra backing (TARGET source)
  <pid>_voice.wav     = FULL MIX (voice + SAME backing performance), common-trimmed
                        to equal frames but offset by a global lag (tens of ms)
                        plus slow drift (~15 samples / 4 min). GCC-PHAT z 25-34.
Construct: piecewise-align voice-mix onto the orchestra timeline, fit scalar
backing gain alpha, then pair = (separator(mix_aligned) -> alpha * orchestra).
"""
import json
import os

import numpy as np
import soundfile as sf
import soxr
from numpy.fft import rfft, irfft, rfftfreq

ROOT = "/mnt/bigdisk/mickg/holefill"
MAN = "/home/mickg/cantolopera_benchmark/benchmark_manifest.v4.json"
SR48 = 48000
SR = 44100                     # processing rate (separator-native); resampled ONCE with soxr HQ
PAIRS44 = f"{ROOT}/work/pairs44"

# alignment params (same values as vad_train/code/extract_residuals.py)
MAXLAG_GLOBAL = 5 * SR48
MAXLAG_BLOCK = 2400            # +-50 ms
BLK = 10 * SR48
CF = int(0.25 * SR48)
Z_MIN = 7.0

# eval bands (Hz) — matches the causal-analysis band vocabulary
BANDS = {
    "low": (60.0, 300.0),
    "mid": (300.0, 1200.0),
    "himid": (1200.0, 4000.0),
    "presence": (4000.0, 8000.0),
    "air": (8000.0, 16000.0),
}

NP2 = lambda n: 1 << (int(n) - 1).bit_length()


def load_manifest():
    return json.load(open(MAN))


def is_senza(pid):
    return ("senza" in pid) or ("completa" in pid)


def eligible_pairs(splits=None):
    m = load_manifest()
    out = []
    for p in m["pairs"]:
        if is_senza(p["pair_id"]):
            continue
        if p["alignment_review"]:
            continue
        if splits is not None and p["split"] not in splits:
            continue
        out.append(p)
    return out


def mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def gcc_phat(a, b, max_lag, sr=SR48, band=(100.0, 6000.0)):
    """peak lag L means b(t) ~ a(t+L): to put b on a's timeline delay b by L."""
    L = NP2(len(a) + len(b))
    A = rfft(a, L)
    B = rfft(b, L)
    X = A * np.conj(B)
    f = rfftfreq(L, 1.0 / sr)
    W = ((f >= band[0]) & (f <= band[1])).astype(np.float64)
    xc = irfft(W * X / (np.abs(X) + 1e-12), L)
    cand = np.concatenate([xc[L - max_lag:], xc[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    k = int(np.argmax(cand))
    z = float((cand[k] - cand.mean()) / (cand.std() + 1e-12))
    return int(lags[k]), z


def shift_delay(x, lag, n):
    """return y with y(t) = x(t - lag), length n (x may be 2D [n, ch])."""
    out = np.zeros((n,) + x.shape[1:], dtype=x.dtype)
    if lag >= 0:
        m = min(n - lag, len(x))
        if m > 0:
            out[lag: lag + m] = x[:m]
    else:
        m = min(n, len(x) + lag)
        if m > 0:
            out[:m] = x[-lag: -lag + m]
    return out


def piecewise_align(v, o_mono, v_mono, glag, n):
    """Per-block lag refinement around glag; return aligned v (2D), lag table, z table."""
    starts = list(range(0, n, BLK))
    lags, zs = [], []
    vg = shift_delay(v_mono, glag, n)
    for s in starts:
        e = min(s + BLK, n)
        if e - s < SR48:
            lags.append(np.nan)
            zs.append(0.0)
            continue
        a = o_mono[s:e]
        b = vg[s:e]
        l2, z = gcc_phat(a, b, MAXLAG_BLOCK)
        lags.append(float(l2))
        zs.append(z)
    lags = np.array(lags, dtype=np.float64)
    zs = np.array(zs)
    good = (zs >= Z_MIN) & np.isfinite(lags)
    if good.sum() == 0:
        lags_f = np.zeros(len(lags))
    else:
        gi = np.where(good)[0]
        lags_f = np.interp(np.arange(len(lags)), gi, lags[gi])
        if len(lags_f) >= 3:
            lags_f = np.array([np.median(lags_f[max(0, i - 1): i + 2]) for i in range(len(lags_f))])
    total = glag + np.round(lags_f).astype(int)
    out = np.zeros((n,) + v.shape[1:], dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    for bi, s in enumerate(starts):
        e = min(s + BLK, n)
        s2 = max(0, s - CF)
        e2 = min(n, e + CF)
        seg = shift_delay(v, int(total[bi]), n)[s2:e2]
        w = np.ones(e2 - s2, dtype=np.float32)
        r = np.arange(1, CF + 1, dtype=np.float32) / (CF + 1)
        if s2 > 0:
            w[:CF] = r
        if e2 < n:
            w[-CF:] = r[::-1]
        out[s2:e2] += seg * w[:, None] if v.ndim == 2 else seg * w
        wsum[s2:e2] += w
    wsum[wsum == 0] = 1.0
    out = out / (wsum[:, None] if v.ndim == 2 else wsum)
    return out.astype(np.float32), total.tolist(), [round(float(z), 1) for z in zs]


def fit_alpha(v_al_mono, o_mono):
    """Scalar backing gain: least squares <v,o>/<o,o> (voice ~ uncorrelated with o)."""
    num = float(np.dot(v_al_mono, o_mono))
    den = float(np.dot(o_mono, o_mono)) + 1e-20
    return num / den


def band_alphas(v_al_mono, o_mono, sr=SR48, nseg=4096):
    """Per-band backing transfer Re<V O*>/<|O|^2> via Welch cross-spectra (QA only)."""
    from scipy.signal import csd, welch
    f, Pvo = csd(v_al_mono, o_mono, fs=sr, nperseg=nseg)
    f, Poo = welch(o_mono, fs=sr, nperseg=nseg)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[name] = float(np.real(Pvo[m]).sum() / (Poo[m].sum() + 1e-20))
    return out


def resample_44(x48):
    """48k -> 44.1k, soxr HQ, float32. x48 [n, ch]."""
    y = soxr.resample(x48.astype(np.float32), SR48, SR, quality="HQ")
    return np.ascontiguousarray(y.astype(np.float32))


def frame_band_energy(x, sr=SR, hop=441, win=1764, nfft=4096):
    """Per-hop per-band mean power. x mono. Returns dict band -> [nframes] linear power,
    plus broadband. 10 ms hop, 40 ms window."""
    n = len(x)
    nfr = max(1, n // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(nfr)[:, None]
    idx = np.minimum(idx, n - 1)
    w = np.hanning(win).astype(np.float32)
    frames = x[idx] * w[None, :]
    X = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    norm = (w.sum() ** 2) / 2.0
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[name] = X[:, m].sum(axis=1) / norm
    m = (f >= 60.0) & (f < 16000.0)
    out["broad"] = X[:, m].sum(axis=1) / norm
    return out


def db(x):
    return 10.0 * np.log10(np.asarray(x) + 1e-12)


def voiced_mask_44(res_mono44, tgt_mono44, hop=441):
    """Voice-over-orchestra ratio labels on the 44.1k / 10ms grid.
    res = mix_aligned - alpha*orchestra (rough isolated voice + eps).
    Returns (voiced bool, orchonly bool, vor dB)."""
    Er = frame_band_energy(res_mono44, hop=hop)["broad"]
    Eo = frame_band_energy(tgt_mono44, hop=hop)["broad"]
    vor = db(Er) - db(Eo)
    lo = db(Eo)
    voiced = (vor > -18.0) & (db(Er) > -70.0)
    orchonly = (vor < -30.0) & (lo > -70.0)
    return voiced, orchonly, vor


def synth_hall_ir(rng, sr=SR):
    """Synthetic hall IR: direct + exp-decay noise tail, RT60 1.0-2.2 s,
    predelay 8-20 ms, direct-to-reverb 6-12 dB, stereo-decorrelated tail."""
    rt60 = rng.uniform(1.0, 2.2)
    pre = int(rng.uniform(0.008, 0.020) * sr)
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    decay = 10.0 ** (-3.0 * t / rt60)
    tail = rng.standard_normal((n, 2)).astype(np.float32) * decay[:, None].astype(np.float32)
    # gentle LP tilt on the tail (air dies faster in halls)
    from scipy.signal import lfilter
    b, a = [0.6], [1.0, -0.4]
    tail = lfilter(b, a, tail, axis=0).astype(np.float32)
    drr_db = rng.uniform(6.0, 12.0)
    tail_e = np.sqrt(float(np.mean(tail ** 2)) * n) + 1e-12
    tail *= 10.0 ** (-drr_db / 20.0) / tail_e
    ir = np.zeros((pre + n, 2), dtype=np.float32)
    ir[0, :] = 1.0
    ir[pre:, :] += tail
    return ir


def conv_ir(x, ir):
    """x [n,2] float32, ir [m,2]; fft conv per channel, same length as x."""
    from scipy.signal import fftconvolve
    y = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 0.99:
        y *= 0.99 / peak
    return y.astype(np.float32)


def write_f32(path, x, sr=SR):
    sf.write(path, x.astype(np.float32), sr, subtype="FLOAT")


def pair_dir(pid, variant=None):
    d = f"{PAIRS44}/{pid}" + (f"__{variant}" if variant else "")
    return d
