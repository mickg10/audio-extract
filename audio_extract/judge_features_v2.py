"""Source-aware reference-free features (v2) for the trained voice-removal judge.

Supersedes the ``judge_features.py`` v0 baseline (kept as one diagnostic input
branch). Both reviewers independently flagged the v0 contract's defects; v2 fixes
them:

* **Required ``(M, Y, D, task)``** — the mixture M and removed signal D=M−Y are
  REQUIRED, never silently zero. A candidate waveform alone cannot tell a naturally
  sparse orchestra from a separator-gouged hole; D reveals what was assigned to the
  singer (oracle §1, gpt56 QC).
* **Absolute band energy, not fraction-of-total** — v0's ``low_end_retention`` /
  ``hf_reduction`` / ``hf_side_energy_ratio`` divided each signal's *fraction* of
  total energy, so a broadband change inverts their meaning. v2 uses absolute band
  energy in dB, referenced to M.
* **Frame distributions, not track means** — a whole-track mean hides a catastrophic
  100–500 ms phrase, so event-relevant features report ``{p50, p90, max, cvar90}``.
* **Mid/side stereo retained** — stereo damage needs L/R, not a mono collapse.
* **Explicit availability masks** — a missing optional input is UNAVAILABLE, never a
  clean-looking 0.0.

Everything is deterministic. Names describe what is measured (e.g. ``voiced_prob``,
not ``singing_voice_prob`` — pYIN fires on violins too).
"""

from __future__ import annotations

import numpy as np

from . import dsp

_EPS = 1e-12

# Ordered scalar contract. Distribution features expand to <name>__{p50,p90,max,cvar90}.
SCALAR_FEATURES = (
    "broadband_reduction_db",          # 20log(rms M / rms Y): total energy removed
    "low_band_abs_reduction_db",       # ABSOLUTE 20-250 Hz band energy, M − Y (dB)
    "hf_band_abs_reduction_db",        # ABSOLUTE 6-20 kHz band energy, M − Y (dB)
    "ms_width_change_db",              # side/mid of Y vs M (dB) — stereo image change
    "interchannel_coherence_change",   # |coh(Y) − coh(M)|
    "voiced_prob_on_Y",                # pYIN voiced-prob on Y (diagnostic; not voice-specific)
    "residual_voice_alignment",        # energy of Y aligned to D's harmonic activity (leakage proxy)
    "ensemble_disagreement",           # RMS spread across member vocals (UNCERTAINTY, not quality)
)
DIST_FEATURES = (
    "accomp_continuity_deficit_db",    # Y accompaniment DIP during D-active frames vs Y's own
                                       # inactive-frame baseline (a true gouge is a discontinuity
                                       # in Y — NOT the expected M>Y gap from removing the voice)
    "hf_musical_noise_excess_db",      # Y HF isolated-peak energy above M's HF (per frame)
    "spectral_flux_excess",            # flux(Y) − flux(M), rectified, per frame
)

# Frozen task vocabulary — one-hot encoded INTO the model input so the judge can tell
# "remove all voices" from "remove one soloist, retain chorus" (oracle P0 #1). Unknown → all-zero.
TASK_VOCAB = ("soloist_vs_rest", "all_vocals", "soloist_retain_chorus", "duet_one_role")

# Features whose `available` flag can be False. An availability bit is appended to the
# model input so a genuine 0.0 is distinguishable from an unavailable stereo/pYIN/ensemble input.
MASKABLE_FEATURES = (
    "ms_width_change_db", "interchannel_coherence_change",
    "voiced_prob_on_Y", "residual_voice_alignment", "ensemble_disagreement",
    "accomp_continuity_deficit_db", "hf_musical_noise_excess_db", "spectral_flux_excess",
)


def _bands():
    return getattr(dsp, "PUMP_BANDS", [(20, 250), (250, 2000), (2000, 6000), (6000, 20000)])


def _abs_band_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Absolute band energy in dBFS-ish (NOT a fraction of total energy)."""
    m = dsp.mono(dsp.as2d(x))
    X = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    f = np.fft.rfftfreq(len(m), 1.0 / sr)
    sel = (f >= lo) & (f < hi)
    e = float(np.sum(X[sel] ** 2)) / (len(m) + _EPS)
    return 10.0 * np.log10(e + _EPS)


def _dist(vals: np.ndarray) -> dict:
    """{p50,p90,max,cvar90} — never a mean; CVaR90 = mean of worst 10% frames."""
    if vals.size == 0:
        return {"p50": 0.0, "p90": 0.0, "max": 0.0, "cvar90": 0.0}
    v = np.sort(vals)
    k = max(1, int(np.ceil(0.10 * len(v))))
    return {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "max": float(v[-1]), "cvar90": float(np.mean(v[-k:]))}


def _framewise_band_sum_db(x: np.ndarray, sr: int, lo: float, hi: float,
                           win_ms: float = 40.0, hop_ms: float = 20.0) -> np.ndarray:
    p, _ = dsp.stft_power(x, sr, win_ms, hop_ms)
    freqs = np.linspace(0, sr / 2, p.shape[1])
    sel = (freqs >= lo) & (freqs < hi)
    band = p[:, sel].sum(axis=1)
    return 10.0 * np.log10(band + _EPS)


def source_aware_features(mixture: np.ndarray, candidate: np.ndarray, sr: int, *,
                          removed: np.ndarray | None = None, task: str = "soloist_vs_rest",
                          member_vocals: list[np.ndarray] | None = None) -> tuple[dict, dict]:
    """Return ``(features, available)``. ``mixture`` (M) and ``candidate`` (Y) are
    REQUIRED; ``removed`` (D) defaults to M−Y. ``available`` maps each feature to a
    bool — a missing optional input is unavailable, never a fabricated zero."""
    M = dsp.as2d(mixture)
    Y = dsp.as2d(candidate)
    n = min(len(M), len(Y))
    if n == 0:
        raise ValueError("source_aware_features requires non-empty M and Y")
    M, Y = M[:n], Y[:n]
    D = (dsp.as2d(removed)[:n] if removed is not None else M - Y)

    feats: dict = {}
    avail: dict = {}

    # --- absolute reduction features (M-referenced, absolute band energy) --------
    em = float(np.sqrt(np.mean(M ** 2))) + _EPS
    ey = float(np.sqrt(np.mean(Y ** 2))) + _EPS
    feats["broadband_reduction_db"] = 20.0 * np.log10(em / ey)
    feats["low_band_abs_reduction_db"] = _abs_band_db(M, sr, 20, 250) - _abs_band_db(Y, sr, 20, 250)
    hi_hi = min(20000.0, sr / 2 - 1)
    feats["hf_band_abs_reduction_db"] = _abs_band_db(M, sr, 6000, hi_hi) - _abs_band_db(Y, sr, 6000, hi_hi)
    for k in ("broadband_reduction_db", "low_band_abs_reduction_db", "hf_band_abs_reduction_db"):
        avail[k] = True

    # --- stereo (mid/side, referenced to M) -------------------------------------
    if M.shape[1] >= 2 and Y.shape[1] >= 2:
        def ms_ratio(z):
            mid = 0.5 * (z[:, 0] + z[:, 1]); side = 0.5 * (z[:, 0] - z[:, 1])
            return (np.sqrt(np.mean(side ** 2)) + _EPS) / (np.sqrt(np.mean(mid ** 2)) + _EPS)
        def coh(z):
            if np.std(z[:, 0]) < _EPS or np.std(z[:, 1]) < _EPS:
                return 1.0
            return float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])
        feats["ms_width_change_db"] = 20.0 * np.log10(ms_ratio(Y) / ms_ratio(M))
        feats["interchannel_coherence_change"] = abs(coh(Y) - coh(M))
        avail["ms_width_change_db"] = avail["interchannel_coherence_change"] = True
    else:
        feats["ms_width_change_db"] = 0.0; feats["interchannel_coherence_change"] = 0.0
        avail["ms_width_change_db"] = avail["interchannel_coherence_change"] = False  # mono: unavailable

    # --- retained voice / leakage (conditioned on D, not Y alone) ---------------
    try:
        import librosa
        d_mono = dsp.mono(D)
        _, dv, _ = librosa.pyin(d_mono, sr=sr, fmin=110, fmax=1000,
                                frame_length=2048, hop_length=max(1, sr // 100))
        d_active = np.nan_to_num(dv, nan=0.0) > 0.5
        _, yv, yp = librosa.pyin(dsp.mono(Y), sr=sr, fmin=110, fmax=1000,
                                 frame_length=2048, hop_length=max(1, sr // 100))
        feats["voiced_prob_on_Y"] = float(np.mean(np.nan_to_num(yp, nan=0.0)))
        # leakage: Y voiced-prob specifically WHERE the removed signal was voiced
        nA = min(len(d_active), len(yp))
        yp_a = np.nan_to_num(yp[:nA], nan=0.0)[d_active[:nA]]
        feats["residual_voice_alignment"] = float(np.mean(yp_a)) if yp_a.size else 0.0
        avail["voiced_prob_on_Y"] = avail["residual_voice_alignment"] = True
    except Exception:
        feats["voiced_prob_on_Y"] = 0.0; feats["residual_voice_alignment"] = 0.0
        avail["voiced_prob_on_Y"] = avail["residual_voice_alignment"] = False

    # --- ensemble disagreement (UNCERTAINTY, optional) --------------------------
    if member_vocals and len(member_vocals) >= 2:
        m0 = min(len(dsp.as2d(v)) for v in member_vocals)
        stack = np.stack([dsp.mono(dsp.as2d(v)[:m0]) for v in member_vocals], axis=0)
        feats["ensemble_disagreement"] = float(np.mean(np.std(stack, axis=0)) /
                                               (np.sqrt(np.mean(dsp.mono(Y) ** 2)) + _EPS))
        avail["ensemble_disagreement"] = True
    else:
        feats["ensemble_disagreement"] = 0.0
        avail["ensemble_disagreement"] = False

    # --- distribution features (frame quantiles/CVaR, NOT means) ----------------
    # accompaniment-continuity deficit: does Y's accompaniment band DIP during D-active
    # frames RELATIVE TO Y's own level when the removed source is INACTIVE? A genuine hole is
    # a discontinuity in Y itself; the old (M − Y) form flagged correct voice removal as a hole
    # (for a perfect candidate M is louder than the accompaniment *because* the voice was removed).
    dfr = _framewise_band_sum_db(D, sr, 250, 6000)
    yfr = _framewise_band_sum_db(Y, sr, 250, 6000)
    T = min(len(dfr), len(yfr))
    if T:
        d_hot = dfr[:T] > np.percentile(dfr[:T], 60)
        if d_hot.any() and (~d_hot).any():
            baseline = float(np.median(yfr[:T][~d_hot]))        # Y accompaniment when singer inactive
            deficit = np.clip(baseline - yfr[:T][d_hot], 0, None)  # only DIPS below that baseline
        else:
            deficit = np.array([])
    else:
        deficit = np.array([])
    feats["accomp_continuity_deficit_db"] = _dist(deficit)
    avail["accomp_continuity_deficit_db"] = deficit.size > 0

    # HF musical-noise excess (Y HF above M HF, per frame)
    yhf = _framewise_band_sum_db(Y, sr, 6000, hi_hi); mhf = _framewise_band_sum_db(M, sr, 6000, hi_hi)
    Th = min(len(yhf), len(mhf))
    feats["hf_musical_noise_excess_db"] = _dist(np.clip(yhf[:Th] - mhf[:Th], 0, None))
    avail["hf_musical_noise_excess_db"] = Th > 0

    # spectral flux excess (Y over M)
    py, _ = dsp.stft_power(Y, sr); pm, _ = dsp.stft_power(M, sr)
    fy = np.clip(np.diff(np.sqrt(py), axis=0), 0, None).sum(axis=1)
    fm = np.clip(np.diff(np.sqrt(pm), axis=0), 0, None).sum(axis=1)
    Tf = min(len(fy), len(fm))
    ref = pm.mean() ** 0.5 + _EPS
    feats["spectral_flux_excess"] = _dist(np.clip(fy[:Tf] - fm[:Tf], 0, None) / ref)
    avail["spectral_flux_excess"] = Tf > 0

    feats["_task"] = task
    return feats, avail


def feature_vector(feats: dict, avail: dict | None = None) -> np.ndarray:
    """Flatten to the ordered numeric contract: scalars + dist{p50,p90,max,cvar90}
    + frozen task one-hot + availability bits. ``avail`` (the second return of
    ``source_aware_features``) SHOULD be passed so a missing optional input is flagged
    rather than looking like a genuine 0.0; when omitted, all maskable inputs are
    assumed available (back-compat)."""
    out = [float(feats.get(k, 0.0)) for k in SCALAR_FEATURES]
    for k in DIST_FEATURES:
        d = feats.get(k, {})
        out += [float(d.get(q, 0.0)) for q in ("p50", "p90", "max", "cvar90")]
    task = feats.get("_task", "soloist_vs_rest")
    out += [1.0 if task == t else 0.0 for t in TASK_VOCAB]          # frozen task one-hot
    av = avail or {}
    out += [1.0 if av.get(k, True) else 0.0 for k in MASKABLE_FEATURES]  # availability bits
    return np.asarray(out, dtype=np.float64)


def feature_names() -> list[str]:
    names = list(SCALAR_FEATURES)
    for k in DIST_FEATURES:
        names += [f"{k}__{q}" for q in ("p50", "p90", "max", "cvar90")]
    names += [f"task__{t}" for t in TASK_VOCAB]
    names += [f"avail__{k}" for k in MASKABLE_FEATURES]
    return names
