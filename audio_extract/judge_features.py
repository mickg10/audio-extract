"""Reference-free features for a TRAINED voice-removal quality judge (mission:
train a judge, then use it to pick parameters).

These are computable on ANY delivered instrumental with no ground truth at
inference — the inputs the trained judge maps to the exact-reference quality
labels (event-hole depth, vocal SI-SIR/leakage, artifact). The label side
(``judge_labels``) generates the *training targets* from self-remix cases where
the exact accompaniment IS known; this module is the *feature* side.

Design-independent by construction: whatever model class the judge ends up being
(regressor / pairwise ranker / per-defect heads), it consumes this fixed,
documented feature vector. Everything here is deterministic.
"""

from __future__ import annotations

import numpy as np

from . import dsp

_EPS = 1e-12

# Stable, ordered feature names — the judge's input contract. Keep append-only.
FEATURE_NAMES = (
    "residual_voice_harmonic_salience",   # harmonic energy near a tracked F0, in the instrumental
    "singing_voice_prob_on_instrumental", # pyin voiced-probability run ON the instrumental (should be ~0)
    "hf_musical_noise",                   # isolated high-band spectral peaks (birdies)
    "spectral_flux_excess",               # transient/flux beyond a smooth baseline
    "hf_modulation_rate",                 # fast HF envelope modulation (median-switch artifacts)
    "ensemble_member_disagreement",       # RMS spread across member vocal estimates (if provided)
    "hf_side_energy_ratio",               # high-freq mid/side (stereo image damage proxy)
    "low_end_retention",                  # low-band energy vs mixture (hollowing proxy)
    "broadband_reduction_db",             # how much energy was removed vs the mixture
    "hf_reduction_db",                    # high-band reduction (over-subtraction / dullness)
)


def _voiced_prob(mono: np.ndarray, sr: int, hop: int) -> np.ndarray:
    try:
        import librosa
        _, _, vprob = librosa.pyin(mono, sr=sr, fmin=110.0, fmax=1000.0,
                                   frame_length=2048, hop_length=hop)
        return np.nan_to_num(vprob, nan=0.0)
    except Exception:
        return np.zeros(max(1, len(mono) // hop))


def _harmonic_salience(mono: np.ndarray, sr: int, hop: int) -> float:
    """Fraction of spectral energy concentrated in harmonic peaks near a tracked
    F0 — a residual singer leaves salient harmonic combs; orchestra is broadband."""
    try:
        import librosa
        f0, voiced, _ = librosa.pyin(mono, sr=sr, fmin=110.0, fmax=1000.0,
                                     frame_length=2048, hop_length=hop)
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced = np.nan_to_num(voiced, nan=0.0).astype(bool)
        if not voiced.any():
            return 0.0
        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop))
        freqs = np.linspace(0, sr / 2, S.shape[0])
        T = min(S.shape[1], len(f0))
        sal = []
        for t in range(T):
            if not voiced[t] or f0[t] <= 0:
                continue
            total = S[:, t].sum() + _EPS
            harm = 0.0
            for k in range(1, 8):
                fc = f0[t] * k
                if fc >= sr / 2:
                    break
                band = np.abs(freqs - fc) <= (f0[t] * 0.06)
                harm += S[band, t].sum()
            sal.append(harm / total)
        return float(np.mean(sal)) if sal else 0.0
    except Exception:
        return 0.0


def reference_free_features(instrumental: np.ndarray, sr: int, *,
                            mixture: np.ndarray | None = None,
                            member_vocals: list[np.ndarray] | None = None) -> dict:
    """The reference-free feature vector. ``instrumental`` is required; ``mixture``
    (the original with-voice) enables the relative reduction/retention features;
    ``member_vocals`` (the per-model vocal estimates) enables the disagreement
    feature. Missing optionals leave their features at a neutral 0.0."""
    inst = dsp.as2d(instrumental)
    mono = dsp.mono(inst)
    hop = max(1, int(sr * 0.010))

    feats = {k: 0.0 for k in FEATURE_NAMES}
    feats["residual_voice_harmonic_salience"] = _harmonic_salience(mono, sr, hop)
    vprob = _voiced_prob(mono, sr, hop)
    feats["singing_voice_prob_on_instrumental"] = float(np.mean(vprob))

    # HF musical noise: isolated peaks in the top band relative to a smoothed floor
    bands_hi = [(6000.0, 12000.0), (12000.0, min(20000.0, sr / 2 - 1))]
    env_hi = dsp.band_envelope_db(inst, sr, bands_hi)
    if env_hi.size:
        flat = env_hi - np.median(env_hi, axis=1, keepdims=True)
        feats["hf_musical_noise"] = float(np.mean(np.clip(flat, 0, None)))

    # spectral flux excess: rectified frame-to-frame magnitude change beyond baseline
    p, _ = dsp.stft_power(inst, sr)
    mag = np.sqrt(p)
    flux = np.clip(np.diff(mag, axis=0), 0, None).sum(axis=1)
    feats["spectral_flux_excess"] = float(np.mean(flux) / (mag.mean() + _EPS))

    # HF modulation rate: fast changes in the HF envelope (median-switch birdies)
    if env_hi.size:
        hf_env = env_hi.mean(axis=0)
        feats["hf_modulation_rate"] = float(np.mean(np.abs(np.diff(hf_env))))

    # HF side energy ratio (stereo image damage proxy)
    if inst.shape[1] >= 2:
        side = 0.5 * (inst[:, 0] - inst[:, -1])
        mid = 0.5 * (inst[:, 0] + inst[:, -1])
        es = dsp.band_energy_ratio(side, sr, 6000.0, min(20000.0, sr / 2 - 1))
        em = dsp.band_energy_ratio(mid, sr, 6000.0, min(20000.0, sr / 2 - 1))
        feats["hf_side_energy_ratio"] = float(es / (em + _EPS))

    # ensemble disagreement (if member vocal estimates supplied)
    if member_vocals and len(member_vocals) >= 2:
        n = min(len(dsp.as2d(v)) for v in member_vocals)
        stack = np.stack([dsp.mono(dsp.as2d(v)[:n]) for v in member_vocals], axis=0)
        spread = np.std(stack, axis=0)
        ref = np.sqrt(np.mean(dsp.mono(inst) ** 2)) + _EPS
        feats["ensemble_member_disagreement"] = float(np.mean(spread) / ref)

    # relative-to-mixture features
    if mixture is not None:
        mix = dsp.as2d(mixture)
        n = min(len(inst), len(mix))
        i2, m2 = inst[:n], mix[:n]
        e_inst = float(np.sqrt(np.mean(i2 ** 2))) + _EPS
        e_mix = float(np.sqrt(np.mean(m2 ** 2))) + _EPS
        feats["broadband_reduction_db"] = float(20 * np.log10(e_mix / e_inst))
        lo_i = dsp.band_energy_ratio(dsp.mono(i2), sr, 20.0, 250.0)
        lo_m = dsp.band_energy_ratio(dsp.mono(m2), sr, 20.0, 250.0)
        feats["low_end_retention"] = float(lo_i / (lo_m + _EPS))
        hi_i = dsp.band_energy_ratio(dsp.mono(i2), sr, 6000.0, min(20000.0, sr / 2 - 1)) + _EPS
        hi_m = dsp.band_energy_ratio(dsp.mono(m2), sr, 6000.0, min(20000.0, sr / 2 - 1)) + _EPS
        feats["hf_reduction_db"] = float(10 * np.log10(hi_m / hi_i))

    return feats


def feature_vector(feats: dict) -> np.ndarray:
    """Ordered numeric vector in FEATURE_NAMES order — the judge's input tensor."""
    return np.array([float(feats.get(k, 0.0)) for k in FEATURE_NAMES], dtype=np.float64)
