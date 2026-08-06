"""Track-conditioned exact-reference challenge engine (v2.1 WP4 / §9).

The key to removing the runtime human judge: manufacture cases where the true
accompaniment IS available, on the same recording.

* **A — counterfactual remix**: ``M = A + g·H(V)`` where ``A`` is a real no-vocal
  control passage from the track and ``H`` applies estimated hall/placement to a
  known clean vocal. The exact target stays ``A`` *by construction* (linear add),
  even when ``H`` is imperfect — H only shapes the interference, never the target.
* **B — no-vocal orchestral-theft assay**: any "vocal" a model extracts from clean
  orchestra is a direct false positive (material the residual route would remove).
* **C — vocal-only bleed assay**: voice appearing in an instrumental output.
* **D/E — causal interventions**: a good accompaniment estimator changes little
  when only vocal energy is added (invariance) and passes an added orchestral
  probe through (equivariance).

Everything is deterministic (seeded); challenge identity is the SHA-256 of the
canonical recipe. Exact-target errors supply the labels for selector calibration —
no single metric becomes "the answer".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import canon, dsp
from .alignment import apply_alignment, estimate_alignment

_EPS = 1e-12
_CHALLENGE_DOMAIN = b"audio-extract-challenge-v1\x00"

PROBE_KINDS = ("high_string_cluster", "brass_onset", "low_mid_pad",
               "wide_hall_tail", "center_transient", "side_hf_air")


def challenge_id(recipe: dict) -> str:
    return "sha256:" + hashlib.sha256(_CHALLENGE_DOMAIN + canon.canonicalize(recipe)).hexdigest()


# ---------------------------------------------------------------------------
# hall estimation + vocal placement (H)
# ---------------------------------------------------------------------------
def estimate_hall_ir(sr: int, *, rt60_s: float = 1.2, length_s: float | None = None,
                     seed: int = 0) -> np.ndarray:
    """Synthesize a hall impulse response with the given RT60 (exp-decay noise +
    direct path). ``rt60_s`` should come from the track's vocal-offset decay slope
    when available; this stays deterministic via ``seed``."""
    rng = np.random.default_rng(seed)
    n = int(sr * (length_s if length_s is not None else min(2.0, rt60_s * 1.5)))
    t = np.arange(n) / sr
    tau = rt60_s / 6.91  # RT60 = time to -60 dB; exp(-t/tau) hits 1e-3 at t=RT60
    ir = np.exp(-t / tau) * rng.standard_normal(n)
    ir *= 0.25 / (np.max(np.abs(ir)) + _EPS)
    ir[0] = 1.0  # direct path
    return ir


def estimate_rt60_from_tail(tail: np.ndarray, sr: int) -> float:
    """Crude RT60 estimate from a decaying tail: fit the dB-energy slope."""
    e = dsp.frame_rms_db(tail, sr, 40.0, 10.0)
    if len(e) < 5:
        return 1.0
    x = np.arange(len(e)) * 0.010
    slope = float(np.polyfit(x, e, 1)[0])  # dB per second (negative)
    if slope >= -1.0:
        return 2.0
    return float(np.clip(-60.0 / slope, 0.2, 4.0))


def place_vocal(vocal: np.ndarray, sr: int, *, ir: np.ndarray, gain_db: float,
                pan: float = 0.0) -> np.ndarray:
    """H(V): convolve the clean vocal with the hall IR, apply gain and (near-)center
    placement. ``pan`` in [-1, 1], 0 = center."""
    v = dsp.mono(vocal)
    n = len(v)
    nfft = 1 << (n + len(ir) - 2).bit_length()
    wet = np.fft.irfft(np.fft.rfft(v, nfft) * np.fft.rfft(ir, nfft), nfft)[:n]
    peak = np.max(np.abs(wet)) + _EPS
    wet = wet / peak * (np.max(np.abs(v)) + _EPS)  # keep the source's scale
    g = 10 ** (gain_db / 20.0)
    left = np.cos((pan + 1) * np.pi / 4)
    right = np.sin((pan + 1) * np.pi / 4)
    return np.column_stack([wet * g * left * np.sqrt(2), wet * g * right * np.sqrt(2)])


# ---------------------------------------------------------------------------
# Challenge A — counterfactual remix cases
# ---------------------------------------------------------------------------
@dataclass
class RemixCase:
    challenge_id: str
    mixture: np.ndarray          # A + g·H(V)  — run candidates on this
    target: np.ndarray           # A           — the EXACT accompaniment target
    injected_vocal: np.ndarray   # g·H(V)      — known interference (diagnostics)
    recipe: dict
    klass: dict = field(default_factory=dict)


def build_track_remix_cases(controls: list[tuple[str, np.ndarray]],
                            vocals: list[tuple[str, np.ndarray]], sr: int, *,
                            levels_db: tuple[float, ...] = (-12.0, -6.0, 0.0, 3.0),
                            rt60_s: float = 1.2, seed: int = 0,
                            max_cases: int = 12) -> list[RemixCase]:
    """Build a diverse subset (not the full Cartesian product) of remix cases.
    ``controls`` are (id, no-vocal accompaniment array); ``vocals`` are
    (id, clean vocal array). Linear construction ⇒ target is exact."""
    ir = estimate_hall_ir(sr, rt60_s=rt60_s, seed=seed)
    cases: list[RemixCase] = []
    k = 0
    for ci, (control_id, a) in enumerate(controls):
        a2 = dsp.as2d(a)
        for vi, (vocal_id, v) in enumerate(vocals):
            # rotate levels so each (control, vocal) pair samples different levels
            for li in range(len(levels_db)):
                if len(cases) >= max_cases:
                    return cases
                g_db = levels_db[(ci + vi + li) % len(levels_db)]
                placed = place_vocal(v, sr, ir=ir, gain_db=g_db,
                                     pan=0.0 if k % 2 == 0 else 0.15)
                n = min(len(a2), len(placed))
                if n < sr // 2:
                    continue
                mixture = a2[:n] + placed[:n]
                recipe = {
                    "schema": "audio-extract/challenge/track-remix/v1",
                    "control_id": control_id, "vocal_id": vocal_id,
                    "gain_micro_db": int(round(g_db * 1000)),
                    "rt60_ms": int(round(rt60_s * 1000)),
                    "pan_milli": 0 if k % 2 == 0 else 150,
                    "seed": seed, "sample_rate_hz": sr, "frames": int(n),
                }
                cases.append(RemixCase(
                    challenge_id=challenge_id(recipe),
                    mixture=mixture, target=a2[:n], injected_vocal=placed[:n],
                    recipe=recipe,
                    klass={"vocal_level_db": g_db,
                           "pan": "center" if k % 2 == 0 else "off_center"},
                ))
                k += 1
                if li >= 1:   # at most 2 levels per (control, vocal) pair -> diversity
                    break
    return cases


# ---------------------------------------------------------------------------
# Challenges B/C — theft + bleed assays
# ---------------------------------------------------------------------------
def theft_ratio(extracted: np.ndarray, source: np.ndarray) -> float:
    """||extracted|| / ||source|| — energy a model claimed from material that
    contains none of its target stem."""
    e = float(np.sqrt(np.mean(dsp.as2d(extracted) ** 2)))
    s = float(np.sqrt(np.mean(dsp.as2d(source) ** 2)))
    return e / (s + _EPS)


def run_no_vocal_theft_assay(vocal_estimator, controls: list[tuple[str, np.ndarray]],
                             sr: int) -> list[dict]:
    """Challenge B: run a vocal estimator on clean orchestra. Any output is theft.
    Reports broadband + per-band + mid/side ratios."""
    out = []
    for control_id, a in controls:
        v_hat = dsp.as2d(vocal_estimator(a))
        a2 = dsp.as2d(a)
        n = min(len(v_hat), len(a2))
        v_hat, a2 = v_hat[:n], a2[:n]
        env_v = dsp.band_envelope_db(v_hat, sr, dsp.PUMP_BANDS)
        env_a = dsp.band_envelope_db(a2, sr, dsp.PUMP_BANDS)
        per_band = [round(float(np.mean(np.clip(ev - ea + 60.0, 0, None))), 3)
                    for ev, ea in zip(env_v, env_a)]
        mid_v = 0.5 * (v_hat[:, 0] + v_hat[:, -1]); side_v = 0.5 * (v_hat[:, 0] - v_hat[:, -1])
        mid_a = 0.5 * (a2[:, 0] + a2[:, -1]); side_a = 0.5 * (a2[:, 0] - a2[:, -1])
        out.append({
            "control_id": control_id,
            "theft_broadband": round(theft_ratio(v_hat, a2), 5),
            "theft_mid": round(theft_ratio(mid_v, mid_a), 5),
            "theft_side": round(theft_ratio(side_v, side_a), 5),
            "band_excess_db": per_band,
        })
    return out


def run_vocal_only_bleed_assay(instrumental_estimator, vocals: list[tuple[str, np.ndarray]],
                               sr: int) -> list[dict]:
    """Challenge C: run an instrumental estimator on clean vocals; output is bleed."""
    del sr
    return [{"vocal_id": vid, "bleed_broadband": round(theft_ratio(instrumental_estimator(v), v), 5)}
            for vid, v in vocals]


# ---------------------------------------------------------------------------
# Challenges D/E — causal interventions
# ---------------------------------------------------------------------------
def make_orchestral_probe(kind: str, sr: int, dur_s: float = 1.0, seed: int = 0,
                          level: float = 0.05) -> np.ndarray:
    """Deterministic orchestral probes (§9.8)."""
    if kind not in PROBE_KINDS:
        raise ValueError(f"unknown probe kind {kind!r}; allowed: {PROBE_KINDS}")
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    if kind == "high_string_cluster":
        sig = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                  for f in (2093.0, 2637.0, 3136.0, 3951.0))
        st = np.column_stack([sig, sig])
    elif kind == "brass_onset":
        saw = 2 * ((220.0 * t) % 1.0) - 1.0
        env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.6)
        sig = saw * env
        st = np.column_stack([sig, sig])
    elif kind == "low_mid_pad":
        sig = sum(np.sin(2 * np.pi * f * t) for f in (110.0, 165.0, 220.0))
        st = np.column_stack([sig, sig])
    elif kind == "wide_hall_tail":
        l = rng.standard_normal(n) * np.exp(-t / 0.5)
        r = rng.standard_normal(n) * np.exp(-t / 0.5)
        st = np.column_stack([l, r])
    elif kind == "center_transient":
        sig = np.zeros(n)
        for s in range(0, n, sr // 4):
            sig[s: s + 32] = np.hanning(64)[:32]
        st = np.column_stack([sig, sig])
    else:  # side_hf_air
        noise = rng.standard_normal(n)
        hf = noise - np.convolve(noise, np.ones(9) / 9, mode="same")  # crude HF
        st = np.column_stack([hf, -hf])
    peak = np.max(np.abs(st)) + _EPS
    return st / peak * level


def vocal_intervention_invariance(candidate_fn, mixture: np.ndarray,
                                  delta_vocal: np.ndarray) -> float:
    """Challenge D: relative change of the accompaniment estimate when only vocal
    energy is added. Lower = better (invariant)."""
    m = dsp.as2d(mixture)
    n = min(len(m), len(dsp.as2d(delta_vocal)))
    base = dsp.as2d(candidate_fn(m[:n]))
    pert = dsp.as2d(candidate_fn(m[:n] + dsp.as2d(delta_vocal)[:n]))
    nn = min(len(base), len(pert))
    num = float(np.sqrt(np.mean((pert[:nn] - base[:nn]) ** 2)))
    den = float(np.sqrt(np.mean(base[:nn] ** 2))) + _EPS
    return num / den


def orchestra_intervention_error(candidate_fn, mixture: np.ndarray,
                                 delta_orch: np.ndarray) -> float:
    """Challenge E: how far the estimator's response to an added orchestral probe is
    from passing it through exactly. 0 = perfect pass-through."""
    m = dsp.as2d(mixture)
    d = dsp.as2d(delta_orch)
    n = min(len(m), len(d))
    base = dsp.as2d(candidate_fn(m[:n]))
    pert = dsp.as2d(candidate_fn(m[:n] + d[:n]))
    nn = min(len(base), len(pert), n)
    resp = pert[:nn] - base[:nn]
    num = float(np.sqrt(np.mean((resp - d[:nn]) ** 2)))
    den = float(np.sqrt(np.mean(d[:nn] ** 2))) + _EPS
    return num / den


# ---------------------------------------------------------------------------
# Exact-reference metrics (§9.9)
# ---------------------------------------------------------------------------
def si_sdr_db(est: np.ndarray, target: np.ndarray) -> float:
    e = dsp.mono(est)
    t = dsp.mono(target)
    n = min(len(e), len(t))
    e, t = e[:n], t[:n]
    alpha = float(np.dot(e, t) / (np.dot(t, t) + _EPS))
    proj = alpha * t
    noise = e - proj
    return float(10 * np.log10((np.sum(proj ** 2) + _EPS) / (np.sum(noise ** 2) + _EPS)))


def multires_stft_distance(est: np.ndarray, target: np.ndarray, sr: int) -> float:
    total = 0.0
    for win_ms in (25.0, 80.0):
        pe, _ = dsp.stft_power(est, sr, win_ms, win_ms / 4)
        pt, _ = dsp.stft_power(target, sr, win_ms, win_ms / 4)
        n = min(pe.shape[0], pt.shape[0])
        le = np.log10(pe[:n] + _EPS)
        lt = np.log10(pt[:n] + _EPS)
        total += float(np.mean(np.abs(le - lt)))
    return total / 2.0


def exact_reference_error(est: np.ndarray, target: np.ndarray, sr: int) -> dict:
    """Error of an estimated accompaniment against the EXACT known target, after
    fixed alignment. Supplies calibration labels; no single number is the answer."""
    al = estimate_alignment(target, est)
    est_a = apply_alignment(dsp.as2d(est), al, target_len=dsp.as2d(target).shape[0])
    tgt = dsp.as2d(target)
    env_e = dsp.band_envelope_db(est_a, sr, dsp.PUMP_BANDS)
    env_t = dsp.band_envelope_db(tgt, sr, dsp.PUMP_BANDS)
    nT = min(env_e.shape[1], env_t.shape[1])
    band_err = float(np.mean(np.abs(env_e[:, :nT] - env_t[:, :nT])))
    width_err = abs(dsp.stereo_side_ratio(est_a) - dsp.stereo_side_ratio(tgt))
    return {
        "si_sdr_db": round(si_sdr_db(est_a, tgt), 3),
        "stft_distance": round(multires_stft_distance(est_a, tgt, sr), 5),
        "band_envelope_err_db": round(band_err, 3),
        "stereo_width_err": round(float(width_err), 5),
        "alignment": {"delay": al.delay_samples, "confidence": round(al.confidence, 4),
                      "residual_db": round(al.residual_db, 2)},
    }
