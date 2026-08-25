import json
import os
import numpy as np

ROOT = "/mnt/bigdisk/mickg/vad_train"
MAN = "/home/mickg/cantolopera_benchmark/benchmark_manifest.v4.json"
SR = 48000
HOP = 480           # 10 ms
WIN = 1200          # 25 ms
NFFT = 2048
NMELS = 128
BAND = (200.0, 8000.0)
VOICE_DB = -25.0    # corrected VOR above -> VOICE
ORCH_DB = -45.0     # corrected VOR below -> ORCHESTRA-ONLY
SIL_DB = -60.0      # frame band-energy dB (rel FS) below -> near-silence
LEAK_KAPPA = 2.0    # headroom factor on the per-pair cancellation floor
LEAK_MIN_DB = -40.0
LEAK_MAX_DB = -10.0
LEAK_DROP_DB = -12.0  # pairs with floor above this are unusable
EPS = 1e-12

# label codes
LAB_ORCH = 0
LAB_VOICE = 1
LAB_HARD = -1       # in VOR dead zone, excluded from loss
LAB_SIL = -2        # both stems near-silent, excluded from loss


def load_manifest():
    return json.load(open(MAN))


def is_senza(pid):
    return ("senza" in pid) or ("completa" in pid)


def load_pairs(splits=None, include_senza=False, include_review=False):
    m = load_manifest()
    out = []
    for p in m["pairs"]:
        pid = p["pair_id"]
        if not include_senza and is_senza(pid):
            continue
        if not include_review and p["alignment_review"]:
            continue
        if splits is not None and p["split"] not in splits:
            continue
        out.append(p)
    return out


def res_path(pid):
    return f"{ROOT}/features/residual48/{pid}.res.wav"


def qa_path(pid):
    return f"{ROOT}/features/qa/{pid}.json"


def energy_path(pid):
    return f"{ROOT}/features/energy/{pid}.npz"


_WIN_CACHE = {}


def _hann(win):
    if win not in _WIN_CACHE:
        _WIN_CACHE[win] = np.hanning(win).astype(np.float32)
    return _WIN_CACHE[win]


def n_frames_for(nsamp, hop=HOP):
    return nsamp // hop + 1


def frame_band_energy(x, sr=SR, hop=HOP, win=WIN, nfft=NFFT, band=BAND):
    """Per-hop mean power in [band] Hz. Center-padded framing; frame count = len//hop + 1.
    Returns float64 array of linear power (rel FS, arbitrary consistent scale)."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    nfr = n_frames_for(n, hop)
    pad = win // 2
    xp = np.pad(x, (pad, pad + win))
    idx = np.arange(win)[None, :] + hop * np.arange(nfr)[:, None]
    frames = xp[idx] * _hann(win)[None, :]
    X = np.fft.rfft(frames, nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    m = (f >= band[0]) & (f <= band[1])
    w = _hann(win)
    # normalize so a full-scale in-band sine reads ~ -3 dB (0.5 mean power)
    norm = (np.sum(w) ** 2) / 2.0
    return (np.abs(X[:, m]) ** 2).sum(axis=1) / norm


def db(x):
    return 10.0 * np.log10(np.asarray(x) + EPS)


def estimate_leak_db(Eo, Er, win=50):
    """Per-pair cancellation floor: p05 over `win`-frame windows of Er/Eo (dB),
    restricted to orchestra-active windows. Eo/Er linear frame band energies."""
    nw = len(Eo) // win
    if nw < 4:
        return LEAK_MAX_DB
    eo = Eo[: nw * win].reshape(nw, win).mean(1)
    er = Er[: nw * win].reshape(nw, win).mean(1)
    act = db(eo) > -50.0
    if act.sum() < 4:
        return LEAK_MAX_DB
    ratio = db(er[act]) - db(eo[act])
    return float(np.clip(np.percentile(ratio, 5), LEAK_MIN_DB, LEAK_MAX_DB))


def labels_from_energies(Eo, Er, leak_db):
    """Leakage-aware labels. Eo, Er: linear per-frame band energies on the same
    timeline (post-IR, post-alpha as applicable). leak_db: per-pair floor of the
    residual extraction (residual contains ~leak*orchestra even with no voice).
    E_v = max(Er - kappa*leak*Eo, 0); VOR = E_v/Eo."""
    leak = 10.0 ** (leak_db / 10.0)
    lo = db(Eo)
    lr = db(Er)
    Ev = np.maximum(Er - LEAK_KAPPA * leak * Eo, 0.0)
    vor = db(Ev) - db(Eo)
    n = len(vor)
    y = np.full(n, LAB_HARD, dtype=np.int8)
    y[vor > VOICE_DB] = LAB_VOICE
    y[(vor < ORCH_DB) & (lo > SIL_DB)] = LAB_ORCH
    # near-silent orchestra: voice over silence counts as voice, else ignore
    qo = lo <= SIL_DB
    y[qo] = LAB_SIL
    y[qo & (lr > np.maximum(lo + 10.0, SIL_DB - 5.0))] = LAB_VOICE
    return y, vor, lo, lr


def vor_labels(orch, res, leak_db=LEAK_MIN_DB, sr=SR):
    """waveform wrapper around labels_from_energies."""
    Eo = frame_band_energy(orch, sr)
    Er = frame_band_energy(res, sr)
    return labels_from_energies(Eo, Er, leak_db)


def mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x
