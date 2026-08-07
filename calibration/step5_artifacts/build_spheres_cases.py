#!/usr/bin/env python3
"""Step 5 measured-hall + stereo case builder (runs on tt-quietbox2).

Constructs LINEAR-EXACT with/without-voice pairs for two NEW independent works:
  * Spheres-Mozart      (The Spheres dataset Mozart multitrack orchestra)
  * Spheres-Tchaikovsky (The Spheres dataset Tchaikovsky multitrack orchestra)

RENDERING RECIPE (documented, exact by construction):
  orchestra_dry = mono downmix of the SUM of every Spheres instrument stem over a
                  fixed 90 s window (48 kHz -> 44.1 kHz).
  voice         = a borrowed anechoic operatic soprano (Aalto Mozart soprano stem,
                  mono, 90 s max-energy singing window), scaled so the voice carries
                  ~12% of mix energy (matches real-opera balance).
  For each condition we apply the SAME measured RIR (one Spheres source seat) to
  BOTH voice and orchestra, so the exact target A stays exact:
      target A          = H(orchestra)
      mix M             = H(orchestra) + H(voice) = H(orchestra + voice)
      voice_ref         = H(voice)
  Conditions per work:
    dry    : H = identity (dual-mono, no reverb) -- the anechoic control.
    hall   : H = convolution with Spheres RIR source_Vln_1, receiver ch 0 (dual-mono,
             MEASURED reverb).
    stereo : H = convolution with the SAME RIR, receiver channels {L,R} (a spaced
             stereo receiver pair) -> genuine measured-hall stereo image.
  A shared peak gain (mix -> -3 dBFS) is applied identically to mix/orch/voice.
  Output: 44.1 kHz stereo float32 WAV + meta.json.

Because H is applied identically to voice and orchestra, mix - target - voice_ref = 0
to float precision (linear_exact), independent of the RIR.
"""
import os, io, json, glob, zipfile
import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly, fftconvolve

SR = 44100
SPH_SR = 48000
WINDOW_S = 90.0
PEAK_DBFS = -3.0
PEAK_LIN = 10 ** (PEAK_DBFS / 20.0)
VOC_FRAC_TARGET = 0.12
_EPS = 1e-12

SPHERES = os.path.expanduser("~/datasets/spheres")
ZIP = os.path.join(SPHERES, "TheSpheresDataset-StereoMix.zip")
RIR_DIR = os.path.join(SPHERES, "RIRs_ext", "TheSpheresDataset-RIRs", "npy")
AALTO = os.path.expanduser("~/datasets/aalto/mozart")
OUT = os.path.expanduser("~/step5_cases")
os.makedirs(OUT, exist_ok=True)

RIR_SOURCE = "source_Vln_1"        # front first-violin seat (documented choice)
RIR_RX_MONO = 0
RIR_RX_STEREO = (0, 12)            # spaced receiver pair for a wide stereo image

# fixed mid-piece windows (seconds) where the orchestra plays tutti
WORK_OFFSET_S = {"mozart": 500.0, "tchaikovsky": 400.0}


def resample_to_sr(x, sr_from, sr_to=SR):
    if sr_from == sr_to:
        return x
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, int(sr_from) // g)


def load_aalto_voice(win_s=WINDOW_S):
    mp3s = sorted(glob.glob(os.path.join(AALTO, "*.mp3")))
    sol = [f for f in mp3s if "sopr" in os.path.basename(f).lower()]
    assert sol, f"no soprano stems in {AALTO}"
    acc = None
    for f in sol:
        x, sr = sf.read(f, dtype="float64", always_2d=True)
        m = resample_to_sr(x.mean(axis=1), sr, SR)
        acc = m if acc is None else (acc[:min(len(acc), len(m))] + m[:min(len(acc), len(m))])
    # max-energy singing window
    W = int(win_s * SR); fr = SR
    if len(acc) > W:
        env = np.array([np.sum(acc[i:i + fr] ** 2) for i in range(0, len(acc) - fr, fr)])
        cs = np.cumsum(np.concatenate([[0], env])); wf = W // fr
        best = int(np.argmax([cs[j + wf] - cs[j] for j in range(len(env) - wf)]))
        acc = acc[best * fr: best * fr + W]
    else:
        acc = np.pad(acc, (0, max(0, W - len(acc))))[:W]
    return acc, [os.path.basename(f) for f in sol]


def load_spheres_orchestra(song, win_s=WINDOW_S):
    """Sum every instrument stem over the fixed window, mono, at 44.1 kHz."""
    z = zipfile.ZipFile(ZIP)
    stems = sorted(n for n in z.namelist()
                   if n.endswith(".flac") and f"/{song.capitalize()}/" in n)
    assert stems, f"no stems for {song}"
    off = int(WORK_OFFSET_S[song.lower()] * SPH_SR)
    nwin = int(win_s * SPH_SR)
    acc48 = np.zeros(nwin, dtype=np.float64)
    used = []
    for n in stems:
        b = z.read(n)
        f = sf.SoundFile(io.BytesIO(b))
        if f.frames <= off + 1:
            continue
        f.seek(off)
        seg = f.read(min(nwin, f.frames - off), dtype="float64", always_2d=True)
        m = seg.mean(axis=1)
        acc48[:len(m)] += m
        used.append(os.path.basename(n)[:-5])
    orch = resample_to_sr(acc48, SPH_SR, SR)[: int(win_s * SR)]
    return orch, used


def load_rir():
    a = np.load(os.path.join(RIR_DIR, RIR_SOURCE + ".npy"))  # (23 receivers, taps) @48k
    assert a.ndim == 2, a.shape
    rir_mono = resample_to_sr(a[RIR_RX_MONO], SPH_SR, SR)
    rir_L = resample_to_sr(a[RIR_RX_STEREO[0]], SPH_SR, SR)
    rir_R = resample_to_sr(a[RIR_RX_STEREO[1]], SPH_SR, SR)
    return rir_mono, rir_L, rir_R, a.shape


def conv(x, ir):
    return fftconvolve(x, ir)[:len(x)]


def to_stereo_f32(L, R):
    n = min(len(L), len(R))
    return np.column_stack([L[:n], R[:n]]).astype(np.float32)


def write_case(work, cond, orch_LR, voice_LR, meta):
    """orch_LR/voice_LR are (L,R) mono tuples already spatialised. Shared peak gain."""
    oL, oR = orch_LR; vL, vR = voice_LR
    n = min(len(oL), len(oR), len(vL), len(vR))
    oL, oR, vL, vR = oL[:n], oR[:n], vL[:n], vR[:n]
    mL, mR = oL + vL, oR + vR
    g = PEAK_LIN / (max(np.max(np.abs(mL)), np.max(np.abs(mR))) + _EPS)
    case = f"spheres_{work}_{cond}"
    d = os.path.join(OUT, case); os.makedirs(d, exist_ok=True)
    sf.write(os.path.join(d, "mix_with_voice.wav"), to_stereo_f32(mL * g, mR * g), SR, subtype="FLOAT")
    sf.write(os.path.join(d, "orchestra_only.wav"), to_stereo_f32(oL * g, oR * g), SR, subtype="FLOAT")
    sf.write(os.path.join(d, "voice_ref.wav"), to_stereo_f32(vL * g, vR * g), SR, subtype="FLOAT")
    residL = mL - oL - vL
    lin_db = 10 * np.log10((np.mean(residL ** 2) + _EPS) / (np.mean(mL ** 2) + _EPS))
    voc_frac = float(np.mean(0.5 * (vL ** 2 + vR ** 2)) / (np.mean(0.5 * (mL ** 2 + mR ** 2)) + _EPS))
    side = 0.5 * (mL - mR); mid = 0.5 * (mL + mR)
    width = float(np.mean(side ** 2) / (np.mean(mid ** 2) + _EPS))
    meta = dict(meta)
    meta.update({"case": case, "work": f"spheres_{work}", "condition": cond, "sr": SR,
                 "duration_s": round(n / SR, 3),
                 "gain_applied_db": round(float(20 * np.log10(g + _EPS)), 3),
                 "linear_exactness_db": round(float(lin_db), 1),
                 "voice_energy_frac_of_mix": round(voc_frac, 4),
                 "mix_stereo_side_mid_ratio": round(width, 6),
                 "peak_dbfs_target": PEAK_DBFS,
                 "files": ["mix_with_voice.wav", "orchestra_only.wav", "voice_ref.wav"]})
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{case}] dur={meta['duration_s']}s lin_exact={meta['linear_exactness_db']}dB "
          f"voc_frac={voc_frac:.3f} width={width:.5f} -> {d}", flush=True)
    return meta


def build_work(song):
    work = song.lower()
    print(f"\n=== BUILD spheres_{work} ===", flush=True)
    voice, sopr_files = load_aalto_voice()
    orch, used = load_spheres_orchestra(song)
    n = min(len(voice), len(orch)); voice, orch = voice[:n], orch[:n]
    # scale voice to target energy fraction (pre-spatialisation; H preserves the ratio)
    o_pow = float(np.mean(orch ** 2)); v_pow = float(np.mean(voice ** 2)) + _EPS
    v_target = VOC_FRAC_TARGET * o_pow / (1.0 - VOC_FRAC_TARGET)
    voice = voice * np.sqrt(v_target / v_pow)
    rir_mono, rir_L, rir_R, rir_shape = load_rir()
    base = {"source": f"The Spheres Dataset ({song}) orchestra + borrowed Aalto soprano",
            "license": "The Spheres Dataset (measured RIRs + multitrack); Aalto soprano "
                       "(Aalto Univ. Acoustics Lab, academic research)",
            "citation": "The Spheres Dataset 2025 (measured RIRs + orchestral multitrack); "
                        "Aalto anechoic orchestra (Patynen/Pulkki/Lokki 2008)",
            "pair_integrity": "linear_exact", "evaluation_task": "soloist_vs_rest",
            "anechoic_source": True, "n_orch_stems": len(used), "orch_stems": used,
            "soprano_files": sopr_files, "window_start_s": WORK_OFFSET_S[work],
            "rir_source_seat": RIR_SOURCE, "rir_shape_receivers_taps": list(rir_shape),
            "voc_frac_target": VOC_FRAC_TARGET,
            "construction": ("sum all Spheres instrument stems (mono) over a fixed 90s window; "
                             "borrowed anechoic soprano scaled to ~12% mix energy; SAME measured "
                             "RIR applied identically to voice and orchestra per condition; "
                             "shared -3dBFS gain; 44.1k f32 stereo")}
    metas = []
    # dry (dual-mono, no reverb)
    m = dict(base); m["rir_applied"] = "none (anechoic dual-mono)"
    metas.append(write_case(work, "dry", (orch, orch), (voice, voice), m))
    # hall mono (dual-mono, measured reverb, receiver 0)
    oh = conv(orch, rir_mono); vh = conv(voice, rir_mono)
    m = dict(base); m["rir_applied"] = f"{RIR_SOURCE} receiver {RIR_RX_MONO} (mono/dual-mono)"
    metas.append(write_case(work, "hall", (oh, oh), (vh, vh), m))
    # hall stereo (measured reverb, spaced receiver pair)
    oL, oR = conv(orch, rir_L), conv(orch, rir_R)
    vL, vR = conv(voice, rir_L), conv(voice, rir_R)
    m = dict(base); m["rir_applied"] = f"{RIR_SOURCE} receivers {RIR_RX_STEREO} (stereo pair)"
    metas.append(write_case(work, "stereo", (oL, oR), (vL, vR), m))
    return metas


if __name__ == "__main__":
    allm = []
    for song in ("Mozart", "Tchaikovsky"):
        allm += build_work(song)
    json.dump(allm, open(os.path.join(OUT, "spheres_manifest.json"), "w"), indent=2)
    print(f"\nWROTE {len(allm)} cases to {OUT}", flush=True)
