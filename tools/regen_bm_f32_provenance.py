"""Voice-gate calibration f32 provenance regeneration + detector per-window recompute + hashing.

Regenerates B_melband_ft and M_maxagg as LOSSLESS f32 (task-required), replays the FROZEN
median-ensemble detector (voice_gate2) per-window on the lossless f32 candidate set, and hashes
every f32 parent + every lossy listening copy. Writes results incrementally to a JSON with a
heartbeat file so a flaky link can monitor. Idempotent-ish: safe to re-run.

Recipes (verbatim from render_ab.py / render_aggressive.py):
  A_champion = mix - median3(MDX23C, MelBand_base, BS)                 (== iter1)
  M_maxagg   = mix - per-sample-per-channel argmax|.| of the 3 vocals  (MelBand base)
  B_melband_ft = mix - median3(MDX23C, MelBand_ft@step600, BS)
Detector (voice_gate2, FROZEN): vmed=median3(MDX,MelBand_base,BS)(cand); 1s win / 0.5s hop;
  program energy we=mean(MIX_win^2); keep we>max(we)*1e-3; rel=10log10(mean(vmed_win^2)/we);
  worst=max(rel over kept); PASS iff worst<=-10. Normalized by ORIGINAL mix.
"""
import os, sys, glob, json, time, hashlib, subprocess, traceback
import numpy as np, soundfile as sf, torch

MODEL_DIR = "/home/mickg/models"
STEM_TMP  = "/mnt/bigdisk/stem_tmp"
CANDDIR   = "/home/mickg/gate_candidates/verdi_slow_voices"
ABDIR     = "/home/mickg/ab_pairs"
MIX_PATH  = "/home/mickg/ae-opera/7_verdi_slow_voices_tobacco_05_14_26_1/source/canonical.f32.wav"
MEL = "vocals_mel_band_roformer.ckpt"; MDX = "MDX23C-8KFFT-InstVoc_HQ.ckpt"; BS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MEL_FT = "/home/mickg/runs/roformer_ft1/melband_ft_step600.ckpt"
SR = 44100; WIN = int(SR); HOP = int(SR // 2); THRESH = -10.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

OUTJSON = f"{CANDDIR}/provenance_regen.json"
HB      = f"{CANDDIR}/regen.heartbeat"
DONE    = f"{CANDDIR}/regen.done"
ERR     = f"{CANDDIR}/regen.error"
os.makedirs(STEM_TMP, exist_ok=True)
RESULT = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": DEV, "stages": {}}


def hb(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        open(HB, "a").write(line + "\n")
    except OSError:
        pass


def flush():
    tmp = OUTJSON + ".tmp"
    json.dump(RESULT, open(tmp, "w"), indent=1)
    os.replace(tmp, OUTJSON)


def pcm_sha256(X):
    return hashlib.sha256(np.ascontiguousarray(X.astype(np.float32)).tobytes()).hexdigest()


def ffmpeg():
    f = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).stdout.strip()
    return f or "ffmpeg"


def decode_any(path):
    """Match voice_gate2.decode: m4a via ffmpeg -ar 44100; wav direct. float64, 2d."""
    if path.endswith(".m4a"):
        w = f"{STEM_TMP}/dec_{os.path.basename(path)}.wav"
        subprocess.run([ffmpeg(), "-y", "-i", path, "-ar", str(SR), w], capture_output=True)
        a, _ = sf.read(w, dtype="float64", always_2d=True)
        try: os.remove(w)
        except OSError: pass
        return a
    a, _ = sf.read(path, dtype="float64", always_2d=True)
    return a


def load_sep(name):
    from audio_separator.separator import Separator
    s = Separator(model_file_dir=MODEL_DIR, output_dir=STEM_TMP, log_level=40)
    s.load_model(model_filename=name)
    return s


def sep_vocals(sep, wav_path):
    before = set(glob.glob(f"{STEM_TMP}/*.wav"))
    sep.separate(wav_path)
    prod = [p for p in glob.glob(f"{STEM_TMP}/*.wav") if p not in before]
    vp = [p for p in prod if "ocal" in os.path.basename(p)]
    v, _ = sf.read(vp[0], dtype="float64", always_2d=True)
    for p in prod:
        try: os.remove(p)
        except OSError: pass
    return v


def median3(a, b, c):
    n = min(len(a), len(b), len(c))
    return np.median(np.stack([a[:n], b[:n], c[:n]], 0), axis=0)


def maxagg3(a, b, c):
    n = min(len(a), len(b), len(c))
    Vst = np.stack([a[:n], b[:n], c[:n]], 0)          # (3,n,2)
    idx = np.argmax(np.abs(Vst), axis=0, keepdims=True)
    return np.take_along_axis(Vst, idx, axis=0)[0]


def windows(n):
    return list(range(0, max(1, n - WIN + 1), HOP))


def detect_perwindow(cand, mix, vmed):
    """Replays voice_gate2.detect; returns per-window rel_db (None where silence-gated) + aggregates."""
    n = min(len(cand), len(vmed), len(mix))
    V = vmed[:n]; M = mix[:n]
    st = windows(n)
    we = np.array([np.mean(M[s:s + WIN] ** 2) for s in st])
    ve = np.array([np.mean(V[s:s + WIN] ** 2) for s in st])
    wemax = float(we.max()) if len(we) else 0.0
    keep = we > (wemax * 1e-3)
    rel = np.full(len(st), np.nan)
    for i in np.where(keep)[0]:
        rel[i] = 10 * np.log10(max(ve[i], 1e-20) / we[i])
    worst = float(np.nanmax(rel[keep])) if keep.any() else float("nan")
    n_novoice = int(np.sum(keep & (rel <= THRESH)))
    per_window = [None if not keep[i] else round(float(rel[i]), 3) for i in range(len(st))]
    return dict(
        worst_db=(None if not np.isfinite(worst) else round(worst, 2)),
        acoustic_pass=bool(np.isfinite(worst) and worst <= THRESH),
        n_windows=int(len(st)), n_kept=int(keep.sum()), n_novoice=n_novoice,
        window_starts_samples=[int(s) for s in st], per_window_rel_db=per_window)


def main():
    for p in (DONE, ERR, HB):
        try: os.remove(p)
        except OSError: pass
    RESULT["ffmpeg_version"] = subprocess.run([ffmpeg(), "-version"], capture_output=True, text=True).stdout.splitlines()[0] if True else ""
    flush()

    hb(f"loading separators MDX23C + BS + MelBand(base) on {DEV}")
    sep_mdx, sep_bs, sep_mel = load_sep(MDX), load_sep(BS), load_sep(MEL)
    hb("separators loaded")

    mix = decode_any(MIX_PATH)
    hb(f"mix {len(mix)} samp {len(mix)/SR:.1f}s ch={mix.shape[1]}")

    # ---- base-mel separation of MIX (shared: recipe A/M + detector iter0) ----
    hb("separating MIX with MDX / BS / MelBand(base)")
    t = time.time(); vmdx_m = sep_vocals(sep_mdx, MIX_PATH); hb(f"  MDX mix {time.time()-t:.1f}s")
    t = time.time(); vbs_m = sep_vocals(sep_bs, MIX_PATH); hb(f"  BS mix {time.time()-t:.1f}s")
    t = time.time(); vmelA_m = sep_vocals(sep_mel, MIX_PATH); hb(f"  MelBase mix {time.time()-t:.1f}s")

    vmed_mix = median3(vmdx_m, vmelA_m, vbs_m)
    n = min(len(mix), len(vmed_mix))
    A_champ = mix[:n] - vmed_mix[:n]
    M_max = mix[:min(len(mix),len(vmdx_m),len(vmelA_m),len(vbs_m))] - maxagg3(vmdx_m, vmelA_m, vbs_m)[:min(len(mix),len(vmdx_m),len(vmelA_m),len(vbs_m))]

    # save M f32 lossless + hash
    Mpath = f"{CANDDIR}/M_maxagg.f32.wav"
    sf.write(Mpath, M_max.astype(np.float32), SR, subtype="FLOAT")
    RESULT["stages"]["M_maxagg"] = dict(recipe="mix - per_sample_argmax_abs(MDX23C,MelBand_base,BS)",
                                        f32_path=Mpath, pcm_sha256=pcm_sha256(M_max),
                                        n_samples=int(len(M_max)))
    hb(f"M_maxagg f32 saved sha={RESULT['stages']['M_maxagg']['pcm_sha256'][:12]}")
    flush()

    # A_champion validation hash (must match iter1 4c229b5...)
    RESULT["stages"]["A_champion_repro"] = dict(recipe="mix - median3(MDX23C,MelBand_base,BS)",
                                                pcm_sha256=pcm_sha256(A_champ),
                                                expected_iter1="4c229b5f568fd9fcd7768daa0fd1f4dfdd61cdc410449e02327948b5001ac9e1")
    RESULT["stages"]["A_champion_repro"]["reproduces_iter1"] = (
        RESULT["stages"]["A_champion_repro"]["pcm_sha256"] == RESULT["stages"]["A_champion_repro"]["expected_iter1"])
    hb(f"A_champion repro sha={RESULT['stages']['A_champion_repro']['pcm_sha256'][:12]} "
       f"reproduces_iter1={RESULT['stages']['A_champion_repro']['reproduces_iter1']}")
    flush()

    # ---- DETECTOR per-window (base mel) on f32 candidates that DON'T need FT ----
    det = RESULT.setdefault("detector_perwindow", {})
    # iter0 = mix: reuse vmed_mix
    det["iter0"] = detect_perwindow(mix, mix, vmed_mix)
    hb(f"detect iter0 worst={det['iter0']['worst_db']} pass={det['iter0']['acoustic_pass']}"); flush()

    def detect_file(tag, path):
        X = decode_any(path)
        vmed = median3(sep_vocals(sep_mdx, path) if path.endswith(".wav") else _sep_arr(sep_mdx, X),
                       sep_vocals(sep_mel, path) if path.endswith(".wav") else _sep_arr(sep_mel, X),
                       sep_vocals(sep_bs, path) if path.endswith(".wav") else _sep_arr(sep_bs, X))
        det[tag] = detect_perwindow(X, mix, vmed)
        hb(f"detect {tag} worst={det[tag]['worst_db']} pass={det[tag]['acoustic_pass']}"); flush()

    def _sep_arr(sep, X):
        tmp = f"{STEM_TMP}/cand_{time.time_ns()}.wav"
        sf.write(tmp, X.astype(np.float32), SR, subtype="FLOAT")
        v = sep_vocals(sep, tmp)
        try: os.remove(tmp)
        except OSError: pass
        return v

    for tag, path in [("iter1", f"{CANDDIR}/iter1.wav"),
                      ("iter2", f"{CANDDIR}/iter2.wav"),
                      ("iter2_Z2_comp", f"{CANDDIR}/iter2_Z2_comp.wav")]:
        detect_file(tag, path)
    # M detector (base mel) via in-memory array
    det["M_maxagg"] = detect_perwindow(M_max, mix, median3(_sep_arr(sep_mdx, M_max), _sep_arr(sep_mel, M_max), _sep_arr(sep_bs, M_max)))
    hb(f"detect M_maxagg worst={det['M_maxagg']['worst_db']} pass={det['M_maxagg']['acoustic_pass']}"); flush()

    # ---- swap MelBand -> FT step600, build B ----
    hb("capturing base MelBand state_dict (clone) then swapping to FT step600")
    base_sd = {k: v.detach().clone() for k, v in sep_mel.model_instance.model_run.state_dict().items()}
    ft = torch.load(MEL_FT, map_location=DEV)
    try:
        sep_mel.model_instance.model_run.load_state_dict(ft)          # render_ab.py path: bare state_dict
    except Exception:
        sep_mel.model_instance.model_run.load_state_dict(ft["state_dict"])
    sep_mel.model_instance.model_run.eval()
    hb("FT loaded; separating MIX with MelBand_ft")
    t = time.time(); vmelB_m = sep_vocals(sep_mel, MIX_PATH); hb(f"  MelFT mix {time.time()-t:.1f}s")
    nb = min(len(mix), len(vmdx_m), len(vmelB_m), len(vbs_m))
    B = mix[:nb] - median3(vmdx_m, vmelB_m, vbs_m)[:nb]
    Bpath = f"{CANDDIR}/B_melband_ft.f32.wav"
    sf.write(Bpath, B.astype(np.float32), SR, subtype="FLOAT")
    RESULT["stages"]["B_melband_ft"] = dict(recipe="mix - median3(MDX23C,MelBand_ft@step600,BS)",
                                            f32_path=Bpath, pcm_sha256=pcm_sha256(B),
                                            melband_ft_ckpt=MEL_FT, n_samples=int(len(B)))
    hb(f"B_melband_ft f32 saved sha={RESULT['stages']['B_melband_ft']['pcm_sha256'][:12]}")
    flush()

    # restore base mel, detect B
    hb("restoring base MelBand; detecting B")
    sep_mel.model_instance.model_run.load_state_dict(base_sd)
    sep_mel.model_instance.model_run.eval()
    det["B_melband_ft"] = detect_perwindow(B, mix, median3(_sep_arr(sep_mdx, B), _sep_arr(sep_mel, B), _sep_arr(sep_bs, B)))
    hb(f"detect B worst={det['B_melband_ft']['worst_db']} pass={det['B_melband_ft']['acoustic_pass']}"); flush()

    # ---- HASHING (CPU): lossless f32 parents + lossy listening copies ----
    hb("hashing f32 parents + lossy listening copies")
    H = RESULT.setdefault("hashes", {})
    # original mix f32 + iter wavs
    for tag, path, lossy in [
        ("original_mix", MIX_PATH, False),
        ("iter0", f"{CANDDIR}/iter0.wav", False),
        ("iter1", f"{CANDDIR}/iter1.wav", False),
        ("iter2", f"{CANDDIR}/iter2.wav", False),
        ("iter2_Z2_comp", f"{CANDDIR}/iter2_Z2_comp.wav", False),
        ("O_original_m4a", f"{ABDIR}/verdi_slow_voices__O_original.m4a", True),
        ("A_champion_m4a", f"{ABDIR}/verdi_slow_voices__A_champion.m4a", True),
        ("B_melband_ft_m4a", f"{ABDIR}/verdi_slow_voices__B_melband_ft.m4a", True),
        ("C_cascade_m4a", f"{ABDIR}/verdi_slow_voices__C_cascade.m4a", True),
        ("M_maxagg_m4a", f"{ABDIR}/verdi_slow_voices__M_maxagg.m4a", True),
        ("V1_inverted_m4a", f"{ABDIR}/verdi_slow_voices__V1_inverted.m4a", True),
        ("V2_mdx23c_m4a", f"{ABDIR}/verdi_slow_voices__V2_mdx23c.m4a", True),
        ("V3_roformer_m4a", f"{ABDIR}/verdi_slow_voices__V3_roformer.m4a", True),
        ("V4_deechorev_m4a", f"{ABDIR}/verdi_slow_voices__V4_deechorev.m4a", True),
        ("V5_ensmax_m4a", f"{ABDIR}/verdi_slow_voices__V5_ensmax.m4a", True),
        ("Z_gate2_iter2_m4a", f"{ABDIR}/verdi_slow_voices__Z_gate2_iter2.m4a", True),
        ("Z2_gate2_iter2_comp_m4a", f"{ABDIR}/verdi_slow_voices__Z2_gate2_iter2_comp.m4a", True),
    ]:
        if not os.path.exists(path):
            H[tag] = dict(path=path, exists=False); flush(); continue
        X = decode_any(path)
        H[tag] = dict(path=path, exists=True, lossy_origin=bool(lossy),
                      pcm_sha256=pcm_sha256(X), n_samples=int(len(X)), channels=int(X.shape[1]))
        hb(f"  hash {tag} sha={H[tag]['pcm_sha256'][:12]} lossy={lossy}")
        flush()

    RESULT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flush()
    open(DONE, "w").write(RESULT["finished"] + "\n")
    hb("REGEN_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        hb("ERROR:\n" + tb)
        open(ERR, "w").write(tb)
        RESULT["error"] = tb
        try: flush()
        except Exception: pass
        sys.exit(1)
