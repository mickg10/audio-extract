"""A/B render on the real library: instrumental = mix - median(MDX23C, MelBand?, BS).
A = champion (MelBand pretrained);  B = MelBand'@step600 (roformer_ft1). MDX23C + BS shared.
Encode m4a pairs <track>__A_champion.m4a / <track>__B_melband_ft.m4a. Heartbeats; detached."""
import os, glob, time, json, subprocess, sys
import numpy as np, soundfile as sf, torch
from audio_separator.separator import Separator

MODEL_DIR = "/home/mickg/models"; STEM_TMP = "/mnt/bigdisk/stem_tmp"; OUT = "/home/mickg/ab_pairs"
MEL = "vocals_mel_band_roformer.ckpt"; MDX = "MDX23C-8KFFT-InstVoc_HQ.ckpt"; BS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MEL_FT = "/home/mickg/runs/roformer_ft1/melband_ft_step600.ckpt"
LIB = "/home/mickg/ae-opera"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
TRACKS = {
    "verdi_slow_voices": "7_verdi_slow_voices_tobacco_05_14_26_1",
    "nessun_dorma": "yt_nessun_suj_2sbsfks",
    "donna": "yt_donna_a8_vzjny10k",
    "habanera": "1_habanera_last_tobacco_2026_06_18",
    "marseillaise_applause": "17_marseillaise_applause_tobacco_05_17_26",
    "barber_agnus_dei": "21_barber_agnus_dei_tobacco_2026_06_22",
}


def hb(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_sep(name):
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
    n = min(len(a), len(b), len(c)); return np.median(np.stack([a[:n], b[:n], c[:n]], 0), axis=0)


def encode_m4a(inst, path_wav, path_m4a):
    sf.write(path_wav, np.clip(inst, -1, 1).astype("float32"), 44100, subtype="FLOAT")
    ff = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).stdout.strip()
    if not ff:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-i", path_wav, "-c:a", "aac", "-b:a", "256k", path_m4a],
                   capture_output=True)
    os.remove(path_wav)
    return os.path.getsize(path_m4a)


def main():
    t0 = time.time()
    hb("loading separators MDX23C + BS + MelBand(pretrained)")
    sep_mdx, sep_bs, sep_mel = load_sep(MDX), load_sep(BS), load_sep(MEL)
    cache = {}
    report = {}
    hb("=== PHASE A: champion (MelBand pretrained) ===")
    for name, d in TRACKS.items():
        mp = f"{LIB}/{d}/source/canonical.f32.wav"
        mix, sr = sf.read(mp, dtype="float64", always_2d=True); assert sr == 44100
        vmdx = sep_vocals(sep_mdx, mp); vbs = sep_vocals(sep_bs, mp); vmelA = sep_vocals(sep_mel, mp)
        cache[name] = (mp, mix, vmdx, vbs)
        n = min(len(mix), len(vmdx), len(vmelA), len(vbs))
        instA = mix[:n] - median3(vmdx, vmelA, vbs)
        sz = encode_m4a(instA, f"{OUT}/{name}_A.wav", f"{OUT}/{name}__A_champion.m4a")
        report.setdefault(name, {})["A_m4a_bytes"] = sz; report[name]["dur_s"] = round(n / 44100, 1)
        hb(f"  A {name}: {round(n/44100,1)}s -> {sz//1024}KB")
    hb("=== swap MelBand -> step600 (fine-tuned) ===")
    sd = torch.load(MEL_FT, map_location=DEV)
    sep_mel.model_instance.model_run.load_state_dict(sd)
    sep_mel.model_instance.model_run.eval()
    hb("=== PHASE B: MelBand'@step600 ===")
    for name, d in TRACKS.items():
        mp, mix, vmdx, vbs = cache[name]
        vmelB = sep_vocals(sep_mel, mp)
        n = min(len(mix), len(vmdx), len(vmelB), len(vbs))
        instB = mix[:n] - median3(vmdx, vmelB, vbs)
        sz = encode_m4a(instB, f"{OUT}/{name}_B.wav", f"{OUT}/{name}__B_melband_ft.m4a")
        # A/B difference magnitude (how much the instrumentals differ)
        instA_path = f"{OUT}/{name}__A_champion.m4a"
        report[name]["B_m4a_bytes"] = sz
        hb(f"  B {name}: -> {sz//1024}KB")
    json.dump(report, open(f"{OUT}/render_report.json", "w"), indent=1)
    hb(f"RENDER_AB_DONE {len(TRACKS)} tracks in {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
