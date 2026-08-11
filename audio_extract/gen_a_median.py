"""Generate the CHAMPION A_median (median MDX23C/MelBand/BS vocal estimates -> residual) for
each classical-v1 TRAIN work, cached alongside M/A/V for the zero-init residual-correction run.
A_median = M - per-sample-median(v_MDX23C, v_MelBand, v_BS).  V_median cached too (optional input).
Matches the held-out champion recipe (overlap=8, same 3 models). Resumable: skips existing."""
import os, sys, glob, time, hashlib, json
import numpy as np, soundfile as sf
from audio_extract.separate import Separator
from audio_extract import dsp

SR = 44100
MODEL_DIR = "/home/mickg/models"
STEM_TMP = "/mnt/bigdisk/stem_tmp"
TRAIN_DATA = "/home/mickg/train_data"
os.makedirs(STEM_TMP, exist_ok=True)
MODELS = {"MDX23C": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
          "MelBand": "vocals_mel_band_roformer.ckpt",
          "BS": "model_bs_roformer_ep_317_sdr_12.9755.ckpt"}
_sep = {}


def sep_vocals(model_name, wavpath):
    s = _sep.get(model_name)
    if s is None:
        s = Separator(MODELS[model_name], overlap=8, model_dir=MODEL_DIR, output_dir=STEM_TMP)
        _sep[model_name] = s
        print(f"  [load] {model_name}", flush=True)
    out = s.separate_file(wavpath)
    v = out.stems.get("vocals")
    assert v is not None, f"{model_name}: no vocals stem; got {list(out.stems)}"
    return np.asarray(v, dtype=np.float64)


def a_median(mix):
    m = dsp.as2d(mix).astype("float64")
    h = hashlib.sha256(m.tobytes()).hexdigest()[:12]
    for p in glob.glob(os.path.join(STEM_TMP, "*.wav")):   # clean stale outputs: wrapper diffs new-vs-before wavs
        try: os.remove(p)
        except OSError: pass
    tmp = os.path.join(STEM_TMP, f"in_{h}.wav")
    sf.write(tmp, m.astype("float32"), SR, subtype="FLOAT")
    try:
        vs = []
        for mn in MODELS:
            v = sep_vocals(mn, tmp)
            if v.ndim == 1:
                v = np.stack([v, v], 1)
            vs.append(v)
        n = min(len(m), *[len(v) for v in vs])
        vs = [v[:n] for v in vs]
        medV = np.median(np.stack(vs, 0), axis=0)          # per-sample median of 3 vocals
        A = m[:n] - medV
        return A, medV, {mn: float(np.sqrt(np.mean((vs[i] - medV) ** 2))) for i, mn in enumerate(MODELS)}
    finally:
        for p in glob.glob(os.path.join(STEM_TMP, "*.wav")):
            try: os.remove(p)
            except OSError: pass


def main():
    if sys.argv[1:]:
        works = sys.argv[1:]
    else:                       # orchestral (donor/freidi) first -> enables earlier training launch; cantoria (long) last
        allw = [os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{TRAIN_DATA}/*/M.flac")]
        pri = {"donor": 0, "freidi": 1, "cantoria": 2}
        works = sorted(allw, key=lambda w: (pri.get(w.split("_")[0], 3), w))
    report = {}
    for i, w in enumerate(works):
        d = f"{TRAIN_DATA}/{w}"
        outp = f"{d}/A_median.flac"
        if os.path.exists(outp):
            print(f"[{i+1}/{len(works)}] {w} SKIP (exists)", flush=True)
            continue
        M, sr = sf.read(f"{d}/M.flac", dtype="float64", always_2d=True)
        assert sr == SR, f"{w} sr={sr}"
        t0 = time.time()
        A, medV, disp = a_median(M)
        sf.write(outp, np.clip(A, -1, 1).astype("float32"), SR, subtype="PCM_24")
        sf.write(f"{d}/V_median.flac", np.clip(medV, -1, 1).astype("float32"), SR, subtype="PCM_24")
        # sanity: reconstruction M ~= A_median + V_median
        n = min(len(M), len(A))
        rec = 20 * np.log10(np.sqrt(np.mean((M[:n] - A - medV[:n]) ** 2)) / (np.sqrt(np.mean(M[:n] ** 2)) + 1e-12) + 1e-12)
        report[w] = dict(frames=len(A), sec=round(len(A) / SR, 1), sep_time_s=round(time.time() - t0, 1),
                         M_eq_Amed_plus_Vmed_db=round(float(rec), 1), model_dispersion_rms=disp)
        print(f"[{i+1}/{len(works)}] {w}  {report[w]['sec']}s  t={report[w]['sep_time_s']}s  recon={report[w]['M_eq_Amed_plus_Vmed_db']}dB", flush=True)
        json.dump(report, open(f"{STEM_TMP}/a_median_report.json", "w"), indent=1)
    json.dump(report, open(f"{STEM_TMP}/a_median_report.json", "w"), indent=1)
    print("GEN_A_MEDIAN_DONE", len(report), "works", flush=True)


if __name__ == "__main__":
    main()
