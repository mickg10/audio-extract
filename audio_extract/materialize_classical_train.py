"""Stage-1 train data materialization (runs on research6): stage cantoria + freidi from
tt-quietbox2, build exact M/A/V full-length -> /home/mickg/train_data/<work>/{M,A,V}.flac +
exact vocal-activity mask. Respects classical-v1 splits (train split only)."""
import os, subprocess, glob, json, hashlib
import numpy as np, soundfile as sf
SR = 44100
SRC = "/home/mickg/train_data_src"
OUT = "/home/mickg/train_data"
os.makedirs(SRC, exist_ok=True); os.makedirs(OUT, exist_ok=True)
RS = ["rsync", "-a", "-e", "sshpass -p ttuser ssh -o StrictHostKeyChecking=no"]


def stage():
    os.makedirs(f"{SRC}/freidi", exist_ok=True); os.makedirs(f"{SRC}/cantoria", exist_ok=True)
    subprocess.run(RS + ["ttuser@100.91.242.69:/home/ttuser/datasets/freidi/built/", f"{SRC}/freidi/"], check=True)
    subprocess.run(RS + ["--include=*_Mix.wav", "--include=*_MixOrgan.wav", "--exclude=*",
                         "ttuser@100.91.242.69:/home/ttuser/datasets/works/cantoria/CantoriaDataset_v1.0.0/Audio/",
                         f"{SRC}/cantoria/"], check=True)


def rd(p):
    a, sr = sf.read(p, dtype="float32", always_2d=True)
    return a, int(sr)


def resample(a, sr):
    if sr == SR:
        return a
    import librosa
    return librosa.resample(a.T, orig_sr=sr, target_sr=SR, axis=1).T


def wr(p, a):
    sf.write(p, np.clip(a, -1, 1).astype("float32"), SR, subtype="PCM_24")


def vmask(V):
    m = np.abs(V).mean(1)
    win = int(0.05 * SR); k = np.ones(win) / win
    env = np.convolve(m, k, mode="same")
    thr = 0.05 * (env.max() + 1e-9)
    return (env > thr).astype("float32")


def emit(work, M, A, V):
    n = min(len(M), len(A), len(V)); M, A, V = M[:n], A[:n], V[:n]
    resid = float(np.sqrt(np.mean((M - A - V) ** 2)) / (np.sqrt(np.mean(M ** 2)) + 1e-12))
    d = f"{OUT}/{work}"; os.makedirs(d, exist_ok=True)
    wr(f"{d}/M.flac", M); wr(f"{d}/A.flac", A); wr(f"{d}/V.flac", V)
    np.save(f"{d}/vmask.npy", vmask(V))
    json.dump(dict(work=work, frames=n, sr=SR, channels=M.shape[1],
                   M_eq_A_plus_V_db=round(20 * np.log10(resid + 1e-12), 1)),
              open(f"{d}/meta.json", "w"))
    print(f"{work} frames={n} ch={M.shape[1]} M=A+V {round(20*np.log10(resid+1e-12),1)}dB", flush=True)


if __name__ == "__main__":
    print("[stage] rsync cantoria+freidi from tt-quietbox2 ...", flush=True)
    stage()
    # FreiDi (direct M/A/V)
    for no in ("06", "08", "09"):
        dd = f"{SRC}/freidi/{no}"
        if not os.path.exists(f"{dd}/mix.flac"):
            continue
        M, s1 = rd(f"{dd}/mix.flac"); A, s2 = rd(f"{dd}/accomp.flac"); V, s3 = rd(f"{dd}/voice.flac")
        emit(f"freidi_no{no}", resample(M, s1), resample(A, s2), resample(V, s3))
    # Cantoria (M=MixOrgan, A=MixOrgan-Mix, V=Mix)
    for p in ["CEA", "EJB1", "EJB2", "HCB", "LBM1", "LBM2", "LJT1", "LJT2", "LNG", "RRC", "SSS", "THM", "VBP", "YSM"]:
        mo, mx = f"{SRC}/cantoria/Cantoria_{p}_MixOrgan.wav", f"{SRC}/cantoria/Cantoria_{p}_Mix.wav"
        if not (os.path.exists(mo) and os.path.exists(mx)):
            continue
        MO, s1 = rd(mo); Voices, s2 = rd(mx)
        MO, Voices = resample(MO, s1), resample(Voices, s2)
        n = min(len(MO), len(Voices)); MO, Voices = MO[:n], Voices[:n]
        emit(f"cantoria_{p}", MO, MO - Voices, Voices)
    print("MATERIALIZE_DONE", len(glob.glob(f"{OUT}/*/M.flac")), "works")
