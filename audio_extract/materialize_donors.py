"""Domain fix: build donor ORCHESTRAL constructions (VocalSet x PHENICX bed x measured Pori RIR)
into the fine-tune set. M=A_bed(stereo)+place_vocal(singer,RIR,g), A=bed, V=placed voice; exact.
-> /home/mickg/train_data/donor_<orch>_<singer>_<rir>/{M,A,V}.flac + vmask + meta. Group-atomic."""
import os, subprocess, json
import numpy as np, soundfile as sf
from audio_extract import challenges as ch, dsp
SR = 44100
AS = "/home/mickg/donor_assets"
OUT = "/home/mickg/train_data"
os.makedirs(AS, exist_ok=True)
subprocess.run(["rsync", "-a", "-e", "sshpass -p ttuser ssh -o StrictHostKeyChecking=no",
                "ttuser@100.91.242.69:/home/ttuser/v3_stage/", f"{AS}/"], check=True)


def r1(p):
    a, sr = sf.read(p, dtype="float64", always_2d=False)
    a = a.mean(1) if a.ndim > 1 else a
    if sr != SR:
        import librosa; a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a


def central(a, s):
    n = int(s * SR)
    return a if len(a) <= n else a[(len(a) - n) // 2:(len(a) - n) // 2 + n]


def vmask(V):
    m = np.abs(V).mean(1); win = int(0.05 * SR)
    env = np.convolve(m, np.ones(win) / win, mode="same")
    return (env > 0.05 * (env.max() + 1e-9)).astype("float32")


def wr(p, a):
    sf.write(p, np.clip(a, -1, 1).astype("float32"), SR, subtype="PCM_24")


rirs = {"pori1": r1(f"{AS}/rir/s1_r1_o.wav"), "pori3": r1(f"{AS}/rir/s3_r3_o.wav")}
beds = {p: r1(f"{AS}/orch/{p}_bed.wav") for p in ("beethoven", "bruckner", "mahler")}
plan = {
    "beethoven": [("female1", "pori1", -3), ("male1", "pori3", 2), ("female2", "pori1", 0), ("male2", "pori3", -6), ("female3", "pori1", 3), ("male3", "pori3", -3)],
    "bruckner": [("male4", "pori1", 0), ("female4", "pori3", -3), ("male5", "pori1", 3), ("female5", "pori3", 0), ("female1", "pori1", -6), ("male1", "pori3", 2)],
    "mahler": [("female2", "pori3", 2), ("male3", "pori1", -3), ("female4", "pori1", 0), ("male5", "pori3", 3), ("female3", "pori3", -6), ("male2", "pori1", 0)],
}
n = 0
for orch, rows in plan.items():
    A_mono = central(beds[orch], 20)
    A_st = np.repeat(dsp.as2d(A_mono), 2, axis=1)
    for singer, rk, g in rows:
        Vc = central(r1(f"{AS}/vocalset/{singer}_caro.wav"), 20)
        placed = ch.place_vocal(Vc, SR, ir=rirs[rk], gain_db=float(g), pan=0.1)
        L = min(len(A_st), len(placed))
        M, A, V = A_st[:L] + placed[:L], A_st[:L], placed[:L]
        wid = f"donor_{orch}_{singer}_{rk}"
        d = f"{OUT}/{wid}"; os.makedirs(d, exist_ok=True)
        wr(f"{d}/M.flac", M); wr(f"{d}/A.flac", A); wr(f"{d}/V.flac", V)
        np.save(f"{d}/vmask.npy", vmask(V))
        resid = 20 * np.log10(np.sqrt(np.mean((M - A - V) ** 2)) / (np.sqrt(np.mean(M ** 2)) + 1e-12) + 1e-12)
        json.dump(dict(work=wid, frames=L, sr=SR, channels=2, group=f"donor_{orch}",
                       family="synthesizable_orchestra", M_eq_A_plus_V_db=round(resid, 1)), open(f"{d}/meta.json", "w"))
        n += 1
print(f"donor constructions materialized: {n}")
print("DONORS_DONE")
