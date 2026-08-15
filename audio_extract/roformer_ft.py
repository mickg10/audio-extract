"""Phase-2/3: ANTI-FORGETTING fine-tune of the Mel-Band-Roformer ensemble member (228M, loaded via
audio-separator RoformerLoader == the inference model_run) + median-ensemble beat/no-beat GATE.
Guards vs catastrophic drift (HTDemucs lesson): tiny LR, L2-SP anchor to pretrained, short schedule,
grad-clip. Gate: median(MDX23C, MelBand', BS) vs current champion on held-out Bologna/Aalto exact ref.
Step-0: MelBand'=pretrained -> ensemble == champion (parity). RAM-preload. systemd-run + heartbeats.
NOT the SHADOW judge. Invoke: ./run.sh python -m audio_extract.roformer_ft --run-dir ... --max-step ..."""
from __future__ import annotations
import argparse, glob, json, os, time, subprocess
import numpy as np, soundfile as sf, torch
from audio_extract.train_classical import cmrstft, exact_metrics, readf, to2, SR, TP, CAND, TRAIN_DATA
from audio_separator.separator import Separator

MODEL_DIR = "/home/mickg/models"; STEM_TMP = "/mnt/bigdisk/stem_tmp"; CACHE = "/mnt/bigdisk/roformer_ft_cache"
MEL = "vocals_mel_band_roformer.ckpt"; MDX = "MDX23C-8KFFT-InstVoc_HQ.ckpt"; BS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_WORKS = ["bologna_verdi", "bologna_puccini", "bologna_donizetti", "aalto_mozart_dry", "aalto_mozart_hall"]
GATE_STEPS = [0, 100, 300, 600]
os.makedirs(CACHE, exist_ok=True)


def hb(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_sep(model_name):
    s = Separator(model_file_dir=MODEL_DIR, output_dir=STEM_TMP, log_level=40)
    s.load_model(model_filename=model_name)
    return s


def sep_vocals(sep, wav_path):
    before = set(glob.glob(f"{STEM_TMP}/*.wav"))
    sep.separate(wav_path)
    prod = [p for p in glob.glob(f"{STEM_TMP}/*.wav") if p not in before]
    vp = [p for p in prod if "ocal" in os.path.basename(p)]
    v = None
    if vp:
        a, _ = sf.read(vp[0], dtype="float64", always_2d=True); v = a
    for p in prod:
        try: os.remove(p)
        except OSError: pass
    return v


def median3(vmdx, vmel, vbs):
    n = min(len(vmdx), len(vmel), len(vbs))
    return np.median(np.stack([vmdx[:n], vmel[:n], vbs[:n]], 0), axis=0)


# ---------------- gate: median(MDX, MelBand', BS) vs champion ----------------
_HELD = {}
def prep_heldout(sep_mel):
    hb("precompute held-out V_MDX23C + V_BS (once) + champion (median w/ pretrained MelBand)")
    sep_mdx = load_sep(MDX); sep_bs = load_sep(BS)
    for w in EVAL_WORKS:
        Mp = f"{TP}/{w}/mix_with_voice.wav"
        M = readf(Mp).T.numpy(); A = readf(f"{TP}/{w}/orchestra_only.wav").T.numpy(); V = readf(f"{TP}/{w}/voice_ref.wav").T.numpy()
        cmdx = f"{CACHE}/{w}_vmdx.npy"; cbs = f"{CACHE}/{w}_vbs.npy"
        vmdx = np.load(cmdx) if os.path.exists(cmdx) else sep_vocals(sep_mdx, Mp)
        if not os.path.exists(cmdx): np.save(cmdx, vmdx)
        vbs = np.load(cbs) if os.path.exists(cbs) else sep_vocals(sep_bs, Mp)
        if not os.path.exists(cbs): np.save(cbs, vbs)
        vmel0 = sep_vocals(sep_mel, Mp)                      # pretrained MelBand vocals -> champion
        A_champ = M[:min(len(M), len(vmdx), len(vmel0), len(vbs))] - median3(vmdx, vmel0, vbs)
        n = min(len(A_champ), len(A), len(V))
        champ = exact_metrics(A_champ[:n], A[:n], V[:n])
        _HELD[w] = dict(Mp=Mp, M=M, A=A, V=V, vmdx=vmdx, vbs=vbs, champion=champ)
        hb(f"  {w}: champion hole_p90={champ['event_hole_db_p90']} sisdr={champ['si_sdr_db']} rv={champ['retained_voice_db_p90']}")
    del sep_mdx, sep_bs; torch.cuda.empty_cache()


def gate_eval(sep_mel, step):
    model = sep_mel.model_instance.model_run; model.eval(); res = {}; beats = 0
    for w in EVAL_WORKS:
        h = _HELD[w]
        vmel = sep_vocals(sep_mel, h["Mp"])                 # MelBand' vocals (real inference, fine-tuned weights)
        A_ref = h["M"][:min(len(h["M"]), len(h["vmdx"]), len(vmel), len(h["vbs"]))] - median3(h["vmdx"], vmel, h["vbs"])
        n = min(len(A_ref), len(h["A"]), len(h["V"]))
        rf = exact_metrics(A_ref[:n], h["A"][:n], h["V"][:n]); ch = h["champion"]
        won = (rf["event_hole_db_p90"] <= ch["event_hole_db_p90"] + 0.3 and rf["si_sdr_db"] >= ch["si_sdr_db"] - 0.3
               and rf["retained_voice_db_p90"] <= ch["retained_voice_db_p90"] + 0.5)
        beats += int(won); res[w] = dict(refined=rf, champion=ch)
        hb(f"  [{w}] MelBand' hole={rf['event_hole_db_p90']} sisdr={rf['si_sdr_db']} rv={rf['retained_voice_db_p90']} | champ hole={ch['event_hole_db_p90']} sisdr={ch['si_sdr_db']} rv={ch['retained_voice_db_p90']} {'BEAT' if won else ''}")
    hb(f"[gate {step}] MelBand'-ensemble BEATS champion on {beats}/{len(EVAL_WORKS)} works")
    model.train(); return res


# ---------------- RAM-preloaded (M,V) sampler, peak-normalized ----------------
class Sampler:
    def __init__(self, crop):
        self.crop = crop; self.works = []
        for d in sorted(glob.glob(f"{TRAIN_DATA}/*/V.flac")):
            dd = os.path.dirname(d)
            M = to2(readf(f"{dd}/M.flac")); V = to2(readf(f"{dd}/V.flac"))
            n = min(M.shape[1], V.shape[1]); self.works.append((M[:, :n], V[:, :n]))
        hb(f"[sampler] pre-loaded {len(self.works)} works (M,V) into RAM")

    def sample(self, rng):
        M, V = self.works[rng.integers(len(self.works))]
        L = M.shape[1]; st = int(rng.integers(0, max(1, L - self.crop))); c = self.crop
        m = M[:, st:st + c]; v = V[:, st:st + c]
        if m.shape[1] < c:
            m = torch.nn.functional.pad(m, (0, c - m.shape[1])); v = torch.nn.functional.pad(v, (0, c - v.shape[1]))
        pk = m.abs().max().clamp(min=1e-6)                  # peak-normalize (matches inference normalize)
        return (m / pk), (v / pk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--max-step", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--l2sp", type=float, default=1.0)
    ap.add_argument("--crop-s", type=float, default=7.0); ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-gate", action="store_true", help="smoke: skip held-out prep + gates, train only")
    a = ap.parse_args(); os.makedirs(a.run_dir, exist_ok=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    commit = subprocess.run(["git", "-C", "/home/mickg/audio-extract", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    hb(f"loading MelBand-Roformer (trainable == inference model_run)")
    sep_mel = load_sep(MEL); model = sep_mel.model_instance.model_run.to(DEV)
    theta0 = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    npar = sum(p.numel() for p in model.parameters())
    hb(f"model params={npar} lr={a.lr} l2sp={a.l2sp} crop_s={a.crop_s} batch={a.batch}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crop = int(a.crop_s * SR)
    if not a.no_gate:
        prep_heldout(sep_mel)
    smp = Sampler(crop)

    gates = {}; hist = []; t0 = time.time()
    for step in range(0, a.max_step + 1):
        if step in GATE_STEPS and not a.no_gate:
            hb(f"[gate {step}] evaluating (median MDX23C/MelBand'/BS vs champion)...")
            gates[step] = gate_eval(sep_mel, step)
            json.dump(gates, open(f"{a.run_dir}/gates.json", "w"), indent=1)
            torch.save(model.state_dict(), f"{a.run_dir}/melband_ft_step{step}.ckpt")
            if step == 0:
                pv = max(abs(gates[0][w]["refined"]["si_sdr_db"] - gates[0][w]["champion"]["si_sdr_db"]) for w in EVAL_WORKS)
                hb(f"[PARITY step0 ensemble==champion] max|dsisdr|={pv:.3e} ({'OK' if pv < 0.05 else 'CHECK'})")
        if step == a.max_step:
            break
        model.train()
        Ms, Vs = [], []
        for _ in range(a.batch):
            m, v = smp.sample(rng); Ms.append(m); Vs.append(v)
        M = torch.stack(Ms).to(DEV); V = torch.stack(Vs).to(DEV)
        Vest = model(M); Aest = M - Vest
        l_v = cmrstft(Vest, V); l_a = cmrstft(Aest, M - V); l_w = (Aest - (M - V)).abs().mean()
        l_sp = sum(((p - theta0[n]) ** 2).sum() for n, p in model.named_parameters() if n in theta0) / npar
        loss = l_v + l_a + l_w + a.l2sp * l_sp
        if not torch.isfinite(loss):
            json.dump({"ABORT": "non-finite", "step": step}, open(f"{a.run_dir}/ABORT.json", "w")); raise SystemExit("ABORT")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(float(loss.detach()))
        if step % 10 == 0:
            hb(f"step {step} loss={loss:.4f} v={float(l_v):.3f} a={float(l_a):.3f} w={float(l_w):.3f} l2sp={float(a.l2sp*l_sp):.4f} gpu_gb={torch.cuda.max_memory_allocated()/1e9:.1f}")

    torch.save(model.state_dict(), f"{a.run_dir}/melband_ft_final.ckpt")
    report = dict(experiment="anti-forgetting FT of MelBand-Roformer member + median-ensemble gate",
                  member=MEL, source_commit=commit, device=DEV, n_params=npar,
                  lr=a.lr, l2sp=a.l2sp, crop_s=a.crop_s, batch=a.batch, max_step=a.max_step, gate_steps=GATE_STEPS,
                  convergence=dict(first20=round(float(np.mean(hist[:20])), 4), last20=round(float(np.mean(hist[-20:])), 4), all_finite=bool(all(np.isfinite(hist)))),
                  gates=gates, train_time_s=round(time.time() - t0, 1),
                  export_format="raw state_dict OrderedDict (matches original ckpt)", do_not_optimize_shadow_judge=True)
    json.dump(report, open(f"{a.run_dir}/run_report.json", "w"), indent=1)
    hb("ROFORMER_FT_DONE")


if __name__ == "__main__":
    main()
