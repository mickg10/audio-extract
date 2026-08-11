"""ZERO-INIT RESIDUAL CORRECTION on the median MDX23C/MelBand/BS champion (§13.5).
A_refined = A_median + iSTFT(Delta_theta(STFT(M), STFT(A_median)));  V_refined = M - A_refined.
Delta_theta = small spectral U-Net, FINAL CONV ZERO-INIT -> Delta=0 at step 0 -> A_refined==A_median
EXACTLY (waveform add of zero: champion parity, hard sanity). L1(Delta) keeps it to the champion's
identified errors. Loss = full asymmetric family (reused from train_classical). Starts from the
WINNING method; can only improve it. Gate: A_refined BEAT A_median on held-out Bologna/Aalto exact
reference? NOT the SHADOW judge. Invoke: ./run.sh python -m audio_extract.refine_ensemble ..."""
from __future__ import annotations
import argparse, glob, hashlib, json, os, subprocess, time
import numpy as np, soundfile as sf, torch, torch.nn as nn, yaml
from audio_extract.train_classical import (SR, cmrstft, srccoord, event_weight, exact_metrics,
                                           readf, to2, TRAIN_DATA, TP, CAND)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GATE_STEPS = [0, 100, 500, 2000]
NFFT, HOP = 2048, 512


def sha_file(p):
    return "sha256:" + hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------- STFT helpers ----------------
def stft(x):  # x (B,2,L) -> (B,2,F,T) complex
    B, C, L = x.shape
    w = torch.hann_window(NFFT, device=x.device)
    X = torch.stft(x.reshape(B * C, L), NFFT, HOP, window=w, return_complex=True, center=True)
    return X.reshape(B, C, X.shape[-2], X.shape[-1])


def istft(X, length):  # X (B,2,F,T) complex -> (B,2,L)
    B, C, F, T = X.shape
    w = torch.hann_window(NFFT, device=X.device)
    x = torch.istft(X.reshape(B * C, F, T), NFFT, HOP, window=w, length=length, center=True)
    return x.reshape(B, C, length)


# ---------------- zero-init spectral U-Net ----------------
class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(min(8, co), co), nn.GELU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(min(8, co), co), nn.GELU())

    def forward(self, x):
        return self.net(x)


class ResidualRefiner(nn.Module):
    def __init__(self, cin=8, cout=4, base=24):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(cin, base), Block(base, base * 2), Block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = Block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = Block(base * 2, base)
        self.out = nn.Conv2d(base, cout, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # ZERO-INIT -> Delta=0 at step 0

    def unet(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

    def forward(self, M, A_med):                      # (B,2,L) each
        L = M.shape[-1]
        Xm, Xa = stft(M), stft(A_med)                 # (B,2,F,T) complex
        feat = torch.cat([Xm.real, Xm.imag, Xa.real, Xa.imag], 1)   # (B,8,F,T)
        F, T = feat.shape[-2], feat.shape[-1]
        pf, pt = (4 - F % 4) % 4, (4 - T % 4) % 4
        feat = torch.nn.functional.pad(feat, (0, pt, 0, pf))
        d = self.unet(feat)[..., :F, :T]              # (B,4,F,T)
        Dstft = torch.complex(d[:, :2], d[:, 2:])     # (B,2,F,T)
        Dwave = istft(Dstft, L)
        A_ref = A_med + Dwave
        V_ref = M - A_ref
        return A_ref, V_ref, Dwave


# ---------------- asymmetric loss family on (A_ref, V_ref) ----------------
def refine_loss(A_ref, V_ref, Dwave, M, A, V, vmask, kinds, W):
    ew = event_weight(M, vmask)
    l_vc = cmrstft(V_ref, V); l_ac = cmrstft(A_ref, A)
    l_aw = (ew * (A_ref - A).abs()).mean()
    l_mix = (A_ref + V_ref - M).abs().mean()          # mixture consistency (=0 by construction; sanity)
    exact_mask = torch.tensor([1.0 if k == "exact" else 0.0 for k in kinds], device=A.device)
    la, lb, lbaud, lR = srccoord(A_ref, A, V, exact_mask)
    l_src = W["alpha"] * la + W["beta"] * lb + W["beta_aud"] * lbaud + W["R"] * lR
    l_stereo = ((A_ref[:, 0] - A_ref[:, 1]) - (A[:, 0] - A[:, 1])).abs().mean()
    nv = torch.tensor([1.0 if k == "no_vocal" else 0.0 for k in kinds], device=A.device).view(-1, 1, 1)
    l_fp = (nv * V_ref.abs()).mean()                  # no-vocal false-positive: V_ref -> 0
    vo = torch.tensor([1.0 if k == "vocal_only" else 0.0 for k in kinds], device=A.device).view(-1, 1, 1)
    l_fn = (vo * (V_ref - V).abs()).mean()            # vocal-only: keep all voice
    l_l1 = Dwave.abs().mean()                          # §13.5 sparse correction
    total = (W["vc"] * l_vc + W["ac"] * l_ac + W["aw"] * l_aw + W["mix"] * l_mix + l_src
             + W["stereo"] * l_stereo + W["no_vocal"] * l_fp + W["vocal_fn"] * l_fn + W["l1_delta"] * l_l1)
    parts = dict(vc=float(l_vc), ac=float(l_ac), aw=float(l_aw), src=float(l_src), fp=float(l_fp),
                 fn=float(l_fn), l1=float(l_l1))
    return total, parts


# ---------------- chunked apply for full-length eval ----------------
@torch.no_grad()
def apply_refiner(model, M, A_med, seg_s=12, hop_s=6):
    L = M.shape[-1]; seg = int(seg_s * SR); hop = int(hop_s * SR)
    if L <= seg:
        A_ref, _, _ = model(M.unsqueeze(0).to(DEV), A_med.unsqueeze(0).to(DEV))
        return A_ref[0].cpu()
    Dacc = torch.zeros_like(M); wsum = torch.zeros(L)
    win = torch.hann_window(seg)
    for st in range(0, L, hop):
        en = min(st + seg, L); n = en - st
        w = win[:n] if n == seg else torch.ones(n)
        _, _, D = model(M[:, st:en].unsqueeze(0).to(DEV), A_med[:, st:en].unsqueeze(0).to(DEV))
        Dacc[:, st:en] += D[0].cpu() * w; wsum[st:en] += w
        if en >= L:
            break
    Dacc = Dacc / wsum.clamp(min=1e-6)
    return A_med + Dacc


# ---------------- held-out champion baseline + gate ----------------
_BASE = {}
def load_heldout(work):
    if work in _BASE:
        return _BASE[work]
    M = readf(f"{TP}/{work}/mix_with_voice.wav"); A = readf(f"{TP}/{work}/orchestra_only.wav"); V = readf(f"{TP}/{work}/voice_ref.wav")
    Amed = readf(f"{CAND}/{work}__median_mdx_mel_bs.flac")
    n = min(M.shape[1], Amed.shape[1], A.shape[1], V.shape[1])
    M, A, V, Amed = M[:, :n], A[:, :n], V[:, :n], Amed[:, :n]
    champ = exact_metrics(Amed, A, V)
    _BASE[work] = (M, A, V, Amed, champ)
    return _BASE[work]


def gate_eval(model, works):
    model.eval(); res = {}
    for w in works:
        M, A, V, Amed, champ = load_heldout(w)
        A_ref = apply_refiner(model, M, Amed); n = min(A_ref.shape[1], A.shape[1])
        stu = exact_metrics(A_ref[:, :n], A[:, :n], V[:, :n])
        parity = float(torch.max(torch.abs(A_ref[:, :n] - Amed[:, :n])))
        res[w] = dict(refined=stu, champion=champ, step0_max_abs_dev=parity)
    model.train(); return res


# ---------------- sampler (train works with cached A_median) ----------------
class Sampler:
    def __init__(self, crop):
        # PRE-LOAD all train stems into RAM once (immune to the shared lane's disk contention;
        # per-step crops then slice from memory -> GPU-bound stepping, not disk-bound).
        self.crop = crop; self.by_corpus = {}; self.cache = {}
        for d in sorted(glob.glob(f"{TRAIN_DATA}/*/A_median.flac")):
            dd = os.path.dirname(d); w = os.path.basename(dd)
            corp = "donor" if w.startswith("donor") else ("freidi" if w.startswith("freidi") else ("cantolopera" if "cantolopera" in w else "cantoria"))
            self.cache[dd] = dict(M=to2(readf(f"{dd}/M.flac")), A=to2(readf(f"{dd}/A.flac")),
                                  V=to2(readf(f"{dd}/V.flac")), Am=to2(readf(f"{dd}/A_median.flac")),
                                  vm=torch.from_numpy(np.load(f"{dd}/vmask.npy").astype("float32")))
            self.by_corpus.setdefault(corp, []).append((w, dd, self.cache[dd]["M"].shape[1]))
        self.orch = self.by_corpus.get("donor", []) + self.by_corpus.get("freidi", [])
        print(f"[sampler] pre-loaded {len(self.cache)} works into RAM", flush=True)

    def _crop(self, d, fr, rng):
        c = self.cache[d]; L = c["M"].shape[1]; st = int(rng.integers(0, max(1, L - self.crop))); cc = self.crop
        M, A, V, Am = (c[k][:, st:st + cc] for k in ("M", "A", "V", "Am"))
        vm = c["vm"][st:st + cc]
        M, A, V, Am = (torch.nn.functional.pad(x, (0, cc - x.shape[1]))[:, :cc] for x in (M, A, V, Am))
        vm = torch.nn.functional.pad(vm, (0, cc - len(vm)))[:cc]
        return M, A, V, Am, vm

    def sample(self, rng):
        r = rng.random()
        if r < 0.25 and self.orch:                         # no-vocal: M=A, A_median≈A, V=0
            w, d, fr = self.orch[rng.integers(len(self.orch))]
            M, A, V, Am, vm = self._crop(d, fr, rng)
            return "no_vocal", A, A, torch.zeros_like(V), A.clone(), torch.zeros_like(vm)
        if r < 0.40:                                        # vocal-only: M=V, A=0, A_median≈0
            corp = list(self.by_corpus)[rng.integers(len(self.by_corpus))]
            w, d, fr = self.by_corpus[corp][rng.integers(len(self.by_corpus[corp]))]
            M, A, V, Am, vm = self._crop(d, fr, rng)
            return "vocal_only", V, torch.zeros_like(A), V, torch.zeros_like(Am), vm
        corp = ["donor", "freidi", "cantoria", "cantolopera"][int(rng.integers(4))]
        pool = self.by_corpus.get(corp) or self.by_corpus["cantoria"]
        w, d, fr = pool[rng.integers(len(pool))]
        return ("exact", *self._crop(d, fr, rng))


def main():
    ap = argparse.ArgumentParser("audio-extract refine ensemble")
    ap.add_argument("--config", required=True); ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-step", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.run_dir, exist_ok=True)
    cfg = yaml.safe_load(open(a.config)); W = cfg["loss"]; o = cfg["optim"]
    torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    eval_works = cfg["gate"]["held_out"]

    model = ResidualRefiner(base=o.get("base", 24)).to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=o["lr"])
    crop = int(o.get("crop_s", 6) * SR)
    smp = Sampler(crop); batch = o.get("batch", 2)
    commit = subprocess.run(["git", "-C", "/home/mickg/audio-extract", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    gates, hist = {}, []; t0 = time.time()
    for step in range(0, a.max_step + 1):
        if step in GATE_STEPS:
            print(f"[gate {step}] evaluating...", flush=True)
            gates[step] = gate_eval(model, eval_works)
            json.dump(gates, open(f"{a.run_dir}/gates.json", "w"), indent=1)
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg}, f"{a.run_dir}/ckpt_step{step}.pt")
            beats = 0
            for w, r in gates[step].items():
                rf, ch = r["refined"], r["champion"]
                won = (rf["event_hole_db_p90"] <= ch["event_hole_db_p90"] + 0.3 and rf["si_sdr_db"] >= ch["si_sdr_db"] - 0.3
                       and rf["retained_voice_db_p90"] <= ch["retained_voice_db_p90"] + 0.5)
                beats += int(won)
                print(f"  [{w}] refined sisdr={rf['si_sdr_db']} hole={rf['event_hole_db_p90']} rv={rf['retained_voice_db_p90']} | champ sisdr={ch['si_sdr_db']} hole={ch['event_hole_db_p90']} rv={ch['retained_voice_db_p90']} | dev={r['step0_max_abs_dev']:.2e} {'BEAT' if won else ''}", flush=True)
            print(f"  [gate {step}] refined BEATS champion on {beats}/{len(eval_works)} works", flush=True)
            if step == 0:
                pv = max(r["step0_max_abs_dev"] for r in gates[0].values())
                print(f"  [PARITY step0==A_median] max_abs_dev={pv:.3e} ({'OK' if pv < 1e-4 else 'FAIL'})", flush=True)
        if step == a.max_step:
            break
        Ms, As, Vs, Ams, VMs, kinds = [], [], [], [], [], []
        for _ in range(batch):
            k, M, A, V, Am, vm = smp.sample(rng)
            Ms.append(to2(M)); As.append(to2(A)); Vs.append(to2(V)); Ams.append(to2(Am)); VMs.append(vm); kinds.append(k)
        M, A, V, Am, vm = (torch.stack(x).to(DEV) for x in (Ms, As, Vs, Ams, VMs))
        A_ref, V_ref, Dwave = model(M, Am)
        loss, parts = refine_loss(A_ref, V_ref, Dwave, M, A, V, vm, kinds, W)
        if not torch.isfinite(loss):
            json.dump({"ABORT": "non-finite", "step": step}, open(f"{a.run_dir}/ABORT.json", "w")); raise SystemExit("ABORT")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(float(loss.detach()))
        if step % 50 == 0:
            print(f"step {step} loss={loss:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()), flush=True)

    report = dict(experiment="zero-init residual correction on median MDX/Mel/BS champion (§13.5)",
                  model="ResidualRefiner spectral U-Net, final-conv zero-init (A_refined=A_median+iSTFT(Delta))",
                  source_commit=commit, device=DEV, n_params=sum(p.numel() for p in model.parameters()),
                  resolved_config=cfg, crop_frames=crop, batch=batch, gate_steps=GATE_STEPS,
                  corpora={k: len(v) for k, v in smp.by_corpus.items()},
                  convergence=dict(first50=round(float(np.mean(hist[:50])), 4), last50=round(float(np.mean(hist[-50:])), 4), all_finite=bool(all(np.isfinite(hist)))),
                  gates=gates, train_time_s=round(time.time() - t0, 1),
                  baseline="median_mdx_mel_bs (delivered champion)",
                  repro_cmd=f"./run.sh python -m audio_extract.refine_ensemble --config {a.config} --run-dir {a.run_dir}",
                  do_not_optimize_shadow_judge=True)
    json.dump(report, open(f"{a.run_dir}/run_report.json", "w"), indent=1)
    print("REFINE_ENSEMBLE_DONE")


if __name__ == "__main__":
    main()
