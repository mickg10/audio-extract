"""audio-extract train classical — §2 first CONTINUATION run: FULL fine-tune of the released
vocal-specialized HTDemucs 04573f0d (loaded via demucs, arch NOT reconstructed). Product
(§2): V_hat=outputs['vocals']; A_hat=mixture-V_hat. §10 exact loss + no-vocal/vocal-only
controls, §11 differentiable source-coordinate alpha/beta/R, §7 hierarchical Gate-2 Phase-A
sampler, §13.2 parent-weight anti-forgetting. Gate at steps 0/100/500/2000 vs untouched-04573f0d,
untouched-955717e8, MDX23C-residual, median; step-0 == untouched-04573f0d (parity). NOT the
SHADOW judge (H4). Invoke: ./run.sh python -m audio_extract.train_classical ..."""
from __future__ import annotations
import argparse, hashlib, json, os, time, subprocess, glob
import numpy as np, soundfile as sf, torch, yaml
from demucs.states import load_model
from demucs.apply import apply_model

SR = 44100
TRAIN_DATA = "/home/mickg/train_data"
TP = "/home/mickg/truth_pairs"
CAND = "/home/mickg/classical_candidates"
CKPT_04 = "/home/mickg/.cache/torch/hub/checkpoints/04573f0d-f3cf25b2.th"
CKPT_95 = "/home/mickg/.cache/torch/hub/checkpoints/955717e8-8726e21a.th"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GATE_STEPS = [0, 100, 500, 2000]


def sha_file(p):
    return "sha256:" + hashlib.sha256(open(p, "rb").read()).hexdigest()


def readf(p, start=None, stop=None):
    a, sr = sf.read(p, dtype="float32", always_2d=True, start=start or 0, stop=stop)
    assert sr == SR, f"ABORT hidden resample {p} sr={sr}"
    return torch.from_numpy(a.T)


def to2(x):
    return x if x.shape[0] == 2 else x.repeat(2, 1)


# ---------- losses (§10, §11) ----------
def cmrstft(x, y):
    loss = 0.0
    for nfft in (512, 1024, 2048):
        w = torch.hann_window(nfft, device=x.device)
        X = torch.stft(x.reshape(-1, x.shape[-1]), nfft, nfft // 4, window=w, return_complex=True)
        Y = torch.stft(y.reshape(-1, y.shape[-1]), nfft, nfft // 4, window=w, return_complex=True)
        loss = loss + (X.abs() - Y.abs()).abs().mean() + (X.real - Y.real).abs().mean() + (X.imag - Y.imag).abs().mean()
    return loss / 3.0


def srccoord(A_hat, A, V, sample_mask):
    # per (B,ch): Y=A_hat; X=[A V]; c=(X^T X + lamI)^-1 X^T Y ; R=Y-aA-bV  (real waveform proxy of §11).
    # §11 kappa masking: only well-conditioned tiles from EXACT crops (both A,V active); controls
    # (V=0 no-vocal, A=0 vocal-only) give a singular Gram -> alpha/beta explode -> EXCLUDED.
    B, C, L = A_hat.shape
    Y = A_hat.reshape(B * C, L, 1); a = A.reshape(B * C, L, 1); v = V.reshape(B * C, L, 1)
    X = torch.cat([a, v], -1)                       # (BC, L, 2)
    G = X.transpose(1, 2) @ X + 1e-3 * torch.eye(2, device=A.device)   # (BC,2,2) always PD
    c = torch.linalg.solve(G, X.transpose(1, 2) @ Y)                   # (BC,2,1)
    alpha, beta = c[:, 0, 0], c[:, 1, 0]
    R = Y - X @ c
    ea = a.pow(2).mean(1).squeeze(-1); ev = v.pow(2).mean(1).squeeze(-1)
    with torch.no_grad():
        smask = sample_mask.view(B, 1).repeat(1, C).reshape(B * C).bool()
        kappa = torch.linalg.cond(G)
        vm = (smask & (kappa < 1e6) & (ea > 1e-8) & (ev > 1e-8)).float()   # invalid rows get weight 0 (no grad)
    den = vm.sum() + 1e-8
    wmean = lambda x: (x * vm).sum() / den
    l_alpha = wmean((alpha - 1.0) ** 2)
    l_beta = wmean(beta ** 2)
    l_baud = wmean((beta ** 2 * ev) / (alpha ** 2 * ea + 1e-6))
    l_R = wmean(R.pow(2).mean(1).squeeze(-1) / (ea + 1e-6))
    return l_alpha, l_beta, l_baud, l_R


def event_weight(M, vmask):
    amp = M.abs().mean(1)
    tutti = (amp > (amp.amax(-1, keepdim=True) * 0.5)).float()
    return (1.0 + 2.0 * vmask + 1.5 * tutti).unsqueeze(1)


# ---------- exact-reference eval (judge label math, not the judge model) ----------
def exact_metrics(Y, A, V):
    from audio_extract import judge_train as jt, judge_labels as jl, challenges as ch
    Y2 = Y.T.numpy() if isinstance(Y, torch.Tensor) else Y
    A2 = A.T.numpy() if isinstance(A, torch.Tensor) else A
    V2 = V.T.numpy() if isinstance(V, torch.Tensor) else V
    n = min(len(Y2), len(A2), len(V2))
    labs = jl.local_source_coordinate_labels(Y2[:n], A2[:n], V2[:n], tile_frames=int(0.5 * SR), hop_frames=int(0.25 * SR))
    tg = jt.label_targets(labs)
    holes = np.array([d["accompaniment_hole_db"] for d in (l.to_dict() for l in labs) if d.get("available") and d.get("accompaniment_hole_db") is not None])
    cvar = float(np.mean(np.sort(holes)[-max(1, len(holes) // 10):])) if holes.size else 0.0
    return dict(retained_voice_db_p90=round(tg["retained_voice_db_p90"], 2), retained_voice_coef_p90=round(tg["retained_voice_coef_p90"], 3),
                event_hole_db_p90=round(tg["event_hole_db_p90"], 2), event_hole_db_max=round(tg["event_hole_db_max"], 2),
                event_hole_cvar90=round(cvar, 2), alpha_error_p90=round(tg["alpha_error_p90"], 3), si_sdr_db=round(ch.si_sdr_db(Y2[:n], A2[:n]), 2))


def student_Ahat(model, M):
    vidx = model.sources.index("vocals")
    with torch.no_grad():
        V_hat = apply_model(model, M.unsqueeze(0).to(DEV), device=DEV, split=True, overlap=0.25)[0][vidx].cpu()
    n = min(V_hat.shape[1], M.shape[1])
    return (M[:, :n] - V_hat[:, :n])


_BASE_CACHE = {}
def baselines_for(work):
    if work in _BASE_CACHE:
        return _BASE_CACHE[work]
    M = readf(f"{TP}/{work}/mix_with_voice.wav"); A = readf(f"{TP}/{work}/orchestra_only.wav"); V = readf(f"{TP}/{work}/voice_ref.wav")
    out = {}
    for tag, ck in (("untouched_04573f0d", CKPT_04), ("untouched_955717e8", CKPT_95)):
        m = load_model(ck).to(DEV).eval()
        out[tag] = exact_metrics(student_Ahat(m, M), A, V); del m; torch.cuda.empty_cache()
    for tag, rec in (("mdx23c_residual", "residual_mdx23c"), ("median_ensemble", "median_mdx_mel_bs")):
        p = f"{CAND}/{work}__{rec}.flac"
        if os.path.exists(p):
            Yb = readf(p); nb = min(Yb.shape[1], A.shape[1], V.shape[1]); out[tag] = exact_metrics(Yb[:, :nb], A[:, :nb], V[:, :nb])
    _BASE_CACHE[work] = (M, A, V, out)
    return _BASE_CACHE[work]


def gate_eval(model, works):
    model.eval(); res = {}
    for w in works:
        M, A, V, base = baselines_for(w)
        A_hat = student_Ahat(model, M); n = min(A_hat.shape[1], A.shape[1], V.shape[1])
        res[w] = dict(parity=bool(A_hat.shape[1] == M.shape[1]), student=exact_metrics(A_hat[:, :n], A[:, :n], V[:, :n]), baselines=base)
    model.train(); return res


# ---------- hierarchical Gate-2 Phase-A sampler ----------
class Sampler:
    def __init__(self, crop):
        self.crop = crop
        self.by_corpus = {}
        for d in sorted(glob.glob(f"{TRAIN_DATA}/*/M.flac")):
            w = os.path.basename(os.path.dirname(d))
            corp = "donor" if w.startswith("donor") else ("freidi" if w.startswith("freidi") else "cantoria")
            self.by_corpus.setdefault(corp, []).append((w, os.path.dirname(d), sf.info(d).frames))
        self.orch = self.by_corpus.get("donor", []) + self.by_corpus.get("freidi", [])

    def _crop(self, d, fr, rng):
        st = int(rng.integers(0, max(1, fr - self.crop)))
        M = to2(readf(f"{d}/M.flac", st, st + self.crop)); A = to2(readf(f"{d}/A.flac", st, st + self.crop))
        V = to2(readf(f"{d}/V.flac", st, st + self.crop)); vm = torch.from_numpy(np.load(f"{d}/vmask.npy")[st:st + self.crop].astype("float32"))
        c = self.crop
        M, A, V = (torch.nn.functional.pad(x, (0, c - x.shape[1]))[:, :c] for x in (M, A, V))
        vm = torch.nn.functional.pad(vm, (0, c - len(vm)))[:c]
        return M, A, V, vm

    def sample(self, rng):
        r = rng.random()
        if r < 0.25 and self.orch:                        # no-vocal hard negative: M=A, V=0
            w, d, fr = self.orch[rng.integers(len(self.orch))]
            M, A, V, vm = self._crop(d, fr, rng)
            return "no_vocal", A, A, torch.zeros_like(V), torch.zeros_like(vm)
        if r < 0.40:                                       # vocal-only: M=V, A=0
            corp = list(self.by_corpus)[rng.integers(len(self.by_corpus))]
            w, d, fr = self.by_corpus[corp][rng.integers(len(self.by_corpus[corp]))]
            M, A, V, vm = self._crop(d, fr, rng)
            return "vocal_only", V, torch.zeros_like(A), V, vm
        corp = ["donor", "freidi", "cantoria"][int(rng.integers(3))]     # exact opera M,A,V (orchestra-weighted)
        pool = self.by_corpus.get(corp) or self.by_corpus["cantoria"]
        w, d, fr = pool[rng.integers(len(pool))]
        return ("exact", *self._crop(d, fr, rng))


def main():
    ap = argparse.ArgumentParser("audio-extract train classical")
    ap.add_argument("--manifest"); ap.add_argument("--split-manifest"); ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--max-step", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.run_dir, exist_ok=True)
    cfg = yaml.safe_load(open(a.config)); W = cfg["loss"]; o = cfg["optim"]
    torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False  # bit-exact step-0 parity
    eval_works = ["bologna_verdi", "bologna_puccini", "bologna_donizetti", "aalto_mozart_dry", "aalto_mozart_hall"]

    model = load_model(CKPT_04).to(DEV).train()
    parent = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=o["lr"])
    crop = model.valid_length(int(o.get("crop_s", 6) * SR))
    smp = Sampler(crop); batch = o.get("batch", 2)
    commit = subprocess.run(["git", "-C", "/home/mickg/audio-extract", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    vidx = model.sources.index("vocals")

    gates = {}; hist = []; t0 = time.time()
    for step in range(0, a.max_step + 1):
        if step in GATE_STEPS:
            print(f"[gate {step}] evaluating...", flush=True)
            gates[step] = gate_eval(model, eval_works)
            json.dump(gates, open(f"{a.run_dir}/gates.json", "w"), indent=1)
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg}, f"{a.run_dir}/ckpt_step{step}.pt")
            for w, r in gates[step].items():
                s = r["student"]; b04 = r["baselines"]["untouched_04573f0d"]; bm = r["baselines"].get("median_ensemble", {})
                print(f"  [{w}] student rv={s['retained_voice_db_p90']} hole_p90={s['event_hole_db_p90']} sisdr={s['si_sdr_db']} | 04573f0d rv={b04['retained_voice_db_p90']} hole={b04['event_hole_db_p90']} sisdr={b04['si_sdr_db']} | median sisdr={bm.get('si_sdr_db')}", flush=True)
            if step == 0:
                pv = all(abs(gates[0][w]["student"]["si_sdr_db"] - gates[0][w]["baselines"]["untouched_04573f0d"]["si_sdr_db"]) < 0.2 for w in eval_works)
                print(f"  [PARITY step0==untouched_04573f0d] {pv}", flush=True)
        if step == a.max_step:
            break
        Ms, tgtV, tgtA, VMs, kinds = [], [], [], [], []
        for _ in range(batch):
            k, M, A, V, vm = smp.sample(rng)
            Ms.append(to2(M)); tgtA.append(to2(A)); tgtV.append(to2(V)); VMs.append(vm); kinds.append(k)
        M = torch.stack(Ms).to(DEV); A = torch.stack(tgtA).to(DEV); V = torch.stack(tgtV).to(DEV); vm = torch.stack(VMs).to(DEV)
        out = model(M); V_hat = out[:, vidx]; A_hat = M - V_hat
        ew = event_weight(M, vm)
        l_exact = W["vc"] * cmrstft(V_hat, V) + W["ac"] * cmrstft(A_hat, A) + W["aw"] * (ew * (A_hat - A).abs()).mean()
        exact_mask = torch.tensor([1.0 if k == "exact" else 0.0 for k in kinds], device=DEV)
        la, lb, lbaud, lR = srccoord(A_hat, A, V, exact_mask)
        l_src = W["alpha"] * la + W["beta"] * lb + W["beta_aud"] * lbaud + W["R"] * lR
        l_stereo = W["stereo"] * ((A_hat[:, 0] - A_hat[:, 1]) - (A[:, 0] - A[:, 1])).abs().mean()
        nv = torch.tensor([1.0 if k == "no_vocal" else 0.0 for k in kinds], device=DEV).view(-1, 1, 1)
        l_ctrl = W["no_vocal"] * (nv * V_hat.abs()).mean()
        l_parent = W["parent_reg"] * sum(((p - parent[n]) ** 2).sum() for n, p in model.named_parameters()) / sum(p.numel() for p in model.parameters())
        loss = l_exact + l_src + l_stereo + l_ctrl + l_parent
        if not torch.isfinite(loss):
            json.dump({"ABORT": "non-finite", "step": step}, open(f"{a.run_dir}/ABORT.json", "w")); raise SystemExit("ABORT non-finite")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(float(loss.detach()))
        if step % 50 == 0:
            print(f"step {step} loss={loss:.4f} exact={float(l_exact):.3f} src={float(l_src):.3f} ctrl={float(l_ctrl):.3f} parent={float(l_parent):.4f}", flush=True)

    report = dict(cli="audio-extract train classical (continuation §2)", base="04573f0d (vocal-specialized HTDemucs-ft, full FT)",
                  source_commit=commit, device=DEV, n_params=sum(p.numel() for p in model.parameters()),
                  resolved_config=cfg, crop_frames=crop, batch=batch, gate_steps=GATE_STEPS,
                  train_works=sum(len(v) for v in smp.by_corpus.values()), corpora={k: len(v) for k, v in smp.by_corpus.items()},
                  convergence=dict(first50=round(float(np.mean(hist[:50])), 4), last50=round(float(np.mean(hist[-50:])), 4), all_finite=bool(all(np.isfinite(hist)))),
                  gates=gates, train_time_s=round(time.time() - t0, 1),
                  baselines=["untouched_04573f0d", "untouched_955717e8", "mdx23c_residual", "median_ensemble"],
                  repro_cmd=f"./run.sh python -m audio_extract.train_classical --config {a.config} --run-dir {a.run_dir}",
                  dataset_M_sha={corp[0][0]: sha_file(f"{corp[0][1]}/M.flac") for corp in smp.by_corpus.values()},
                  do_not_optimize_shadow_judge=True)
    json.dump(report, open(f"{a.run_dir}/run_report.json", "w"), indent=1)
    print("TRAIN_CLASSICAL_DONE")


if __name__ == "__main__":
    main()
