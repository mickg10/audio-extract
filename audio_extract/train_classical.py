"""audio-extract train classical — deterministic HTDemucs student trainer on the frozen
classical-v1 manifest+splits. Stage-1 curriculum (dry exact linear stems), full asymmetric
loss family, held-out exact-reference eval vs frozen baselines, on-disk export bridge, and the
bigoracle run-artifact contract. Invoke: ./run.sh python -m audio_extract.train_classical ...

Success/abort: finite losses, exact frame/channel/rate parity, resumable checkpoints, held-out
eval vs MDX23C-residual/median baselines on the CONSTRAINED objective (retained voice below
ceiling AND orchestral hole/theft no worse AND stereo no worse). ABORT on NaN/divergence.
Does NOT optimize the SHADOW judge (gpt56 H4)."""
from __future__ import annotations
import argparse, hashlib, json, os, time, subprocess
import numpy as np, soundfile as sf, torch
import yaml
from demucs.htdemucs import HTDemucs
from demucs.apply import apply_model

SR = 44100
TRAIN_DATA = "/home/mickg/train_data"           # materialized cantoria+freidi M/A/V + vmask
TP = "/home/mickg/truth_pairs"                   # held-out Bologna/Aalto (exact, on research6)
CAND = "/home/mickg/classical_candidates"        # frozen baselines (Job 3): <work>__<recipe>.flac
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sha_arr(a):
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(np.asarray(a, "float32")).tobytes()).hexdigest()


def sha_file(p):
    return "sha256:" + hashlib.sha256(open(p, "rb").read()).hexdigest()


def readf(p, start=None, stop=None):
    a, sr = sf.read(p, dtype="float32", always_2d=True, start=start or 0, stop=stop)
    assert sr == SR, f"ABORT hidden resample {p} sr={sr}"
    return torch.from_numpy(a.T)                  # (ch, frames)


# ---------------- loss family (asymmetric; theft expensive) ----------------
def mrstft(x, y):
    loss = 0.0
    for nfft in (512, 1024, 2048):
        win = torch.hann_window(nfft, device=x.device)
        X = torch.stft(x.reshape(-1, x.shape[-1]), nfft, nfft // 4, window=win, return_complex=True)
        Y = torch.stft(y.reshape(-1, y.shape[-1]), nfft, nfft // 4, window=win, return_complex=True)
        loss = loss + (X.abs() - Y.abs()).abs().mean() + (X.real - Y.real).abs().mean() + (X.imag - Y.imag).abs().mean()
    return loss / 3.0


def loss_family(out, M, A, V, vmask, w):
    A_hat, V_hat = out[:, 0], out[:, 1]            # (B,ch,L)
    tutti = (M.abs().mean(1) > (M.abs().mean(1).amax(-1, keepdim=True) * 0.5)).float()
    ew = (1.0 + w["event_vocal"] * vmask + w["event_tutti"] * tutti).unsqueeze(1)   # (B,1,L)
    l_wav = (ew * ((A_hat - A).abs() + (V_hat - V).abs())).mean()
    l_stft = mrstft(A_hat, A) + mrstft(V_hat, V)
    l_mix = (A_hat + V_hat - M).abs().mean()
    beta = (A_hat * V).sum(-1) / (V * V).sum(-1).clamp_min(1e-6)      # voice-in-accompaniment (theft)
    l_theft = (beta ** 2).mean()
    l_fn = (vmask.unsqueeze(1) * (V_hat - V).abs()).mean()            # vocal false-negative
    alpha = (A_hat * A).sum(-1) / (A * A).sum(-1).clamp_min(1e-6)     # accompaniment scale (source-coord)
    l_alpha = ((alpha - 1.0) ** 2).mean()
    l_stereo = ((A_hat[:, 0] - A_hat[:, 1]) - (A[:, 0] - A[:, 1])).abs().mean()   # side coherence
    total = (w["wav"] * l_wav + w["stft"] * l_stft + w["mix"] * l_mix + w["theft"] * l_theft
             + w["fn"] * l_fn + w["alpha"] * l_alpha + w["stereo"] * l_stereo)
    return total, dict(wav=float(l_wav), stft=float(l_stft), mix=float(l_mix), theft=float(l_theft),
                       fn=float(l_fn), alpha=float(l_alpha), stereo=float(l_stereo))


# ---------------- exact-reference eval (uses the judge label math, not the judge model) ----------------
def exact_metrics(Y, A, V):
    from audio_extract import judge_train as jt, judge_labels as jl, challenges as ch
    import numpy as _np
    Y2 = Y.T.numpy() if isinstance(Y, torch.Tensor) else Y
    A2 = A.T.numpy() if isinstance(A, torch.Tensor) else A
    V2 = V.T.numpy() if isinstance(V, torch.Tensor) else V
    n = min(len(Y2), len(A2), len(V2))
    labs = jl.local_source_coordinate_labels(Y2[:n], A2[:n], V2[:n], tile_frames=int(0.5 * SR), hop_frames=int(0.25 * SR))
    tg = jt.label_targets(labs)
    holes = [l.to_dict()["accompaniment_hole_db"] for l in labs if getattr(l, "available", False)]
    holes = _np.array([h for h in holes if h is not None])
    cvar = float(_np.mean(_np.sort(holes)[-max(1, len(holes) // 10):])) if holes.size else 0.0
    return dict(retained_voice_db_p90=round(tg["retained_voice_db_p90"], 2),
                retained_voice_coef_p90=round(tg["retained_voice_coef_p90"], 3),
                event_hole_db_p90=round(tg["event_hole_db_p90"], 2), event_hole_db_max=round(tg["event_hole_db_max"], 2),
                event_hole_cvar90=round(cvar, 2), alpha_error_p90=round(tg["alpha_error_p90"], 3),
                si_sdr_db=round(ch.si_sdr_db(Y2[:n], A2[:n]), 2), avail_tiles=tg["_available_tiles"])


def eval_work(model, work, run):
    M = readf(f"{TP}/{work}/mix_with_voice.wav")
    A = readf(f"{TP}/{work}/orchestra_only.wav")
    V = readf(f"{TP}/{work}/voice_ref.wav")
    with torch.no_grad():
        est = apply_model(model, M.unsqueeze(0).to(DEV), device=DEV, split=True, overlap=0.25)[0].cpu()
    A_hat = est[0]
    n = min(A_hat.shape[1], M.shape[1])
    parity = bool(A_hat.shape[1] == M.shape[1] and A_hat.shape[0] == M.shape[0])
    res = {"parity": parity, "student": exact_metrics(A_hat[:, :n], A[:, :n], V[:, :n])}
    for recipe in ("residual_mdx23c", "median_mdx_mel_bs"):
        p = f"{CAND}/{work}__{recipe}.flac"
        if os.path.exists(p):
            Yb = readf(p)
            nb = min(Yb.shape[1], A.shape[1], V.shape[1])
            res[recipe] = exact_metrics(Yb[:, :nb], A[:, :nb], V[:, :nb])
    return res


# ---------------- streaming dataset (train split only) ----------------
class ClassicalDS:
    def __init__(self, works, crop):
        self.crop = crop
        self.items = []
        for w in works:
            d = f"{TRAIN_DATA}/{w}"
            if os.path.exists(f"{d}/M.flac"):
                info = sf.info(f"{d}/M.flac")
                self.items.append((w, d, info.frames))
        assert self.items, "no materialized train works found"

    def sample(self, rng):
        w, d, fr = self.items[rng.integers(len(self.items))]
        st = int(rng.integers(0, max(1, fr - self.crop)))
        M = readf(f"{d}/M.flac", st, st + self.crop)
        A = readf(f"{d}/A.flac", st, st + self.crop)
        V = readf(f"{d}/V.flac", st, st + self.crop)
        vm = torch.from_numpy(np.load(f"{d}/vmask.npy")[st:st + self.crop].astype("float32"))
        M, A, V = (x if x.shape[0] == 2 else x.repeat(2, 1) for x in (M, A, V))   # mono->stereo (cantoria)
        L = min(M.shape[1], A.shape[1], V.shape[1], len(vm))
        if L < self.crop:
            pad = self.crop - L
            M = torch.nn.functional.pad(M[:, :L], (0, pad)); A = torch.nn.functional.pad(A[:, :L], (0, pad))
            V = torch.nn.functional.pad(V[:, :L], (0, pad)); vm = torch.nn.functional.pad(vm[:L], (0, pad))
        return M[:, :self.crop], A[:, :self.crop], V[:, :self.crop], vm[:self.crop]


def export_bridge(model, cfg, run):
    """Prove a HTDemucs checkpoint round-trips through demucs' inference (== audio-separator's
    demucs_separator math): serialize -> save .th -> reload -> apply_model on a probe."""
    from demucs import states
    from omegaconf import OmegaConf
    th = f"{run}/htdemucs_export.th"
    pkg = states.serialize_model(model, OmegaConf.create({"sources": cfg["sources"], "samplerate": SR}), half=False)
    torch.save(pkg, th)
    loaded = states.load_model(th).to(DEV).eval()
    probe = torch.randn(1, 2, SR).to(DEV)
    with torch.no_grad():
        y = apply_model(loaded, probe, device=DEV, split=False)
    ok = bool(torch.isfinite(y).all() and y.shape[1] == len(cfg["sources"]))
    return dict(export_path=f"research6:{th}", export_sha256=sha_file(th), roundtrip_infer_ok=ok,
                note="reloaded via demucs.states.load_model + apply_model (audio-separator demucs_separator uses the same demucs inference); audio-separator model-registry packaging is the final step")


def main():
    ap = argparse.ArgumentParser("audio-extract train classical")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    splits = json.load(open(a.split_manifest))
    manifest = [json.loads(l) for l in open(a.manifest) if l.strip()]
    torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)

    train_splits = set(cfg["data"].get("train_splits", ["train"]))
    train_works = sorted({r["work_id"] for r in manifest
                          if splits["group_split"].get(r["group_id"]) in train_splits
                          and os.path.exists(f"{TRAIN_DATA}/{r['work_id']}/M.flac")})
    print(f"[data] train works ({len(train_works)}): {train_works}", flush=True)
    eval_works = ["bologna_verdi", "bologna_puccini", "bologna_donizetti", "aalto_mozart_dry"]

    m = cfg["model"]; o = cfg["optim"]
    crop = int(o.get("crop_s", 6) * SR)
    model = HTDemucs(sources=cfg["sources"], audio_channels=cfg["audio_channels"], samplerate=SR,
                     channels=m["channels"], depth=m["depth"]).to(DEV)
    crop = model.valid_length(crop)
    opt = torch.optim.Adam(model.parameters(), lr=o["lr"])
    ds = ClassicalDS(train_works, crop)
    W = cfg["loss"]
    steps = a.steps or o.get("steps_first_run", 1500)
    ckpt = f"{a.run_dir}/checkpoint.pt"
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=DEV); model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"]); start = s["step"]
        print(f"[resume] from step {start}", flush=True)
    else:
        start = 0
    commit = subprocess.run(["git", "-C", "/home/mickg/audio-extract", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    batch = o.get("batch", 4)
    hist = []
    evals = []
    t0 = time.time()
    for step in range(start, steps):
        Ms, As, Vs, VMs = [], [], [], []
        for _ in range(batch):
            M, A, V, vm = ds.sample(rng)
            Ms.append(M); As.append(A); Vs.append(V); VMs.append(vm)
        M = torch.stack(Ms).to(DEV); A = torch.stack(As).to(DEV); V = torch.stack(Vs).to(DEV); vm = torch.stack(VMs).to(DEV)
        out = model(M)
        loss, parts = loss_family(out, M, A, V, vm, W)
        if not torch.isfinite(loss):
            json.dump({"ABORT": "non-finite loss", "step": step, "parts": parts}, open(f"{a.run_dir}/ABORT.json", "w"))
            raise SystemExit(f"ABORT non-finite loss at step {step}")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        hist.append(float(loss))
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step} loss={loss:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()), flush=True)
        if step > 0 and (step % max(1, steps // 3) == 0 or step == steps - 1):
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step + 1, "cfg": cfg}, ckpt)
            model.eval()
            ev = {"step": step, "works": {w: eval_work(model, w, a.run_dir) for w in eval_works}}
            evals.append(ev)
            model.train()
            for w, r in ev["works"].items():
                s = r["student"]; b = r.get("residual_mdx23c", {})
                print(f"  [eval {w}] student rv_db={s['retained_voice_db_p90']} hole_p90={s['event_hole_db_p90']} "
                      f"sisdr={s['si_sdr_db']} | baseline(mdx) rv_db={b.get('retained_voice_db_p90')} hole_p90={b.get('event_hole_db_p90')} sisdr={b.get('si_sdr_db')}", flush=True)

    export = export_bridge(model, cfg, a.run_dir)
    # convergence signal: mean of first vs last 100 steps
    conv = dict(first100=round(float(np.mean(hist[:100])), 4), last100=round(float(np.mean(hist[-100:])), 4),
                decreasing=bool(np.mean(hist[-100:]) < np.mean(hist[:100])), all_finite=bool(all(np.isfinite(hist))))
    report = dict(cli="audio-extract train classical", arch="HTDemucs", source_commit=commit, device=DEV,
                  resolved_config=cfg, steps=steps, batch=batch, crop_frames=crop, n_params=sum(p.numel() for p in model.parameters()),
                  train_works=train_works, train_work_M_sha={w: sha_file(f"{TRAIN_DATA}/{w}/M.flac") for w in train_works},
                  splits_used={s: [g for g, sp in splits["group_split"].items() if sp == s] for s in train_splits},
                  loss_family=list(W.keys()), convergence=conv, evals=evals,
                  checkpoint={"path": f"research6:{ckpt}", "sha256": sha_file(ckpt), "resumable": True},
                  export_bridge=export, train_time_s=round(time.time() - t0, 1),
                  repro_cmd=f"./run.sh python -m audio_extract.train_classical --manifest {a.manifest} --split-manifest {a.split_manifest} --config {a.config} --run-dir {a.run_dir}",
                  do_not_optimize_shadow_judge=True)
    json.dump(report, open(f"{a.run_dir}/run_report.json", "w"), indent=1)
    print("RUN_REPORT", json.dumps({"convergence": conv, "export_ok": export["roundtrip_infer_ok"]}))
    print("TRAIN_CLASSICAL_DONE")


if __name__ == "__main__":
    main()
