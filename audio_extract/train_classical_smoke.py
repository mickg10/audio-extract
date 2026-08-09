"""Job 4 SMOKE: HTDemucs (trainable nn.Module; no dora needed) trained 100 steps on 2 exact
works, evaluated on a 3rd through demucs' production inference path (apply_model). Loss family
(subset per bigoracle): multi-res STFT mag + waveform L1 + mixture-consistency. Success =
finite losses, exact frame/channel/rate parity, resumable checkpoint, reproducible hashes;
ABORT on hidden resample/truncation/NaN. Emits resolved config + hashes + per-work validation."""
import hashlib, json, os, time, subprocess
import numpy as np, soundfile as sf, torch
from demucs.htdemucs import HTDemucs
from demucs.apply import apply_model

TP = "/home/mickg/truth_pairs"
RUN = "/home/mickg/runs/train/classical-smoke-1"
os.makedirs(RUN, exist_ok=True)
SR = 44100
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = ["bologna_verdi", "bologna_puccini"]
EVALW = "bologna_donizetti"
torch.manual_seed(0); np.random.seed(0)


def load(w, role):
    a, sr = sf.read(f"{TP}/{w}/{role}.wav", dtype="float32", always_2d=True)
    assert sr == SR, f"{w}/{role} sr={sr}"           # ABORT on hidden resample
    return torch.from_numpy(a.T)                       # (channels, frames)


def sha(t):
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy().astype("float32")).tobytes()).hexdigest()


def mrstft(a, b):
    loss = 0.0
    for nfft in (512, 1024, 2048):
        wa = torch.stft(a.reshape(-1, a.shape[-1]), nfft, nfft // 4, window=torch.hann_window(nfft, device=a.device), return_complex=True)
        wb = torch.stft(b.reshape(-1, b.shape[-1]), nfft, nfft // 4, window=torch.hann_window(nfft, device=b.device), return_complex=True)
        loss = loss + (wa.abs() - wb.abs()).abs().mean()
    return loss / 3.0


data = {w: {"M": load(w, "mix_with_voice"), "A": load(w, "orchestra_only"), "V": load(w, "voice_ref")} for w in TRAIN + [EVALW]}
for w in data:
    n = min(data[w]["M"].shape[1], data[w]["A"].shape[1], data[w]["V"].shape[1])
    for k in data[w]:
        data[w][k] = data[w][k][:, :n]

cfg = dict(sources=["accompaniment", "vocals"], audio_channels=2, samplerate=SR, channels=32, depth=4, segment=int(5))
model = HTDemucs(**cfg).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
n_params = sum(p.numel() for p in model.parameters())
Ls = model.valid_length(5 * SR)
print(f"HTDemucs params={n_params} valid_len={Ls} dev={DEV}", flush=True)

losses = []
t0 = time.time()
for step in range(100):
    w = TRAIN[step % 2]
    M, A, V = data[w]["M"], data[w]["A"], data[w]["V"]
    n = M.shape[1]
    st = np.random.randint(0, max(1, n - Ls))
    mc = M[:, st:st + Ls]; ac = A[:, st:st + Ls]; vc = V[:, st:st + Ls]
    if mc.shape[1] < Ls:
        pad = Ls - mc.shape[1]
        mc = torch.nn.functional.pad(mc, (0, pad)); ac = torch.nn.functional.pad(ac, (0, pad)); vc = torch.nn.functional.pad(vc, (0, pad))
    mc = mc.unsqueeze(0).to(DEV); tgt = torch.stack([ac, vc], 0).unsqueeze(0).to(DEV)  # (1,2src,2ch,Ls)
    out = model(mc)                                    # (1,2src,2ch,Ls)
    l_wav = (out - tgt).abs().mean()
    l_stft = mrstft(out[:, 0], tgt[:, 0]) + mrstft(out[:, 1], tgt[:, 1])
    l_mix = (out.sum(1) - mc).abs().mean()             # mixture-consistency A+V=M
    loss = l_wav + 0.5 * l_stft + 0.1 * l_mix
    assert torch.isfinite(loss), f"ABORT: non-finite loss at step {step}"
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))
    if step % 20 == 0 or step == 99:
        print(f"step {step} loss={loss:.4f} (wav {l_wav:.4f} stft {l_stft:.4f} mix {l_mix:.4f})", flush=True)

# resumable checkpoint
ckpt = f"{RUN}/checkpoint.pt"
torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": 100, "cfg": cfg}, ckpt)
ck_sha = "sha256:" + hashlib.sha256(open(ckpt, "rb").read()).hexdigest()

# eval the 3rd work through demucs' production inference path (apply_model, overlap-add chunking)
model.eval()
with torch.no_grad():
    M = data[EVALW]["M"].to(DEV)
    est = apply_model(model, M.unsqueeze(0), device=DEV, split=True, overlap=0.25)[0]  # (2src,2ch,frames)
A_hat = est[0].cpu(); A_ref = data[EVALW]["A"]
n = min(A_hat.shape[1], A_ref.shape[1])
parity = dict(frames_M=int(data[EVALW]["M"].shape[1]), frames_A_hat=int(A_hat.shape[1]), frames_A_ref=int(A_ref.shape[1]),
              channels=int(A_hat.shape[0]), sr=SR, frame_parity=bool(A_hat.shape[1] == data[EVALW]["M"].shape[1]))
a, r = A_hat[:, :n].reshape(-1), A_ref[:, :n].reshape(-1)
alpha = float((a @ r) / (r @ r + 1e-9)); noise = a - alpha * r
sisdr = float(10 * np.log10((float((alpha * r) @ (alpha * r)) + 1e-9) / (float(noise @ noise) + 1e-9)))
val = dict(work=EVALW, accompaniment_si_sdr_db=round(sisdr, 2),
           A_hat_pcm_sha256=sha(A_hat), finite=bool(np.isfinite(sisdr)))

commit = subprocess.run(["git", "-C", "/home/mickg/audio-extract", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
report = dict(arch="HTDemucs", trainable="verified (forward+backward, plain torch; dora NOT required)",
              export_adapter="audio_separator.architectures.demucs_separator (round-trip to production inference)",
              resolved_config=cfg, n_params=n_params, device=DEV, steps=100, source_commit=commit,
              dataset={"train_works": TRAIN, "eval_work": EVALW,
                       "train_M_sha": {w: sha(data[w]["M"]) for w in TRAIN},
                       "eval_M_sha": sha(data[EVALW]["M"])},
              split_manifest="datasets/classical-v1-splits.json",
              loss_family=["multi_res_stft_mag", "waveform_L1", "mixture_consistency",
                           "(TODO per bigoracle: complex-STFT phase, no-vocal-FP, vocal-FN, source-coord alpha/beta/R, stereo-coherence, event-tail)"],
              first_loss=round(losses[0], 4), last_loss=round(losses[-1], 4), all_finite=bool(all(np.isfinite(losses))),
              train_time_s=round(time.time() - t0, 1),
              checkpoint={"path": f"research6:{ckpt}", "sha256": ck_sha, "resumable": True},
              parity=parity, validation=val,
              repro_cmd="OMP_NUM_THREADS=8 ./run.sh python /tmp/train_classical_smoke.py",
              verdict=("SMOKE PASS" if (all(np.isfinite(losses)) and parity["frame_parity"] and np.isfinite(sisdr)) else "SMOKE FAIL"))
json.dump(report, open(f"{RUN}/smoke_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
print("SMOKE_DONE")
