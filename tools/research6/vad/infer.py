#!/usr/bin/env python
"""Long-file inference: chunked 8 s windows, hop 4 s, center-stitched, per-chunk CMVN
(matches training). Usage: infer.py <ckpt> <audio> <out.npz> [--cpu]"""
import os
import sys
import time
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SR, HOP
from model import CRNN

CHUNK = 8 * SR
HOPC = 4 * SR
TFR = CHUNK // HOP + 1  # 801


def load_audio_48k_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CRNN().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, float(ck.get("temperature", 1.0)), float(ck.get("threshold", 0.5)), ck


@torch.no_grad()
def infer_wave(model, x, device, temperature=1.0, batch=16, use_amp=True):
    n = len(x)
    nfr_total = n // HOP + 1
    starts = list(range(0, max(1, n - CHUNK) + 1, HOPC))
    if starts[-1] < n - CHUNK:
        starts.append(n - CHUNK)
    if n < CHUNK:
        x = np.pad(x, (0, CHUNK - n))
        starts = [0]
    out = np.zeros(nfr_total, dtype=np.float32)
    got = np.zeros(nfr_total, dtype=bool)
    for b0 in range(0, len(starts), batch):
        bs = starts[b0: b0 + batch]
        wav = torch.from_numpy(np.stack([x[s: s + CHUNK] for s in bs])).to(device)
        if use_amp and device != "cpu":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(wav)
        else:
            logits = model(wav)
        probs = torch.sigmoid(logits.float() / temperature).cpu().numpy()
        for i, s in enumerate(bs):
            fs = s // HOP
            ta = 0 if s == 0 else 200
            tb = TFR if s >= n - CHUNK else 600
            sl = slice(fs + ta, min(fs + tb, nfr_total))
            m = sl.stop - sl.start
            if m > 0:
                out[sl] = probs[i, ta: ta + m]
                got[sl] = True
    if not got.all():
        out[~got] = out[got][-1] if got.any() else 0.0
    return out


def main():
    ckpt, audio, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    device = "cpu" if "--cpu" in sys.argv else "cuda"
    if device == "cpu":
        torch.set_num_threads(8)
    model, T, thr, ck = load_model(ckpt, device)
    x = load_audio_48k_mono(audio)
    if device != "cpu":
        infer_wave(model, x[: 10 * SR], device, T)  # warmup
        torch.cuda.synchronize()
    t0 = time.time()
    p = infer_wave(model, x, device, T)
    if device != "cpu":
        torch.cuda.synchronize()
    dt = time.time() - t0
    dur = len(x) / SR
    np.savez_compressed(outp, prob=p.astype(np.float16), hop_s=HOP / SR,
                        threshold=thr, temperature=T)
    print(f"{os.path.basename(audio)}: {dur:.1f}s audio, {dt:.2f}s infer on {device}, "
          f"RTF {dur/dt:.1f}x, mean_p {p.mean():.3f}")


if __name__ == "__main__":
    main()
