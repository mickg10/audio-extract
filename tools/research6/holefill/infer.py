"""Chunked overlap-add inference for long files. Windows 8 s, hop 6 s, triangular
crossfade in the 2 s overlap; per-window RMS normalization (matches training)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_common as H
from model import HoleFiller

WIN_S, HOP_S = 8.0, 6.0


def load_model(ckpt_path, dev="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HoleFiller()
    sd = ck.get("ema") or ck["model"]
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(dev).eval()
    return model


@torch.no_grad()
def process(model, x, dev="cuda", batch=4):
    """x [n, 2] float32 44.1k -> y [n, 2] float32"""
    n = len(x)
    win = int(WIN_S * H.SR)
    hop = int(HOP_S * H.SR)
    ov = win - hop
    starts = list(range(0, max(1, n - ov), hop))
    w = np.ones(win, dtype=np.float32)
    r = np.arange(1, ov + 1, dtype=np.float32) / (ov + 1)
    w[:ov] = r
    w[-ov:] = np.minimum(w[-ov:], r[::-1])
    out = np.zeros((n, 2), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for b0 in range(0, len(starts), batch):
        bs = starts[b0:b0 + batch]
        chunks, scales, lens = [], [], []
        for s in bs:
            seg = x[s:s + win]
            lens.append(len(seg))
            if len(seg) < win:
                seg = np.pad(seg, ((0, win - len(seg)), (0, 0)))
            rms = float(np.sqrt(np.mean(seg ** 2)))
            sc = 1.0 / max(rms, 1e-4)
            chunks.append(seg.T * sc)
            scales.append(sc)
        xb = torch.from_numpy(np.stack(chunks).astype(np.float32)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yb = model(xb)
        yb = yb.float().cpu().numpy()
        for j, s in enumerate(bs):
            L = lens[j]
            wl = w[:L].copy()
            if s == 0:
                wl[:ov] = 1.0  # no left neighbor
            y = (yb[j].T[:L] / scales[j]) * wl[:, None]
            out[s:s + L] += y
            wsum[s:s + L] += wl
    wsum[wsum == 0] = 1.0
    return (out / wsum[:, None]).astype(np.float32)


def process_file(model, in_path, out_path, dev="cuda"):
    import soundfile as sf
    x, sr = sf.read(in_path, dtype="float32", always_2d=True)
    assert sr == H.SR, f"expected 44100, got {sr}"
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    y = process(model, x, dev)
    H.write_f32(out_path, y)
    return y
