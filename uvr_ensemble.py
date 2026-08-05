#!/usr/bin/env python3
"""Spectrogram ensemble of separated stems — reproduces UVR's min/max/avg spec.

min / max pick, per time-frequency bin, the *complex* STFT value from whichever
model has the smallest / largest magnitude there, so phase stays coherent (this
mirrors UVR's min_mag / max_mag). avg averages the waveforms.

    python uvr_ensemble.py --algo max out.wav stemA.wav stemB.wav [stemC.wav ...]
"""
import argparse
import numpy as np
import soundfile as sf
import librosa

NFFT, HOP = 4096, 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["min", "max", "avg"], default="max")
    ap.add_argument("out")
    ap.add_argument("stems", nargs="+")
    a = ap.parse_args()

    ys = [sf.read(s, always_2d=True)[0] for s in a.stems]
    sr = sf.info(a.stems[0]).samplerate
    n = min(len(y) for y in ys)                 # align to shortest
    ch = ys[0].shape[1]
    ys = [y[:n, :ch] for y in ys]

    if a.algo == "avg":
        out = np.mean(ys, axis=0)
    else:
        out = np.zeros((n, ch), dtype=np.float32)
        for c in range(ch):
            S = np.stack([librosa.stft(np.ascontiguousarray(y[:, c]),
                                       n_fft=NFFT, hop_length=HOP) for y in ys])
            mag = np.abs(S)
            idx = mag.argmin(0) if a.algo == "min" else mag.argmax(0)
            sel = np.take_along_axis(S, idx[None], axis=0)[0]   # winner keeps its phase
            out[:, c] = librosa.istft(sel, hop_length=HOP, length=n)

    sf.write(a.out, out, sr)
    print(f"wrote {a.out}  ({a.algo}-spec of {len(a.stems)} models, {sr} Hz, {ch}ch)")


if __name__ == "__main__":
    main()
