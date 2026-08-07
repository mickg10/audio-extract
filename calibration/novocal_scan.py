#!/usr/bin/env python3
"""Flag library tracks with >=3 s orchestral-only (vocal-free) spans.

Reuses the passage miner's *defining* no_vocal_control gate (audio_extract.passages):
a genuine vocal-absent span must satisfy the ABSOLUTE gate
    _vocal_ratio(span) = rms(vocal) / rms(vocal + accomp) <= no_vocal_abs_ratio_max (0.15)
(see passages.py lines 293-320). We compute a continuous per-frame version of that
ratio with the repo's dsp.frame_rms_db framing (win40/hop10), find contiguous runs
that stay vocal-absent for >= MIN_SPAN_S, then re-confirm each run with the exact
span-level _vocal_ratio formula. Vocal stem = the separated roformer/mdx vocals
(the miner also operates on a *provisional* vocal stem); mix = vocal + inverted
(exact-complement) instrumental so rms(v+a) == rms(original mix).
"""
import sys, glob, os, json
import numpy as np
import soundfile as sf
sys.path.insert(0, "/Users/mickg10/audio-extract")
from audio_extract import dsp
from audio_extract.passages import MinerConfig

cfg = MinerConfig()
RATIO_MAX = cfg.no_vocal_abs_ratio_max      # 0.15, the miner's absolute vocal-absence gate
MIN_SPAN_S = 3.0                            # task: >=3 s orchestral-only spans
WIN_MS, HOP_MS = 40.0, 10.0

VOC_PRI = ["roformer_vocals", "mdx23c_vocals", "vocals"]
ACC_PRI = ["inverted_instrumental", "ensemble_max_instrumental", "mdx23c_instrumental",
           "deecho_dereverb_instrumental", "denoise_instrumental", "instrumental"]

def pick(outdir, pri):
    for tag in pri:
        hits = sorted(glob.glob(os.path.join(outdir, f"*{tag}*.wav")))
        if hits:
            return hits[0]
    return None

def load_mono(path):
    x, sr = sf.read(path, always_2d=True)
    return dsp.mono(x.astype(np.float64)), sr

def scan(track_dir):
    outd = os.path.join(track_dir, "out")
    vp, ap = pick(outd, VOC_PRI), pick(outd, ACC_PRI)
    if not vp or not ap:
        return {"track": os.path.basename(track_dir), "error": "no stem pair"}
    v, sr = load_mono(vp)
    a, sr2 = load_mono(ap)
    n = min(len(v), len(a)); v, a = v[:n], a[:n]
    if sr != sr2:
        return {"track": os.path.basename(track_dir), "error": f"sr mismatch {sr}/{sr2}"}
    dur = n / sr
    # per-frame linear RMS via the repo's dsp framing (dB -> linear)
    v_db = dsp.frame_rms_db(v, sr, WIN_MS, HOP_MS)
    m_db = dsp.frame_rms_db(v + a, sr, WIN_MS, HOP_MS)
    T = min(len(v_db), len(m_db))
    v_lin = 10.0 ** (v_db[:T] / 20.0)
    m_lin = 10.0 ** (m_db[:T] / 20.0)
    r = v_lin / (m_lin + 1e-12)             # continuous per-frame _vocal_ratio proxy
    hop = int(HOP_MS / 1000.0 * sr)
    inactive = r <= RATIO_MAX
    min_frames = int(MIN_SPAN_S * 1000.0 / HOP_MS)

    def span_vocal_ratio(s0, s1):           # exact passages.py _vocal_ratio (samples)
        vs = v[s0:s1]; as_ = a[s0:s1]
        ev = float(np.sqrt(np.mean(vs ** 2))) if vs.size else 0.0
        em = float(np.sqrt(np.mean((vs + as_[:len(vs)]) ** 2))) if vs.size else 1.0
        return ev / (em + 1e-12)

    spans = []
    i = 0
    while i < T:
        if inactive[i]:
            j = i
            while j < T and inactive[j]:
                j += 1
            if j - i >= min_frames:
                s0, s1 = i * hop, min(n, j * hop)
                if span_vocal_ratio(s0, s1) <= RATIO_MAX:     # re-confirm exact gate
                    spans.append((round(s0 / sr, 2), round(s1 / sr, 2),
                                  round((s1 - s0) / sr, 2)))
            i = j
        else:
            i += 1
    total = round(sum(s[2] for s in spans), 1)
    longest = round(max((s[2] for s in spans), default=0.0), 1)
    return {"track": os.path.basename(track_dir), "dur_s": round(dur, 1),
            "n_novocal_spans>=3s": len(spans), "novocal_total_s": total,
            "longest_span_s": longest, "vocal_stem": os.path.basename(vp),
            "spans": spans}

rows = []
for d in sorted(glob.glob("/Users/mickg10/audio-extract/lib/*")):
    if os.path.isdir(os.path.join(d, "out")):
        rows.append(scan(d))

rows_ok = [r for r in rows if "error" not in r]
rows_ok.sort(key=lambda r: r["novocal_total_s"], reverse=True)
print(json.dumps({"rows": rows_ok + [r for r in rows if "error" in r]}, indent=1))
have = [r for r in rows_ok if r["n_novocal_spans>=3s"] > 0]
print(f"\n# TRACKS WITH >=3s VOCAL-FREE SPANS: {len(have)}/{len(rows_ok)}", file=sys.stderr)
for r in have:
    print(f"  {r['track']}: {r['n_novocal_spans>=3s']} spans, "
          f"{r['novocal_total_s']}s total, longest {r['longest_span_s']}s", file=sys.stderr)
