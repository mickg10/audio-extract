#!/usr/bin/env python
"""Generic upload pipeline driver: VAD spans -> build_track A-E -> timelines -> comparator.

Usage: auto_track.py <src_path> <base_name>
Prints STEP: lines (flushed) for the Mac orchestrator to relay into the track thread.
"""
import glob
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/home/mickg/tobacco_variants")
sys.path.insert(0, "/mnt/bigdisk/mickg/vad_train/code")
import render_tobacco as rt  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from common import SR as VSR, HOP as VHOP  # noqa: E402
from infer import load_model, infer_wave  # noqa: E402

CKPT = "/mnt/bigdisk/mickg/vad_train/ckpt/best.pt"
SCAN = "/mnt/bigdisk/mickg/vad_scan"
ENTER, EXIT, MINSEG_S, MERGEGAP_S = 0.6, 0.3, 1.0, 2.0
HOP_S = VHOP / VSR


def step(msg):
    print("STEP: " + msg, flush=True)


def spans_analyze(src, wd, base):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "quiet", "-i", str(src),
                          "-f", "f32le", "-ac", "1", "-ar", str(VSR), "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    dur = len(x) / VSR
    model, T, thr, _ = load_model(CKPT, "cuda")
    p = infer_wave(model, x, "cuda", T)
    ps = median_filter(p, 51)
    rawsp, state, start = [], False, 0
    for i, v in enumerate(ps):
        if not state and v > ENTER:
            state, start = True, i
        elif state and v < EXIT:
            rawsp.append([start, i]); state = False
    if state:
        rawsp.append([start, len(ps)])
    merged = []
    for s in rawsp:
        if merged and (s[0] - merged[-1][1]) * HOP_S < MERGEGAP_S:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    final = [s for s in merged if (s[1] - s[0]) * HOP_S >= MINSEG_S]
    spans = [{"t0_s": round(s[0] * HOP_S, 2), "t1_s": round(s[1] * HOP_S, 2),
              "mean_p": round(float(ps[s[0]:s[1]].mean()), 3)} for s in final]
    cov = sum(s["t1_s"] - s["t0_s"] for s in spans) / max(dur, 1e-9)
    mode = "full" if cov > 0.60 else "splice"
    pad = []
    for s in spans:
        a, b = max(0.0, s["t0_s"] - 0.5), min(dur, s["t1_s"] + 0.5)
        if pad and a <= pad[-1][1]:
            pad[-1][1] = b
        else:
            pad.append([a, b])
    doc = {"tracks": {base: {"duration_s": round(dur, 3), "spans": spans}},
           "decision": {"mode": mode, "coverage_frac": round(cov, 4),
                        "splice_spans_s": [[round(a, 2), round(b, 2)] for a, b in pad]}}
    (wd / "vad_spans.json").write_text(json.dumps(doc, indent=1))
    return doc


def timelines_and_comparator(wd, base):
    m4as = sorted(glob.glob(str(wd / (base + "__*.m4a"))))
    V = sys.executable
    for m in m4as:
        b = os.path.basename(m)[:-4]
        wav = f"{SCAN}/wav/{b}.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", m,
                        "-ar", "48000", "-ac", "1", wav], check=True)
        subprocess.run([V, "/mnt/bigdisk/mickg/vad_train/code/infer.py", CKPT,
                        wav, f"{SCAN}/out/{b}.npz"], capture_output=True, check=True)
    out = {}
    for p in sorted(glob.glob(SCAN + "/out/*.npz")):
        b = os.path.basename(p)[:-4]
        d = np.load(p)
        pr = d[list(d.keys())[0]].astype(float)
        ps = median_filter(pr, 51)
        n = len(ps) // 25
        out[b + ".m4a"] = {"hop": 0.25, "thr": 0.3785,
                           "p": [round(float(ps[i * 25:(i + 1) * 25].mean()), 3) for i in range(n)]}
    json.dump(out, open(SCAN + "/vad_timelines.json", "w"))
    comp = []
    for m in m4as:
        b = os.path.basename(m)[:-4]
        d = np.load(f"{SCAN}/out/{b}.npz")
        pr = d[list(d.keys())[0]].astype(float)
        ps = median_filter(pr, 51)
        comp.append({"variant": b.split("__")[1], "meanP": round(float(ps.mean()), 3)})
    comp.sort(key=lambda r: r["meanP"])
    (wd / "auto_comparator.json").write_text(json.dumps(comp, indent=1))
    return comp


def main():
    src = Path(sys.argv[1]); base = sys.argv[2]
    name = base
    wd = rt.BASE / name
    wd.mkdir(parents=True, exist_ok=True)
    step(f"analyzing voice activity of the original ({src.name})...")
    doc = spans_analyze(src, wd, base)
    tr = doc["tracks"][base]; dec = doc["decision"]
    sp = "; ".join(f"{s['t0_s']}-{s['t1_s']}s p={s['mean_p']}" for s in tr["spans"]) or "none"
    step(f"voiced spans: {sp} | coverage {round(100*dec['coverage_frac'])}% -> mode: {dec['mode'].upper()}")
    cfg = {"name": name, "base": base, "src": src, "mode": dec["mode"],
           "src_note": f"uploaded via /v; VAD spans {sp}; coverage {dec['coverage_frac']}"}
    if dec["mode"] == "splice":
        cfg["spans"] = [tuple(s) for s in dec["splice_spans_s"]]
    adapter = rt.REPO / "audio_extract" / "separate.py"
    if rt.sha256_file(adapter) != rt.ADAPTER_SHA_PINNED:
        step("FATAL: adapter sha mismatch"); sys.exit(1)
    models_present = {m: (rt.MODELS_DIR / m).exists() for m in (rt.MDX, rt.BSR, rt.MEL)}
    if not all(models_present.values()):
        step(f"FATAL: checkpoints missing {models_present}"); sys.exit(1)
    pins = rt.collect_pins(models_present)
    step("rendering variants A (mdx23c), B (ensemble-max), C (roformer), D (subtract), E (melband) on the GPU...")
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "models_present": models_present, "mode": dec["mode"],
               "tracks": {}, "track_errors": {}}
    try:
        rt.build_track(cfg, pins, models_present, summary)
    except Exception:
        tb = traceback.format_exc()
        step("FATAL: render failed: " + tb.splitlines()[-1])
        (wd / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        sys.exit(2)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (wd / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    step("render complete; running the voice detector over every variant...")
    comp = timelines_and_comparator(wd, base)
    step("detector comparator: " + ", ".join(f"{r['variant']} {r['meanP']}" for r in comp))
    step("DONE " + json.dumps({"cleanest": comp[0]["variant"] if comp else None,
                               "mode": dec["mode"], "m4a_dir": str(wd)}))


if __name__ == "__main__":
    main()
