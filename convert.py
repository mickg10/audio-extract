#!/usr/bin/env python3
"""Batch stem-separation pipeline with a per-file library, manifest, and viz assets.

Layout produced (consumed by the web GUI):

  lib/
    index.json                       # summary of every file for the listing page
    <slug>/
      original.<ext>                 # HARD LINK to the input audio (mp3/wav only)
      manifest.jsonl                 # one JSON line per conversion/experiment
      out/
        <exp>.wav                    # each experiment's audio output
        <exp>.png                    # spectrogram
        <exp>.peaks.json             # downsampled waveform envelope for the UI
      original.png / original.peaks.json

Design:
  * Separations use the audio_separator API, LOADING EACH MODEL ONCE and running
    it over every file (models are heavy; per-file CLI reload would be wasteful).
  * Ensemble / diff / spectrogram / peaks are pure numpy+librosa.
  * Idempotent: an experiment whose output already exists is skipped, so re-runs
    pick up newly-downloaded de-echo/de-reverb models without redoing work.

manifest.jsonl line schema (the GUI contract — see SCHEMA.md):
  {
    "id": "ensemble_max_instrumental",
    "kind": "separate|ensemble|diff",
    "stem": "Instrumental|Vocals|residual|...",
    "description": "human text",
    "models": ["..."],            # models involved
    "flags": {...},               # e.g. {"algo":"max","segment_size":256}
    "input": "original" | "<exp id>" | ["<exp a>","<exp b>"],
    "output": "out/<id>.wav",
    "spectrogram": "out/<id>.png",
    "peaks": "out/<id>.peaks.json",
    "status": "done|error",
    "duration_s": 94.4, "sr": 44100, "channels": 2,
    "peak": 0.52, "rms": 0.033, "lufs_approx": -20.1
  }
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import traceback
import numpy as np
import soundfile as sf
import librosa
import librosa.display  # noqa: F401  (registers specshow)

ROOT = os.path.expanduser("~/audio-extract")
LIB = os.path.join(ROOT, "lib")
MODELS = os.path.join(ROOT, "models")
AUDIO_EXTS = (".mp3", ".wav")

ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MDX23C = "MDX23C-8KFFT-InstVoc_HQ.ckpt"
# De-reverb / de-echo models (used only if present in models/); stem name we keep.
CLEANUP_MODELS = [
    ("UVR-De-Echo-Normal.pth", "deecho", "No Echo", "De-Echo only (echo removed, reverb kept)"),
    ("Reverb_HQ_By_FoxJoy.onnx", "dereverb", "No Reverb", "De-Reverb only (hall reverb removed, echo kept)"),
    ("UVR-DeEcho-DeReverb.pth", "deecho_dereverb", "No Reverb", "De-Echo + De-Reverb (removes both)"),
    ("UVR-DeNoise.pth", "denoise", "No Noise", "De-Noise (hiss/artefact cleanup)"),
]

NFFT, HOP = 4096, 1024
NORM_DBFS = float(os.environ.get("NORM_DBFS", "-5"))   # peak-normalize instrumentals to this


# ------------------------------------------------------------------ helpers
def slugify(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")


EXCLUDE_DIRS = {".venv", "lib", "models", "output", "drive_pull", "__pycache__", ".git"}


def discover_inputs():
    """mp3/wav under input/ (recursive) + project-root top level, deduped by (name,size)."""
    seen, files = set(), []

    def add(p):
        n = os.path.basename(p)
        if n.startswith(".") or not n.lower().endswith(AUDIO_EXTS):
            return
        key = (n, os.path.getsize(p))
        if key not in seen:
            seen.add(key); files.append(p)

    inp = os.path.join(ROOT, "input")
    for dirpath, dirs, names in os.walk(inp):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for n in names:
            add(os.path.join(dirpath, n))
    for n in os.listdir(ROOT):                       # root top level only (not recursive)
        add(os.path.join(ROOT, n))
    return sorted(files)


def hardlink_original(src, slugdir):
    ext = os.path.splitext(src)[1].lower()
    dst = os.path.join(slugdir, "original" + ext)
    if not os.path.exists(dst):
        try:
            os.link(src, dst)          # hard link (same volume)
        except OSError:
            import shutil; shutil.copy2(src, dst)
    return dst


def load_audio(path, sr=44100):
    y, _sr = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y  # [ch, n]


def write_wav(path, y_ch_n, sr=44100):
    sf.write(path, y_ch_n.T.astype(np.float32), sr)


def metrics(y):
    peak = float(np.abs(y).max()) if y.size else 0.0
    rms = float(np.sqrt((y ** 2).mean())) if y.size else 0.0
    lufs = 20 * np.log10(rms + 1e-9)  # crude loudness proxy
    return {"peak": round(peak, 4), "rms": round(rms, 5), "lufs_approx": round(lufs, 1)}


def make_peaks(y, out, buckets=1600):
    """Downsampled min/max envelope per channel for the waveform UI."""
    mono = y.mean(0)
    n = len(mono)
    step = max(1, n // buckets)
    mins, maxs = [], []
    for i in range(0, n, step):
        seg = mono[i:i + step]
        if len(seg):
            mins.append(round(float(seg.min()), 4)); maxs.append(round(float(seg.max()), 4))
    json.dump({"min": mins, "max": maxs}, open(out, "w"))


def make_spectrogram(y, out, sr=44100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mono = y.mean(0)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=NFFT, hop_length=HOP)), ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=90)
    librosa.display.specshow(S, sr=sr, hop_length=HOP, x_axis="time", y_axis="log", ax=ax, cmap="magma")
    ax.set_ylim(20, sr / 2); ax.margins(0)
    fig.tight_layout(pad=0.3); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def assets(y, base_noext, sr=44100):
    make_peaks(y, base_noext + ".peaks.json")
    try:
        make_spectrogram(y, base_noext + ".png", sr)
    except Exception as e:
        print(f"    spectrogram failed: {e}", flush=True)


def align2(a, b):
    n = min(a.shape[1], b.shape[1])
    return a[:, :n], b[:, :n]


# ------------------------------------------------------------------ manifest
def read_manifest(slugdir):
    p = os.path.join(slugdir, "manifest.jsonl")
    done = {}
    if os.path.exists(p):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln); done[r["id"]] = r
                except Exception:
                    pass
    return done


def write_manifest(slugdir, records):
    p = os.path.join(slugdir, "manifest.jsonl")
    with open(p, "w") as f:
        for r in records.values():
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------ pipeline
def separate_all(inputs, model, tag):
    """Load `model` once, separate every input; return {slug: {stem: wavpath}}."""
    from audio_separator.separator import Separator
    results = {}
    if not os.path.exists(os.path.join(MODELS, model)):
        print(f"model {model} not present; skipping {tag}", flush=True)
        return results
    # skip the (slow) model load entirely if every input is already separated
    def _done(slug):
        od = os.path.join(LIB, slug, "out")
        return all(os.path.exists(os.path.join(od, f"{tag}_{s}.wav")) for s in ("instrumental", "vocals"))
    if inputs and all(_done(slug) for _, slug in inputs):
        for _, slug in inputs:
            od = os.path.join(LIB, slug, "out")
            results[slug] = {f"{tag}_instrumental": os.path.join(od, f"{tag}_instrumental.wav"),
                             f"{tag}_vocals": os.path.join(od, f"{tag}_vocals.wav")}
        print(f"=== {tag}: all {len(inputs)} already separated (no model load) ===", flush=True)
        return results
    print(f"=== loading {model} ({tag}) ===", flush=True)
    sep = Separator(output_dir="/tmp/_sep", output_format="WAV", model_file_dir=MODELS)
    sep.load_model(model_filename=model)
    for src, slug in inputs:
        outdir = os.path.join(LIB, slug, "out"); os.makedirs(outdir, exist_ok=True)
        want = {f"{tag}_instrumental": None, f"{tag}_vocals": None}
        if all(os.path.exists(os.path.join(outdir, k + ".wav")) for k in want):
            results.setdefault(slug, {})
            results[slug][f"{tag}_instrumental"] = os.path.join(outdir, f"{tag}_instrumental.wav")
            results[slug][f"{tag}_vocals"] = os.path.join(outdir, f"{tag}_vocals.wav")
            print(f"  [skip] {slug}", flush=True); continue
        try:
            os.makedirs("/tmp/_sep", exist_ok=True)
            outs = sep.separate(src)
            results.setdefault(slug, {})
            for op in outs:
                low = op.lower()
                canonical = None
                if "(instrumental)" in low or "_instrumental" in low or "no vocals" in low:
                    canonical = f"{tag}_instrumental"
                elif "(vocals)" in low or "_vocals" in low:
                    canonical = f"{tag}_vocals"
                if canonical:
                    dst = os.path.join(outdir, canonical + ".wav")
                    srcp = op if os.path.isabs(op) else os.path.join("/tmp/_sep", op)
                    if os.path.exists(srcp):
                        os.replace(srcp, dst); results[slug][canonical] = dst
            print(f"  [ok] {slug}: {list(results[slug])}", flush=True)
        except Exception as e:
            print(f"  [ERR] {slug}: {e}", flush=True)
    return results


def build_ensembles(inputs):
    """Max-spec ensemble of the two instrumentals for each file (Phase A.5)."""
    for _src, slug in inputs:
        outdir = os.path.join(LIB, slug, "out")
        ri = os.path.join(outdir, "roformer_instrumental.wav")
        mi = os.path.join(outdir, "mdx23c_instrumental.wav")
        ens = os.path.join(outdir, "ensemble_max_instrumental.wav")
        if not (os.path.exists(ri) and os.path.exists(mi)) or os.path.exists(ens):
            continue
        a, b = align2(load_audio(ri), load_audio(mi))
        out = np.zeros_like(a)
        for c in range(a.shape[0]):
            S = np.stack([librosa.stft(a[c], n_fft=NFFT, hop_length=HOP),
                          librosa.stft(b[c], n_fft=NFFT, hop_length=HOP)])
            idx = np.abs(S).argmax(0)
            out[c] = librosa.istft(np.take_along_axis(S, idx[None], 0)[0], hop_length=HOP, length=a.shape[1])
        write_wav(ens, out)
        print(f"  [ensemble] {slug}", flush=True)


def cleanup_all(inputs, model, tag, keep_stem):
    """Run a de-echo/de-reverb model on each file's ensemble instrumental."""
    from audio_separator.separator import Separator
    results = {}
    if not os.path.exists(os.path.join(MODELS, model)):
        return results
    print(f"=== loading {model} ({tag}) ===", flush=True)
    sep = Separator(output_dir="/tmp/_sep", output_format="WAV", model_file_dir=MODELS)
    sep.load_model(model_filename=model)
    for _src, slug in inputs:
        outdir = os.path.join(LIB, slug, "out")
        base = os.path.join(outdir, "ensemble_max_instrumental.wav")
        if not os.path.exists(base):
            continue
        dst = os.path.join(outdir, f"{tag}_instrumental.wav")
        if os.path.exists(dst):
            results[slug] = dst; continue
        try:
            os.makedirs("/tmp/_sep", exist_ok=True)
            outs = sep.separate(base)
            pick = None
            for op in outs:
                if keep_stem.lower().replace(" ", "") in op.lower().replace(" ", "") or "no" in op.lower():
                    pick = op; break
            pick = pick or (outs[0] if outs else None)
            if pick:
                srcp = pick if os.path.isabs(pick) else os.path.join("/tmp/_sep", pick)
                if os.path.exists(srcp):
                    os.replace(srcp, dst); results[slug] = dst
            print(f"  [ok] {tag} {slug}", flush=True)
        except Exception as e:
            print(f"  [ERR] {tag} {slug}: {e}", flush=True)
    return results


def main(only=None):
    os.makedirs(LIB, exist_ok=True)
    if only:                                   # single-file / subset job
        raw = [os.path.abspath(p) for p in only if os.path.exists(p)]
    else:
        raw = discover_inputs()
        lim = int(os.environ.get("LIMIT", "0"))
        if lim:
            raw = raw[:lim]
    inputs = [(p, slugify(p)) for p in raw]
    print(f"JOB: processing {len(inputs)} file(s)" if only
          else f"discovered {len(inputs)} audio files (mp3/wav)", flush=True)
    for src, slug in inputs:
        sd = os.path.join(LIB, slug); os.makedirs(os.path.join(sd, "out"), exist_ok=True)
        hardlink_original(src, sd)

    # Phase A: separations, one model load each
    rof = separate_all(inputs, ROFORMER, "roformer")
    mdx = separate_all(inputs, MDX23C, "mdx23c")

    # Phase A.5: build the max-spec instrumental ensemble for every file
    build_ensembles(inputs)

    # Phase A.6: de-echo / de-reverb passes on the ensemble (only models present run)
    for model, tag, keep, _desc in CLEANUP_MODELS:
        cleanup_all(inputs, model, tag, keep)

    # Phase B: per-file derive (diffs, cleanup diffs) + assets + manifest
    index = []
    for src, slug in inputs:
        sd = os.path.join(LIB, slug); outdir = os.path.join(sd, "out")
        rec = read_manifest(sd)
        try:
            # the AUDIO original specifically (not original.png / .peaks.json, which
            # assets() creates — picking by listdir order grabbed the png otherwise)
            orig_name = next(f for f in os.listdir(sd)
                             if f.startswith("original.") and f.lower().endswith(AUDIO_EXTS))
            orig = load_audio(os.path.join(sd, orig_name))
        except (StopIteration, Exception) as e:
            print(f"load original failed {slug}: {e}"); continue
        if not os.path.exists(os.path.join(sd, "original.png")):
            assets(orig, os.path.join(sd, "original"))
        rec["original"] = {"id": "original", "kind": "source", "stem": "original",
                           "description": "Original mix", "output": orig_name,
                           "spectrogram": "original.png", "peaks": "original.peaks.json",
                           "status": "done", **audio_meta(orig)}

        # canonical separation outputs
        def add_sep(tagdict, tag, model):
            for stem in ("instrumental", "vocals"):
                key = f"{tag}_{stem}"; wav = os.path.join(outdir, key + ".wav")
                if os.path.exists(wav):
                    if key not in rec or not os.path.exists(os.path.join(outdir, key + ".png")):
                        y = load_audio(wav); assets(y, os.path.join(outdir, key))
                    else:
                        y = load_audio(wav)
                    rec[key] = {"id": key, "kind": "separate", "stem": stem.capitalize(),
                                "description": f"{model} — {stem}", "models": [model],
                                "flags": {"segment_size": 256}, "input": "original",
                                "output": f"out/{key}.wav", "spectrogram": f"out/{key}.png",
                                "peaks": f"out/{key}.peaks.json", "status": "done", **audio_meta(y)}
        add_sep(rof, "roformer", ROFORMER)
        add_sep(mdx, "mdx23c", MDX23C)

        # ensemble max of the two instrumentals (built in Phase A.5)
        ens = os.path.join(outdir, "ensemble_max_instrumental.wav")
        if os.path.exists(ens) and not os.path.exists(os.path.join(outdir, "ensemble_max_instrumental.png")):
            assets(load_audio(ens), os.path.join(outdir, "ensemble_max_instrumental"))
        if os.path.exists(ens):
            y = load_audio(ens)
            rec["ensemble_max_instrumental"] = {"id": "ensemble_max_instrumental", "kind": "ensemble",
                "stem": "Instrumental", "description": "Max-spec ensemble of BS-Roformer + MDX23C instrumentals",
                "models": [ROFORMER, MDX23C], "flags": {"algo": "max"},
                "input": ["roformer_instrumental", "mdx23c_instrumental"],
                "output": "out/ensemble_max_instrumental.wav", "spectrogram": "out/ensemble_max_instrumental.png",
                "peaks": "out/ensemble_max_instrumental.peaks.json", "status": "done", **audio_meta(y)}

        # diff: original - ensemble instrumental = extracted voice + reverb
        if os.path.exists(ens):
            dpath = os.path.join(outdir, "residual_voice.wav")
            if not os.path.exists(dpath):
                a, b = align2(orig, load_audio(ens)); d = a - b
                write_wav(dpath, d); assets(d, os.path.join(outdir, "residual_voice"))
            y = load_audio(dpath)
            rec["residual_voice"] = {"id": "residual_voice", "kind": "diff", "stem": "residual",
                "description": "Original − instrumental ensemble (the removed voice + hall reverb)",
                "flags": {"op": "subtract"}, "input": ["original", "ensemble_max_instrumental"],
                "output": "out/residual_voice.wav", "spectrogram": "out/residual_voice.png",
                "peaks": "out/residual_voice.peaks.json", "status": "done", **audio_meta(y)}

        # Spectral-inversion instrumental: original − BS-Roformer vocal. Because it
        # SUBTRACTS the isolated voice from the untouched original (no mask applied to
        # the orchestra), the instrument levels/dynamics are preserved exactly — no
        # volume pumping under operatic peaks. This is the guide's "invert" trick.
        rv = os.path.join(outdir, "roformer_vocals.wav")
        if os.path.exists(rv):
            inv = os.path.join(outdir, "inverted_instrumental.wav")
            if not os.path.exists(inv):
                a, b = align2(orig, load_audio(rv)); d = a - b
                write_wav(inv, d); assets(d, os.path.join(outdir, "inverted_instrumental"))
            y = load_audio(inv)
            rec["inverted_instrumental"] = {"id": "inverted_instrumental", "kind": "separate",
                "stem": "Instrumental",
                "description": "Spectral inversion — original − BS-Roformer vocal "
                               "(orchestral dynamics untouched, no volume pumping)",
                "models": [ROFORMER], "flags": {"method": "spectral_inversion"},
                "input": ["original", "roformer_vocals"],
                "output": "out/inverted_instrumental.wav",
                "spectrogram": "out/inverted_instrumental.png",
                "peaks": "out/inverted_instrumental.peaks.json", "status": "done", **audio_meta(y)}

        # cleanup outputs (added when models exist) + "reverb removed" diff
        for model, tag, keep, desc in CLEANUP_MODELS:
            wav = os.path.join(outdir, f"{tag}_instrumental.wav")
            if os.path.exists(wav):
                if not os.path.exists(os.path.join(outdir, f"{tag}_instrumental.png")):
                    assets(load_audio(wav), os.path.join(outdir, f"{tag}_instrumental"))
                y = load_audio(wav)
                rec[f"{tag}_instrumental"] = {"id": f"{tag}_instrumental", "kind": "separate",
                    "stem": "Instrumental", "description": desc, "models": [model],
                    "input": "ensemble_max_instrumental", "output": f"out/{tag}_instrumental.wav",
                    "spectrogram": f"out/{tag}_instrumental.png", "peaks": f"out/{tag}_instrumental.peaks.json",
                    "status": "done", **audio_meta(y)}
                # reverb-removed diff = ensemble - cleaned
                dr = os.path.join(outdir, f"{tag}_removed.wav")
                if os.path.exists(ens) and not os.path.exists(dr):
                    a, b = align2(load_audio(ens), y); d = a - b
                    write_wav(dr, d); assets(d, os.path.join(outdir, f"{tag}_removed"))
                if os.path.exists(dr):
                    yy = load_audio(dr)
                    rec[f"{tag}_removed"] = {"id": f"{tag}_removed", "kind": "diff", "stem": "residual",
                        "description": f"What {desc.split('(')[0].strip()} removed (ensemble − cleaned)",
                        "flags": {"op": "subtract"}, "input": ["ensemble_max_instrumental", f"{tag}_instrumental"],
                        "output": f"out/{tag}_removed.wav", "spectrogram": f"out/{tag}_removed.png",
                        "peaks": f"out/{tag}_removed.peaks.json", "status": "done", **audio_meta(yy)}

        write_manifest(sd, rec)
        index.append({"slug": slug, "title": os.path.splitext(os.path.basename(src))[0],
                      "original": rec["original"]["output"],
                      "n_variants": len([k for k in rec if k != "original"]),
                      "duration_s": rec["original"].get("duration_s"),
                      "variants": [k for k in rec if k != "original"]})
        print(f"[manifest] {slug}: {len(rec)} entries", flush=True)

    # Phase C: peak-normalize instrumentals to NORM_DBFS (after diffs are computed
    # from the un-normalized stems, so residuals keep their meaning)
    normalize_library()
    # Phase D: AAC streams for the web GUI (WAV stays as the lossless download)
    transcode_library()
    # index.json is always a full scan of lib/, so single-file jobs never clobber it
    total = rebuild_index()
    print(f"\nDONE: processed {len(index)} file(s); index has {total}", flush=True)


def rebuild_index():
    """Regenerate lib/index.json from ALL manifests (robust to single-file jobs)."""
    import glob
    files = []
    for md in sorted(glob.glob(os.path.join(LIB, "*", "manifest.jsonl"))):
        slug = os.path.basename(os.path.dirname(md))
        byid = {r["id"]: r for r in (json.loads(l) for l in open(md) if l.strip())}
        orig = byid.get("original", {})
        files.append({"slug": slug, "title": slug, "original": orig.get("output", "original.mp3"),
                      "duration_s": orig.get("duration_s"),
                      "n_variants": len([k for k in byid if k != "original"]),
                      "variants": [k for k in byid if k != "original"]})
    json.dump({"generated": int(time.time()), "count": len(files),
               "files": sorted(files, key=lambda f: f["slug"])},
              open(os.path.join(LIB, "index.json"), "w"), indent=1)
    return len(files)


def audio_meta(y, sr=44100):
    return {"duration_s": round(y.shape[1] / sr, 2), "sr": sr, "channels": int(y.shape[0]), **metrics(y)}


FFMPEG = "/opt/homebrew/bin/ffmpeg"


def transcode_library(bitrate="256k"):
    """Make a streamable AAC (.m4a) sibling for every .wav variant so the web GUI
    serves compact audio (~3MB) instead of multi-MB WAV. Records it as each entry's
    'stream' field; the WAV stays as the lossless download. Idempotent."""
    import glob
    import subprocess
    n = 0
    for md in sorted(glob.glob(os.path.join(LIB, "*", "manifest.jsonl"))):
        sd = os.path.dirname(md)
        rec = read_manifest(sd)
        changed = False
        for eid, e in rec.items():
            out = e.get("output", "")
            src = os.path.join(sd, out)
            if not os.path.exists(src):
                continue
            if out.lower().endswith(".wav"):
                m4a_rel = out[:-4] + ".m4a"
                m4a = os.path.join(sd, m4a_rel)
                if not os.path.exists(m4a) or os.path.getmtime(m4a) < os.path.getmtime(src):
                    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src,
                                    "-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart", m4a],
                                   check=True)
                    n += 1
                if e.get("stream") != m4a_rel:
                    e["stream"] = m4a_rel; changed = True
            elif e.get("stream") != out:      # already-compressed originals stream as-is
                e["stream"] = out; changed = True
        if changed:
            write_manifest(sd, rec)
    print(f"transcoded {n} new AAC streams", flush=True)


def _peak_normalize(y, target_dbfs):
    peak = float(np.abs(y).max())
    if peak <= 1e-9:
        return y, 1.0
    gain = (10.0 ** (target_dbfs / 20.0)) / peak
    return y * gain, gain


def normalize_library(target_dbfs=NORM_DBFS):
    """Peak-normalize every INSTRUMENTAL variant to target_dbfs, in place, losslessly.

    Instrumentals are the voice-subtracted deliverables; vocals and diff/residual
    tracks are left untouched. Output is written as 24-bit WAV (no requantization
    loss from the gain). Idempotent: a track already at target gets gain≈1.0.
    """
    import glob
    n = 0
    for md in sorted(glob.glob(os.path.join(LIB, "*", "manifest.jsonl"))):
        sd = os.path.dirname(md)
        rec = read_manifest(sd)
        changed = False
        for eid, e in list(rec.items()):
            if e.get("stem") != "Instrumental" or e.get("status") != "done":
                continue
            if e.get("normalized_dbfs") == target_dbfs:      # already normalized — skip (fast re-runs)
                continue
            wav = os.path.join(sd, e.get("output", ""))
            if not os.path.exists(wav):
                continue
            y = load_audio(wav)
            yn, gain = _peak_normalize(y, target_dbfs)
            sf.write(wav, yn.T.astype(np.float32), 44100, subtype="PCM_24")   # lossless
            assets(yn, wav[:-4])                                              # redo png + peaks
            e.update(audio_meta(yn))
            e["normalized_dbfs"] = target_dbfs
            if "normaliz" not in e.get("description", "").lower():
                e["description"] = e.get("description", "") + f" · normalized to {target_dbfs:g} dBFS peak"
            changed = True
            n += 1
            print(f"  [norm {target_dbfs:g}dBFS  {20*np.log10(gain):+.1f}dB] "
                  f"{os.path.basename(sd)}/{eid}", flush=True)
        if changed:
            write_manifest(sd, rec)
    print(f"normalized {n} instrumental tracks to {target_dbfs:g} dBFS", flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "normalize":
            normalize_library()          # reprocess existing library only
        elif len(sys.argv) > 1 and sys.argv[1] == "transcode":
            transcode_library()          # (re)build AAC streams only
        elif len(sys.argv) > 1 and sys.argv[1] == "reindex":
            print("index:", rebuild_index())
        elif len(sys.argv) > 2 and sys.argv[1] == "file":
            main(only=sys.argv[2:])      # process one or more specific files (job runner)
        else:
            main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
