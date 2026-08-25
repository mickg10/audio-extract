#!/usr/bin/env python3
"""Tobacco variants batch (owner-ordered) — /home/mickg/tobacco_variants.

Float32 44.1k stereo throughout. Model inference goes ONLY through the
float-gated adapter audio_extract.separate.Separator (working copy ==
mac commit 2c859f6 blob, sha256 78885d56...). Ensemble-max reproduces
/home/mickg/v2_canonical_ext/render_dag.py ensemble_max byte-for-byte in
approach: members [bsroformer, mdx23c], per-channel complex STFT n_fft=4096
hop=1024, |S|.argmax(0) winner-takes-phase, istft length=n, float32 FLOAT.

Variants: A=MDX23C native (Instrumental); B=ensemble-max{MDX23C,BSR};
C=BSR native (Instrumental); D=N0 - MDX23C (Vocals) exact-grid subtract;
E=MelBand non-vocal stem. verdi+habanera: ONLY the subtraction, delivered
as letter E label "subtract". scarydog+valkyries: all letters spliced into
N0 over the given spans with 80 ms equal-power sin/cos fades placed strictly
INSIDE the spans (outside spans the output is bit-identical N0 by construction).
No gain anywhere. Write-once: existing node files are never overwritten
(reused only if they verify as FLOAT).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/mickg/audio-extract")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

BASE = Path("/home/mickg/tobacco_variants")
DELIV = BASE / "deliverables"
MODELS_DIR = Path("/home/mickg/models")
LIB_IN = Path("/home/mickg/lib_in")
MDX = "MDX23C-8KFFT-InstVoc_HQ.ckpt"
BSR = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MEL = "vocals_mel_band_roformer.ckpt"
ADAPTER_SHA_PINNED = "78885d562938521e36d6c198143190d48041cc0a9e5aa8007763f1ac2d6948aa"
ADAPTER_COMMIT_NOTE = "working copy == mac repo commit 2c859f6 fix(separate): float-gate backend stems before read"
VERDI_CANON = Path("/home/mickg/v2_provenance/verdi_dag/verdi_A_decoded.f32.wav")
SR = 44100
NFFT, HOP = 4096, 1024  # donna_ensmax / convert.py ensemble constants
FADE_S = 0.080

TRACKS = [
    {"name": "la_wally", "base": "6_la_wally",
     "src": LIB_IN / "6_LA_WALLY_CHILDREN_Tobacco_2026_07_06 (1).mp3",
     "mode": "full"},
    {"name": "scarydog", "base": "12_13_scarydog_stabat",
     "src": LIB_IN / "12_13_SCARYDOG_STABAT_MATER_Tobacco_2026_07_06.mp3",
     "mode": "splice", "spans": [(50.0, 87.5)]},
    {"name": "valkyries", "base": "22_valkyries",
     "src": LIB_IN / "22_VALKYRIES_NEW_7242026.mp3",
     "mode": "splice", "spans": [(42.0, 55.0), (65.0, 83.0)]},
    {"name": "verdi", "base": "7_verdi_slow_voices",
     "src": VERDI_CANON, "src_is_float": True,
     "src_note": "canonical float decode from v2_provenance (read-only); "
                 "lib_in mp3 sha256 3c8a7645... == provenance historical source",
     "mode": "sub_only"},
    {"name": "habanera", "base": "1_habanera",
     "src": LIB_IN / "1_HABANERA_LAST_Tobacco_2026_06_18.mp3",
     "mode": "sub_only"},
]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(cmd[:4])}...): {r.stderr[-2000:]}")
    return r.stdout


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_f32(p: Path, label: str) -> None:
    codec = run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(p)]).strip()
    if codec != "pcm_f32le":
        raise RuntimeError(f"{label} is {codec!r}, not pcm_f32le: {p}")
    info = sf.info(str(p))
    if info.subtype != "FLOAT":
        raise RuntimeError(f"{label} soundfile subtype {info.subtype!r} != FLOAT: {p}")
    if int(info.samplerate) != SR or int(info.channels) != 2:
        raise RuntimeError(f"{label} grid {info.samplerate}Hz/{info.channels}ch != 44100/2: {p}")


def read_f64(p: Path):
    arr, sr = sf.read(str(p), dtype="float64", always_2d=True)
    return arr, int(sr)


def write_f32(p: Path, arr, sr: int) -> None:
    if p.exists():
        raise RuntimeError(f"write-once violation: {p} already exists")
    sf.write(str(p), np.asarray(arr, dtype=np.float32), sr, subtype="FLOAT")


def stats(p: Path) -> dict:
    arr, sr = read_f64(p)
    return {
        "frames": int(arr.shape[0]),
        "duration_s": round(arr.shape[0] / sr, 6),
        "rms": round(float(np.sqrt(np.mean(arr ** 2))), 8),
        "peak": round(float(np.abs(arr).max()), 8),
        "sha256": sha256_file(p),
    }


def node_ok(p: Path) -> bool:
    if not p.exists():
        return False
    assert_f32(p, f"resume-check {p.name}")
    return True


def collect_pins(models_present: dict) -> dict:
    import importlib.metadata as md
    head = run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip()
    dirty = run(["git", "-C", str(REPO), "status", "--short"]).strip()
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version",
               "--format=csv,noheader"]).strip()
    gpu_name, driver = [x.strip() for x in smi.split(",", 1)]
    return {
        "research6_repo_head": head,
        "repo_dirty_files": [line for line in dirty.splitlines() if line],
        "adapter_path": str(REPO / "audio_extract" / "separate.py"),
        "adapter_sha256": sha256_file(REPO / "audio_extract" / "separate.py"),
        "adapter_commit_note": ADAPTER_COMMIT_NOTE,
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "ckpt_sha256": {name: sha256_file(MODELS_DIR / name)
                        for name, present in models_present.items() if present},
        "audio_separator_version": md.version("audio-separator"),
        "torch_version": md.version("torch"),
        "nvidia_driver": driver,
        "gpu_name": gpu_name,
        "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0],
        "ensemble_reference": "/home/mickg/v2_canonical_ext/render_dag.py ensemble_max "
                              "(convert.py build_ensembles Phase A.5, float-safe reproduction)",
    }


def decode_n0(src: Path, n0: Path) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(src), "-vn",
         "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", str(n0)])
    assert_f32(n0, "N0")


def stem_tag(p: Path) -> str:
    m = re.search(r"\(([^)]+)\)", p.stem)
    return m.group(1) if m else p.stem


def run_separator(model: str, n0: Path, sep_dir: Path) -> dict:
    """One adapter run; returns {normalized_name: {path, tag}} of on-disk stems."""
    from audio_extract.separate import Separator, _normalize_stem_name

    sep_dir.mkdir(parents=True, exist_ok=True)
    leftover = list(sep_dir.glob("*.wav"))
    for p in leftover:  # stale partial run: separator dir is disposable
        p.unlink()
    sep = Separator(model, model_dir=MODELS_DIR, output_dir=sep_dir)
    out = sep.separate_file(n0)  # float gate lives inside separate_file
    produced = sorted(sep_dir.glob("*.wav"))
    stems = {}
    for p in produced:
        stems[_normalize_stem_name(p.stem)] = {"path": p, "tag": stem_tag(p)}
    del sep
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    log(f"  {model}: stems {sorted((k, v['tag']) for k, v in stems.items())} "
        f"model_sha={out.model_sha256[:23]}")
    return stems


def keep(stem: dict, node: Path) -> None:
    if node.exists():
        raise RuntimeError(f"write-once violation: {node} already exists")
    Path(stem["path"]).replace(node)


def cleanup_sepdir(sep_dir: Path) -> None:
    if sep_dir.is_dir():
        shutil.rmtree(sep_dir)


def ensemble_max(rof_path: Path, mdx_path: Path, out_path: Path) -> dict:
    """render_dag.py / convert.py build_ensembles, reproduced exactly (float32)."""
    import librosa

    a = sf.read(str(rof_path), dtype="float32", always_2d=True)[0].T  # [ch, n]
    b = sf.read(str(mdx_path), dtype="float32", always_2d=True)[0].T
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    out = np.zeros_like(a)
    for c in range(a.shape[0]):
        S = np.stack([librosa.stft(np.ascontiguousarray(a[c]), n_fft=NFFT, hop_length=HOP),
                      librosa.stft(np.ascontiguousarray(b[c]), n_fft=NFFT, hop_length=HOP)])
        idx = np.abs(S).argmax(0)
        out[c] = librosa.istft(np.take_along_axis(S, idx[None], 0)[0],
                               hop_length=HOP, length=n)
    write_f32(out_path, out.T, SR)
    return {"algo": "max_spec", "member_order": ["bsroformer", "mdx23c"],
            "n_fft": NFFT, "hop": HOP, "aligned_len_samples": int(n),
            "source": "convert.py build_ensembles (Phase A.5) via render_dag.py, "
                      "float-safe reproduction"}


def splice(n0_arr, v_arr, spans, sr: int):
    """N0 outside spans, variant inside, 80 ms equal-power fades INSIDE span edges."""
    L = int(round(FADE_S * sr))
    out = n0_arr.copy()
    t = np.arange(L) / (L - 1)
    g_in = np.sin(t * np.pi / 2.0)[:, None]   # variant rises (fade-in)
    g_out = np.cos(t * np.pi / 2.0)[:, None]  # complement (equal power)
    zones = []
    for (t0, t1) in spans:
        s0, s1 = int(round(t0 * sr)), int(round(t1 * sr))
        if not (0 <= s0 and s1 <= len(n0_arr) and s1 - s0 > 2 * L):
            raise RuntimeError(f"span {t0}-{t1}s invalid for len {len(n0_arr)}")
        out[s0:s0 + L] = g_out * n0_arr[s0:s0 + L] + g_in * v_arr[s0:s0 + L]
        out[s0 + L:s1 - L] = v_arr[s0 + L:s1 - L]
        out[s1 - L:s1] = g_in * n0_arr[s1 - L:s1] + g_out * v_arr[s1 - L:s1]
        zones.append({"span_s": [t0, t1], "span_samples": [s0, s1],
                      "fade_samples": L})
    return out, zones


def verify_splice(spliced_path: Path, n0_path: Path, spans, sr: int) -> dict:
    a, _ = read_f64(spliced_path)
    b, _ = read_f64(n0_path)
    if a.shape != b.shape:
        raise RuntimeError(f"splice shape {a.shape} != N0 {b.shape}")
    mask = np.ones(len(a), dtype=bool)
    for (t0, t1) in spans:
        mask[int(round(t0 * sr)):int(round(t1 * sr))] = False
    outside = float(np.abs(a[mask] - b[mask]).max()) if mask.any() else 0.0
    return {"frames_equal_n0": True,
            "outside_span_max_abs_diff": outside,
            "outside_span_ok": bool(outside < 1e-6)}


def encode_m4a(node: Path, m4a: Path) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(node),
         "-c:a", "aac", "-b:a", "256k", "-ar", "44100", "-ac", "2", str(m4a)])


def m4a_duration(p: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", str(p)]).strip())


def deliver(node: Path, d: Path, name: str, n0_dur: float, manifest: dict) -> dict:
    m4a = d / name
    if not m4a.exists():
        encode_m4a(node, m4a)
    dur = m4a_duration(m4a)
    fst = stats(node)
    row = {
        "file": name,
        "m4a_sha256": sha256_file(m4a),
        "m4a_duration_s": round(dur, 4),
        "float_master": node.name,
        "float_rms": fst["rms"],
        "float_peak": fst["peak"],
        "checks": {
            "duration_within_0.1s_of_source": bool(abs(dur - n0_dur) <= 0.1),
            "duration_delta_s": round(dur - n0_dur, 4),
            "rms_gt_0.003": bool(fst["rms"] > 0.003),
        },
    }
    DELIV.mkdir(parents=True, exist_ok=True)
    staged = DELIV / name
    if not staged.exists():
        os.link(m4a, staged)
    manifest["deliverables"].append(row)
    return row


def build_track(cfg: dict, pins: dict, models_present: dict, summary: dict) -> None:
    name, base, mode = cfg["name"], cfg["base"], cfg["mode"]
    d = BASE / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"track": name, "trackbase": base, "mode": mode,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_path": str(cfg["src"]),
                "source_sha256": sha256_file(cfg["src"]),
                "splice_spans_s": cfg.get("spans"),
                "src_note": cfg.get("src_note"),
                "nodes": {}, "variants": {}, "deliverables": [],
                "failures": [], "pins": pins}

    n0 = d / "N0_source.f32.wav"
    if not node_ok(n0):
        log(f"=== {name}: N0 ===")
        if cfg.get("src_is_float"):
            shutil.copyfile(cfg["src"], n0)
            assert_f32(n0, "N0 (canonical copy)")
        else:
            decode_n0(cfg["src"], n0)
    n0_arr, _ = read_f64(n0)
    n0_st = stats(n0)
    manifest["nodes"]["N0_source.f32.wav"] = n0_st
    n0_dur = n0_st["duration_s"]
    log(f"=== {name}: N0 {n0_dur:.3f}s frames={n0_st['frames']} rms={n0_st['rms']} ===")

    # verdi cross-check: fresh decode of the lib_in twin vs canonical (diagnostic)
    if name == "verdi":
        try:
            tmp = d / "_verdi_libin_decode.f32.wav"
            if not tmp.exists():
                decode_n0(LIB_IN / "7_VERDI_SLOW_Voices_Tobacco_05_14_26 (1).mp3", tmp)
            fresh, _ = read_f64(tmp)
            nn = min(len(fresh), len(n0_arr))
            diff = float(np.abs(fresh[:nn] - n0_arr[:nn]).max())
            manifest["verdi_canonical_vs_libin_decode"] = {
                "max_abs_diff": diff, "frames": [len(fresh), len(n0_arr)]}
            tmp.unlink()
            log(f"  verdi canonical vs fresh lib_in decode: max_abs_diff={diff:.3e}")
        except Exception as e:  # diagnostic only
            manifest["verdi_canonical_vs_libin_decode"] = {"error": str(e)}

    a_node = d / "A_mdx23c_instrumental.f32.wav"
    voc_node = d / "mdx23c_vocals.f32.wav"
    c_node = d / "C_bsroformer_instrumental.f32.wav"
    b_node = d / "B_ensmax.f32.wav"
    dd_node = d / "D_subtract.f32.wav"
    e_node = d / "E_melband_instrumental.f32.wav"

    def fail(letter: str, why: str) -> None:
        manifest["failures"].append({"letter": letter, "reason": why})
        log(f"  FAIL {name} {letter}: {why}")

    # ---- MDX23C (always needed: A and/or the vocals for D) ----
    mdx_tags = None
    if not (node_ok(a_node) and node_ok(voc_node)):
        log(f"=== {name}: separate MDX23C ===")
        stems = run_separator(MDX, n0, d / "_sep_mdx")
        if "instrumental" not in stems or "vocals" not in stems:
            raise RuntimeError(f"MDX23C stems missing: {sorted(stems)}")
        mdx_tags = {k: v["tag"] for k, v in stems.items()}
        keep(stems["instrumental"], a_node)
        keep(stems["vocals"], voc_node)
        cleanup_sepdir(d / "_sep_mdx")
    a_st, voc_st = stats(a_node), stats(voc_node)
    manifest["nodes"][a_node.name] = a_st
    manifest["nodes"][voc_node.name] = voc_st
    manifest["variants"]["A"] = {
        "label": "mdx23c", "model": MDX,
        "construction": "native '(Instrumental)' stem via float-gated adapter",
        "backend_stem_tags": mdx_tags, "frames_equal_n0": a_st["frames"] == n0_st["frames"]}
    if a_st["frames"] != n0_st["frames"] or voc_st["frames"] != n0_st["frames"]:
        raise RuntimeError(f"MDX23C stem frames {a_st['frames']}/{voc_st['frames']} "
                           f"!= N0 {n0_st['frames']}")

    # ---- D: N0 minus MDX23C vocals (exact-grid float subtract) ----
    if not node_ok(dd_node):
        log(f"=== {name}: D subtract ===")
        voc_arr, _ = read_f64(voc_node)
        if voc_arr.shape != n0_arr.shape:
            raise RuntimeError(f"D: vocals shape {voc_arr.shape} != N0 {n0_arr.shape}")
        write_f32(dd_node, n0_arr - voc_arr, SR)
    d_st = stats(dd_node)
    manifest["nodes"][dd_node.name] = d_st
    inst_arr, _ = read_f64(a_node)
    voc_arr, _ = read_f64(voc_node)
    additivity = float(np.abs(n0_arr - voc_arr - inst_arr).max())
    manifest["variants"]["D"] = {
        "label": "subtract", "model": MDX,
        "construction": "N0 minus MDX23C '(Vocals)' stem, frames verified equal, "
                        "float64 math, float32 store, no alignment, no gain",
        "frames_equal_n0": True,
        "diagnostic_mdx23c_additivity_max_abs(N0-voc-inst)": additivity}
    del inst_arr, voc_arr

    full_letters = {"A": a_node, "D": dd_node}

    if mode in ("full", "splice"):
        # ---- C: BS-Roformer ----
        if models_present[BSR]:
            try:
                if not node_ok(c_node):
                    log(f"=== {name}: separate BS-Roformer ===")
                    stems = run_separator(BSR, n0, d / "_sep_bsr")
                    if "instrumental" not in stems:
                        raise RuntimeError(f"BSR produced no instrumental: {sorted(stems)}")
                    manifest["variants"].setdefault("C", {})["backend_stem_tags"] = {
                        k: v["tag"] for k, v in stems.items()}
                    keep(stems["instrumental"], c_node)
                    cleanup_sepdir(d / "_sep_bsr")
                c_st = stats(c_node)
                manifest["nodes"][c_node.name] = c_st
                manifest["variants"].setdefault("C", {}).update({
                    "label": "roformer", "model": BSR,
                    "construction": "native '(Instrumental)' stem via float-gated adapter "
                                    "(same checkpoint as donna_ensmax member)",
                    "frames_equal_n0": c_st["frames"] == n0_st["frames"]})
                if c_st["frames"] != n0_st["frames"]:
                    raise RuntimeError(f"C frames {c_st['frames']} != N0 {n0_st['frames']}")
                full_letters["C"] = c_node
            except Exception as e:
                fail("C", f"{e}")
        else:
            fail("C", f"checkpoint absent: {BSR}")

        # ---- B: ensemble-max of {BSR, MDX23C} instrumentals ----
        if "C" in full_letters:
            try:
                if not node_ok(b_node):
                    log(f"=== {name}: ensemble_max ===")
                    ens = ensemble_max(c_node, a_node, b_node)
                else:
                    ens = {"note": "resumed existing node"}
                b_st = stats(b_node)
                manifest["nodes"][b_node.name] = b_st
                manifest["variants"]["B"] = {
                    "label": "ensmax", "models": [BSR, MDX],
                    "construction": "ensemble-max (donna_ensmax recipe)", "ensemble": ens,
                    "frames_equal_n0": b_st["frames"] == n0_st["frames"]}
                if b_st["frames"] != n0_st["frames"]:
                    raise RuntimeError(f"B frames {b_st['frames']} != N0 {n0_st['frames']}")
                full_letters["B"] = b_node
            except Exception as e:
                fail("B", f"{e}")
        else:
            fail("B", "BS-Roformer member unavailable, ensemble impossible")

        # ---- E: MelBand ----
        if models_present[MEL]:
            try:
                if not node_ok(e_node):
                    log(f"=== {name}: separate MelBand ===")
                    stems = run_separator(MEL, n0, d / "_sep_mel")
                    pick = None
                    if "instrumental" in stems:
                        pick, note = stems["instrumental"], "native '(Instrumental)' stem"
                    elif "other" in stems:
                        pick, note = stems["other"], (
                            "backend non-vocal stem tagged '(Other)' — this checkpoint is "
                            "vocals-target (instruments [vocals, other]); the non-vocal stem "
                            "audio-separator writes is its instrumental output")
                    if pick is None:
                        raise RuntimeError(f"MelBand produced no non-vocal stem: {sorted(stems)}")
                    manifest["variants"].setdefault("E", {}).update({
                        "backend_stem_tags": {k: v["tag"] for k, v in stems.items()},
                        "construction": note + " via float-gated adapter"})
                    keep(pick, e_node)
                    cleanup_sepdir(d / "_sep_mel")
                e_st = stats(e_node)
                manifest["nodes"][e_node.name] = e_st
                manifest["variants"].setdefault("E", {}).update({
                    "label": "melband", "model": MEL,
                    "frames_equal_n0": e_st["frames"] == n0_st["frames"]})
                if e_st["frames"] != n0_st["frames"]:
                    raise RuntimeError(f"E frames {e_st['frames']} != N0 {n0_st['frames']}")
                full_letters["E"] = e_node
            except Exception as e:
                fail("E", f"{e}")
        else:
            fail("E", f"checkpoint absent: {MEL}")

    # ---- deliverables ----
    letter_label = {"A": "mdx23c", "B": "ensmax", "C": "roformer",
                    "D": "subtract", "E": "melband"}
    if mode == "sub_only":
        # deliver the subtraction as letter E, label "subtract"
        nm = f"{base}__E_subtract.m4a"
        if not (d / nm).exists():
            log(f"=== {name}: encode {nm} ===")
        deliver(dd_node, d, nm, n0_dur, manifest)
        manifest["variants"]["E_delivery_note"] = (
            "delivered as letter E with label 'subtract' per owner order; "
            "construction is the D-subtraction (N0 - MDX23C vocals)")
    elif mode == "full":
        for letter in "ABCDE":
            node = full_letters.get(letter)
            if node is None:
                continue
            nm = f"{base}__{letter}_{letter_label[letter]}.m4a"
            log(f"=== {name}: encode {nm} ===")
            deliver(node, d, nm, n0_dur, manifest)
    elif mode == "splice":
        spans = cfg["spans"]
        for letter in "ABCDE":
            node = full_letters.get(letter)
            if node is None:
                continue
            try:
                sp_node = d / f"{letter}_{letter_label[letter]}_spliced.f32.wav"
                if not node_ok(sp_node):
                    log(f"=== {name}: splice {letter} ===")
                    v_arr, _ = read_f64(node)
                    if v_arr.shape != n0_arr.shape:
                        raise RuntimeError(f"splice {letter}: variant shape {v_arr.shape} "
                                           f"!= N0 {n0_arr.shape}")
                    out, zones = splice(n0_arr, v_arr, spans, SR)
                    write_f32(sp_node, out, SR)
                    del v_arr, out
                else:
                    zones = None
                ver = verify_splice(sp_node, n0, spans, SR)
                sp_st = stats(sp_node)
                manifest["nodes"][sp_node.name] = sp_st
                manifest["variants"][f"{letter}_spliced"] = {
                    "label": letter_label[letter] + "_spliced",
                    "spans_s": spans, "fade_ms": 80.0,
                    "fade_shape": "equal-power sin/cos, fades inside span edges",
                    "fade_zones": zones, "verify": ver,
                    "duration_equals_n0": sp_st["frames"] == n0_st["frames"]}
                if not ver["outside_span_ok"]:
                    raise RuntimeError(f"splice {letter} outside-span diff "
                                       f"{ver['outside_span_max_abs_diff']}")
                if sp_st["frames"] != n0_st["frames"]:
                    raise RuntimeError(f"splice {letter} frames {sp_st['frames']} "
                                       f"!= N0 {n0_st['frames']}")
                nm = f"{base}__{letter}_{letter_label[letter]}_spliced.m4a"
                log(f"=== {name}: encode {nm} ===")
                row = deliver(sp_node, d, nm, n0_dur, manifest)
                row["checks"]["outside_span_bit_close"] = ver["outside_span_ok"]
                row["checks"]["outside_span_max_abs_diff"] = ver["outside_span_max_abs_diff"]
            except Exception as e:
                fail(f"{letter}_spliced", f"{e}")

    mp = d / "manifest.json"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    summary["tracks"][name] = {
        "deliverables": manifest["deliverables"],
        "failures": manifest["failures"],
        "n0_duration_s": n0_dur,
    }
    log(f"=== {name}: DONE ({len(manifest['deliverables'])} deliverables, "
        f"{len(manifest['failures'])} failures) ===")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    DELIV.mkdir(parents=True, exist_ok=True)

    adapter = REPO / "audio_extract" / "separate.py"
    actual = sha256_file(adapter)
    if actual != ADAPTER_SHA_PINNED:
        print(f"STOP: adapter sha {actual} != pinned {ADAPTER_SHA_PINNED}", flush=True)
        sys.exit(1)

    models_present = {m: (MODELS_DIR / m).exists() for m in (MDX, BSR, MEL)}
    if not models_present[MDX]:
        print(f"STOP: required checkpoint absent: {MDX}", flush=True)
        sys.exit(1)
    for cfg in TRACKS:
        if not Path(cfg["src"]).is_file():
            print(f"STOP: source missing for {cfg['name']}: {cfg['src']}", flush=True)
            sys.exit(1)

    pins = collect_pins(models_present)
    log("PINS: " + json.dumps(pins, indent=2))
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "models_present": models_present, "tracks": {}, "track_errors": {}}
    for cfg in TRACKS:
        try:
            build_track(cfg, pins, models_present, summary)
        except Exception:
            tb = traceback.format_exc()
            summary["track_errors"][cfg["name"]] = tb
            log(f"TRACK ERROR {cfg['name']}:\n{tb}")
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (BASE / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    log("ALL DONE")


if __name__ == "__main__":
    main()
