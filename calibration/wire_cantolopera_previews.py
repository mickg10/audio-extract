#!/usr/bin/env python3
"""Wire the Cantolopera 30s preview pairs into the judge's transfer-validation set.

Previews are cut per-version at different offsets, so each (full, target) pair is
FULL-RANGE aligned + trimmed to overlap, then audited. As lossy 30s clips they sit
at the same-take null threshold (oracle: preview = transfer-DIRECTION grade, not
reference-grade), so acceptance uses the aligned xcorr peak + null residual and the
pair is flagged reference_grade='lossy_preview'."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

from audio_extract import challenges as ch
from audio_extract import dsp

D = Path("~/audio-extract/calibration/input_data").expanduser()
OUT = Path("~/audio-extract/calibration/cantolopera_previews.json").expanduser()
SR = 44100
VOICE_TAGS = {"soprano": "soprano", "mezzosoprano": "mezzo", "tenore": "tenor",
              "baritono": "baritone", "basso": "bass", "duca": "tenor", "rigoletto": "baritone"}


def load(p):
    a, sr = sf.read(str(p), dtype="float64", always_2d=True)
    if sr != SR:
        a = ss.resample(a, int(len(a) * SR / sr))
    return a


def align_trim(full, orch):
    q = SR // 8000
    fm = ss.decimate(full.mean(1), q, ftype="fir"); om = ss.decimate(orch.mean(1), q, ftype="fir")
    F = fm - fm.mean(); O = om - om.mean()
    xc = ss.fftconvolve(F, O[::-1], mode="full")
    lag = int(round((xc.argmax() - (len(fm) - 1)) * q))
    peak = float(xc.max() / (np.linalg.norm(F) * np.linalg.norm(O) + 1e-9))
    if lag >= 0:
        f, o = full[lag:], orch[:len(full) - lag]
    else:
        o, f = orch[-lag:], full[:len(orch) + lag]
    n = min(len(f), len(o))
    return f[:n], o[:n], lag, peak


def parse(fn):
    base = fn.replace("_voice_preview.wav", "|voice").replace("_orchestra_preview.wav", "|orchestra")
    name, role = base.split("|")
    version = "full"
    if role == "orchestra" or "strumentale" in name:
        version = "orchestra"
    m = re.search(r"senza-([a-z]+)", name)
    if m:
        version = "senza_" + VOICE_TAGS.get(m.group(1), m.group(1))
    work = re.sub(r"-?(senza-[a-z]+|strumentale)-?", "-", name).strip("-")
    return work, version


def coverage(w):
    voice = ("soprano" if any(k in w for k in ["vissi-darte","mi-chiamano","un-bel-di","non-mi-dir",
             "mi-tradi","o-mio-babbino","caro-nome","in-questa-reggia","ritorna-vincitor","deh-vieni",
             "der-holle","spargi","dolce-suono"]) else
             "tenor" if any(k in w for k in ["celeste-aida","che-gelida","e-lucevan","recondita",
             "la-donna-e-mobile","questa-o-quella","nessun-dorma"]) else
             "mezzo" if "oiseau" in w or "una-voce" in w else
             "baritone" if any(k in w for k in ["largo-al-factotum","cortigiani","il-balen","bella-figlia"]) else
             "bass" if any(k in w for k in ["la-calunnia","vecchia-zimarra"]) else "mixed")
    orch = "transparent" if any(c in w for c in ["mozart","barbiere","flauto"]) else "dense"
    return {"voice": voice, "orchestration": orch,
            "coloratura": any(k in w for k in ["der-holle","spargi","dolce-suono","caro-nome","una-voce"])}


files = sorted(D.glob("*.wav"))
works = defaultdict(dict)
for p in files:
    work, version = parse(p.name)
    works[work][version] = p

print(f"{len(files)} files -> {len(works)} arias  (align + trim, lossy-preview acceptance)\n")
records, confirmed, n_pairs = [], 0, 0
for work, versions in sorted(works.items()):
    if "full" not in versions:
        continue
    full = load(versions["full"]); cov = coverage(work)
    for v, p in sorted(versions.items()):
        if v == "full":
            continue
        f, o, lag, peak = align_trim(full, load(p))
        vest = dsp.mono(f) - dsp.mono(o)
        env = np.convolve(np.abs(vest), np.ones(SR // 100) / (SR // 100), mode="same")
        mask = env < np.percentile(env, 30)
        a = ch.audit_pair(f, o, SR, solo_inactive_mask=mask)
        null = a["residual_on_solo_inactive_db"]
        n_pairs += 1
        # lossy-preview acceptance: aligned strongly AND orchestra genuinely cancels
        same = (a["recommended_integrity"] in ("linear_exact", "same_take_paired_target")
                or (peak >= 0.45 and null <= -9.0))
        confirmed += same
        records.append({
            "work": work, "task": "all_voices_vs_nonvocal" if v == "orchestra" else "soloist_vs_rest",
            "removed": v, "coverage": cov, "reference_grade": "lossy_preview",
            "aligned_lag_samples": int(lag), "xcorr_peak": round(peak, 3),
            "null_residual_db": round(null, 2), "integrity_audit": a["recommended_integrity"],
            "same_take_confirmed": bool(same),
            "files": {"full": versions["full"].name, v: p.name}})
        print(f"{'OK ' if same else '.. '}{work[:44]:44s} vs {v:14s} peak={peak:.2f} "
              f"null={null:6.1f}dB lag={lag/SR:+.2f}s  {a['recommended_integrity']}")

OUT.write_text(json.dumps({"schema": "cantolopera-previews/v1", "reference_grade": "lossy_preview",
    "note": "30s streaming previews, full-range aligned; transfer-DIRECTION grade per oracle "
            "(not reference-grade thresholds — upgrade with WAV purchases).",
    "sample_rate": SR, "n_arias": len(works), "n_pairs": n_pairs,
    "n_same_take_confirmed": confirmed, "records": records}, indent=2,
    default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))

cells = defaultdict(int)
for r in records:
    if r["same_take_confirmed"]:
        cells[f'{r["coverage"]["voice"]} x {r["coverage"]["orchestration"]}'] += 1
print(f"\n{confirmed}/{n_pairs} pairs confirmed same-take (aligned; lossy-preview acceptance)")
print("coverage cells covered by confirmed pairs:")
for c, k in sorted(cells.items()):
    print(f"  {c:26s} {k}")
print(f"\nmanifest -> {OUT}")
