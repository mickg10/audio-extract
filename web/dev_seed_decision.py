#!/usr/bin/env python3
"""Seed synthetic data-model-v2 facts so the GUI's autonomous decision panel
(v2.1 WP13) can be developed and tested without running the full pipeline.

Creates (by default) three runs under ``--lib /tmp/guilib3``, one per terminal
selector status, so every banner state of the panel is visible:

* ``dectest``       — ``final`` / ``clear_winner``; v2 tables written into the
                      run's ``manifest.sqlite`` **alongside** the v1 tables
* ``dectest-safe``  — ``final`` / ``best_safe`` (bounds overlap, not certain);
                      v2 tables written to a **separate** ``manifest_v2.sqlite``
* ``dectest-none``  — ``no_acceptable_candidate`` (every candidate fails a gate)

Each run's audio/passages/candidates come from an existing run found under
``/tmp`` (default base: ``/tmp/guilib/guitest``); if none exists the script
falls back to ``audio-extract ingest`` on a small mp3 under ``./input``, and
finally to a fully synthetic 4-second tone source. The ``selection_decision``
report is **real**: it is produced by ``audio_extract.selector.select`` on
synthetic per-defect severity cells (so shapes/rounding match production).

Usage:  uv run python web/dev_seed_decision.py [--lib /tmp/guilib3] [--base DIR]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from audio_extract import challenges, selector          # noqa: E402
from audio_extract.manifest_v2 import ManifestV2        # noqa: E402

DEFECTS = ("orchestral_theft", "event_hole", "vocal_leakage", "fullness",
           "brightness", "hall", "generic_quality")
GATE_KEYS = ("orchestral_theft", "vocal_leakage", "event_hole")


# ---------------------------------------------------------------------------
# base run acquisition: copy an existing run / CLI ingest / synthesize
# ---------------------------------------------------------------------------
def find_base(explicit: str | None) -> str | None:
    cands = [explicit] if explicit else []
    for root in ("/tmp/guilib", "/tmp/guilib2", "/tmp/guitest", "/tmp/lib"):
        cands.extend(sorted(glob.glob(os.path.join(root, "*"))))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "source", "source.json")) \
                and os.path.isfile(os.path.join(c, "source", "canonical.f32.wav")):
            return c
    return None


def _copy_sqlite(src: str, dst: str) -> None:
    """Copy a sqlite db via the backup API (WAL-safe)."""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    with d:
        s.backup(d)
    s.close()
    d.close()


def copy_base_run(base: str, dest: str, track_id: str) -> None:
    for sub in ("source", "passages", "candidates"):
        sp = os.path.join(base, sub)
        if os.path.isdir(sp):
            shutil.copytree(sp, os.path.join(dest, sub), dirs_exist_ok=True)
    man = os.path.join(base, "manifest.sqlite")
    if os.path.isfile(man):                       # keep the v1 tables (job/metric/...)
        _copy_sqlite(man, os.path.join(dest, "manifest.sqlite"))
    # keep source.json's track_id honest for the copied run
    sj = os.path.join(dest, "source", "source.json")
    try:
        with open(sj, encoding="utf-8") as fh:
            src = json.load(fh)
        src["track_id"] = track_id
        with open(sj, "w", encoding="utf-8") as fh:
            json.dump(src, fh, indent=2)
    except (OSError, ValueError):
        pass


def try_cli_ingest(lib: str, run_id: str) -> str | None:
    """Fallback #2: ingest a small mp3 from ./input via the real CLI."""
    small = [p for p in sorted(glob.glob(os.path.join(REPO, "input", "*.mp3")))
             if os.path.getsize(p) < 10 * 1024 * 1024]
    if not small:
        return None
    try:
        subprocess.run(
            [sys.executable, "-m", "audio_extract.cli", "--lib", lib,
             "ingest", small[0], "--run-id", run_id],
            check=True, cwd=REPO, timeout=300)
    except (subprocess.SubprocessError, OSError):
        return None
    run = os.path.join(lib, run_id)
    return run if os.path.isfile(os.path.join(run, "source", "source.json")) else None


def synthesize_run(dest: str, track_id: str) -> None:
    """Fallback #3: a fully synthetic 4 s stereo source + 3 candidate stems."""
    import soundfile as sf
    sr, dur = 44100, 4.0
    t = np.arange(int(sr * dur)) / sr
    left = 0.4 * np.sin(2 * np.pi * 220 * t) * np.exp(-t / 3.0) \
        + 0.15 * np.sin(2 * np.pi * 2093 * t) * (t > 1.5)
    right = 0.4 * np.sin(2 * np.pi * 277 * t) * np.exp(-t / 3.0) \
        + 0.1 * np.random.default_rng(0).standard_normal(len(t)) * (t < 0.5)
    x = np.column_stack([left, right]).astype(np.float32)
    os.makedirs(os.path.join(dest, "source"), exist_ok=True)
    os.makedirs(os.path.join(dest, "passages"), exist_ok=True)
    sf.write(os.path.join(dest, "source", "canonical.f32.wav"), x, sr, subtype="FLOAT")
    with open(os.path.join(dest, "source", "source.json"), "w", encoding="utf-8") as fh:
        json.dump({"track_id": track_id, "original_filename": "synthetic-tone.wav",
                   "sample_rate_hz": sr, "channel_layout": ["FL", "FR"],
                   "frames": len(x), "sample_format": "float32-le-interleaved"}, fh, indent=2)
    for i in range(3):
        hexd = hashlib.sha256(f"dev-seed-cand-{i}".encode()).hexdigest()
        cdir = os.path.join(dest, "candidates", "sha256_" + hexd)
        os.makedirs(cdir, exist_ok=True)
        sf.write(os.path.join(cdir, "output.f32.wav"),
                 (x * (0.9 - 0.25 * i)).astype(np.float32), sr, subtype="FLOAT")
        with open(os.path.join(cdir, "recipe.json"), "w", encoding="utf-8") as fh:
            json.dump({"schema": "audio-extract/recipe/v2",
                       "model": {"model_id": f"synthetic-model-{i}"},
                       "operation": {"type": "separate", "target": "instrumental",
                                     "construction": "native"},
                       "effective_config": {"overlap_factor": 4}}, fh, indent=2)


def ensure_passages(run_dir: str) -> None:
    pj = os.path.join(run_dir, "passages", "passages.v1.json")
    if os.path.isfile(pj):
        return
    try:
        with open(os.path.join(run_dir, "source", "source.json"), encoding="utf-8") as fh:
            frames = int(json.load(fh).get("frames") or 0)
    except (OSError, ValueError):
        frames = 0
    os.makedirs(os.path.dirname(pj), exist_ok=True)
    mk = lambda pid, a, b, tag: {"passage_id": pid, "start_sample": int(frames * a),
                                 "end_sample": int(frames * b), "tags": [tag],
                                 "features": {}, "reason": "dev seed"}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"schema": "audio-extract/passages/v1",
                   "passages": [mk("p_0000", 0.10, 0.40, "no_vocal_control"),
                                mk("p_0001", 0.50, 0.80, "vocal_overlap")]}, fh, indent=2)


def candidate_ids(run_dir: str) -> list[str]:
    out = []
    cdir = os.path.join(run_dir, "candidates")
    if os.path.isdir(cdir):
        for d in sorted(os.listdir(cdir)):
            if d.startswith("sha256_") and \
                    os.path.isfile(os.path.join(cdir, d, "output.f32.wav")):
                out.append("sha256:" + d[len("sha256_"):])
    return out


# ---------------------------------------------------------------------------
# synthetic severity cells -> a REAL selector report
# ---------------------------------------------------------------------------
def _cells(rng, centers: dict[str, float], spread: float, u: float, n: int = 8):
    cells = {}
    for d in DEFECTS:
        c = centers.get(d, 0.10)
        cells[d] = [(float(np.clip(rng.normal(c, spread), 0.0, 1.0)),
                     float(abs(rng.normal(u, u / 3 + 1e-9)))) for _ in range(n)]
    return cells


def _gates_from_cells(cells, extra: dict[str, float] | None = None):
    g = {"technical": 0.0, "severe_artifact": 0.05}
    for k in GATE_KEYS:
        g[k] = round(max(s for s, _ in cells[k]), 4)
    g.update(extra or {})
    return g


def build_candidates(scenario: str, cids: list[str]) -> dict[str, dict]:
    """Three severity profiles per scenario, keyed by the run's real recipe ids."""
    rng = np.random.default_rng(42)
    lo = {d: 0.08 for d in DEFECTS}
    profiles: list[dict]
    if scenario == "clear_winner":
        profiles = [
            {"cells": _cells(rng, lo, 0.015, 0.01)},                                  # winner
            {"cells": _cells(rng, {**lo, "hall": 0.55, "fullness": 0.40}, 0.01, 0.02)},
            {"cells": _cells(rng, {**lo, "vocal_leakage": 0.60, "brightness": 0.45}, 0.01, 0.02)},
        ]
    elif scenario == "best_safe":
        profiles = [
            {"cells": _cells(rng, {**lo, "hall": 0.38, "fullness": 0.30}, 0.06, 0.05)},
            {"cells": _cells(rng, {**lo, "hall": 0.35, "orchestral_theft": 0.30,
                                   "event_hole": 0.28}, 0.06, 0.05)},
            {"cells": _cells(rng, {**lo, "vocal_leakage": 0.80}, 0.02, 0.02)},        # gate-failer
        ]
    else:  # no_acceptable
        profiles = [
            {"cells": _cells(rng, {**lo, "vocal_leakage": 0.85, "hall": 0.40}, 0.02, 0.03)},
            {"cells": _cells(rng, {**lo, "orchestral_theft": 0.75, "event_hole": 0.80}, 0.02, 0.03)},
            {"cells": _cells(rng, {**lo, "brightness": 0.50}, 0.02, 0.03),
             "extra_gates": {"severe_artifact": 0.92}},
        ]
    out = {}
    for cid, prof in zip(cids, profiles):
        out[cid] = {"cells": prof["cells"],
                    "gates": _gates_from_cells(prof["cells"], prof.get("extra_gates"))}
    return out


# ---------------------------------------------------------------------------
# challenge cases + results (shapes mirror audio_extract/challenges.py)
# ---------------------------------------------------------------------------
def build_challenge_rows(sr: int, frames: int, cids: list[str], quality_rank: list[str]):
    """Returns (cases, results): result_json shapes mirror the real engine —
    exact_reference_error / theft assay / bleed assay / probe pass-through."""
    rng = np.random.default_rng(7)
    remix = []
    for level, pan_milli in ((-6.0, 0), (0.0, 150), (3.0, 0)):
        recipe = {"schema": "audio-extract/challenge/track-remix/v1",
                  "control_id": "p_0000", "vocal_id": "aux_v0",
                  "gain_micro_db": int(level * 1000), "rt60_ms": 1200,
                  "pan_milli": pan_milli, "seed": 0,
                  "sample_rate_hz": sr, "frames": frames}
        remix.append({
            "challenge_id": challenges.challenge_id(recipe),
            "challenge_type": "track_remix",
            "mixture_node_id": "sha256:" + hashlib.sha256(
                f"mix-{level}-{pan_milli}".encode()).hexdigest(),
            "target_node_id": "sha256:" + hashlib.sha256(
                f"tgt-{level}-{pan_milli}".encode()).hexdigest(),
            "recipe": recipe,
            "klass": {"vocal_level_db": level,
                      "pan": "center" if pan_milli == 0 else "off_center"},
        })
    aux = [
        {"challenge_id": challenges.challenge_id({"schema": "audio-extract/challenge/no-vocal-theft/v1",
                                                  "control_id": "p_0000", "sample_rate_hz": sr}),
         "challenge_type": "no_vocal_theft", "mixture_node_id": "sha256:" + "a1" * 32,
         "target_node_id": None,
         "recipe": {"schema": "audio-extract/challenge/no-vocal-theft/v1", "control_id": "p_0000"},
         "klass": {"control_id": "p_0000"}},
        {"challenge_id": challenges.challenge_id({"schema": "audio-extract/challenge/vocal-only-bleed/v1",
                                                  "vocal_id": "aux_v0", "sample_rate_hz": sr}),
         "challenge_type": "vocal_only_bleed", "mixture_node_id": "sha256:" + "b2" * 32,
         "target_node_id": None,
         "recipe": {"schema": "audio-extract/challenge/vocal-only-bleed/v1", "vocal_id": "aux_v0"},
         "klass": {"vocal_id": "aux_v0"}},
        {"challenge_id": challenges.challenge_id({"schema": "audio-extract/challenge/orchestra-probe/v1",
                                                  "probe": "brass_onset", "sample_rate_hz": sr}),
         "challenge_type": "orchestra_probe", "mixture_node_id": "sha256:" + "c3" * 32,
         "target_node_id": None,
         "recipe": {"schema": "audio-extract/challenge/orchestra-probe/v1", "probe": "brass_onset"},
         "klass": {"probe": "brass_onset"}},
    ]
    cases = remix + aux

    results = []   # (challenge_id, cid, result_dict)
    for case in cases:
        for cid in cids:
            k = quality_rank.index(cid) if cid in quality_rank else len(quality_rank)
            j = float(rng.normal(0, 0.4))
            if case["challenge_type"] == "track_remix":
                level = case["klass"]["vocal_level_db"]
                res = {"si_sdr_db": round(16.0 - 5.5 * k - 0.35 * max(0.0, level + 6) + j, 3),
                       "stft_distance": round(0.05 + 0.045 * k + 0.002 * abs(level) + abs(j) * 0.01, 5),
                       "band_envelope_err_db": round(0.8 + 1.1 * k + abs(j) * 0.3, 3),
                       "stereo_width_err": round(0.01 + 0.02 * k, 5),
                       "alignment": {"delay": 0, "confidence": round(0.99 - 0.03 * k, 4),
                                     "residual_db": round(-38.0 + 6.0 * k, 2)}}
            elif case["challenge_type"] == "no_vocal_theft":
                res = {"control_id": "p_0000",
                       "theft_broadband": round(0.01 + 0.06 * k + abs(j) * 0.005, 5),
                       "theft_mid": round(0.012 + 0.07 * k, 5),
                       "theft_side": round(0.008 + 0.05 * k, 5),
                       "band_excess_db": [round(0.1 + 0.4 * k + 0.05 * b, 3) for b in range(5)]}
            elif case["challenge_type"] == "vocal_only_bleed":
                res = {"vocal_id": "aux_v0",
                       "bleed_broadband": round(0.02 + 0.09 * k + abs(j) * 0.01, 5)}
            else:  # orchestra_probe pass-through error
                res = {"probe": "brass_onset",
                       "passthrough_error": round(0.06 + 0.11 * k + abs(j) * 0.02, 5)}
            results.append((case["challenge_id"], cid, res))
    return cases, results


def build_actions(cids: list[str]):
    c = (cids + ["sha256:" + "0" * 64] * 3)[:3]
    return [
        dict(action_id="act-0001", round_no=0, status="executed",
             proposed={"type": "run_model_variant", "model_id": "mel_band_roformer_big",
                       "target": "vocals"},
             validated_recipe_id=c[1], reason=None),
        dict(action_id="act-0002", round_no=0, status="executed",
             proposed={"type": "run_construction", "construction": "mixture_minus_primary"},
             validated_recipe_id=c[2], reason=None),
        dict(action_id="act-0003", round_no=0, status="rejected",
             proposed={"type": "change_overlap", "overlap_factor": 8},
             parent_recipe_id=c[0],
             reason="duplicate of a cached recipe (one material variable per experiment)"),
        dict(action_id="act-0004", round_no=1, status="executed",
             proposed={"type": "run_track_remix_challenge", "levels_db": [-6, 0, 3]},
             reason="discriminating probe: risk bounds of the top-2 candidates overlap"),
        dict(action_id="act-0005", round_no=1, status="rejected",
             proposed={"type": "build_weighted_ensemble", "members": 2},
             reason="ensemble budget exhausted"),
        dict(action_id="act-0006", round_no=1, status="executed",
             proposed={"type": "stop_with_reason"},
             reason="selector bounds separable after remix probe"),
    ]


# ---------------------------------------------------------------------------
# seeding one run
# ---------------------------------------------------------------------------
def seed_run(lib: str, run_id: str, scenario: str, base: str | None,
             separate_v2_file: bool) -> None:
    dest = os.path.join(lib, run_id)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    if base:
        copy_base_run(base, dest, run_id)
        origin = f"copied from {base}"
    else:
        got = try_cli_ingest(lib, run_id)
        origin = "cli ingest" if got else "synthesized"
        if not got:
            synthesize_run(dest, run_id)
    ensure_passages(dest)

    cids = candidate_ids(dest)
    if len(cids) < 3:                       # decision needs 3 profiles to be interesting
        synthesize_run(dest, run_id)
        cids = candidate_ids(dest)
    cids = cids[:3]

    with open(os.path.join(dest, "source", "source.json"), encoding="utf-8") as fh:
        src = json.load(fh)
    sr = int(src.get("sample_rate_hz") or 44100)
    frames = int(src.get("frames") or sr * 4)

    # --- REAL selector decision over synthetic severity cells ---
    cands = build_candidates(scenario, cids)
    decision = selector.select(cands, seed=0)
    expect = {"clear_winner": ("final", "clear_winner"),
              "best_safe": ("final", "best_safe"),
              "no_acceptable": ("no_acceptable_candidate", None)}[scenario]
    got = (decision["status"], decision.get("mode"))
    if got != expect:
        print(f"  WARNING: scenario {scenario!r} produced {got}, expected {expect}")

    quality_rank = sorted(cids, key=lambda c: decision["candidates"][c]["risk_mean"])
    cases, results = build_challenge_rows(sr, frames, cids, quality_rank)

    man_path = os.path.join(dest, "manifest_v2.sqlite" if separate_v2_file
                            else "manifest.sqlite")
    with ManifestV2(man_path) as m:
        # a couple of artifact nodes + observations (completeness; panel-optional)
        for i, cid in enumerate(cids):
            m.add_artifact_node(
                node_id=cid, node_type="candidate", role="accompaniment",
                cohort="instrumental-production-v1", operation="separate",
                parents=["sha256:src"], recipe={"seed": i}, status="complete",
                created_at="2026-08-06T09:00:00Z", sample_rate_hz=sr, frames=frames)
        m.record_observation(node_id=cids[0], scope_id="p_0000", metric="leakage/v2",
                             value=0.12, available=True, uncertainty=0.02)
        m.record_observation(node_id=cids[0], scope_id="p_0001", metric="hall/v2",
                             value=None, available=False)

        for case in cases:
            m.add_challenge_case(challenge_id=case["challenge_id"],
                                 challenge_type=case["challenge_type"],
                                 mixture_node_id=case["mixture_node_id"],
                                 target_node_id=case["target_node_id"],
                                 recipe=case["recipe"], klass=case["klass"])
        for chid, cid, res in results:
            m.add_challenge_result(challenge_id=chid, candidate_recipe_id=cid, result=res)

        for a in build_actions(cids):
            m.record_action(**a)

        m.add_judge_calibration(judge_id="mos-judge-v1", calibration_id="cal-2026-08",
                                report={"spearman": {"leakage": 0.83, "pumping": 0.78}},
                                passed_axes=["leakage", "pumping"])
        m.add_judge_calibration(judge_id="artifact-judge-v1", calibration_id="cal-2026-08",
                                report={"spearman": {"artifacts": 0.71}},
                                passed_axes=["artifacts"])

        # an earlier, stricter round-0 decision so "latest" selection is exercised
        early = selector.select(cands, max_acceptable_risk=0.02, delta=0.5, seed=0)
        m.add_selection_decision(
            decision_id=f"dec-{scenario}-r0", status=early["status"],
            selector_version=selector.SELECTOR_VERSION, report=early,
            created_at="2026-08-06T10:00:00Z",
            candidate_recipe_id=early.get("candidate_id"))
        m.add_selection_decision(
            decision_id=f"dec-{scenario}-r1", status=decision["status"],
            selector_version=selector.SELECTOR_VERSION, report=decision,
            created_at="2026-08-06T12:00:00Z",
            candidate_recipe_id=decision.get("candidate_id"))
        m.set_run_state(run_id,
                        "SELECTED" if decision["status"] == "final" else "NO_ACCEPTABLE",
                        updated_at="2026-08-06T12:00:00Z")

    where = os.path.basename(man_path)
    print(f"  {run_id}: {origin}; {len(cids)} candidates; decision={got[0]}"
          f"{('/' + got[1]) if got[1] else ''} -> {where}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", default="/tmp/guilib3", help="target GUI lib root")
    ap.add_argument("--base", default=None,
                    help="existing run dir to copy audio/candidates from "
                         "(default: first run found under /tmp/guilib*)")
    args = ap.parse_args()

    os.makedirs(args.lib, exist_ok=True)
    base = find_base(args.base)
    print(f"seeding {args.lib} (base run: {base or 'none — will ingest/synthesize'})")
    seed_run(args.lib, "dectest", "clear_winner", base, separate_v2_file=False)
    seed_run(args.lib, "dectest-safe", "best_safe", base, separate_v2_file=True)
    seed_run(args.lib, "dectest-none", "no_acceptable", base, separate_v2_file=False)
    print("done. serve with:\n  uv run python web/server_v2.py --lib %s --port 8752" % args.lib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
