"""``audio-extract`` v2 CLI.

Two families of subcommands:

* **Deterministic tools wired today** — ``recipe canon``, ``recipe id``,
  ``panel show``, ``ingest``, ``run inspect``, ``run finalize``, ``fingerprint``.
* **Conductor contracts** the Pi + DeepSeek layer will call — ``passages mine``,
  ``panel render``, ``qa score``, ``candidate render``. These already emit a
  stable JSON envelope; the model-dependent bodies land in Milestones 2–3.

Every command emits a single JSON object on stdout (the conductor consumes JSON;
humans can read it too). The envelope shape is frozen now so the tool contract is
stable before the internals are complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__, identity, recipe as recipe_mod
from .storage import TrackLayout


# The Pi conductor consumes JSON on stdout, but audio-separator / onnxruntime write
# progress to fd 1. main() redirects fd 1 -> stderr and points this at the real stdout.
_STDOUT = sys.stdout


def _emit(obj: dict[str, Any], *, code: int = 0) -> int:
    json.dump(obj, _STDOUT, ensure_ascii=False)
    _STDOUT.write("\n")
    _STDOUT.flush()
    return code


def _envelope(command: str, status: str, run_id: str | None = None, **payload: Any) -> dict[str, Any]:
    env: dict[str, Any] = {"ok": status == "ok", "command": command, "status": status}
    if run_id is not None:
        env["run_id"] = run_id
    env.update(payload)
    return env


def _not_implemented(command: str, run_id: str | None, milestone: str, message: str) -> int:
    return _emit(
        _envelope(command, "not_implemented", run_id, milestone=milestone, message=message),
        code=0,
    )


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# deterministic commands
# --------------------------------------------------------------------------
def cmd_recipe_canon(args: argparse.Namespace) -> int:
    from . import canon

    obj = _load_json(args.recipe)
    normalized = recipe_mod.normalize_recipe(obj)
    canonical = canon.canonicalize_str(normalized)
    rid = identity.recipe_id(obj)
    return _emit(
        _envelope("recipe.canon", "ok", recipe_id=rid, canonical=canonical,
                  problems=recipe_mod.validate_recipe(obj))
    )


def cmd_recipe_id(args: argparse.Namespace) -> int:
    obj = _load_json(args.recipe)
    return _emit(_envelope("recipe.id", "ok", recipe_id=identity.recipe_id(obj)))


def cmd_panel_show(args: argparse.Namespace) -> int:
    from .panel import load_panel

    panel = load_panel(args.panel)
    models = [
        {
            "id": m.id,
            "family": m.family,
            "adapter": m.adapter,
            "target": m.target,
            "checkpoint": m.checkpoint_filename,
            "sha256": m.checkpoint_sha256,
            "constructions": m.constructions,
            "needs_import": m.needs_import,
        }
        for m in panel.models
    ]
    return _emit(
        _envelope("panel.show", "ok", schema=panel.schema, defaults=panel.defaults,
                  model_count=len(models), models=models)
    )


def cmd_fingerprint(_args: argparse.Namespace) -> int:
    return _emit(_envelope("fingerprint", "ok", fingerprint=identity.execution_fingerprint()))


def cmd_train_classical(args: argparse.Namespace) -> int:
    """Run the deterministic pretrained classical-separator training contract."""
    from .train_classical import run_training

    report = run_training(args)
    return _emit(_envelope("train.classical", "ok", report=report))


def _resolve_locked_model(lock_path: str, name: str, model_dir: str):
    """If a model lock exists, resolve+verify ``name`` through it (v2.1 §6.5/§6.7).
    Returns (registry_filename, executed_bundle_id, expected_sha256). Without a lock
    file the legacy hash-at-run path applies (dev mode)."""
    from .model_lock import ModelLock

    lock = ModelLock(lock_path)
    if not lock.exists:
        return name, None, None
    bundle = lock.verify(name, model_dir)   # raises on unknown/mismatch/missing
    weights = next(f for f in bundle["files"] if f["role"] == "weights")
    return bundle["registry"]["alias"], bundle["bundle_sha256"], weights["sha256"]


def cmd_models_import(args: argparse.Namespace) -> int:
    """Resolve registry models to executed bundles and write the immutable lock."""
    from importlib.metadata import version as _pkg_ver

    from .model_lock import ModelLock, build_bundle, default_logical_id, guess_family

    try:
        adapter_version = _pkg_ver("audio-separator")
    except Exception:
        adapter_version = "unknown"

    lock = ModelLock(args.write_lock)
    imported, errors = [], []
    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        alias, _, target = spec.partition("=")
        target = target or "vocals"
        path = Path(args.model_dir) / alias
        try:
            bundle = build_bundle(
                logical_id=default_logical_id(alias),
                family=guess_family(alias),
                target_stem=target,
                registry_alias=alias,
                files=[("weights", path)],
                adapter_version=adapter_version,
                adapter_revision="audio-separator+audio-extract-adapter-v1",
                effective_defaults={
                    "model_sample_rate_hz": 44100, "segment_size": 256, "overlap_factor": 8,
                    "normalization_threshold": "1.000000", "amplification_threshold": "0.000000",
                    "use_soundfile": True,
                },
            )
            lock.add(bundle)
            imported.append({"logical_id": bundle["logical_id"], "alias": alias,
                             "bundle_sha256": bundle["bundle_sha256"],
                             "weights_sha256": bundle["files"][0]["sha256"]})
        except Exception as exc:
            errors.append({"model": alias, "error": f"{type(exc).__name__}: {exc}"})
    if imported:
        lock.write()
    return _emit(_envelope("models.import", "ok" if imported and not errors else
                           ("error" if not imported else "partial"),
                           lock=str(args.write_lock), imported=imported, errors=errors),
                 code=0 if imported else 2)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Decode a source file to canonical float32 PCM, hash it, lay out storage."""
    import numpy as np
    import soundfile as sf

    src = Path(args.source)
    if not src.exists():
        return _emit(_envelope("ingest", "error", args.run_id, message=f"no such file: {src}"), code=2)

    raw = src.read_bytes()
    source_blob = identity.blob_sha256(raw)

    # Prefer soundfile (native SR, float). Fall back to librosa for formats
    # libsndfile can't decode on this platform (e.g. some MP3s).
    try:
        samples, sr = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception:
        import librosa

        y, sr = librosa.load(str(src), sr=None, mono=False)
        arr = np.asarray(y, dtype="float32")
        samples = arr.T if arr.ndim == 2 else arr.reshape(-1, 1)

    frames, channels = int(samples.shape[0]), int(samples.shape[1])
    layout = {1: ["FC"], 2: ["FL", "FR"]}.get(channels, [f"CH{i}" for i in range(channels)])

    layout_obj = TrackLayout(args.lib, args.run_id).ensure()
    canonical_wav = layout_obj.source_dir / "canonical.f32.wav"
    sf.write(str(canonical_wav), samples, int(sr), subtype="FLOAT")

    input_pcm = identity.artifact_pcm_sha256(samples, int(sr), layout, frames)
    source_record = {
        "track_id": args.run_id,
        "original_filename": src.name,
        "source_blob_sha256": source_blob,
        "input_pcm_sha256": input_pcm,
        "sample_rate_hz": int(sr),
        "channel_layout": layout,
        "frames": frames,
        "sample_format": "float32-le-interleaved",
    }
    (layout_obj.source_dir / "source.json").write_text(json.dumps(source_record, indent=2))

    from .manifest import Manifest

    with Manifest(layout_obj.manifest_sqlite) as man:
        man.set_state(args.run_id, "INGESTED")

    return _emit(_envelope("ingest", "ok", args.run_id, **source_record,
                           canonical_pcm=str(canonical_wav)))


def cmd_run_inspect(args: argparse.Namespace) -> int:
    from .manifest import Manifest

    layout = TrackLayout(args.lib, args.run_id)
    if not layout.root.exists():
        return _not_implemented("run.inspect", args.run_id, "M1",
                                f"run {args.run_id!r} not ingested yet")
    source = None
    src_json = layout.source_dir / "source.json"
    if src_json.exists():
        source = json.loads(src_json.read_text())
    with Manifest(layout.manifest_sqlite) as man:
        state = man.get_state(args.run_id)
        candidates = man.list_candidates()
    return _emit(
        _envelope("run.inspect", "ok", args.run_id, state=state, source=source,
                  candidate_count=len(candidates), candidates=candidates)
    )


def cmd_run_finalize(args: argparse.Namespace) -> int:
    from .manifest import Manifest

    layout = TrackLayout(args.lib, args.run_id)
    if not layout.root.exists():
        return _emit(_envelope("run.finalize", "error", args.run_id,
                               message=f"run {args.run_id!r} not found"), code=2)
    with Manifest(layout.manifest_sqlite) as man:
        cand = man.get_candidate(args.candidate_id)
        if cand is None:
            return _emit(_envelope("run.finalize", "error", args.run_id,
                                   message=f"candidate {args.candidate_id!r} not in manifest"), code=2)
        man.set_state(args.run_id, "COMPLETE")
    return _emit(_envelope("run.finalize", "ok", args.run_id, candidate_id=args.candidate_id,
                           finalized=cand))


# --------------------------------------------------------------------------
# conductor contracts (stable envelope; bodies land in later milestones)
# --------------------------------------------------------------------------
def cmd_passages_mine(args: argparse.Namespace) -> int:
    """Provisional-separate (kim_vocal_2) then mine hard passages (docs/v2 §2)."""
    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    canonical = layout.source_dir / "canonical.f32.wav"
    if not canonical.exists():
        return _emit(_envelope("passages.mine", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)

    from .separate import provisional_vocal
    from .passages import activity_timebase, mine_passages, write_passages, MinerConfig
    from .manifest import Manifest

    vocal, accomp, sr, out = provisional_vocal(
        canonical, model_filename=args.model, overlap=args.overlap, model_dir=args.model_dir
    )
    import soundfile as sf

    sf.write(str(layout.source_dir / "provisional_vocal.f32.wav"),
             vocal.astype("float32"), sr, subtype="FLOAT")  # leakage reference for qa score
    cfg = MinerConfig(seed=args.seed)
    passages = mine_passages(vocal, accomp, sr, cfg)

    layout.passages_dir.mkdir(parents=True, exist_ok=True)
    write_passages(passages, layout.passages_dir / "passages.v1.json",
                   timebase=activity_timebase(sr, cfg))
    with Manifest(layout.manifest_sqlite) as man:
        man.upsert_passages([asdict(p) for p in passages])
        man.set_state(args.run_id, "MINING_PASSAGES")

    summary = [
        {"passage_id": p.passage_id, "start_sample": p.start_sample,
         "end_sample": p.end_sample, "tags": p.tags} for p in passages
    ]
    return _emit(_envelope("passages.mine", "ok", args.run_id,
                           provisional_model=out.model_filename,
                           provisional_model_sha256=out.model_sha256,
                           passage_count=len(passages), passages=summary))


def cmd_panel_render(args: argparse.Namespace) -> int:
    """Render one comparable candidate per model (a panel round). Driven by
    explicit audio-separator registry names; the panel.yaml -> registry importer is
    a follow-up. Each model yields the same construction so candidates are comparable."""
    from .separate import render_candidate
    from .manifest import Manifest

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("panel.render", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    source_record = json.loads(src_json.read_text())
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rendered, errors = [], []
    for model_name in models:
        try:
            filename, bundle_id, expected = _resolve_locked_model(args.lock, model_name, args.model_dir)
            rec = render_candidate(
                layout, source_record, model_filename=filename, target=args.target,
                construction=args.construction, overlap=args.overlap, code_commit=_code_commit(),
                model_dir=args.model_dir, executed_bundle_id=bundle_id, expected_sha256=expected,
            )
            rendered.append({"model": model_name, "recipe_id": rec["recipe_id"],
                             "cached": rec.get("cached", False), "locked": bundle_id is not None})
        except Exception as exc:
            errors.append({"model": model_name, "error": f"{type(exc).__name__}: {exc}"})

    with Manifest(layout.manifest_sqlite) as man:
        man.set_state(args.run_id, "SCREENING")
    return _emit(_envelope("panel.render", "ok" if rendered else "error", args.run_id,
                           construction=args.construction, rendered=rendered, errors=errors))


def _candidate_role(construction: str, target: str) -> str:
    """Semantic role of a separator candidate's output stem, so only accompaniment
    candidates enter one ranking round (oracle review P0 #1)."""
    if construction == "native_primary":
        out = target
    else:  # native_secondary / mixture_minus_*: the complement of the target stem
        out = "instrumental" if target == "vocals" else "vocals"
    return "accompaniment" if out == "instrumental" else "vocal"


def _score_store(layout: "TrackLayout", sr: int, *, write_metrics: bool = True):
    """Score every accompaniment candidate in the store against a consensus
    reference. Returns ``(scored, report)``; shared by `qa score` and `conduct`."""
    import numpy as np
    import soundfile as sf

    from . import metrics as mx
    from . import panel_runner as pr
    from .manifest import Manifest

    # Source peak: float audio can legitimately exceed 1.0 (mp3 decode overshoot);
    # residuals inherit it. Clipping is judged against the source's own headroom.
    source_peak = None
    canonical = layout.source_dir / "canonical.f32.wav"
    if canonical.exists():
        src_arr, _ = sf.read(str(canonical), dtype="float64", always_2d=True)
        source_peak = float(np.max(np.abs(src_arr))) if src_arr.size else None

    cands: dict[str, Any] = {}
    rejected: list[dict] = []
    for d in sorted(layout.candidates_dir.glob("sha256_*")):
        wav = d / "output.f32.wav"
        if not wav.exists():
            continue
        rid = d.name.replace("sha256_", "sha256:")
        # Only accompaniment-eligible candidates enter one ranking round — never mix
        # native vocals or auxiliary outputs into the instrumental consensus.
        rj = d / "recipe.json"
        if rj.exists():
            op = json.loads(rj.read_text()).get("operation", {})
            if _candidate_role(op.get("construction", ""), op.get("target", "")) != "accompaniment":
                rejected.append({"recipe_id": rid, "reason": "role", "detail": "not accompaniment"})
                continue
        arr, _ = sf.read(str(wav), dtype="float64", always_2d=True)
        hc = mx.hard_checks(arr, sr, source_peak=source_peak)
        if not hc["ok"]:   # hard-reject BEFORE Pareto ranking — but never silently
            rejected.append({"recipe_id": rid, "reason": "hard_checks", "detail": hc["problems"]})
            continue
        cands[rid] = arr
    if not cands:
        return [], {"ranking": [], "pareto_frontier": [], "candidate_count": 0, "axes": [],
                    "rejected": rejected}

    reference = pr.consensus_reference(list(cands.values())) if len(cands) > 1 else next(iter(cands.values()))
    pv = layout.source_dir / "provisional_vocal.f32.wav"
    vocal = sf.read(str(pv), dtype="float64", always_2d=True)[0] if pv.exists() else None
    windows = None
    pj = layout.passages_dir / "passages.v1.json"
    if pj.exists():
        windows = [(p["start_sample"], p["end_sample"]) for p in json.loads(pj.read_text())["passages"]]

    scored = [pr.Scored(rid, pr.score_candidate(arr, sr, reference=reference, vocal_ref=vocal, windows=windows))
              for rid, arr in cands.items()]
    if write_metrics:
        with Manifest(layout.manifest_sqlite) as man:
            for s in scored:
                for axis, val in s.costs.items():
                    man.upsert_metric({"recipe_id": s.recipe_id, "passage_id": "__aggregate__",
                                       "metric": f"{axis}/v1", "value": val, "unit": "", "details": {}})
    report = {
        "candidate_count": len(scored),
        "axes": pr._axes_for(scored),
        "pareto_frontier": [s.recipe_id for s in pr.pareto_frontier(scored)],
        "ranking": [{"recipe_id": s.recipe_id, "costs": s.costs} for s in pr.rank_by_scalarized(scored)],
        "rejected": rejected,
    }
    return scored, report


def cmd_qa_score(args: argparse.Namespace) -> int:
    """Score store candidates -> metric rows + Pareto frontier + scalarized ranking."""
    from .manifest import Manifest

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("qa.score", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    sr = json.loads(src_json.read_text())["sample_rate_hz"]
    scored, report = _score_store(layout, sr, write_metrics=True)
    if not scored:
        return _not_implemented("qa.score", args.run_id, "M3",
                                "no candidates rendered yet (run `candidate render` / `panel render`)")
    with Manifest(layout.manifest_sqlite) as man:
        man.set_state(args.run_id, "MEASURING")
    return _emit(_envelope("qa.score", "ok", args.run_id, **report))


def cmd_conduct(args: argparse.Namespace) -> int:
    """Run the bounded Pi/DeepSeek conductor loop over the run (docs/v2 §6)."""
    from . import conductor as cd
    from .manifest import Manifest
    from .separate import render_candidate

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("conduct", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    source_record = json.loads(src_json.read_text())
    sr = source_record["sample_rate_hz"]
    code_commit = _code_commit()

    def execute(action: dict) -> dict | None:
        if action["type"] == "build_weighted_ensemble":
            from .separate import render_ensemble_candidate

            rec = render_ensemble_candidate(
                layout, source_record,
                member_recipe_ids=list(action.get("members", [])),
                algo=action.get("algo", "median"),
                weights=action.get("weights"), code_commit=code_commit)
            return {"recipe_id": rec["recipe_id"]}
        rec = render_candidate(
            layout, source_record, model_filename=action.get("model", args.model),
            target=action.get("target", "vocals"), construction=action.get("construction", "native_primary"),
            overlap=int(action.get("overlap", 8)), code_commit=code_commit, model_dir=args.model_dir,
        )
        return {"recipe_id": rec["recipe_id"]}

    def score(_cands):
        _s, report = _score_store(layout, sr, write_metrics=True)
        return report

    if args.planner == "deepseek":
        planner: cd.Planner = cd.DeepSeekPlanner()
    else:
        planner = cd.ScriptedPlanner(json.loads(Path(args.script).read_text()))

    _scored, baseline = _score_store(layout, sr, write_metrics=False)
    conductor = cd.Conductor(cd.Budget(), execute, score, planner)
    decision = conductor.run([{"recipe_id": r["recipe_id"]} for r in baseline["ranking"]], baseline)

    if decision.get("status") == "final" and decision.get("candidate_id"):
        # §19.6: selection is FINALIST_SELECTED, never COMPLETE — delivery + QC finish the run
        with Manifest(layout.manifest_sqlite) as man:
            man.set_state(args.run_id, "FINALIST_SELECTED")
    from .cli_autonomous import record_decision

    decision_id = record_decision(layout, decision)
    return _emit(_envelope("conduct", "ok", args.run_id, decision=decision,
                           decision_id=decision_id, rounds_log=conductor.log))


def _real_estimator_factory(model_dir: str, lock_path: str, sr: int):
    """model name -> estimator(audio)->stems, via the locked Separator (cached)."""
    import os
    import tempfile

    import numpy as np
    import soundfile as sf

    from .separate import Separator

    cache: dict[str, Any] = {}

    def factory(name: str):
        if name not in cache:
            filename, _bid, expected = _resolve_locked_model(lock_path, name, model_dir)
            del expected
            cache[name] = Separator(filename, model_dir=model_dir)
        sep = cache[name]

        def estimate(audio):
            fd, tmp = tempfile.mkstemp(suffix=".f32.wav")
            os.close(fd)
            try:
                sf.write(tmp, np.asarray(audio, dtype="float32"), sr, subtype="FLOAT")
                return sep.separate_file(tmp).stems
            finally:
                os.unlink(tmp)
        return estimate

    return factory


def cmd_challenges(args: argparse.Namespace) -> int:
    from . import cli_autonomous as auto

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope(f"challenges.{args.cmd}", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    sr = json.loads(src_json.read_text())["sample_rate_hz"]
    if args.cmd == "build":
        res = auto.build_challenges(layout, sr, count=args.count, seed=args.seed)
    elif args.cmd == "run":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        res = auto.run_challenges(layout, sr, models,
                                  _real_estimator_factory(args.model_dir, args.lock, sr))
    else:  # report
        res = {"cases": len(auto.severity_cells_from_store(layout)),
               "cells": auto.severity_cells_from_store(layout)}
    return _emit(_envelope(f"challenges.{args.cmd}", "ok", args.run_id, **res))


def cmd_candidate_ensemble(args: argparse.Namespace) -> int:
    """Render a vocal-ensemble residual candidate (oracle §5 ε-minimization)."""
    from .separate import render_ensemble_candidate

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("candidate.ensemble", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    weights = ([float(x) for x in args.weights.split(",")] if args.weights else None)
    rec = render_ensemble_candidate(
        layout, json.loads(src_json.read_text()), member_recipe_ids=members,
        algo=args.algo, weights=weights, code_commit=_code_commit(),
        mixture_recipe_id=args.mixture_recipe_id)
    return _emit(_envelope("candidate.ensemble", "ok", args.run_id, candidate=rec))


def cmd_candidate_channel_map(args: argparse.Namespace) -> int:
    """Render an explicit immutable channel-map candidate."""
    from .separate import render_channel_map_candidate

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("candidate.channel-map", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    rec = render_channel_map_candidate(
        layout, json.loads(src_json.read_text()), mode=args.mode,
        code_commit=_code_commit())
    return _emit(_envelope("candidate.channel-map", "ok", args.run_id, candidate=rec))


def cmd_candidate_resample(args: argparse.Namespace) -> int:
    """Render an explicit immutable sample-rate conversion candidate."""
    from .resample import render_resample_candidate

    layout = TrackLayout(args.lib, args.run_id)
    rec = render_resample_candidate(
        layout, args.parent_recipe_id, target_rate_hz=args.target_rate_hz,
        algo=args.algo, code_commit=_code_commit())
    return _emit(_envelope("candidate.resample", "ok", args.run_id, candidate=rec))


def cmd_finalists_render(args: argparse.Namespace) -> int:
    """Certified delivery is bound to an immutable decision (--decision-id). Direct
    candidate rendering is allowed ONLY as an explicit --uncertified-preview (§9)."""
    from .cli_autonomous import finalize_and_deliver, finalize_from_decision

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("finalists.render", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    sr = json.loads(src_json.read_text())["sample_rate_hz"]
    if args.decision_id:
        res = finalize_from_decision(layout, sr, args.decision_id,
                                     calibration_path=args.calibration,
                                     target_dbfs=args.target_dbfs, bitrate=args.bitrate,
                                     code_commit=_code_commit())
    elif args.uncertified_preview and args.candidate_id:
        res = finalize_and_deliver(layout, sr, args.candidate_id,
                                   target_dbfs=args.target_dbfs, bitrate=args.bitrate,
                                   code_commit=_code_commit())
        res["certified"] = False
        res["rollout_level"] = "engineering_preview"
    else:
        return _emit(_envelope("finalists.render", "error", args.run_id,
                               message="certified render requires --decision-id; a direct "
                               "--candidate-id render requires --uncertified-preview"), code=2)
    ok = res["status"] == "delivered"
    return _emit(_envelope("finalists.render", "ok" if ok else "error", args.run_id, **res),
                 code=0 if ok else 3)


def cmd_select_autonomous(args: argparse.Namespace) -> int:
    from . import cli_autonomous as auto

    layout = TrackLayout(args.lib, args.run_id)
    decision = auto.select_autonomous(
        layout, calibration_path=args.calibration, target_risk=args.target_risk,
        task=args.task, domain=args.domain)
    return _emit(_envelope("select.autonomous", "ok", args.run_id, decision=decision))


def cmd_run_report(args: argparse.Namespace) -> int:
    from . import cli_autonomous as auto

    layout = TrackLayout(args.lib, args.run_id)
    return _emit(_envelope("run.report", "ok", args.run_id, report=auto.run_report(layout)))


def cmd_deliver(args: argparse.Namespace) -> int:
    """Render the delivery DAG (gain → dither/AAC children) for a finalist (docs/v2 §7)."""
    from .delivery import deliver

    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("deliver", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    if not (layout.candidate_dir(args.candidate_id) / "output.f32.wav").exists():
        return _emit(_envelope("deliver", "error", args.run_id,
                               message=f"candidate {args.candidate_id!r} not in store"), code=2)
    report = deliver(layout, json.loads(src_json.read_text()), args.candidate_id,
                     target_dbfs=args.target_dbfs, bitrate=args.bitrate, code_commit=_code_commit())
    return _emit(_envelope("deliver", "ok", args.run_id, delivery=report))


def _code_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return __version__


def cmd_candidate_render(args: argparse.Namespace) -> int:
    """Render one immutable separator candidate, keyed by recipe_id (docs/v2 §1.4)."""
    layout = TrackLayout(args.lib, args.run_id)
    src_json = layout.source_dir / "source.json"
    if not src_json.exists():
        return _emit(_envelope("candidate.render", "error", args.run_id,
                               message=f"run {args.run_id!r} not ingested"), code=2)
    source_record = json.loads(src_json.read_text())

    from .separate import render_candidate

    filename, bundle_id, expected = _resolve_locked_model(args.lock, args.model, args.model_dir)
    rec = render_candidate(
        layout, source_record, model_filename=filename, target=args.target,
        construction=args.construction, overlap=args.overlap, code_commit=_code_commit(),
        model_dir=args.model_dir, executed_bundle_id=bundle_id, expected_sha256=expected,
        input_recipe_id=args.input_recipe_id,
    )
    return _emit(_envelope("candidate.render", "ok", args.run_id, candidate=rec,
                           locked=bundle_id is not None))


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-extract", description="audio-extract v2 deterministic core")
    p.add_argument("--version", action="version", version=f"audio-extract {__version__}")
    p.add_argument("--lib", default="lib", help="library root (default: lib)")
    sub = p.add_subparsers(dest="group", required=True)

    # recipe
    g_recipe = sub.add_parser("recipe", help="canonical recipe identity").add_subparsers(dest="cmd", required=True)
    sp = g_recipe.add_parser("canon", help="normalize + canonicalize a recipe.json, print recipe_id")
    sp.add_argument("--recipe", required=True)
    sp.set_defaults(func=cmd_recipe_canon)
    sp = g_recipe.add_parser("id", help="print recipe_id for a recipe.json")
    sp.add_argument("--recipe", required=True)
    sp.set_defaults(func=cmd_recipe_id)

    # panel
    g_panel = sub.add_parser("panel", help="model panel").add_subparsers(dest="cmd", required=True)
    sp = g_panel.add_parser("show", help="validate + list panel bundles")
    sp.add_argument("--panel", required=True)
    sp.set_defaults(func=cmd_panel_show)
    sp = g_panel.add_parser("render", help="render one comparable candidate per model")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--models", default="Kim_Vocal_2.onnx",
                    help="comma-separated audio-separator registry model names")
    sp.add_argument("--target", default="vocals", choices=["vocals", "instrumental"])
    sp.add_argument("--construction", default="mixture_minus_primary",
                    choices=["native_primary", "native_secondary", "mixture_minus_primary"])
    sp.add_argument("--overlap", type=int, default=8)
    sp.add_argument("--model-dir", default=str(Path.home() / "audio-extract" / "models"))
    sp.add_argument("--lock", default="configs/model-lock.json")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_panel_render)

    # passages
    g_pass = sub.add_parser("passages", help="passage miner").add_subparsers(dest="cmd", required=True)
    sp = g_pass.add_parser("mine", help="provisional-separate + mine hard excerpts")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--model", default="Kim_Vocal_2.onnx", help="provisional vocal model")
    sp.add_argument("--overlap", type=int, default=8)
    sp.add_argument("--model-dir", default=str(Path.home() / "audio-extract" / "models"))
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_passages_mine)

    # qa
    g_qa = sub.add_parser("qa", help="objective QA").add_subparsers(dest="cmd", required=True)
    sp = g_qa.add_parser("score", help="[M2] score candidates on mined passages")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_qa_score)

    # candidate
    g_cand = sub.add_parser("candidate", help="candidate DAG").add_subparsers(dest="cmd", required=True)
    sp = g_cand.add_parser("ensemble", help="vocal-ensemble residual candidate (median/mean of vocal estimates)")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--members", required=True, help="comma-separated VOCAL candidate recipe_ids")
    sp.add_argument("--algo", default="median",
                    choices=["median", "mean", "stft_geometric_median"])
    sp.add_argument("--weights", help="comma-separated weights (mean only)")
    sp.add_argument("--mixture-recipe-id",
                    help="explicit channel-map candidate used as the mixture/input")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_candidate_ensemble)
    sp = g_cand.add_parser("channel-map", help="explicit immutable mono-to-stereo transform")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--mode", default="duplicate_mono_to_stereo",
                    choices=["duplicate_mono_to_stereo"])
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_candidate_channel_map)
    sp = g_cand.add_parser("resample", help="explicit immutable sample-rate conversion")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--parent-recipe-id", required=True)
    sp.add_argument("--target-rate-hz", type=int, default=48_000)
    sp.add_argument("--algo", default="scipy_polyphase", choices=["scipy_polyphase"])
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_candidate_resample)
    sp = g_cand.add_parser("render", help="render an immutable separator candidate")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--model", required=True, help="separator model filename (audio-separator registry)")
    sp.add_argument("--target", default="vocals", choices=["vocals", "instrumental"])
    sp.add_argument("--construction", default="native_primary",
                    choices=["native_primary", "native_secondary", "mixture_minus_primary"])
    sp.add_argument("--overlap", type=int, default=8)
    sp.add_argument("--model-dir", default=str(Path.home() / "audio-extract" / "models"))
    sp.add_argument("--lock", default="configs/model-lock.json")
    sp.add_argument("--input-recipe-id",
                    help="explicit channel-map candidate to feed to the separator")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_candidate_render)

    # run
    g_run = sub.add_parser("run", help="run lifecycle").add_subparsers(dest="cmd", required=True)
    sp = g_run.add_parser("inspect", help="report run state + candidates")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_run_inspect)
    sp = g_run.add_parser("finalize", help="mark a candidate as the finalist")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--candidate-id", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_run_finalize)
    sp = g_run.add_parser("report", help="consolidated §17.3 autonomous-run report")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_run_report)

    # challenges — exact-reference challenge engine (WP4/WP12)
    g_ch = sub.add_parser("challenges", help="exact-reference challenges").add_subparsers(dest="cmd", required=True)
    sp = g_ch.add_parser("build", help="build remix cases from genuine no-vocal controls")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--count", type=int, default=8)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_challenges)
    sp = g_ch.add_parser("run", help="run candidate models over built challenges")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--models", required=True, help="comma-separated lock logical ids")
    sp.add_argument("--model-dir", default=str(Path.home() / "audio-extract" / "models"))
    sp.add_argument("--lock", default="configs/model-lock.json")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_challenges)
    sp = g_ch.add_parser("report", help="stored challenge cells per candidate")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_challenges)

    # select — the deterministic terminal authority (WP8/WP12)
    g_sel = sub.add_parser("select", help="autonomous selection").add_subparsers(dest="cmd", required=True)
    sp = g_sel.add_parser("autonomous", help="run the robust selector over stored challenge results")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--calibration", help="frozen calibration artifact (certified path); "
                    "omit for an uncertified engineering preview")
    sp.add_argument("--target-risk", default="0.1", choices=["0.1", "0.15", "0.2"])
    sp.add_argument("--task", default="soloist_vs_rest")
    sp.add_argument("--domain", help="target domain/coverage class descriptor")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_select_autonomous)

    # ingest / fingerprint
    sp = sub.add_parser("ingest", help="decode a source to canonical float32 PCM + hashes")
    sp.add_argument("source")
    sp.add_argument("--run-id", required=True)
    sp.set_defaults(func=cmd_ingest)
    sp = sub.add_parser("fingerprint", help="print the execution fingerprint")
    sp.set_defaults(func=cmd_fingerprint)

    # train — deterministic model training; the conductor never manipulates tensors
    g_train = sub.add_parser("train", help="deterministic separator training").add_subparsers(
        dest="cmd", required=True
    )
    sp = g_train.add_parser("classical", help="pretrained HTDemucs opera continuation")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--split-manifest", required=True)
    sp.add_argument("--config", required=True)
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--truth-root", default="/home/mickg/truth_pairs")
    sp.add_argument("--steps", type=int)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--device")
    sp.add_argument("--resume", help="immutable checkpoint-step-XXXXXX.pt to resume")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_train_classical)

    # models — executed model lock
    g_models = sub.add_parser("models", help="executed model lock").add_subparsers(dest="cmd", required=True)
    sp = g_models.add_parser("import", help="hash models into the immutable lock (v2.1 §6)")
    sp.add_argument("--models", required=True,
                    help="comma-separated registry filenames, optional =target (e.g. Kim_Vocal_2.onnx=vocals)")
    sp.add_argument("--model-dir", default=str(Path.home() / "audio-extract" / "models"))
    sp.add_argument("--write-lock", default="configs/model-lock.json")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_models_import)

    # conduct — the bounded Pi/DeepSeek conductor loop
    sp = sub.add_parser("conduct", help="run the bounded conductor loop over a run")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--planner", default="deepseek", choices=["deepseek", "scripted"])
    sp.add_argument("--script", help="JSON file of proposals for --planner scripted")
    sp.add_argument("--model", default="Kim_Vocal_2.onnx")
    sp.add_argument("--model-dir", default=str(Path.home() / "audio-extract" / "models"))
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_conduct)

    # finalists — revalidate on mined passages, then deliver (the last mile)
    g_fin = sub.add_parser("finalists", help="finalist revalidation + delivery").add_subparsers(dest="cmd", required=True)
    sp = g_fin.add_parser("render", help="decision-bound certified delivery (--decision-id), "
                          "or --uncertified-preview --candidate-id")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--decision-id", help="immutable selection decision to render (certified path)")
    sp.add_argument("--candidate-id", help="direct candidate (requires --uncertified-preview)")
    sp.add_argument("--uncertified-preview", action="store_true",
                    help="render a candidate directly, labeled uncertified")
    sp.add_argument("--calibration", help="artifact to re-verify the decision's hash against")
    sp.add_argument("--target-dbfs", type=float, default=-5.0)
    sp.add_argument("--bitrate", default="256k")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_finalists_render)

    # deliver — render delivery children for a finalist
    sp = sub.add_parser("deliver", help="render delivery children (gain/dither/AAC) for a finalist")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--candidate-id", required=True)
    sp.add_argument("--target-dbfs", type=float, default=-5.0)
    sp.add_argument("--bitrate", default="256k")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_deliver)

    return p


def main(argv: list[str] | None = None) -> int:
    global _STDOUT
    import os

    parser = build_parser()
    args = parser.parse_args(argv)  # --version/--help/errors happen here, before redirect
    try:
        _STDOUT = os.fdopen(os.dup(1), "w", encoding="utf-8")
        os.dup2(2, 1)  # library output on fd 1 now goes to stderr; stdout stays JSON-only
    except Exception:
        _STDOUT = sys.stdout
    try:
        return args.func(args)
    except Exception as exc:  # surface as a JSON error envelope, never a bare traceback
        return _emit(
            _envelope(getattr(args, "group", "?"), "error", getattr(args, "run_id", None),
                      error=type(exc).__name__, message=str(exc)),
            code=1,
        )
    finally:
        try:
            _STDOUT.flush()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
