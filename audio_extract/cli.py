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
from pathlib import Path
from typing import Any

from . import __version__, identity, recipe as recipe_mod
from .storage import TrackLayout


def _emit(obj: dict[str, Any], *, code: int = 0) -> int:
    json.dump(obj, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
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
    return _not_implemented(
        "passages.mine", args.run_id, "M2",
        "passage miner requires a provisional separator + PESTO/pYIN fusion (docs/v2 §2)",
    )


def cmd_panel_render(args: argparse.Namespace) -> int:
    return _not_implemented(
        "panel.render", args.run_id, "M3",
        "panel runner requires Tier A separator adapters + excerpt scheduler (docs/v2 §3)",
    )


def cmd_qa_score(args: argparse.Namespace) -> int:
    return _not_implemented(
        "qa.score", args.run_id, "M2",
        "objective metric bank (leakage/fullness/pumping/brightness/hall/stereo) not yet wired (docs/v2 §4)",
    )


def cmd_candidate_render(args: argparse.Namespace) -> int:
    return _not_implemented(
        "candidate.render", args.run_id, "M3",
        "candidate render requires separator adapters + float32 writer (docs/v2 §1.4, §7)",
    )


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
    sp = g_panel.add_parser("render", help="[M3] run the panel on mined excerpts")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--panel", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_panel_render)

    # passages
    g_pass = sub.add_parser("passages", help="passage miner").add_subparsers(dest="cmd", required=True)
    sp = g_pass.add_parser("mine", help="[M2] mine hard excerpts")
    sp.add_argument("--run-id", required=True)
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
    sp = g_cand.add_parser("render", help="[M3] render a candidate from a recipe")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--recipe", required=True)
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

    # ingest / fingerprint
    sp = sub.add_parser("ingest", help="decode a source to canonical float32 PCM + hashes")
    sp.add_argument("source")
    sp.add_argument("--run-id", required=True)
    sp.set_defaults(func=cmd_ingest)
    sp = sub.add_parser("fingerprint", help="print the execution fingerprint")
    sp.set_defaults(func=cmd_fingerprint)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # surface as a JSON error envelope, never a bare traceback
        return _emit(
            _envelope(getattr(args, "group", "?"), "error", getattr(args, "run_id", None),
                      error=type(exc).__name__, message=str(exc)),
            code=1,
        )


if __name__ == "__main__":
    raise SystemExit(main())
