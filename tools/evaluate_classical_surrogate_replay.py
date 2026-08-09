#!/usr/bin/env python3
"""Run the preregistered no-optimizer classical surrogate replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_extract.classical_surrogate_alignment_v2 import SurrogateConfigV2
from audio_extract.classical_surrogate_replay_v2 import (
    ReplayDecisionConfigV2,
    evaluate_replay_v2,
    load_rows_v2,
)

TOOL_SCHEMA = "audio-extract/classical-surrogate-replay-tool/v1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _commit(repository: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot establish source revision: {exc}") from exc
    if status:
        raise RuntimeError("surrogate replay requires a clean repository checkout")
    return head


def run(manifest: Path, output: Path, repository: Path) -> dict:
    manifest = manifest.resolve(strict=True)
    repository = repository.resolve(strict=True)
    rows = load_rows_v2(manifest)
    report = evaluate_replay_v2(
        rows,
        surrogate_config=SurrogateConfigV2(),
        decision_config=ReplayDecisionConfigV2(),
    )
    result = {
        "schema": TOOL_SCHEMA,
        "source_commit": _commit(repository),
        "input_manifest": str(manifest),
        "input_manifest_sha256": _sha_file(manifest),
        "row_count": len(rows),
        "report": report,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != payload:
            raise RuntimeError(f"refusing to rewrite differing output: {output}")
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("evaluate-classical-surrogate-replay")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.manifest, args.output, args.repository)
    print(json.dumps({
        "decision": result["report"]["decision"],
        "row_count": result["row_count"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
