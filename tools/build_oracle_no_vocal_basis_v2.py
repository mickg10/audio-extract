#!/usr/bin/env python3
"""Complete the strict eight-alias Aalto no-vocal routing basis.

The input is the five-member JSON report produced by
`tools/render_oracle_no_vocal_basis.py`. The command verifies those immutable
singles, constructs median/geometric-median/uniform accompaniment ensembles on
the exact orchestra-only source, and writes a strict basis-v2 JSONL manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess

from audio_extract.oracle_no_vocal_basis_v2 import (
    build_complete_basis,
    write_strict_manifest,
)
from audio_extract.oracle_routing_basis_v2 import (
    basis_report,
    deduplicate_basis,
    verify_basis,
    write_jsonl,
)
from audio_extract.oracle_routing_runner_v2 import REQUIRED_ALIASES


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise RuntimeError(f"refusing to replace differing report: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the complete certified no-vocal basis v2"
    )
    parser.add_argument("--control-manifest", required=True, type=Path)
    parser.add_argument("--source-audio", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--strict-manifest", required=True, type=Path)
    parser.add_argument("--deduplicated-manifest", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--work-id")
    parser.add_argument("--code-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code_commit = args.code_commit or _git_commit()
    rows = build_complete_basis(
        control_manifest=args.control_manifest,
        source_audio=args.source_audio,
        output_root=args.output_root,
        code_commit=code_commit,
        work_id_override=args.work_id,
    )
    write_strict_manifest(args.strict_manifest, rows)
    verified = verify_basis(rows)
    deduplicated = deduplicate_basis(
        verified, required_aliases=REQUIRED_ALIASES
    )
    dedup_path = (
        args.deduplicated_manifest
        or args.strict_manifest.with_name("no-vocal-basis-v2-deduplicated.jsonl")
    )
    audit_path = (
        args.audit_report
        or args.strict_manifest.with_name("no-vocal-basis-v2-audit.json")
    )
    write_jsonl(dedup_path, deduplicated)
    audit = basis_report(verified, deduplicated)
    audit.update({
        "code_commit": code_commit,
        "control_manifest": str(args.control_manifest.resolve()),
        "source_audio": str(args.source_audio.resolve()),
        "strict_manifest": str(args.strict_manifest.resolve()),
        "deduplicated_manifest": str(dedup_path.resolve()),
    })
    _write_json(audit_path, audit)
    print(json.dumps({
        "status": "complete",
        "strict_rows": len(rows),
        "unique_decoded_candidates": len(deduplicated),
        "strict_manifest": str(args.strict_manifest.resolve()),
        "deduplicated_manifest": str(dedup_path.resolve()),
        "audit_report": str(audit_path.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
