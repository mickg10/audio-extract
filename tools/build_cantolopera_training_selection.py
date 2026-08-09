#!/usr/bin/env python3
"""Build the frozen Cantolopera Tier-A selection and leakage-safe splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from audio_extract.cantolopera_training import (
    _file_sha256,
    build_selection,
    load_audit,
    write_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--splits", required=True)
    args = parser.parse_args(argv)
    audit = Path(args.audit)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    decisions, manifest, summary, splits = build_selection(
        load_audit(audit), audit_sha256=_file_sha256(audit),
        selector_code_commit=commit,
    )
    write_outputs(
        decisions,
        manifest,
        summary,
        splits,
        selection_path=Path(args.selection),
        manifest_path=Path(args.manifest),
        summary_path=Path(args.summary),
        splits_path=Path(args.splits),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
