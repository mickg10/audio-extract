#!/usr/bin/env python3
"""Audit the acquired Cantolopera lossless pairs without modifying audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_extract.cantolopera_pairs import audit_inventory, write_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--probe-seconds", type=float, default=4.0)
    parser.add_argument("--probe-count", type=int, default=5)
    parser.add_argument("--max-shift-samples", type=int, default=8192)
    parser.add_argument("--null-block-seconds", type=float, default=0.5)
    parser.add_argument("--skip-container-hashes", action="store_true")
    args = parser.parse_args(argv)
    if args.probe_seconds <= 0 or args.probe_count <= 0:
        parser.error("probe seconds/count must be positive")
    if args.max_shift_samples < 0 or args.null_block_seconds <= 0:
        parser.error("shift must be non-negative and block seconds positive")
    rows, summary = audit_inventory(
        args.inventory,
        args.audio_root,
        verify_container_hashes=not args.skip_container_hashes,
        probe_seconds=args.probe_seconds,
        probe_count=args.probe_count,
        max_shift_samples=args.max_shift_samples,
        null_block_seconds=args.null_block_seconds,
    )
    write_audit(rows, summary, args.output_jsonl, args.output_summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
