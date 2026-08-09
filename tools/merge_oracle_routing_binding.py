#!/usr/bin/env python3
"""Merge independently archived per-work binding-routing fragments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.oracle_routing_binding import DEFAULT_WORKS, SCHEMA, _decision
from tools.oracle_routing_envelope import _sha_file


def run(fragment_paths: list[Path], output: Path) -> dict:
    if not fragment_paths:
        raise ValueError("at least one binding fragment is required")
    fragments = []
    works = {}
    invariant = None
    for path in fragment_paths:
        document = json.loads(path.read_text())
        if document.get("schema") != SCHEMA:
            raise ValueError(f"wrong binding fragment schema: {path}")
        facts = {
            key: document[key]
            for key in (
                "code_commit", "candidate_manifest_sha256",
                "frozen_resolutions_seconds", "basis_requested",
            )
        }
        if invariant is None:
            invariant = facts
        elif facts != invariant:
            raise ValueError(f"binding fragment identity differs: {path}")
        overlap = set(works) & set(document.get("works", {}))
        if overlap:
            raise ValueError(f"duplicate works across binding fragments: {sorted(overlap)}")
        works.update(document.get("works", {}))
        fragments.append({
            "path": str(path.resolve()), "sha256": _sha_file(path),
            "works": sorted(document.get("works", {})),
        })
    missing = sorted(set(DEFAULT_WORKS) - set(works))
    extra = sorted(set(works) - set(DEFAULT_WORKS))
    if missing or extra:
        raise ValueError(f"binding work set differs: missing={missing}, extra={extra}")
    assert invariant is not None
    report = {
        "schema": SCHEMA,
        "status": "final",
        "claim": "exact-reference diagnostic only",
        **invariant,
        "source_fragments": fragments,
        "works": {work: works[work] for work in DEFAULT_WORKS},
    }
    report["decision"] = _decision(report["works"])
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text() != payload:
        raise RuntimeError(f"refusing to rewrite differing merged report: {output}")
    if not output.exists():
        output.write_text(payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args.fragment, args.output)
    print(json.dumps({
        "status": report["status"],
        "learned_gate_authorized": report["decision"]["learned_gate_authorized"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
