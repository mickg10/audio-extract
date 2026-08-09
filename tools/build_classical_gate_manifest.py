#!/usr/bin/env python3
"""Build and independently rehash the exact input manifest for a gate fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_extract.train_classical_gate import (
    MANIFEST_SCHEMA,
    _sha_file,
    audio_record,
    load_manifest,
)


TRUTH_FILES = {
    "mixture": "mix_with_voice.wav",
    "accompaniment": "orchestra_only.wav",
    "vocal": "voice_ref.wav",
}


def _rows(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["work_id"]] = row
    return result


def _candidate_rows(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            result[(row["work_id"], row["candidate"])] = row
    return result


def _local_path(value: str) -> Path:
    return Path(value.removeprefix("research6:"))


def _basis_record(row: dict, label: str) -> dict:
    record = audio_record(
        _local_path(row["path"]), recipe_id=row["recipe_id"],
        executed_bundle_hash=row.get("executed_bundle_hash"),
    )
    for key in ("container_sha256", "artifact_pcm_sha256"):
        if row.get(key) != record[key]:
            raise ValueError(f"{label}/{key} mismatch: {row.get(key)} != {record[key]}")
    return record


def _control_member(report: dict, member: str, label: str) -> dict:
    matches = [row for row in report.get("members", []) if row.get("name") == member]
    if len(matches) != 1:
        raise ValueError(f"{label} does not uniquely contain {member}")
    row = matches[0]
    record = dict(row["accompaniment"])
    bundle_hashes = row.get("executed_bundle_hashes") or [row["executed_bundle_hash"]]
    record["executed_bundle_hashes"] = bundle_hashes
    # Reopen/re-hash rather than trusting the control report transitively.
    actual = audio_record(
        Path(record["path"]), recipe_id=record["recipe_id"],
    )
    actual["executed_bundle_hashes"] = bundle_hashes
    for key in ("container_sha256", "artifact_pcm_sha256"):
        if actual[key] != record[key]:
            raise ValueError(f"{label}/{member}/{key} mismatch")
    return actual


def run(args: argparse.Namespace) -> dict:
    oracle = json.loads(args.oracle_report.read_text())
    if oracle.get("schema") != "audio-extract/oracle-routing-envelope/v1":
        raise ValueError("wrong oracle routing report schema")
    dataset_rows = _rows(args.dataset_manifest)
    candidate_rows = (
        _candidate_rows(args.candidate_manifest) if args.candidate_manifest else {}
    )
    members = {
        "conservative": args.conservative,
        "aggressive": args.aggressive,
    }
    works = []
    requested = args.works or list(oracle["works"])
    for work in requested:
        if work not in oracle["works"] or work not in dataset_rows:
            raise ValueError(f"work absent from oracle/dataset manifests: {work}")
        basis = {row["name"]: row for row in oracle["works"][work]["basis"]}
        available = set(basis)
        if (work, "median_mdx_mel_bs") in candidate_rows:
            available.add("median_mdx_mel_bs")
        missing = sorted(set(members.values()) - available)
        if missing:
            raise ValueError(f"{work} oracle basis lacks members: {missing}")
        controls = {}
        for control in ("no_vocal", "vocal_only"):
            path = args.control_root / f"{work}.{control}.json"
            report = json.loads(path.read_text())
            if (report.get("schema") != "audio-extract/classical-gate-controls/v1"
                    or report.get("status") != "complete"
                    or report.get("work_id") != work
                    or report.get("control") != control):
                raise ValueError(f"invalid control report: {path}")
            controls[control] = {
                role: _control_member(report, name, f"{work}/{control}")
                for role, name in members.items()
            }
        truth = {
            role: audio_record(args.truth_root / work / filename)
            for role, filename in TRUTH_FILES.items()
        }
        activity = args.materialized_root / work / "vocal_activity.npy"
        works.append({
            "work_id": work, "group_id": dataset_rows[work]["group_id"],
            "truth": truth,
            "estimates": {
                "mixture": {
                    role: _basis_record(
                        basis[name] if name in basis else candidate_rows[(work, name)],
                        f"{work}/{role}",
                    )
                    for role, name in members.items()
                },
                **controls,
            },
            "vocal_activity_path": str(activity.resolve()),
        })
    document = {
        "schema": MANIFEST_SCHEMA, "status": "complete", "members": members,
        "source_oracle_report": str(args.oracle_report.resolve()),
        "source_oracle_report_sha256": _sha_file(args.oracle_report),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": _sha_file(args.dataset_manifest),
        "works": works,
    }
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != payload:
        raise RuntimeError(f"refusing to rewrite differing gate manifest: {args.output}")
    if not args.output.exists():
        args.output.write_text(payload)
    # A second pass uses the same strict loader as the trainer.
    load_manifest(args.output)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-report", required=True, type=Path)
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--materialized-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--conservative", required=True)
    parser.add_argument("--aggressive", required=True)
    parser.add_argument("--works", nargs="*")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({
        "status": report["status"], "works": len(report["works"]),
        "members": report["members"], "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
