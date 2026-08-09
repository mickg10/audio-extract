#!/usr/bin/env python3
"""Assemble, audit, run, and decide the certified oracle-routing v2 experiment.

Example:

    python tools/oracle_routing_v2_certified.py \
      --audited-candidate-manifest datasets/candidates-v2.jsonl \
      --strict-basis-manifest /runs/oracle/single-members-v2.jsonl \
      --no-vocal-strict-basis-manifest /runs/oracle/no-vocal-basis-v2.jsonl \
      --allowed-host-alias research6 \
      --truth-root /home/mickg/classical_training/exact \
      --output-root /home/mickg/runs/oracle-routing-v2-certified

The command is fail-closed. It refuses missing required aliases, changed hashes,
non-FLOAT grids, absent Aalto no-vocal controls, uncertified O2/O3 results, and
attempts to rewrite a differing completed report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import subprocess

from audio_extract.oracle_routing_basis_sources_v2 import assemble_basis
from audio_extract.oracle_routing_basis_v2 import basis_report, write_jsonl
from audio_extract.oracle_routing_decision_v2 import RoutingGateConfig
from audio_extract.oracle_routing_runner_v2 import (
    CertifiedRoutingRunConfig,
    DEFAULT_WORKS,
    REQUIRED_ALIASES,
    run_experiment,
)
from audio_extract.oracle_routing_spectral_v2 import RoutingSpectralConfig


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def load_run_config(
    path: Path | None,
    *,
    resolutions: tuple[float, ...] | None,
) -> CertifiedRoutingRunConfig:
    values = _load_json(path)
    spectral_values = dict(values.pop("spectral", {}))
    if "band_edges_hz" in spectral_values:
        spectral_values["band_edges_hz"] = tuple(
            int(value) for value in spectral_values["band_edges_hz"]
        )
    spectral = RoutingSpectralConfig(**spectral_values)
    if "resolutions_seconds" in values:
        values["resolutions_seconds"] = tuple(
            float(value) for value in values["resolutions_seconds"]
        )
    if resolutions is not None:
        values["resolutions_seconds"] = resolutions
    result = CertifiedRoutingRunConfig(spectral=spectral, **values)
    result.validate()
    return result


def load_decision_config(path: Path | None) -> RoutingGateConfig:
    result = RoutingGateConfig(**_load_json(path))
    result.validate()
    return result


def _write_immutable_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise RuntimeError(f"refusing to replace differing file: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run certified exact O1/O2/O3 opera routing"
    )
    parser.add_argument(
        "--audited-candidate-manifest", action="append", type=Path, default=[],
        help="legacy audited candidates-v2 JSONL; may be repeated",
    )
    parser.add_argument(
        "--strict-basis-manifest", action="append", type=Path, default=[],
        help="strict basis-v2 JSONL; may be repeated",
    )
    parser.add_argument(
        "--no-vocal-audited-candidate-manifest",
        action="append", type=Path, default=[],
    )
    parser.add_argument(
        "--no-vocal-strict-basis-manifest",
        action="append", type=Path, default=[],
    )
    parser.add_argument(
        "--allowed-host-alias", action="append", default=[],
        help="explicitly treat host:/absolute/path values for this alias as local",
    )
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work", action="append", default=[])
    parser.add_argument("--resolution", action="append", type=float)
    parser.add_argument("--run-config-json", type=Path)
    parser.add_argument("--decision-config-json", type=Path)
    parser.add_argument("--code-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.audited_candidate_manifest or args.strict_basis_manifest):
        raise SystemExit("at least one voiced candidate manifest is required")
    if not (
        args.no_vocal_audited_candidate_manifest
        or args.no_vocal_strict_basis_manifest
    ):
        raise SystemExit(
            "a binding run requires the complete Aalto no-vocal basis manifest"
        )
    resolutions = (
        None if args.resolution is None
        else tuple(float(value) for value in args.resolution)
    )
    run_config = load_run_config(
        args.run_config_json, resolutions=resolutions
    )
    decision_config = load_decision_config(args.decision_config_json)
    code_commit = args.code_commit or _git_commit()
    works = tuple(args.work) if args.work else DEFAULT_WORKS

    verified, basis = assemble_basis(
        audited_candidate_manifests=tuple(args.audited_candidate_manifest),
        strict_basis_manifests=tuple(args.strict_basis_manifest),
        allowed_host_aliases=tuple(args.allowed_host_alias),
        required_aliases=REQUIRED_ALIASES,
    )
    no_vocal_verified, no_vocal_basis = assemble_basis(
        audited_candidate_manifests=tuple(
            args.no_vocal_audited_candidate_manifest
        ),
        strict_basis_manifests=tuple(args.no_vocal_strict_basis_manifest),
        allowed_host_aliases=tuple(args.allowed_host_alias),
        required_aliases=REQUIRED_ALIASES,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "basis-v2.jsonl", basis)
    write_jsonl(args.output_root / "no-vocal-basis-v2.jsonl", no_vocal_basis)
    voiced_audit = basis_report(verified, basis)
    no_vocal_audit = basis_report(no_vocal_verified, no_vocal_basis)
    _write_immutable_json(args.output_root / "basis-audit-v2.json", voiced_audit)
    _write_immutable_json(
        args.output_root / "no-vocal-basis-audit-v2.json", no_vocal_audit
    )
    source_manifests = [
        *args.audited_candidate_manifest,
        *args.strict_basis_manifest,
        *args.no_vocal_audited_candidate_manifest,
        *args.no_vocal_strict_basis_manifest,
    ]
    inputs = {
        "schema": "audio-extract/oracle-routing-run-inputs/v2",
        "code_commit": code_commit,
        "works": list(works),
        "run_config": {
            **asdict(run_config),
            "spectral": asdict(run_config.spectral),
        },
        "decision_config": asdict(decision_config),
        "truth_root": str(args.truth_root.resolve()),
        "source_manifests": [
            {"path": str(path.resolve()), "sha256": _sha_file(path)}
            for path in source_manifests
        ],
        "basis_audit_sha256": _sha_file(
            args.output_root / "basis-audit-v2.json"
        ),
        "no_vocal_basis_audit_sha256": _sha_file(
            args.output_root / "no-vocal-basis-audit-v2.json"
        ),
    }
    _write_immutable_json(args.output_root / "run-inputs-v2.json", inputs)

    report, decision = run_experiment(
        basis_rows=basis,
        no_vocal_basis_rows=no_vocal_basis,
        truth_root=args.truth_root,
        output_root=args.output_root,
        code_commit=code_commit,
        works=works,
        config=run_config,
        decision_config=decision_config,
    )
    summary = {
        "schema": "audio-extract/oracle-routing-command-result/v2",
        "status": "complete",
        "report_schema": report["schema"],
        "decision": decision["decision"],
        "recommendation": decision["recommendation"],
        "selected": decision.get("selected"),
        "output_root": str(args.output_root.resolve()),
        "report": str(
            (args.output_root / "oracle-routing-envelope-v2.json").resolve()
        ),
        "decision_report": str(
            (args.output_root / "binding-decision-v2.json").resolve()
        ),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
