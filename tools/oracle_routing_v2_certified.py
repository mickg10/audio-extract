#!/usr/bin/env python3
"""Assemble, audit, and run the certified oracle-routing v2 diagnostic.

Example:

    python tools/oracle_routing_v2_certified.py \
      --audited-candidate-manifest datasets/candidates-v2.jsonl \
      --strict-basis-manifest /runs/oracle/single-members-v2.jsonl \
      --no-vocal-strict-basis-manifest /runs/oracle/no-vocal-basis-v2.jsonl \
      --allowed-host-alias research6 \
      --truth-root /home/mickg/classical_training/exact \
      --output-root /home/mickg/runs/oracle-routing-v2-certified

Before rendering, the command writes the exact compiled binding policy, separate
voiced/no-vocal source-manifest groups, and independent source-lineage audits
that prove every candidate recipe names the exact truth-source PCM and grid.
The exploratory decision emitted by the runner is not final authority; use
``tools/verify_oracle_routing_binding_v2.py`` on the completed report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
import hashlib
import json
import subprocess

from audio_extract.oracle_routing_basis_sources_v2 import assemble_basis
from audio_extract.oracle_routing_basis_v2 import basis_report, write_jsonl
from audio_extract.oracle_routing_binding_policy_v2 import (
    CANONICAL_POLICY_SHA256,
    canonical_binding_policy,
)
from audio_extract.oracle_routing_decision_v2 import RoutingGateConfig
from audio_extract.oracle_routing_runner_v2 import (
    CertifiedRoutingRunConfig,
    DEFAULT_WORKS,
    REQUIRED_ALIASES,
    run_experiment,
)
from audio_extract.oracle_routing_source_lineage_v2 import (
    audit_basis_source_lineage,
    expected_source_from_audio,
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


def _manifest_records(paths: Sequence[Path]) -> list[dict[str, str]]:
    if not paths:
        raise ValueError("source manifest group must not be empty")
    result = []
    seen: set[Path] = set()
    for raw_path in paths:
        if raw_path.is_symlink():
            raise ValueError(f"refusing symlinked source manifest: {raw_path}")
        path = raw_path.resolve(strict=True)
        if path in seen:
            raise ValueError(f"duplicate source manifest: {path}")
        seen.add(path)
        result.append({"path": str(path), "sha256": _sha_file(path)})
    return result


def _expected_sources(
    truth_root: Path,
    works: Sequence[str],
    *,
    role: str,
) -> dict[str, Any]:
    filename = (
        "mix_with_voice.wav"
        if role == "voiced_mixture"
        else "orchestra_only.wav"
    )
    return {
        work: expected_source_from_audio(
            truth_root / work / filename,
            work_id=work,
            role=role,
        )
        for work in works
    }


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
    voiced_source_paths = (
        *args.audited_candidate_manifest,
        *args.strict_basis_manifest,
    )
    no_vocal_source_paths = (
        *args.no_vocal_audited_candidate_manifest,
        *args.no_vocal_strict_basis_manifest,
    )
    if not voiced_source_paths:
        raise SystemExit("at least one voiced candidate manifest is required")
    if not no_vocal_source_paths:
        raise SystemExit(
            "a binding run requires the complete Aalto no-vocal basis manifest"
        )
    voiced_records = _manifest_records(voiced_source_paths)
    no_vocal_records = _manifest_records(no_vocal_source_paths)
    if {
        record["path"] for record in voiced_records
    } & {
        record["path"] for record in no_vocal_records
    }:
        raise SystemExit(
            "voiced and no-vocal source-manifest groups must be disjoint"
        )

    resolutions = (
        None if args.resolution is None
        else tuple(float(value) for value in args.resolution)
    )
    run_config = load_run_config(
        args.run_config_json, resolutions=resolutions
    )
    decision_config = load_decision_config(args.decision_config_json)
    binding_policy = canonical_binding_policy()
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
    voiced_expected = _expected_sources(
        args.truth_root, works, role="voiced_mixture"
    )
    no_vocal_works = tuple(sorted({
        row.declaration.work_id for row in no_vocal_verified
    }))
    no_vocal_expected = _expected_sources(
        args.truth_root,
        no_vocal_works,
        role="no_vocal_accompaniment",
    )
    voiced_lineage = audit_basis_source_lineage(
        verified, voiced_expected, role="voiced_mixture"
    )
    no_vocal_lineage = audit_basis_source_lineage(
        no_vocal_verified,
        no_vocal_expected,
        role="no_vocal_accompaniment",
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
    _write_immutable_json(
        args.output_root / "source-lineage-audit-v2.json",
        voiced_lineage,
    )
    _write_immutable_json(
        args.output_root / "no-vocal-source-lineage-audit-v2.json",
        no_vocal_lineage,
    )
    _write_immutable_json(
        args.output_root / "binding-policy-v2.json",
        binding_policy.identity_dict(),
    )
    inputs = {
        "schema": "audio-extract/oracle-routing-run-inputs/v2",
        "code_commit": code_commit,
        "works": list(works),
        "run_config": {
            **asdict(run_config),
            "spectral": asdict(run_config.spectral),
        },
        "decision_config": asdict(decision_config),
        "binding_policy": binding_policy.identity_dict(),
        "binding_policy_sha256": CANONICAL_POLICY_SHA256,
        "truth_root": str(args.truth_root.resolve(strict=True)),
        "source_manifest_groups": {
            "voiced": voiced_records,
            "no_vocal": no_vocal_records,
        },
        "basis_audit_sha256": _sha_file(
            args.output_root / "basis-audit-v2.json"
        ),
        "no_vocal_basis_audit_sha256": _sha_file(
            args.output_root / "no-vocal-basis-audit-v2.json"
        ),
        "source_lineage_audit_sha256": _sha_file(
            args.output_root / "source-lineage-audit-v2.json"
        ),
        "no_vocal_source_lineage_audit_sha256": _sha_file(
            args.output_root / "no-vocal-source-lineage-audit-v2.json"
        ),
    }
    _write_immutable_json(args.output_root / "run-inputs-v2.json", inputs)

    report, exploratory_decision = run_experiment(
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
        "status": "diagnostic_complete__strict_verification_required",
        "report_schema": report["schema"],
        "exploratory_decision": exploratory_decision["decision"],
        "binding_policy_sha256": CANONICAL_POLICY_SHA256,
        "output_root": str(args.output_root.resolve()),
        "report": str(
            (args.output_root / "oracle-routing-envelope-v2.json").resolve()
        ),
        "run_inputs": str(
            (args.output_root / "run-inputs-v2.json").resolve()
        ),
        "binding_policy": str(
            (args.output_root / "binding-policy-v2.json").resolve()
        ),
        "strict_verifier": "tools/verify_oracle_routing_binding_v2.py",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
