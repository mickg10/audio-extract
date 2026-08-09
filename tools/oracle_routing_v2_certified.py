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
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audio_extract.oracle_binding_preregistration import (
    PreregistrationError,
    binding_fields,
    canonical_json,
    sha256_bytes,
    verify_source_manifests,
)
from audio_extract.oracle_binding_preregistration import (
    load as load_preregistration,
)
from audio_extract.oracle_routing_basis_sources_v2 import assemble_basis
from audio_extract.oracle_routing_basis_v2 import basis_report, write_jsonl
from audio_extract.oracle_routing_binding_policy_v2 import (
    CANONICAL_POLICY_SHA256,
    canonical_binding_policy,
)
from audio_extract.oracle_routing_decision_v2 import RoutingGateConfig
from audio_extract.oracle_routing_run_contract_v2 import build_truth_manifest
from audio_extract.oracle_routing_run_contract_v3 import (
    CANONICAL_RESOLUTIONS,
    claim_run_input,
    preflight_run,
    write_immutable_json_artifact,
)
from audio_extract.oracle_routing_run_contract_v3 import (
    RunContractError as RunContractV3Error,
)
from audio_extract.oracle_routing_run_contract_v3 import (
    build_run_input as build_run_input_v3,
)
from audio_extract.oracle_routing_run_contract_v3 import (
    write_run_input as write_run_input_v3,
)
from audio_extract.oracle_routing_runner_v2 import (
    DEFAULT_WORKS,
    REQUIRED_ALIASES,
    CertifiedRoutingRunConfig,
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


def _git_commit(requested: str | None = None) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if dirty:
        raise RuntimeError(
            "binding execution requires a clean Git worktree; commit the exact "
            "runner and dependency state first"
        )
    if requested is not None and requested.lower() != commit.lower():
        raise RuntimeError(
            f"--code-commit {requested} differs from checked-out HEAD {commit}"
        )
    return commit


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key in {path}: {key!r}")
            result[key] = item
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a JSON object: {path}")
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


def _semantic_sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _binding_basis_audit(
    voiced: Mapping[str, Any],
    no_vocal: Mapping[str, Any],
    voiced_lineage: Mapping[str, Any],
    no_vocal_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "audio-extract/oracle-routing-basis-audit-binding/v1",
        "voiced": dict(voiced),
        "no_vocal": dict(no_vocal),
        "voiced_source_lineage": dict(voiced_lineage),
        "no_vocal_source_lineage": dict(no_vocal_lineage),
    }


def _routing_config_identity(
    run_config: CertifiedRoutingRunConfig,
    decision_config: RoutingGateConfig,
    works: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "audio-extract/oracle-routing-config-binding/v1",
        "works": list(works),
        "run_config": {
            **asdict(run_config),
            "spectral": asdict(run_config.spectral),
        },
        "decision_config": asdict(decision_config),
    }


def _expected_sources(
    truth_root: Path,
    works: Sequence[str],
    *,
    role: str,
) -> dict[str, Any]:
    filename = (
        "mix_with_voice.wav" if role == "voiced_mixture" else "orchestra_only.wav"
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
        "--audited-candidate-manifest",
        action="append",
        type=Path,
        default=[],
        help="legacy audited candidates-v2 JSONL; may be repeated",
    )
    parser.add_argument(
        "--strict-basis-manifest",
        action="append",
        type=Path,
        default=[],
        help="strict basis-v2 JSONL; may be repeated",
    )
    parser.add_argument(
        "--no-vocal-audited-candidate-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--no-vocal-strict-basis-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--allowed-host-alias",
        action="append",
        default=[],
        help="explicitly treat host:/absolute/path values for this alias as local",
    )
    parser.add_argument("--truth-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        help="preexisting immutable binding witness",
    )
    parser.add_argument(
        "--run-input-v3",
        type=Path,
        help="immutable v3 run input (written by --prepare-only)",
    )
    parser.add_argument(
        "--run-claim-v3",
        type=Path,
        help="immutable externally anchored v3 claim",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="write the immutable v3 input, print its SHA, and stop",
    )
    modes.add_argument(
        "--claim-only",
        action="store_true",
        help="bind --run-input-v3 to --external-anchor and stop",
    )
    parser.add_argument(
        "--external-anchor",
        help="exact GitHub issue-comment URL; valid only with --claim-only",
    )
    parser.add_argument("--work", action="append", default=[])
    parser.add_argument("--resolution", action="append", type=float)
    parser.add_argument("--run-config-json", type=Path)
    parser.add_argument("--decision-config-json", type=Path)
    parser.add_argument("--code-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resolution is not None:
        raise SystemExit(
            "--resolution is forbidden for binding runs; resolutions come "
            "only from the immutable run input"
        )
    if args.run_input_v3 is None:
        raise SystemExit("--run-input-v3 is required")
    if args.claim_only:
        if args.run_claim_v3 is None or args.external_anchor is None:
            raise SystemExit(
                "--claim-only requires --run-claim-v3 and --external-anchor"
            )
        try:
            claim = claim_run_input(
                args.run_input_v3,
                args.run_claim_v3,
                external_anchor=args.external_anchor,
            )
        except RunContractV3Error as exc:
            raise SystemExit(f"v3 claim failed: {exc}") from exc
        print(
            json.dumps(
                {
                    "schema": "audio-extract/oracle-routing-claim-result/v3",
                    "claim_path": claim["claim_path"],
                    "claim_sha256": claim["claim_sha256"],
                    "run_input_sha256": claim["run_input_sha256"],
                    "external_anchor": claim["external_anchor"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.external_anchor is not None:
        raise SystemExit("--external-anchor is valid only with --claim-only")
    for option, value in (
        ("--preregistration", args.preregistration),
        ("--truth-root", args.truth_root),
        ("--output-root", args.output_root),
    ):
        if value is None:
            raise SystemExit(f"{option} is required")
    run_preflight = None
    if not args.prepare_only:
        if args.run_claim_v3 is None:
            raise SystemExit("a binding run requires --run-claim-v3")
        # This is intentionally the first filesystem-dependent operation in a
        # run. It must precede witness reads, audio opens, and output creation.
        try:
            run_preflight = preflight_run(args.run_input_v3, args.run_claim_v3)
        except RunContractV3Error as exc:
            raise SystemExit(f"v3 run preflight failed: {exc}") from exc
        if Path(run_preflight.run_input.document["output_root"]) != (
            args.output_root.resolve(strict=False)
        ):
            raise SystemExit("--output-root differs from the v3 run input")

    try:
        preregistration = load_preregistration(args.preregistration)
        verify_source_manifests(preregistration)
    except PreregistrationError as exc:
        raise SystemExit(f"invalid preregistration: {exc}") from exc
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
    supplied_groups = {
        "voiced": [str(path.resolve(strict=True)) for path in voiced_source_paths],
        "no_vocal": [str(path.resolve(strict=True)) for path in no_vocal_source_paths],
    }
    frozen_groups = {
        group: [
            row["path"] for row in preregistration.document["source_manifests"][group]
        ]
        for group in ("voiced", "no_vocal")
    }
    if supplied_groups != frozen_groups:
        raise SystemExit(
            "CLI source manifests differ from the preregistered groups/order"
        )

    legacy_resolutions = tuple(
        float(value) for value in preregistration.document["resolutions_seconds"]
    )
    resolutions = tuple(item.seconds for item in CANONICAL_RESOLUTIONS)
    if set(legacy_resolutions) != set(resolutions):
        raise SystemExit("legacy witness resolution set differs from run-contract v3")
    run_config = load_run_config(args.run_config_json, resolutions=resolutions)
    decision_config = load_decision_config(args.decision_config_json)
    binding_policy = canonical_binding_policy()
    code_commit = _git_commit(args.code_commit)
    works = tuple(args.work) if args.work else DEFAULT_WORKS
    if code_commit.lower() != preregistration.document["source_commit"]:
        raise SystemExit(
            "execution code commit differs from preregistered source_commit"
        )

    verified, basis = assemble_basis(
        audited_candidate_manifests=tuple(args.audited_candidate_manifest),
        strict_basis_manifests=tuple(args.strict_basis_manifest),
        allowed_host_aliases=tuple(args.allowed_host_alias),
        required_aliases=REQUIRED_ALIASES,
    )
    no_vocal_verified, no_vocal_basis = assemble_basis(
        audited_candidate_manifests=tuple(args.no_vocal_audited_candidate_manifest),
        strict_basis_manifests=tuple(args.no_vocal_strict_basis_manifest),
        allowed_host_aliases=tuple(args.allowed_host_alias),
        required_aliases=REQUIRED_ALIASES,
    )
    voiced_expected = _expected_sources(args.truth_root, works, role="voiced_mixture")
    no_vocal_works = tuple(
        sorted({row.declaration.work_id for row in no_vocal_verified})
    )
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

    voiced_audit = basis_report(verified, basis)
    no_vocal_audit = basis_report(no_vocal_verified, no_vocal_basis)
    combined_basis_audit = _binding_basis_audit(
        voiced_audit, no_vocal_audit, voiced_lineage, no_vocal_lineage
    )
    truth_manifest = build_truth_manifest(
        args.truth_root,
        works,
        identity_tolerance=run_config.truth_identity_tolerance,
    )
    routing_config_identity = _routing_config_identity(
        run_config, decision_config, works
    )
    current_hashes = {
        "truth_manifest_sha256": _semantic_sha(truth_manifest),
        "basis_audit_sha256": _semantic_sha(combined_basis_audit),
        "routing_config_sha256": _semantic_sha(routing_config_identity),
    }
    for field, actual in current_hashes.items():
        expected = preregistration.document[field]
        if actual != expected:
            raise SystemExit(
                f"{field} differs from preregistration: {actual} != {expected}"
            )

    run_input_path = args.run_input_v3.resolve(strict=False)
    artifact_root = run_input_path.parent / f"{run_input_path.stem}.artifacts"
    output_root = args.output_root.resolve(strict=False)
    if artifact_root == output_root or artifact_root.is_relative_to(output_root):
        raise SystemExit("v3 precommit artifacts must be outside --output-root")
    precommit_paths = {
        "truth_manifest": artifact_root / "truth-manifest-v2.json",
        "basis_audit": artifact_root / "binding-basis-audit-v1.json",
        "routing_config": artifact_root / "routing-config-binding-v1.json",
    }
    source_records = {
        group: [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in preregistration.document["source_manifests"][group]
        ]
        for group in ("voiced", "no_vocal")
    }
    candidate_records = [
        dict(row) for group in ("voiced", "no_vocal") for row in source_records[group]
    ]
    legacy_record = {
        "path": str(args.preregistration.resolve(strict=True)),
        "sha256": _sha_file(args.preregistration),
    }

    if args.prepare_only:
        truth_record = write_immutable_json_artifact(
            precommit_paths["truth_manifest"], truth_manifest
        )
        basis_record = write_immutable_json_artifact(
            precommit_paths["basis_audit"], combined_basis_audit
        )
        routing_record = write_immutable_json_artifact(
            precommit_paths["routing_config"], routing_config_identity
        )
        try:
            document_v3 = build_run_input_v3(
                source_commit=code_commit,
                source_manifests=source_records,
                truth_manifest=truth_record,
                basis_audit=basis_record,
                routing_config=routing_record,
                legacy_preregistration=legacy_record,
                candidate_manifests=candidate_records,
                output_root=str(output_root),
            )
            run_input_sha = write_run_input_v3(run_input_path, document_v3)
        except RunContractV3Error as exc:
            raise SystemExit(f"v3 run-input preparation failed: {exc}") from exc
        print(
            json.dumps(
                {
                    "schema": "audio-extract/oracle-routing-prepare-result/v3",
                    "status": "prepared__external_claim_required",
                    "run_input_path": str(run_input_path),
                    "run_input_sha256": run_input_sha,
                    "output_root": str(output_root),
                    "precommit_artifacts": {
                        name: str(path) for name, path in precommit_paths.items()
                    },
                },
                sort_keys=True,
            )
        )
        return 0

    assert run_preflight is not None
    run_document = run_preflight.run_input.document

    def basic_record(value: Mapping[str, Any]) -> dict[str, str]:
        return {"path": str(value["path"]), "sha256": str(value["sha256"])}

    bound_groups = {
        group: [basic_record(row) for row in run_document["source_manifests"][group]]
        for group in ("voiced", "no_vocal")
    }
    if bound_groups != source_records:
        raise SystemExit("v3 source manifests differ from the legacy witness")
    expected_records = {
        "truth_manifest": {
            "record": basic_record(run_document["truth_manifest"]),
            "path": precommit_paths["truth_manifest"],
            "value": truth_manifest,
        },
        "basis_audit": {
            "record": basic_record(run_document["basis_audit"]),
            "path": precommit_paths["basis_audit"],
            "value": combined_basis_audit,
        },
        "routing_config": {
            "record": basic_record(run_document["routing_config"]),
            "path": precommit_paths["routing_config"],
            "value": routing_config_identity,
        },
    }
    for name, expected_record in expected_records.items():
        path = expected_record["path"]
        if expected_record["record"] != {
            "path": str(path.resolve(strict=True)),
            "sha256": _sha_file(path),
        }:
            raise SystemExit(f"v3 {name} record differs from canonical precommit path")
        if _load_json(path) != expected_record["value"]:
            raise SystemExit(f"v3 {name} content differs from recomputed input")
    if basic_record(run_document["legacy_preregistration"]) != legacy_record:
        raise SystemExit("v3 legacy preregistration record differs")
    if [basic_record(row) for row in run_document["candidate_manifests"]] != (
        candidate_records
    ):
        raise SystemExit("v3 candidate manifest set/order differs")

    report, exploratory_decision = run_experiment(
        basis_rows=basis,
        no_vocal_basis_rows=no_vocal_basis,
        truth_root=args.truth_root,
        output_root=args.output_root,
        code_commit=code_commit,
        preregistration=preregistration,
        run_input_path=run_input_path,
        run_claim_path=args.run_claim_v3,
        works=works,
        config=run_config,
        decision_config=decision_config,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "basis-v2.jsonl", basis)
    write_jsonl(args.output_root / "no-vocal-basis-v2.jsonl", no_vocal_basis)
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
    _write_immutable_json(args.output_root / "truth-manifest-v2.json", truth_manifest)
    _write_immutable_json(
        args.output_root / "binding-basis-audit-v1.json",
        combined_basis_audit,
    )
    _write_immutable_json(
        args.output_root / "routing-config-binding-v1.json",
        routing_config_identity,
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
        "source_manifest_groups": preregistration.document["source_manifests"],
        "preregistration": binding_fields(preregistration),
        "run_input_binding": dict(run_preflight.binding),
        "preregistered_artifacts": {
            "truth_manifest": {
                "path": str((args.output_root / "truth-manifest-v2.json").resolve()),
                "semantic_sha256": current_hashes["truth_manifest_sha256"],
            },
            "basis_audit": {
                "path": str(
                    (args.output_root / "binding-basis-audit-v1.json").resolve()
                ),
                "semantic_sha256": current_hashes["basis_audit_sha256"],
            },
            "routing_config": {
                "path": str(
                    (args.output_root / "routing-config-binding-v1.json").resolve()
                ),
                "semantic_sha256": current_hashes["routing_config_sha256"],
            },
        },
        "basis_audit_sha256": _sha_file(args.output_root / "basis-audit-v2.json"),
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
    summary = {
        "schema": "audio-extract/oracle-routing-command-result/v2",
        "status": "diagnostic_complete__strict_verification_required",
        "report_schema": report["schema"],
        "exploratory_decision": exploratory_decision["decision"],
        "binding_policy_sha256": CANONICAL_POLICY_SHA256,
        "preregistration": binding_fields(preregistration),
        "run_input_binding": dict(run_preflight.binding),
        "run_input_v3": str(run_input_path),
        "run_claim_v3": str(args.run_claim_v3.resolve(strict=True)),
        "output_root": str(args.output_root.resolve()),
        "report": str((args.output_root / "oracle-routing-envelope-v2.json").resolve()),
        "run_inputs": str((args.output_root / "run-inputs-v2.json").resolve()),
        "binding_policy": str((args.output_root / "binding-policy-v2.json").resolve()),
        "strict_verifier": "tools/verify_oracle_routing_binding_v2.py",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
