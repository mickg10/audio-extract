#!/usr/bin/env python3
"""Independently apply the frozen oracle-routing binding policy.

This command consumes a completed diagnostic report plus the immutable run-input
record written before the experiment.  It verifies provenance, the exact
compiled preregistration, source-manifest groups, frozen metric thresholds, and
the closed method/resolution evidence matrix.

The policy path is a human-readable witness, not a source of authority.  Its
semantic content must equal the policy compiled into the exact verifier commit;
JSON key order and whitespace cannot change that comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_extract.oracle_routing_binding_policy_v2 import (
    CANONICAL_POLICY_SHA256,
    BindingPolicyConfig,
    canonical_binding_policy,
    evaluate_report_strict,
    policy_semantic_sha256,
)
from audio_extract.oracle_routing_decision_v2 import RoutingGateConfig


RUN_INPUT_SCHEMA = "audio-extract/oracle-routing-run-inputs/v2"
VERIFICATION_SCHEMA = "audio-extract/oracle-routing-binding-verification/v2"
SOURCE_MANIFEST_GROUPS = ("voiced", "no_vocal")


class BindingVerificationError(RuntimeError):
    pass


def _signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _stable_bytes(path: Path) -> tuple[bytes, str]:
    try:
        if path.is_symlink():
            raise BindingVerificationError(f"refusing symlinked input: {path}")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except BindingVerificationError:
        raise
    except OSError as exc:
        raise BindingVerificationError(f"cannot read {path}: {exc}") from exc
    if (
        _signature(before) != _signature(after)
        or _signature(before) != _signature(current)
    ):
        raise BindingVerificationError(
            f"input changed or path was replaced: {path}"
        )
    return payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def _json(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    payload, digest = _stable_bytes(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingVerificationError(
            f"invalid JSON in {label}: {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise BindingVerificationError(f"{label} must be a JSON object")
    return value, digest


def _sha_file(path: Path) -> str:
    return _stable_bytes(path)[1]


def _sha_identity(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not result.startswith("sha256:") or len(result) != 71:
        raise BindingVerificationError(
            f"{label} must be sha256:<64 hex>"
        )
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise BindingVerificationError(
            f"{label} must be sha256:<64 hex>"
        ) from exc
    return result


def _verify_policy(
    value: Mapping[str, Any],
) -> tuple[BindingPolicyConfig, str]:
    try:
        policy = BindingPolicyConfig.from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise BindingVerificationError(
            f"policy differs from compiled preregistration: {exc}"
        ) from exc
    semantic_sha = policy_semantic_sha256(policy.identity_dict())
    if semantic_sha != CANONICAL_POLICY_SHA256:
        raise BindingVerificationError(
            "policy semantic hash differs from compiled preregistration: "
            f"{semantic_sha} != {CANONICAL_POLICY_SHA256}"
        )
    if policy != canonical_binding_policy():
        raise BindingVerificationError(
            "policy value differs from compiled preregistration"
        )
    return policy, semantic_sha


def _source_records(
    inputs: Mapping[str, Any],
) -> tuple[tuple[str, Path, str], ...]:
    groups = inputs.get("source_manifest_groups")
    if not isinstance(groups, Mapping):
        raise BindingVerificationError(
            "run inputs lack source_manifest_groups"
        )
    if set(groups) != set(SOURCE_MANIFEST_GROUPS):
        raise BindingVerificationError(
            "source_manifest_groups must contain exactly voiced and no_vocal"
        )

    result: list[tuple[str, Path, str]] = []
    seen_paths: set[Path] = set()
    for group in SOURCE_MANIFEST_GROUPS:
        records = groups.get(group)
        if not isinstance(records, Sequence) or isinstance(
            records, (str, bytes, bytearray)
        ) or not records:
            raise BindingVerificationError(
                f"source manifest group {group!r} must be a non-empty array"
            )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or set(record) != {
                "path", "sha256"
            }:
                raise BindingVerificationError(
                    f"{group} source manifest record {index} must contain "
                    "exactly path and sha256"
                )
            path_text = str(record.get("path") or "")
            if not path_text:
                raise BindingVerificationError(
                    f"{group} source manifest record {index} has no path"
                )
            try:
                path = Path(path_text).resolve(strict=True)
            except OSError as exc:
                raise BindingVerificationError(
                    f"cannot resolve source manifest {path_text}: {exc}"
                ) from exc
            if path in seen_paths:
                raise BindingVerificationError(
                    f"source manifest path is reused across groups: {path}"
                )
            seen_paths.add(path)
            expected = _sha_identity(
                record.get("sha256"),
                f"{group} source manifest SHA",
            )
            result.append((group, path, expected))
    return tuple(result)


def _verify_run_inputs(
    report: Mapping[str, Any],
    inputs: Mapping[str, Any],
    policy: BindingPolicyConfig,
    run_inputs_path: Path,
) -> RoutingGateConfig:
    if inputs.get("schema") != RUN_INPUT_SCHEMA:
        raise BindingVerificationError(
            f"wrong run-input schema: {inputs.get('schema')!r}"
        )
    if inputs.get("binding_policy") != policy.identity_dict():
        raise BindingVerificationError(
            "run inputs do not contain the compiled binding policy"
        )
    if _sha_identity(
        inputs.get("binding_policy_sha256"),
        "run-input binding policy SHA",
    ) != CANONICAL_POLICY_SHA256:
        raise BindingVerificationError(
            "run-input binding policy SHA differs from preregistration"
        )
    if report.get("code_commit") != inputs.get("code_commit"):
        raise BindingVerificationError(
            "report/run-input code commit mismatch"
        )
    if report.get("works") != inputs.get("works"):
        raise BindingVerificationError(
            "report/run-input work set or order mismatch"
        )
    works = inputs.get("works")
    if not isinstance(works, list) or not works:
        raise BindingVerificationError("run inputs have no works")
    if any(not isinstance(work, str) or not work for work in works):
        raise BindingVerificationError("run inputs contain an invalid work ID")
    if len(set(works)) != len(works):
        raise BindingVerificationError("run inputs contain duplicate work IDs")
    if report.get("config") != inputs.get("run_config"):
        raise BindingVerificationError(
            "report/run-input run configuration mismatch"
        )

    resolutions = report.get("resolutions")
    if not isinstance(resolutions, Mapping):
        raise BindingVerificationError("report lacks resolutions")
    if set(resolutions) != set(policy.required_resolutions):
        raise BindingVerificationError(
            "report resolution set differs from preregistered policy: "
            f"{sorted(resolutions)} != {sorted(policy.required_resolutions)}"
        )
    expected_works = tuple(works)
    for resolution, row in resolutions.items():
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("works"), Mapping)
        ):
            raise BindingVerificationError(
                f"resolution {resolution} lacks works"
            )
        if set(row["works"]) != set(expected_works):
            raise BindingVerificationError(
                f"resolution {resolution} work set differs from run inputs"
            )

    decision_config = inputs.get("decision_config")
    if not isinstance(decision_config, Mapping):
        raise BindingVerificationError(
            "run inputs lack frozen decision_config"
        )
    try:
        thresholds = RoutingGateConfig(**dict(decision_config))
        thresholds.validate()
    except (TypeError, ValueError) as exc:
        raise BindingVerificationError(
            f"invalid frozen decision_config: {exc}"
        ) from exc

    for group, path, expected in _source_records(inputs):
        if _sha_file(path) != expected:
            raise BindingVerificationError(
                f"{group} source manifest changed: {path}"
            )

    for filename, key in (
        ("basis-audit-v2.json", "basis_audit_sha256"),
        ("no-vocal-basis-audit-v2.json", "no_vocal_basis_audit_sha256"),
    ):
        path = run_inputs_path.parent / filename
        expected = _sha_identity(inputs.get(key), key)
        if _sha_file(path) != expected:
            raise BindingVerificationError(f"basis audit changed: {path}")
    return thresholds


def run(
    *,
    report_path: Path,
    run_inputs_path: Path,
    policy_path: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    report, report_sha = _json(report_path, "routing report")
    inputs, inputs_sha = _json(run_inputs_path, "run inputs")
    policy_value, policy_file_sha = _json(policy_path, "binding policy")
    policy, policy_semantic_sha = _verify_policy(policy_value)
    thresholds = _verify_run_inputs(
        report, inputs, policy, run_inputs_path
    )
    decision = evaluate_report_strict(
        report, policy=policy, metric_config=thresholds
    )
    result = {
        "schema": VERIFICATION_SCHEMA,
        "status": (
            "verified"
            if decision["decision"] != "INCOMPLETE_EVIDENCE"
            else "incomplete_evidence"
        ),
        "routing_report": str(report_path.resolve(strict=True)),
        "routing_report_sha256": report_sha,
        "run_inputs": str(run_inputs_path.resolve(strict=True)),
        "run_inputs_sha256": inputs_sha,
        "binding_policy": str(policy_path.resolve(strict=True)),
        "binding_policy_file_sha256": policy_file_sha,
        "binding_policy_semantic_sha256": policy_semantic_sha,
        "compiled_binding_policy_sha256": CANONICAL_POLICY_SHA256,
        "code_commit": inputs["code_commit"],
        "decision": decision,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.read_text() != payload:
            raise BindingVerificationError(
                f"refusing to replace differing verification: {output_path}"
            )
        if not output_path.exists():
            output_path.write_text(payload)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "verify-oracle-routing-binding-v2"
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-inputs", required=True, type=Path)
    parser.add_argument(
        "--policy",
        required=True,
        type=Path,
        help=(
            "human-readable witness; semantic content must equal the policy "
            "compiled into this exact verifier commit"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        report_path=args.report,
        run_inputs_path=args.run_inputs,
        policy_path=args.policy,
        output_path=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"]["decision"],
        "report_sha256": result["routing_report_sha256"],
        "policy_semantic_sha256": result[
            "binding_policy_semantic_sha256"
        ],
    }, sort_keys=True))
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
