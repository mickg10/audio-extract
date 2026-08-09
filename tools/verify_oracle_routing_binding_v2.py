#!/usr/bin/env python3
"""Independently apply the frozen oracle-routing binding policy.

This command consumes a completed diagnostic report plus the immutable run-input
record written before the experiment. It verifies provenance, the exact
compiled preregistration, source-manifest groups, frozen metric thresholds,
source-lineage recipes, and the closed method/resolution evidence matrix.

The policy path is a human-readable witness, not a source of authority. Its
semantic content must equal the policy compiled into the exact verifier commit;
JSON key order and whitespace cannot change that comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_extract.oracle_binding_preregistration import (
    PreregisteredRun,
    PreregistrationError,
    bind_report,
    binding_fields,
    canonical_json,
    sha256_bytes,
    stable_file_record,
    verify_source_manifests,
)
from audio_extract.oracle_binding_preregistration import (
    load as load_preregistration,
)
from audio_extract.oracle_routing_binding_policy_v2 import (
    CANONICAL_POLICY_SHA256,
    BindingPolicyConfig,
    canonical_binding_policy,
    evaluate_report_strict,
    policy_semantic_sha256,
)
from audio_extract.oracle_routing_decision_v2 import RoutingGateConfig
from audio_extract.oracle_routing_run_contract_v2 import (
    RunContractError,
    verify_truth_manifest,
)
from audio_extract.oracle_routing_run_contract_v3 import (
    RunContractError as RunContractV3Error,
)
from audio_extract.oracle_routing_run_contract_v3 import (
    verify_report_binding as verify_v3_report_binding,
)
from audio_extract.oracle_routing_source_lineage_v2 import (
    LINEAGE_SCHEMA,
    _recipe_identity,
    _recipe_source,
    _stable_json,
    expected_source_from_audio,
)

RUN_INPUT_SCHEMA = "audio-extract/oracle-routing-run-inputs/v2"
VERIFICATION_SCHEMA = "audio-extract/oracle-routing-binding-verification/v2"
SOURCE_MANIFEST_GROUPS = ("voiced", "no_vocal")
NO_VOCAL_WORKS = ("aalto_mozart_dry",)


class BindingVerificationError(RuntimeError):
    pass


def _signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BindingVerificationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


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
    if _signature(before) != _signature(after) or _signature(before) != _signature(
        current
    ):
        raise BindingVerificationError(f"input changed or path was replaced: {path}")
    return payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def _json(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    payload, digest = _stable_bytes(path)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
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
        raise BindingVerificationError(f"{label} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise BindingVerificationError(f"{label} must be sha256:<64 hex>") from exc
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
        raise BindingVerificationError("run inputs lack source_manifest_groups")
    if set(groups) != set(SOURCE_MANIFEST_GROUPS):
        raise BindingVerificationError(
            "source_manifest_groups must contain exactly voiced and no_vocal"
        )

    result: list[tuple[str, Path, str]] = []
    seen_paths: set[Path] = set()
    for group in SOURCE_MANIFEST_GROUPS:
        records = groups.get(group)
        if (
            not isinstance(records, Sequence)
            or isinstance(records, (str, bytes, bytearray))
            or not records
        ):
            raise BindingVerificationError(
                f"source manifest group {group!r} must be a non-empty array"
            )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "sha256",
                "size",
                "device",
                "inode",
                "mtime_ns",
            }:
                raise BindingVerificationError(
                    f"{group} source manifest record {index} must contain "
                    "the exact preregistered path/hash/stat fields"
                )
            path_text = str(record.get("path") or "")
            if not path_text:
                raise BindingVerificationError(
                    f"{group} source manifest record {index} has no path"
                )
            raw_path = Path(path_text)
            if raw_path.is_symlink():
                raise BindingVerificationError(
                    f"refusing symlinked source manifest: {raw_path}"
                )
            try:
                path = raw_path.resolve(strict=True)
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
            try:
                current = stable_file_record(path).to_dict()
            except PreregistrationError as exc:
                raise BindingVerificationError(str(exc)) from exc
            if current != dict(record):
                raise BindingVerificationError(
                    f"{group} source manifest changed: {path}"
                )
            result.append((group, path, expected))
    return tuple(result)


def _verify_generated_recipe_bindings(
    report: Mapping[str, Any],
    preregistration: PreregisteredRun,
    run_input_binding: Mapping[str, Any],
) -> None:
    expected = binding_fields(preregistration)
    output_root = Path(str(run_input_binding.get("output_root") or ""))
    if not output_root.is_absolute():
        raise BindingVerificationError("v3 binding output_root is invalid")
    resolutions = report.get("resolutions")
    if not isinstance(resolutions, Mapping):
        raise BindingVerificationError("report lacks resolutions")
    checked = 0
    for resolution in resolutions.values():
        works = resolution.get("works") if isinstance(resolution, Mapping) else None
        if not isinstance(works, Mapping):
            continue
        for work in works.values():
            if not isinstance(work, Mapping):
                continue
            method_groups = [work.get("methods"), work.get("no_vocal")]
            for methods in method_groups:
                if not isinstance(methods, Mapping):
                    continue
                for evidence in methods.values():
                    artifact = (
                        evidence.get("artifact")
                        if isinstance(evidence, Mapping)
                        else None
                    )
                    if not isinstance(artifact, Mapping) or not artifact.get(
                        "recipe_id"
                    ):
                        continue
                    artifact_path = Path(str(artifact.get("path") or ""))
                    try:
                        resolved_artifact = artifact_path.resolve(strict=True)
                    except OSError as exc:
                        raise BindingVerificationError(
                            f"cannot resolve generated artifact: {artifact_path}"
                        ) from exc
                    if not resolved_artifact.is_relative_to(output_root):
                        raise BindingVerificationError(
                            "generated artifact is outside v3 output_root: "
                            f"{resolved_artifact}"
                        )
                    recipe_path = artifact_path.parent / "recipe.json"
                    recipe, _ = _json(recipe_path, "generated artifact recipe")
                    if recipe.get("preregistration") != expected:
                        raise BindingVerificationError(
                            "generated artifact recipe lacks the exact "
                            f"preregistration binding: {recipe_path}"
                        )
                    if recipe.get("run_input_binding") != run_input_binding:
                        raise BindingVerificationError(
                            "generated artifact recipe lacks the exact v3 "
                            f"run-input binding: {recipe_path}"
                        )
                    checked += 1
    if checked == 0:
        raise BindingVerificationError(
            "report contains no generated artifact recipe bindings"
        )


def _verify_preregistered_artifacts(
    inputs: Mapping[str, Any], preregistration: PreregisteredRun
) -> dict[str, Mapping[str, Any]]:
    records = inputs.get("preregistered_artifacts")
    expected_fields = {
        "truth_manifest": "truth_manifest_sha256",
        "basis_audit": "basis_audit_sha256",
        "routing_config": "routing_config_sha256",
    }
    if not isinstance(records, Mapping) or set(records) != set(expected_fields):
        raise BindingVerificationError(
            "run inputs lack the exact preregistered artifact set"
        )
    values: dict[str, Mapping[str, Any]] = {}
    for name, witness_field in expected_fields.items():
        record = records[name]
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "semantic_sha256",
        }:
            raise BindingVerificationError(
                f"invalid preregistered artifact record: {name}"
            )
        value, _ = _json(Path(str(record["path"])), name)
        actual = sha256_bytes(canonical_json(value))
        declared = _sha_identity(record["semantic_sha256"], f"{name} semantic SHA")
        expected = preregistration.document[witness_field]
        if actual != declared or declared != expected:
            raise BindingVerificationError(f"{name} differs from preregistration")
        values[name] = value
    return values


def _verify_report_truth(
    report: Mapping[str, Any], truth_manifest: Mapping[str, Any]
) -> None:
    records = truth_manifest.get("records")
    if not isinstance(records, Mapping):
        raise BindingVerificationError("truth manifest lacks records")
    for resolution_key, resolution in report["resolutions"].items():
        works = resolution.get("works")
        for work_id, work in works.items():
            declared = records.get(work_id)
            if not isinstance(declared, Mapping):
                raise BindingVerificationError(
                    f"report work is absent from truth manifest: {work_id}"
                )
            roles = declared.get("roles")
            truth = work.get("truth")
            if not isinstance(roles, Mapping) or not isinstance(truth, Mapping):
                raise BindingVerificationError(
                    f"invalid truth binding at {resolution_key}/{work_id}"
                )
            expected_pcm = {
                role: roles[role]["artifact_pcm_sha256"]
                for role in ("mixture", "accompaniment", "vocal")
            }
            grids = {
                (
                    role["frames"],
                    role["sample_rate_hz"],
                    tuple(role["channels"]),
                    role["subtype"],
                )
                for role in roles.values()
            }
            expected_truth = {
                "schema": "audio-extract/oracle-routing-truth/v2",
                "frames": next(iter(grids))[0] if len(grids) == 1 else None,
                "sample_rate_hz": next(iter(grids))[1] if len(grids) == 1 else None,
                "channels": ["FL", "FR"],
                "pcm_identities": expected_pcm,
            }
            if len(grids) != 1 or dict(truth) != expected_truth:
                raise BindingVerificationError(
                    f"report truth differs from frozen manifest: "
                    f"{resolution_key}/{work_id}"
                )


def _verify_frozen_execution_inputs(
    inputs: Mapping[str, Any],
    run_inputs_path: Path,
    frozen: Mapping[str, Mapping[str, Any]],
) -> None:
    routing = frozen["routing_config"]
    expected_routing = {
        "schema": "audio-extract/oracle-routing-config-binding/v1",
        "works": inputs.get("works"),
        "run_config": inputs.get("run_config"),
        "decision_config": inputs.get("decision_config"),
    }
    if dict(routing) != expected_routing:
        raise BindingVerificationError(
            "run-input execution configuration differs from preregistration"
        )

    basis = frozen["basis_audit"]
    components = {
        "voiced": "basis-audit-v2.json",
        "no_vocal": "no-vocal-basis-audit-v2.json",
        "voiced_source_lineage": "source-lineage-audit-v2.json",
        "no_vocal_source_lineage": "no-vocal-source-lineage-audit-v2.json",
    }
    if basis.get("schema") != ("audio-extract/oracle-routing-basis-audit-binding/v1"):
        raise BindingVerificationError("wrong frozen basis-audit schema")
    for field, filename in components.items():
        current, _ = _json(run_inputs_path.parent / filename, field)
        if basis.get(field) != current:
            raise BindingVerificationError(
                f"{field} differs from preregistered combined basis audit"
            )


def _candidate_lineage_fields() -> set[str]:
    return {
        "work_id",
        "candidate_name",
        "recipe_id",
        "recipe_path",
        "recipe_container_sha256",
        "recipe_schema",
        "source_role",
        "source_pcm_sha256",
        "source_frames",
        "source_sample_rate_hz",
        "source_channels",
        "source_sample_format",
    }


def _verify_lineage_audit(
    path: Path,
    *,
    expected_sha256: str,
    role: str,
    expected_works: Sequence[str],
) -> None:
    value, digest = _json(path, f"{role} source-lineage audit")
    if digest != expected_sha256:
        raise BindingVerificationError(f"source-lineage audit changed: {path}")
    if value.get("schema") != LINEAGE_SCHEMA or value.get("status") != "pass":
        raise BindingVerificationError(
            f"invalid source-lineage audit status/schema: {path}"
        )
    if value.get("source_role") != role:
        raise BindingVerificationError(f"source-lineage audit role mismatch: {path}")
    works = value.get("works")
    if not isinstance(works, list) or works != sorted(expected_works):
        raise BindingVerificationError(
            f"source-lineage audit work set mismatch: {works} != "
            f"{sorted(expected_works)}"
        )
    expected_values = value.get("expected_sources")
    if not isinstance(expected_values, Mapping) or set(expected_values) != set(
        expected_works
    ):
        raise BindingVerificationError(
            "source-lineage audit expected-source set is incomplete"
        )

    current_sources: dict[str, dict[str, Any]] = {}
    for work in expected_works:
        record = expected_values[work]
        if not isinstance(record, Mapping):
            raise BindingVerificationError(
                f"source-lineage expected source {work} is not an object"
            )
        if record.get("work_id") != work or record.get("role") != role:
            raise BindingVerificationError(
                f"source-lineage expected source identity mismatch: {work}"
            )
        source_path = Path(str(record.get("path") or ""))
        try:
            current = expected_source_from_audio(
                source_path, work_id=work, role=role
            ).to_dict()
        except (OSError, ValueError) as exc:
            raise BindingVerificationError(
                f"cannot reverify expected source {work}: {exc}"
            ) from exc
        if current != dict(record):
            raise BindingVerificationError(
                f"expected source changed since lineage audit: {work}"
            )
        current_sources[work] = current

    candidates = value.get("candidates")
    count = value.get("candidate_count")
    if (
        not isinstance(candidates, list)
        or not candidates
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(candidates)
    ):
        raise BindingVerificationError("source-lineage candidate list/count is invalid")
    seen: set[tuple[str, str, str]] = set()
    candidate_works: set[str] = set()
    for index, record in enumerate(candidates):
        if not isinstance(record, Mapping) or set(record) != (
            _candidate_lineage_fields()
        ):
            raise BindingVerificationError(
                f"source-lineage candidate {index} has the wrong field set"
            )
        work = str(record.get("work_id") or "")
        name = str(record.get("candidate_name") or "")
        recipe_id = _sha_identity(
            record.get("recipe_id"), f"lineage candidate {index} recipe ID"
        )
        key = (work, name, recipe_id)
        if work not in current_sources or not name or key in seen:
            raise BindingVerificationError(
                f"invalid or duplicate source-lineage candidate: {key}"
            )
        seen.add(key)
        candidate_works.add(work)
        expected = current_sources[work]
        if record.get("source_role") != role:
            raise BindingVerificationError(f"candidate source role mismatch: {key}")
        recipe_path = Path(str(record.get("recipe_path") or ""))
        try:
            recipe, recipe_digest = _stable_json(recipe_path)
            computed_recipe_id = _recipe_identity(recipe)
            source = _recipe_source(recipe)
        except (OSError, ValueError) as exc:
            raise BindingVerificationError(
                f"cannot reverify candidate recipe {key}: {exc}"
            ) from exc
        if computed_recipe_id != recipe_id:
            raise BindingVerificationError(f"candidate recipe identity changed: {key}")
        if recipe_digest != _sha_identity(
            record.get("recipe_container_sha256"),
            f"lineage candidate {index} recipe container",
        ):
            raise BindingVerificationError(f"candidate recipe bytes changed: {key}")
        if str(recipe.get("schema")) != record.get("recipe_schema"):
            raise BindingVerificationError(f"candidate recipe schema changed: {key}")
        recipe_work = recipe.get("work_id")
        if recipe_work is not None and str(recipe_work) != work:
            raise BindingVerificationError(f"candidate recipe work changed: {key}")
        source_pcm, frames, sample_rate, channels, sample_format = source
        if (
            source_pcm != expected["artifact_pcm_sha256"]
            or frames != expected["frames"]
            or sample_rate != expected["sample_rate_hz"]
            or list(channels) != expected["channels"]
            or sample_format != "float32-le-interleaved"
        ):
            raise BindingVerificationError(
                f"candidate recipe no longer binds the expected source: {key}"
            )
        declared_source = (
            _sha_identity(
                record.get("source_pcm_sha256"),
                f"lineage candidate {index} source PCM",
            ),
            record.get("source_frames"),
            record.get("source_sample_rate_hz"),
            record.get("source_channels"),
            record.get("source_sample_format"),
        )
        actual_source = (
            source_pcm,
            frames,
            sample_rate,
            list(channels),
            sample_format,
        )
        if declared_source != actual_source:
            raise BindingVerificationError(
                f"candidate lineage row differs from recipe: {key}"
            )
    if candidate_works != set(expected_works):
        raise BindingVerificationError(
            "source-lineage candidates do not cover every expected work"
        )


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
    if (
        _sha_identity(
            inputs.get("binding_policy_sha256"),
            "run-input binding policy SHA",
        )
        != CANONICAL_POLICY_SHA256
    ):
        raise BindingVerificationError(
            "run-input binding policy SHA differs from preregistration"
        )
    if report.get("code_commit") != inputs.get("code_commit"):
        raise BindingVerificationError("report/run-input code commit mismatch")
    if report.get("works") != inputs.get("works"):
        raise BindingVerificationError("report/run-input work set or order mismatch")
    works = inputs.get("works")
    if not isinstance(works, list) or not works:
        raise BindingVerificationError("run inputs have no works")
    if any(not isinstance(work, str) or not work for work in works):
        raise BindingVerificationError("run inputs contain an invalid work ID")
    if len(set(works)) != len(works):
        raise BindingVerificationError("run inputs contain duplicate work IDs")
    if report.get("config") != inputs.get("run_config"):
        raise BindingVerificationError("report/run-input run configuration mismatch")

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
        if not isinstance(row, Mapping) or not isinstance(row.get("works"), Mapping):
            raise BindingVerificationError(f"resolution {resolution} lacks works")
        if set(row["works"]) != set(expected_works):
            raise BindingVerificationError(
                f"resolution {resolution} work set differs from run inputs"
            )

    decision_config = inputs.get("decision_config")
    if not isinstance(decision_config, Mapping):
        raise BindingVerificationError("run inputs lack frozen decision_config")
    try:
        thresholds = RoutingGateConfig(**dict(decision_config))
        thresholds.validate()
    except (TypeError, ValueError) as exc:
        raise BindingVerificationError(
            f"invalid frozen decision_config: {exc}"
        ) from exc

    for group, path, expected in _source_records(inputs):
        if _sha_file(path) != expected:
            raise BindingVerificationError(f"{group} source manifest changed: {path}")

    for filename, key in (
        ("basis-audit-v2.json", "basis_audit_sha256"),
        ("no-vocal-basis-audit-v2.json", "no_vocal_basis_audit_sha256"),
    ):
        path = run_inputs_path.parent / filename
        expected = _sha_identity(inputs.get(key), key)
        if _sha_file(path) != expected:
            raise BindingVerificationError(f"basis audit changed: {path}")

    _verify_lineage_audit(
        run_inputs_path.parent / "source-lineage-audit-v2.json",
        expected_sha256=_sha_identity(
            inputs.get("source_lineage_audit_sha256"),
            "source_lineage_audit_sha256",
        ),
        role="voiced_mixture",
        expected_works=works,
    )
    _verify_lineage_audit(
        run_inputs_path.parent / "no-vocal-source-lineage-audit-v2.json",
        expected_sha256=_sha_identity(
            inputs.get("no_vocal_source_lineage_audit_sha256"),
            "no_vocal_source_lineage_audit_sha256",
        ),
        role="no_vocal_accompaniment",
        expected_works=NO_VOCAL_WORKS,
    )
    return thresholds


def run(
    *,
    preregistration_path: Path,
    run_input_v3_path: Path,
    run_claim_v3_path: Path,
    report_path: Path,
    run_inputs_path: Path,
    policy_path: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    report, report_sha = _json(report_path, "routing report")
    try:
        validated_v3 = verify_v3_report_binding(
            report, run_input_v3_path, run_claim_v3_path
        )
    except RunContractV3Error as exc:
        raise BindingVerificationError(f"invalid v3 run-input binding: {exc}") from exc
    run_input_binding = report["run_input_binding"]
    try:
        preregistration = load_preregistration(preregistration_path)
        verify_source_manifests(preregistration)
    except PreregistrationError as exc:
        raise BindingVerificationError(f"invalid preregistration: {exc}") from exc
    legacy_record = validated_v3.document["legacy_preregistration"]
    if legacy_record["path"] != str(preregistration_path.resolve(strict=True)):
        raise BindingVerificationError(
            "v3 run input binds a different legacy preregistration path"
        )
    if legacy_record["sha256"] != _sha_file(preregistration_path):
        raise BindingVerificationError(
            "v3 run input binds a different legacy preregistration digest"
        )
    try:
        bind_report(report, preregistration)
    except PreregistrationError as exc:
        raise BindingVerificationError(str(exc)) from exc
    inputs, inputs_sha = _json(run_inputs_path, "run inputs")
    if inputs.get("run_input_binding") != run_input_binding:
        raise BindingVerificationError("run-inputs-v2/v3 precommit binding mismatch")
    expected_binding = binding_fields(preregistration)
    v3_source_groups = {
        group: [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in validated_v3.document["source_manifests"][group]
        ]
        for group in ("voiced", "no_vocal")
    }
    legacy_source_groups = {
        group: [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in preregistration.document["source_manifests"][group]
        ]
        for group in ("voiced", "no_vocal")
    }
    if v3_source_groups != legacy_source_groups:
        raise BindingVerificationError(
            "v3 source groups differ from legacy preregistration"
        )
    if validated_v3.document["source_commit"] != inputs.get("code_commit"):
        raise BindingVerificationError("v3/input code commit mismatch")
    if validated_v3.document["required_works"] != inputs.get("works"):
        raise BindingVerificationError("v3/input work set/order mismatch")
    candidate_records = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in validated_v3.document["candidate_manifests"]
    ]
    expected_candidates = [
        dict(row)
        for group in ("voiced", "no_vocal")
        for row in legacy_source_groups[group]
    ]
    if candidate_records != expected_candidates:
        raise BindingVerificationError("v3 candidate manifest set/order mismatch")
    for name, witness_field in (
        ("truth_manifest", "truth_manifest_sha256"),
        ("basis_audit", "basis_audit_sha256"),
        ("routing_config", "routing_config_sha256"),
    ):
        record = validated_v3.document[name]
        value, container_sha = _json(Path(record["path"]), f"v3 {name}")
        if container_sha != record["sha256"]:
            raise BindingVerificationError(f"v3 {name} container digest mismatch")
        if (
            sha256_bytes(canonical_json(value))
            != preregistration.document[witness_field]
        ):
            raise BindingVerificationError(
                f"v3 {name} differs from legacy preregistration"
            )
    if inputs.get("preregistration") != expected_binding:
        raise BindingVerificationError("run-input/preregistration binding mismatch")
    if (
        inputs.get("source_manifest_groups")
        != preregistration.document["source_manifests"]
    ):
        raise BindingVerificationError(
            "run-input source_manifest_groups differ from preregistration; "
            "voiced and no_vocal must remain non-empty arrays"
        )
    preregistered_artifacts = _verify_preregistered_artifacts(inputs, preregistration)
    try:
        verify_truth_manifest(
            preregistered_artifacts["truth_manifest"],
            truth_root=Path(str(inputs.get("truth_root") or "")),
        )
    except RunContractError as exc:
        raise BindingVerificationError(
            f"frozen truth manifest failed replay: {exc}"
        ) from exc
    _verify_frozen_execution_inputs(inputs, run_inputs_path, preregistered_artifacts)
    policy_value, policy_file_sha = _json(policy_path, "binding policy")
    policy, policy_semantic_sha = _verify_policy(policy_value)
    thresholds = _verify_run_inputs(report, inputs, policy, run_inputs_path)
    _verify_report_truth(report, preregistered_artifacts["truth_manifest"])
    _verify_generated_recipe_bindings(report, preregistration, run_input_binding)
    decision = evaluate_report_strict(report, policy=policy, metric_config=thresholds)
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
        "preregistration": expected_binding,
        "run_input_binding": run_input_binding,
        "run_input_v3": validated_v3.path,
        "run_input_v3_sha256": validated_v3.sha256,
        "run_claim_v3": str(run_claim_v3_path.resolve(strict=True)),
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
    try:
        final_preregistration = load_preregistration(preregistration_path)
        verify_source_manifests(final_preregistration)
    except PreregistrationError as exc:
        raise BindingVerificationError(
            f"preregistration changed during verification: {exc}"
        ) from exc
    if final_preregistration != preregistration:
        raise BindingVerificationError("preregistration changed during verification")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("verify-oracle-routing-binding-v2")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--run-input-v3", required=True, type=Path)
    parser.add_argument("--run-claim-v3", required=True, type=Path)
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
        preregistration_path=args.preregistration,
        run_input_v3_path=args.run_input_v3,
        run_claim_v3_path=args.run_claim_v3,
        report_path=args.report,
        run_inputs_path=args.run_inputs,
        policy_path=args.policy,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"]["decision"],
                "report_sha256": result["routing_report_sha256"],
                "policy_semantic_sha256": result["binding_policy_semantic_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
