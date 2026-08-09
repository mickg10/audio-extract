import json
from pathlib import Path

import pytest

from audio_extract.oracle_binding_preregistration import (
    binding_fields,
    build_document,
    canonical_json,
    load,
    sha256_bytes,
    write_once,
)
from audio_extract.oracle_routing_binding_policy_v2 import (
    CANONICAL_POLICY_SHA256,
    canonical_binding_policy,
)
from audio_extract.oracle_routing_run_contract_v3 import (
    build_run_input,
    claim_run_input,
    report_binding,
    validate_run_input,
    write_run_input,
)
from audio_extract.oracle_routing_work_contract_v3 import (
    REQUIRED_WORKS,
    WORK_CONTRACT_SHA256,
)
from audio_extract.oracle_routing_work_contract_v3 import (
    identity_dict as work_contract_identity,
)
from tools import verify_oracle_routing_binding_v2 as tool

ANCHOR = "https://github.com/mickg10/audio-extract/issues/1#issuecomment-5232987267"
_V3_PATHS: dict[Path, tuple[Path, Path]] = {}


def _run(**kwargs):
    run_input, claim = _V3_PATHS[kwargs["report_path"]]
    return tool.run(
        run_input_v3_path=run_input,
        run_claim_v3_path=claim,
        **kwargs,
    )


def _write(path: Path, value, *, sort_keys=True):
    path.write_text(json.dumps(value, indent=2, sort_keys=sort_keys) + "\n")


def _fixture(tmp_path, monkeypatch, *, reordered_preregistration=False):
    voiced_source = tmp_path / "voiced-source.jsonl"
    no_vocal_source = tmp_path / "no-vocal-source.jsonl"
    voiced_source.write_text('{"candidate":"voiced"}\n')
    no_vocal_source.write_text('{"candidate":"no_vocal"}\n')
    basis = tmp_path / "basis-audit-v2.json"
    no_vocal = tmp_path / "no-vocal-basis-audit-v2.json"
    _write(basis, {"status": "pass"})
    _write(no_vocal, {"status": "pass"})
    policy = canonical_binding_policy().identity_dict()
    frozen_documents = {
        "truth_manifest": {"schema": "test/truth", "works": ["fixture"]},
        "basis_audit": {"schema": "test/basis", "status": "pass"},
        "routing_config": {"schema": "test/config", "frozen": True},
    }
    frozen_paths = {}
    frozen_hashes = {}
    for name, value in frozen_documents.items():
        path = tmp_path / f"{name}.json"
        _write(path, value)
        frozen_paths[name] = path
        frozen_hashes[name] = sha256_bytes(canonical_json(value))
    preregistration = write_once(
        tmp_path / "preregistration.json",
        build_document(
            experiment_id="verifier-test",
            source_groups={
                "voiced": [voiced_source],
                "no_vocal": [no_vocal_source],
            },
            source_commit="a" * 40,
            truth_manifest_sha256=frozen_hashes["truth_manifest"],
            basis_audit_sha256=frozen_hashes["basis_audit"],
            routing_config_sha256=frozen_hashes["routing_config"],
        ),
    )
    if reordered_preregistration:
        path = Path(preregistration.path)
        document = json.loads(path.read_text())
        path.chmod(0o644)
        path.write_text(
            json.dumps(
                {key: document[key] for key in reversed(tuple(document))},
                separators=(",", ":"),
            )
            + "\n"
        )
        preregistration = load(path)
    witness = binding_fields(preregistration)
    source_records = {
        group: [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in preregistration.document["source_manifests"][group]
        ]
        for group in ("voiced", "no_vocal")
    }
    output_root = tmp_path / "bound-output"
    v3_document = build_run_input(
        source_commit="a" * 40,
        source_manifests=source_records,
        truth_manifest={
            "path": str(frozen_paths["truth_manifest"].resolve()),
            "sha256": tool._sha_file(frozen_paths["truth_manifest"]),
        },
        basis_audit={
            "path": str(frozen_paths["basis_audit"].resolve()),
            "sha256": tool._sha_file(frozen_paths["basis_audit"]),
        },
        routing_config={
            "path": str(frozen_paths["routing_config"].resolve()),
            "sha256": tool._sha_file(frozen_paths["routing_config"]),
        },
        legacy_preregistration={
            "path": str(Path(preregistration.path).resolve()),
            "sha256": tool._sha_file(Path(preregistration.path)),
        },
        candidate_manifests=[
            dict(row)
            for group in ("voiced", "no_vocal")
            for row in source_records[group]
        ],
        output_root=str(output_root.resolve()),
        prepared_at_utc="2026-08-09T12:00:00Z",
    )
    run_input_v3 = tmp_path / "run-input-v3.json"
    run_claim_v3 = tmp_path / "run-claim-v3.json"
    write_run_input(run_input_v3, v3_document)
    claim = claim_run_input(run_input_v3, run_claim_v3, external_anchor=ANCHOR)
    v3_binding = report_binding(validate_run_input(run_input_v3), claim)
    inputs = {
        "schema": tool.RUN_INPUT_SCHEMA,
        "code_commit": "a" * 40,
        "works": list(REQUIRED_WORKS),
        "run_config": {"resolutions_seconds": [2.0, 1.0, 0.5]},
        "decision_config": {},
        "binding_policy": policy,
        "binding_policy_sha256": CANONICAL_POLICY_SHA256,
        "source_manifest_groups": preregistration.document["source_manifests"],
        "preregistration": witness,
        "run_input_binding": v3_binding,
        "preregistered_artifacts": {
            name: {
                "path": str(path),
                "semantic_sha256": frozen_hashes[name],
            }
            for name, path in frozen_paths.items()
        },
        "basis_audit_sha256": tool._sha_file(basis),
        "no_vocal_basis_audit_sha256": tool._sha_file(no_vocal),
        "source_lineage_audit_sha256": "sha256:" + "5" * 64,
        "no_vocal_source_lineage_audit_sha256": "sha256:" + "6" * 64,
    }
    report = {
        "schema": "audio-extract/oracle-routing-envelope/v2",
        "code_commit": inputs["code_commit"],
        "works": inputs["works"],
        "work_contract": work_contract_identity(),
        "work_contract_sha256": WORK_CONTRACT_SHA256,
        "config": inputs["run_config"],
        "resolutions": {
            value: {"works": {work: {} for work in inputs["works"]}}
            for value in ("1.0", "2.0", "0.5")
        },
        **witness,
        "run_input_binding": v3_binding,
    }
    generated = output_root / "generated"
    generated.mkdir(parents=True)
    (generated / "output.f32.wav").write_bytes(b"fixture")
    _write(
        generated / "recipe.json",
        {"preregistration": witness, "run_input_binding": v3_binding},
    )
    report["resolutions"]["1.0"]["works"][inputs["works"][0]] = {
        "methods": {
            "O2_global_medoid": {
                "artifact": {
                    "path": str(generated / "output.f32.wav"),
                    "recipe_id": "sha256:" + "4" * 64,
                }
            }
        }
    }
    report_path = tmp_path / "report.json"
    inputs_path = tmp_path / "run-inputs-v2.json"
    policy_path = tmp_path / "policy.json"
    _write(report_path, report)
    _write(inputs_path, inputs)
    _write(policy_path, policy)
    _V3_PATHS[report_path] = (run_input_v3, run_claim_v3)
    monkeypatch.setattr(
        tool,
        "evaluate_report_strict",
        lambda *args, **kwargs: {
            "decision": "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS",
            "recommendation": ("CHANGE_BASIS_OR_BUILD_TARGET_SINGER_CORRECTION"),
        },
    )
    monkeypatch.setattr(tool, "_verify_lineage_audit", lambda *a, **k: None)
    monkeypatch.setattr(tool, "_verify_report_truth", lambda *a, **k: None)
    monkeypatch.setattr(tool, "_verify_frozen_execution_inputs", lambda *a, **k: None)
    monkeypatch.setattr(tool, "verify_truth_manifest", lambda *a, **k: None)
    return preregistration.path, report_path, inputs_path, policy_path


def test_verifier_binds_report_inputs_policy_and_writes_immutably(
    tmp_path, monkeypatch
):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "verification.json"
    result = _run(
        preregistration_path=Path(preregistration),
        report_path=report,
        run_inputs_path=inputs,
        policy_path=policy,
        output_path=output,
    )
    assert result["status"] == "verified"
    assert result["routing_report_sha256"] == tool._sha_file(report)
    assert result["run_inputs_sha256"] == tool._sha_file(inputs)
    assert result["binding_policy_file_sha256"] == tool._sha_file(policy)
    assert result["binding_policy_semantic_sha256"] == (CANONICAL_POLICY_SHA256)
    assert result["compiled_binding_policy_sha256"] == (CANONICAL_POLICY_SHA256)
    assert json.loads(output.read_text()) == result
    assert (
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=output,
        )
        == result
    )


def test_policy_json_order_and_whitespace_do_not_change_authority(
    tmp_path, monkeypatch
):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    value = json.loads(policy.read_text())
    reordered = {key: value[key] for key in reversed(tuple(value))}
    policy.write_text(json.dumps(reordered, separators=(",", ":")) + "\n")
    result = _run(
        preregistration_path=Path(preregistration),
        report_path=report,
        run_inputs_path=inputs,
        policy_path=policy,
        output_path=None,
    )
    assert result["status"] == "verified"
    assert result["binding_policy_semantic_sha256"] == (CANONICAL_POLICY_SHA256)


def test_preregistration_key_order_is_semantic_not_authority(tmp_path, monkeypatch):
    preregistration, report, inputs, policy = _fixture(
        tmp_path, monkeypatch, reordered_preregistration=True
    )
    result = _run(
        preregistration_path=Path(preregistration),
        report_path=report,
        run_inputs_path=inputs,
        policy_path=policy,
        output_path=None,
    )
    assert result["status"] == "verified"


def test_missing_preregistration_is_refused(tmp_path, monkeypatch):
    _, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    with pytest.raises(tool.BindingVerificationError, match="preregistration"):
        _run(
            preregistration_path=tmp_path / "never-created.json",
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_witness_changed_after_report_generation_is_refused(tmp_path, monkeypatch):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    path = Path(preregistration)
    document = json.loads(path.read_text())
    document["experiment_id"] = "post-hoc-replacement"
    path.chmod(0o644)
    _write(path, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="binding mismatch|legacy_preregistration",
    ):
        _run(
            preregistration_path=path,
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


@pytest.mark.parametrize("mutation", ["method", "resolution", "policy_digest"])
def test_rehashed_post_hoc_preregistration_policy_is_refused(
    tmp_path, monkeypatch, mutation
):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    path = Path(preregistration)
    document = json.loads(path.read_text())
    if mutation == "method":
        document["policy"]["selected_method"] = "O3_certified_convex"
    elif mutation == "resolution":
        document["resolutions_seconds"][0] = "1.04"
    else:
        document["policy_semantic_sha256"] = "sha256:" + "0" * 64
    path.chmod(0o644)
    _write(path, document)
    with pytest.raises(tool.BindingVerificationError, match="preregistration"):
        _run(
            preregistration_path=path,
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "different_task"),
        ("selected_method", "O3_certified_convex"),
        ("primary_resolution", "2.0"),
        ("sensitivity_resolutions", ["1.0", "0.5"]),
        ("required_methods", ["O2_global_medoid"]),
    ],
)
def test_schema_valid_post_hoc_policy_variants_are_refused(
    tmp_path, monkeypatch, field, value
):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    document = json.loads(policy.read_text())
    document[field] = value
    _write(policy, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="compiled preregistration",
    ):
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_report_run_input_mismatch_is_refused(tmp_path, monkeypatch):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    value = json.loads(report.read_text())
    value["code_commit"] = "different"
    _write(report, value)
    with pytest.raises(
        tool.BindingVerificationError,
        match="commit mismatch",
    ):
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_changed_source_manifest_is_refused(tmp_path, monkeypatch):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    run_inputs = json.loads(inputs.read_text())
    path = Path(run_inputs["source_manifest_groups"]["voiced"][0]["path"])
    path.write_text("changed\n")
    with pytest.raises(
        tool.BindingVerificationError,
        match="voiced source manifest changed|source_manifests.voiced",
    ):
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


@pytest.mark.parametrize("replacement", [None, {}, {"voiced": [], "no_vocal": []}])
def test_missing_or_empty_source_manifest_groups_are_refused(
    tmp_path, monkeypatch, replacement
):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    document = json.loads(inputs.read_text())
    if replacement is None:
        document.pop("source_manifest_groups")
    else:
        document["source_manifest_groups"] = replacement
    _write(inputs, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="source_manifest_groups|non-empty array",
    ):
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_one_empty_source_manifest_group_is_refused(tmp_path, monkeypatch):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    document = json.loads(inputs.read_text())
    document["source_manifest_groups"]["no_vocal"] = []
    _write(inputs, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="no_vocal.*non-empty array",
    ):
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_run_input_policy_digest_is_refused_if_changed(tmp_path, monkeypatch):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    document = json.loads(inputs.read_text())
    document["binding_policy_sha256"] = "sha256:" + "0" * 64
    _write(inputs, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="differs from preregistration",
    ):
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_resolution_set_is_exactly_policy_closed(tmp_path, monkeypatch):
    preregistration, report, inputs, policy = _fixture(tmp_path, monkeypatch)
    value = json.loads(report.read_text())
    value["resolutions"]["0.25"] = value["resolutions"]["0.5"]
    _write(report, value)
    with pytest.raises(
        tool.BindingVerificationError,
        match="resolution set differs",
    ):
        _run(
            preregistration_path=Path(preregistration),
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )
