import json
from pathlib import Path

import pytest

from audio_extract.oracle_routing_binding_policy_v2 import (
    CANONICAL_POLICY_SHA256,
    canonical_binding_policy,
)
from tools import verify_oracle_routing_binding_v2 as tool


def _write(path: Path, value, *, sort_keys=True):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=sort_keys) + "\n"
    )


def _fixture(tmp_path, monkeypatch):
    voiced_source = tmp_path / "voiced-source.jsonl"
    no_vocal_source = tmp_path / "no-vocal-source.jsonl"
    voiced_source.write_text('{"candidate":"voiced"}\n')
    no_vocal_source.write_text('{"candidate":"no_vocal"}\n')
    basis = tmp_path / "basis-audit-v2.json"
    no_vocal = tmp_path / "no-vocal-basis-audit-v2.json"
    _write(basis, {"status": "pass"})
    _write(no_vocal, {"status": "pass"})
    policy = canonical_binding_policy().identity_dict()
    inputs = {
        "schema": tool.RUN_INPUT_SCHEMA,
        "code_commit": "abc123",
        "works": [
            "bologna_verdi",
            "bologna_donizetti",
            "bologna_puccini",
            "aalto_mozart_dry",
        ],
        "run_config": {"resolutions_seconds": [2.0, 1.0, 0.5]},
        "decision_config": {},
        "binding_policy": policy,
        "binding_policy_sha256": CANONICAL_POLICY_SHA256,
        "source_manifest_groups": {
            "voiced": [{
                "path": str(voiced_source),
                "sha256": tool._sha_file(voiced_source),
            }],
            "no_vocal": [{
                "path": str(no_vocal_source),
                "sha256": tool._sha_file(no_vocal_source),
            }],
        },
        "basis_audit_sha256": tool._sha_file(basis),
        "no_vocal_basis_audit_sha256": tool._sha_file(no_vocal),
    }
    report = {
        "schema": "audio-extract/oracle-routing-envelope/v2",
        "code_commit": inputs["code_commit"],
        "works": inputs["works"],
        "config": inputs["run_config"],
        "resolutions": {
            value: {
                "works": {work: {} for work in inputs["works"]}
            }
            for value in ("1.0", "2.0", "0.5")
        },
    }
    report_path = tmp_path / "report.json"
    inputs_path = tmp_path / "run-inputs-v2.json"
    policy_path = tmp_path / "policy.json"
    _write(report_path, report)
    _write(inputs_path, inputs)
    _write(policy_path, policy)
    monkeypatch.setattr(
        tool,
        "evaluate_report_strict",
        lambda *args, **kwargs: {
            "decision": "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS",
            "recommendation": (
                "CHANGE_BASIS_OR_BUILD_TARGET_SINGER_CORRECTION"
            ),
        },
    )
    return report_path, inputs_path, policy_path


def test_verifier_binds_report_inputs_policy_and_writes_immutably(
    tmp_path, monkeypatch
):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "verification.json"
    result = tool.run(
        report_path=report,
        run_inputs_path=inputs,
        policy_path=policy,
        output_path=output,
    )
    assert result["status"] == "verified"
    assert result["routing_report_sha256"] == tool._sha_file(report)
    assert result["run_inputs_sha256"] == tool._sha_file(inputs)
    assert result["binding_policy_file_sha256"] == tool._sha_file(policy)
    assert result["binding_policy_semantic_sha256"] == (
        CANONICAL_POLICY_SHA256
    )
    assert result["compiled_binding_policy_sha256"] == (
        CANONICAL_POLICY_SHA256
    )
    assert json.loads(output.read_text()) == result
    assert tool.run(
        report_path=report,
        run_inputs_path=inputs,
        policy_path=policy,
        output_path=output,
    ) == result


def test_policy_json_order_and_whitespace_do_not_change_authority(
    tmp_path, monkeypatch
):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    value = json.loads(policy.read_text())
    reordered = {
        key: value[key] for key in reversed(tuple(value))
    }
    policy.write_text(
        json.dumps(reordered, separators=(",", ":")) + "\n"
    )
    result = tool.run(
        report_path=report,
        run_inputs_path=inputs,
        policy_path=policy,
        output_path=None,
    )
    assert result["status"] == "verified"
    assert result["binding_policy_semantic_sha256"] == (
        CANONICAL_POLICY_SHA256
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
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    document = json.loads(policy.read_text())
    document[field] = value
    _write(policy, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="compiled preregistration",
    ):
        tool.run(
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_report_run_input_mismatch_is_refused(tmp_path, monkeypatch):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    value = json.loads(report.read_text())
    value["code_commit"] = "different"
    _write(report, value)
    with pytest.raises(
        tool.BindingVerificationError,
        match="commit mismatch",
    ):
        tool.run(
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_changed_source_manifest_is_refused(tmp_path, monkeypatch):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    run_inputs = json.loads(inputs.read_text())
    path = Path(
        run_inputs["source_manifest_groups"]["voiced"][0]["path"]
    )
    path.write_text("changed\n")
    with pytest.raises(
        tool.BindingVerificationError,
        match="voiced source manifest changed",
    ):
        tool.run(
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


@pytest.mark.parametrize("replacement", [None, {}, {"voiced": [], "no_vocal": []}])
def test_missing_or_empty_source_manifest_groups_are_refused(
    tmp_path, monkeypatch, replacement
):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
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
        tool.run(
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_one_empty_source_manifest_group_is_refused(tmp_path, monkeypatch):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    document = json.loads(inputs.read_text())
    document["source_manifest_groups"]["no_vocal"] = []
    _write(inputs, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="no_vocal.*non-empty array",
    ):
        tool.run(
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_run_input_policy_digest_is_refused_if_changed(
    tmp_path, monkeypatch
):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    document = json.loads(inputs.read_text())
    document["binding_policy_sha256"] = "sha256:" + "0" * 64
    _write(inputs, document)
    with pytest.raises(
        tool.BindingVerificationError,
        match="differs from preregistration",
    ):
        tool.run(
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )


def test_resolution_set_is_exactly_policy_closed(tmp_path, monkeypatch):
    report, inputs, policy = _fixture(tmp_path, monkeypatch)
    value = json.loads(report.read_text())
    value["resolutions"]["0.25"] = value["resolutions"]["0.5"]
    _write(report, value)
    with pytest.raises(
        tool.BindingVerificationError,
        match="resolution set differs",
    ):
        tool.run(
            report_path=report,
            run_inputs_path=inputs,
            policy_path=policy,
            output_path=None,
        )
