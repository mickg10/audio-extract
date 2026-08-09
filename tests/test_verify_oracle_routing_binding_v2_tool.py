import json
from pathlib import Path

import pytest

from tools import verify_oracle_routing_binding_v2 as tool


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text('{"candidate":"x"}\n')
    basis = tmp_path / "basis-audit-v2.json"
    no_vocal = tmp_path / "no-vocal-basis-audit-v2.json"
    _write(basis, {"status": "pass"})
    _write(no_vocal, {"status": "pass"})
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
        "source_manifests": [{
            "path": str(source),
            "sha256": tool._sha_file(source),
        }],
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
    policy = {
        "schema": "audio-extract/oracle-routing-binding-policy/v2",
        "task_id": "soloist_vs_rest",
        "selected_method": "O2_global_medoid",
        "primary_resolution": "1.0",
        "sensitivity_resolutions": ["2.0", "0.5"],
        "required_methods": [
            "O2_global_medoid", "O3_certified_convex"
        ],
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
    assert result["binding_policy_sha256"] == tool._sha_file(policy)
    assert json.loads(output.read_text()) == result
    assert tool.run(
        report_path=report,
        run_inputs_path=inputs,
        policy_path=policy,
        output_path=output,
    ) == result


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
    Path(run_inputs["source_manifests"][0]["path"]).write_text(
        "changed\n"
    )
    with pytest.raises(
        tool.BindingVerificationError,
        match="source manifest changed",
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
