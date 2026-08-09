import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

import audio_extract.oracle_routing_run_contract_v3 as contract
from audio_extract.oracle_routing_run_contract_v3 import (
    CANONICAL_RESOLUTIONS,
    REPORT_BINDING_SCHEMA,
    RunContractError,
    build_run_input,
    claim_run_input,
    policy_sha256,
    preflight_run,
    report_binding,
    require_canonical_resolutions,
    validate_run_input,
    verify_report_binding,
    write_run_input,
)

ANCHOR = "https://github.com/mickg10/audio-extract/issues/1#issuecomment-5232987267"


def _record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _files(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for name, payload in (
        ("voiced", b"voiced-manifest\n"),
        ("no_vocal", b"no-vocal-manifest\n"),
        ("truth", b'{"kind":"truth"}\n'),
        ("basis", b'{"kind":"basis"}\n'),
        ("routing", b'{"kind":"routing"}\n'),
        ("legacy", b'{"kind":"legacy"}\n'),
        ("candidate_a", b"candidate-manifest-a\n"),
        ("candidate_b", b"candidate-manifest-b\n"),
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(payload)
        result[name] = path
    return result


def _document(tmp_path: Path, *, files=None):
    files = files or _files(tmp_path)
    return build_run_input(
        source_commit="1" * 40,
        source_manifests={
            "voiced": [_record(files["voiced"])],
            "no_vocal": [_record(files["no_vocal"])],
        },
        truth_manifest=_record(files["truth"]),
        basis_audit=_record(files["basis"]),
        routing_config=_record(files["routing"]),
        legacy_preregistration=_record(files["legacy"]),
        candidate_manifests=[
            _record(files["candidate_a"]),
            _record(files["candidate_b"]),
        ],
        output_root=str((tmp_path / "outputs").resolve()),
        prepared_nonce="a" * 64,
        prepared_at_utc="2026-08-09T12:00:00Z",
    )


def test_exact_canonical_resolutions_are_required_not_formatted_aliases():
    assert require_canonical_resolutions((2.0, 1.0, 0.5)) == (2.0, 1.0, 0.5)
    assert require_canonical_resolutions(("2.0", "1.0", "0.5")) == (2.0, 1.0, 0.5)
    for values in (
        (2.04, 1.0, 0.5),
        (2.0, 1.04, 0.5),
        (2.0, 1.0, 0.54),
        (1.0, 2.0, 0.5),
        (2.0, 1.0),
    ):
        with pytest.raises(RunContractError):
            require_canonical_resolutions(values)
    assert [item.tile_frames for item in CANONICAL_RESOLUTIONS] == [
        88_200,
        44_100,
        22_050,
    ]


def test_run_input_is_immutable_and_revalidated_from_dependency_bytes(tmp_path):
    document = _document(tmp_path)
    path = tmp_path / "run-input.json"
    digest = write_run_input(path, document)
    validated = validate_run_input(path)
    assert validated.sha256 == digest
    assert validated.document == document
    assert policy_sha256() == document["frozen_policy_sha256"]
    assert path.with_suffix(".json.sha256").read_text().strip() == digest
    assert write_run_input(path, document) == digest

    candidate = Path(document["candidate_manifests"][0]["path"])
    candidate.write_bytes(b"mutated after precommit\n")
    with pytest.raises(RunContractError, match="hash mismatch"):
        validate_run_input(path)


def test_missing_null_or_empty_source_group_is_refused(tmp_path):
    files = _files(tmp_path)
    common = {
        "source_commit": "1" * 40,
        "truth_manifest": _record(files["truth"]),
        "basis_audit": _record(files["basis"]),
        "routing_config": _record(files["routing"]),
        "legacy_preregistration": _record(files["legacy"]),
        "candidate_manifests": [_record(files["candidate_a"])],
        "output_root": str((tmp_path / "outputs").resolve()),
    }
    for groups in (
        {"voiced": [_record(files["voiced"])]},
        {"voiced": [], "no_vocal": [_record(files["no_vocal"])]},
        {"voiced": [_record(files["voiced"])], "no_vocal": []},
        {"voiced": [_record(files["voiced"])], "no_vocal": None},
    ):
        with pytest.raises(RunContractError):
            build_run_input(source_manifests=groups, **common)


def test_hardlink_alias_across_source_groups_is_refused(tmp_path):
    files = _files(tmp_path)
    alias = tmp_path / "different-name-same-inode.json"
    os.link(files["voiced"], alias)
    with pytest.raises(RunContractError, match="physical.*inode"):
        build_run_input(
            source_commit="1" * 40,
            source_manifests={
                "voiced": [_record(files["voiced"])],
                "no_vocal": [
                    {
                        "path": str(alias.absolute()),
                        "sha256": _record(files["voiced"])["sha256"],
                    }
                ],
            },
            truth_manifest=_record(files["truth"]),
            basis_audit=_record(files["basis"]),
            routing_config=_record(files["routing"]),
            legacy_preregistration=_record(files["legacy"]),
            candidate_manifests=[_record(files["candidate_a"])],
            output_root=str((tmp_path / "outputs").resolve()),
        )


def test_symlinked_manifest_is_refused(tmp_path):
    files = _files(tmp_path)
    alias = tmp_path / "symlinked-no-vocal.json"
    alias.symlink_to(files["no_vocal"])
    with pytest.raises(RunContractError, match="symlink"):
        build_run_input(
            source_commit="1" * 40,
            source_manifests={
                "voiced": [_record(files["voiced"])],
                "no_vocal": [
                    {
                        "path": str(alias.absolute()),
                        "sha256": _record(files["no_vocal"])["sha256"],
                    }
                ],
            },
            truth_manifest=_record(files["truth"]),
            basis_audit=_record(files["basis"]),
            routing_config=_record(files["routing"]),
            legacy_preregistration=_record(files["legacy"]),
            candidate_manifests=[_record(files["candidate_a"])],
            output_root=str((tmp_path / "outputs").resolve()),
        )


def test_claim_and_report_binding_are_exact(tmp_path):
    document = _document(tmp_path)
    input_path = tmp_path / "run-input.json"
    write_run_input(input_path, document)
    validated = validate_run_input(input_path)
    claim_path = tmp_path / "run-claim.json"
    claim = claim_run_input(
        input_path,
        claim_path,
        external_anchor=ANCHOR,
    )
    binding = report_binding(validated, claim)
    assert binding["schema"] == REPORT_BINDING_SCHEMA
    report = {"run_input_binding": binding}
    verified = verify_report_binding(report, input_path, claim_path)
    assert verified.sha256 == validated.sha256

    mutated = deepcopy(report)
    mutated["run_input_binding"]["run_input_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(RunContractError, match="binding mismatch"):
        verify_report_binding(mutated, input_path, claim_path)


def test_direct_api_report_without_precommit_binding_is_refused(tmp_path):
    document = _document(tmp_path)
    input_path = tmp_path / "run-input.json"
    write_run_input(input_path, document)
    claim_path = tmp_path / "run-claim.json"
    claim_run_input(
        input_path,
        claim_path,
        external_anchor=ANCHOR,
    )
    with pytest.raises(RunContractError, match="lacks run_input_binding"):
        verify_report_binding({}, input_path, claim_path)


def test_posthoc_substitution_of_another_valid_run_input_is_refused(tmp_path):
    files = _files(tmp_path)
    first = _document(tmp_path, files=files)
    first_path = tmp_path / "run-input-a.json"
    write_run_input(first_path, first)
    first_validated = validate_run_input(first_path)
    claim_path = tmp_path / "run-claim.json"
    claim = claim_run_input(
        first_path,
        claim_path,
        external_anchor=ANCHOR,
    )
    report = {"run_input_binding": report_binding(first_validated, claim)}

    second = deepcopy(first)
    second["prepared_nonce"] = "b" * 64
    second_path = tmp_path / "run-input-b.json"
    write_run_input(second_path, second)
    with pytest.raises(RunContractError):
        verify_report_binding(report, second_path, claim_path)


def test_policy_and_resolution_fields_cannot_be_changed_posthoc(tmp_path):
    document = _document(tmp_path)
    for mutation in (
        lambda value: value.update(task_id="all_voices_vs_nonvocal"),
        lambda value: value.update(required_methods=["O3_certified_convex"]),
        lambda value: value["resolutions"][1].update(seconds_decimal="1.04"),
        lambda value: value.update(frozen_policy_sha256="sha256:" + "0" * 64),
    ):
        changed = deepcopy(document)
        mutation(changed)
        path = tmp_path / f"changed-{hash(json.dumps(changed, sort_keys=True))}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(RunContractError):
            validate_run_input(path)


def test_byte_identical_separate_source_files_across_groups_are_refused(tmp_path):
    files = _files(tmp_path)
    copy = tmp_path / "separate-inode-identical-bytes.json"
    copy.write_bytes(files["voiced"].read_bytes())
    with pytest.raises(RunContractError, match="byte-identical"):
        build_run_input(
            source_commit="1" * 40,
            source_manifests={
                "voiced": [_record(files["voiced"])],
                "no_vocal": [_record(copy)],
            },
            truth_manifest=_record(files["truth"]),
            basis_audit=_record(files["basis"]),
            routing_config=_record(files["routing"]),
            legacy_preregistration=_record(files["legacy"]),
            candidate_manifests=[_record(files["candidate_a"])],
            output_root=str((tmp_path / "outputs").resolve()),
        )


def test_dependency_replacement_with_same_bytes_is_refused(tmp_path):
    files = _files(tmp_path)
    document = _document(tmp_path, files=files)
    path = tmp_path / "run-input.json"
    write_run_input(path, document)
    dependency = files["candidate_a"]
    payload = dependency.read_bytes()
    dependency.unlink()
    dependency.write_bytes(payload)
    with pytest.raises(RunContractError, match="(inode|ctime_ns) changed"):
        validate_run_input(path)


def test_dependency_mode_change_is_refused(tmp_path):
    files = _files(tmp_path)
    document = _document(tmp_path, files=files)
    path = tmp_path / "run-input.json"
    write_run_input(path, document)
    files["basis"].chmod(0o400)
    with pytest.raises(RunContractError, match="(ctime_ns|mode) changed"):
        validate_run_input(path)


def test_duplicate_json_keys_are_refused_before_sidecar_use(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"first","schema":"second"}\n')
    with pytest.raises(RunContractError, match="duplicate JSON key"):
        validate_run_input(path)


def test_duplicate_key_in_bound_routing_config_is_refused_before_publish(tmp_path):
    files = _files(tmp_path)
    files["routing"].write_text('{"x":1,"x":2}\n')
    document = _document(tmp_path, files=files)
    path = tmp_path / "run-input.json"
    with pytest.raises(RunContractError, match="duplicate JSON key"):
        write_run_input(path, document)
    assert not path.exists()


def test_run_input_and_claim_paths_must_be_outside_output_root(tmp_path):
    document = _document(tmp_path)
    output = Path(document["output_root"])
    with pytest.raises(RunContractError, match="outside output_root"):
        write_run_input(output / "run-input.json", document)
    input_path = tmp_path / "run-input.json"
    write_run_input(input_path, document)
    with pytest.raises(RunContractError, match="outside output_root"):
        claim_run_input(
            input_path,
            output / "claim.json",
            external_anchor=ANCHOR,
        )
    assert not output.exists()


def test_run_input_sidecar_is_required_and_exact(tmp_path):
    document = _document(tmp_path)
    path = tmp_path / "run-input.json"
    write_run_input(path, document)
    sidecar = path.with_suffix(".json.sha256")
    sidecar.chmod(0o644)
    sidecar.write_text("sha256:" + "0" * 64 + "\n")
    with pytest.raises(RunContractError, match="sidecar mismatch"):
        validate_run_input(path)


def test_short_write_cannot_poison_immutable_destination(tmp_path, monkeypatch):
    document = _document(tmp_path)
    path = tmp_path / "run-input.json"
    real_write = contract.os.write
    calls = 0

    def fail_after_partial_write(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[:3])
        raise OSError("injected publication failure")

    monkeypatch.setattr(contract.os, "write", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        write_run_input(path, document)
    assert not path.exists()
    assert not path.with_suffix(".json.sha256").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_output_root_must_not_exist_at_prepare_claim_or_preflight(tmp_path):
    files = _files(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    with pytest.raises(RunContractError, match="output_root already exists"):
        _document(tmp_path, files=files)

    output.rmdir()
    document = _document(tmp_path, files=files)
    input_path = tmp_path / "run-input.json"
    write_run_input(input_path, document)
    output.mkdir()
    with pytest.raises(RunContractError, match="output_root already exists"):
        claim_run_input(input_path, tmp_path / "claim.json", external_anchor=ANCHOR)

    output.rmdir()
    claim_path = tmp_path / "claim.json"
    claim_run_input(input_path, claim_path, external_anchor=ANCHOR)
    output.mkdir()
    with pytest.raises(RunContractError, match="output_root already exists"):
        preflight_run(input_path, claim_path)


def test_preflight_binds_claim_but_completed_report_can_be_verified(tmp_path):
    document = _document(tmp_path)
    input_path = tmp_path / "run-input.json"
    write_run_input(input_path, document)
    claim_path = tmp_path / "claim.json"
    claim_run_input(input_path, claim_path, external_anchor=ANCHOR)
    preflight = preflight_run(input_path, claim_path)
    assert preflight.binding["output_root"] == document["output_root"]
    report = {"run_input_binding": dict(preflight.binding)}
    Path(document["output_root"]).mkdir()
    assert (
        verify_report_binding(report, input_path, claim_path).sha256
        == preflight.run_input.sha256
    )


def test_claim_requires_real_issue_comment_url_and_no_symlink(tmp_path):
    document = _document(tmp_path)
    input_path = tmp_path / "run-input.json"
    write_run_input(input_path, document)
    with pytest.raises(RunContractError, match="issue-comment URL"):
        claim_run_input(
            input_path,
            tmp_path / "bad-claim.json",
            external_anchor="github:issue-1-comment-TEST",
        )
    claim_path = tmp_path / "claim.json"
    claim = claim_run_input(input_path, claim_path, external_anchor=ANCHOR)
    report = {
        "run_input_binding": report_binding(validate_run_input(input_path), claim)
    }
    alias = tmp_path / "claim-alias.json"
    alias.symlink_to(claim_path)
    alias.with_suffix(".json.sha256").symlink_to(claim_path.with_suffix(".json.sha256"))
    with pytest.raises(RunContractError, match="symlink"):
        verify_report_binding(report, input_path, alias)
