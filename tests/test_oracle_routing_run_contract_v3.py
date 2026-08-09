from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os

import pytest

from audio_extract.oracle_routing_run_contract_v3 import (
    CANONICAL_RESOLUTIONS,
    REPORT_BINDING_SCHEMA,
    RunContractError,
    build_run_input,
    claim_run_input,
    policy_sha256,
    report_binding,
    require_canonical_resolutions,
    validate_run_input,
    verify_report_binding,
    write_run_input,
)


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
        ("truth", b"truth-manifest\n"),
        ("basis", b"basis-audit\n"),
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
        candidate_manifests=[
            _record(files["candidate_a"]),
            _record(files["candidate_b"]),
        ],
        output_root=str((tmp_path / "outputs").resolve()),
        prepared_nonce="a" * 64,
        prepared_at_utc="2026-08-09T12:00:00Z",
    )


def test_exact_canonical_resolutions_are_required_not_formatted_aliases():
    assert require_canonical_resolutions((2.0, 1.0, 0.5)) == (
        2.0, 1.0, 0.5
    )
    assert require_canonical_resolutions(("2.0", "1.0", "0.5")) == (
        2.0, 1.0, 0.5
    )
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
        88_200, 44_100, 22_050
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
    with pytest.raises(FileExistsError):
        write_run_input(path, document)

    candidate = Path(document["candidate_manifests"][0]["path"])
    candidate.write_bytes(b"mutated after precommit\n")
    with pytest.raises(RunContractError, match="hash mismatch"):
        validate_run_input(path)


def test_missing_null_or_empty_source_group_is_refused(tmp_path):
    files = _files(tmp_path)
    common = dict(
        source_commit="1" * 40,
        truth_manifest=_record(files["truth"]),
        basis_audit=_record(files["basis"]),
        candidate_manifests=[_record(files["candidate_a"])],
        output_root=str((tmp_path / "outputs").resolve()),
    )
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
                "no_vocal": [_record(alias)],
            },
            truth_manifest=_record(files["truth"]),
            basis_audit=_record(files["basis"]),
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
                "no_vocal": [_record(alias)],
            },
            truth_manifest=_record(files["truth"]),
            basis_audit=_record(files["basis"]),
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
        external_anchor="github:issue-1-comment-TEST",
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
        external_anchor="github:issue-1-comment-TEST",
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
        external_anchor="github:issue-1-comment-TEST",
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
