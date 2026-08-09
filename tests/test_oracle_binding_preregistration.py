import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

from audio_extract.oracle_binding_preregistration import (
    CANONICAL_POLICY_SEMANTIC_SHA256,
    PRIMARY_RESOLUTION,
    REQUIRED_METHODS,
    SELECTED_METHOD,
    SENSITIVITY_RESOLUTIONS,
    TASK_ID,
    PreregistrationError,
    bind_report,
    binding_fields,
    build_document,
    canonical_json,
    canonical_policy,
    canonical_resolution,
    canonical_resolution_sequence,
    load,
    sha256_bytes,
    source_manifest_groups,
    stable_file_record,
    write_once,
)
from tools import oracle_routing_v2_certified as certified_tool

SHA1 = "sha256:" + "1" * 64
SHA2 = "sha256:" + "2" * 64
SHA3 = "sha256:" + "3" * 64
COMMIT = "a" * 40


def _files(tmp_path: Path):
    voiced = tmp_path / "voiced.jsonl"
    no_vocal = tmp_path / "no-vocal.jsonl"
    voiced.write_text('{"work":"verdi"}\n')
    no_vocal.write_text('{"work":"aalto-control"}\n')
    return voiced, no_vocal


def _document(tmp_path: Path, **kwargs):
    voiced, no_vocal = _files(tmp_path)
    values = {
        "experiment_id": "oracle-routing-binding-001",
        "source_groups": {"voiced": [voiced], "no_vocal": [no_vocal]},
        "source_commit": COMMIT,
        "truth_manifest_sha256": SHA1,
        "basis_audit_sha256": SHA2,
        "routing_config_sha256": SHA3,
    }
    values.update(kwargs)
    return build_document(**values)


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0, "1.0"),
        ("1.0", "1.0"),
        (Decimal("2.0"), "2.0"),
        ("0.5", "0.5"),
    ],
)
def test_exact_resolution_values(value, expected):
    assert canonical_resolution(value) == expected


@pytest.mark.parametrize(
    "value",
    [1.04, 2.04, 0.54, 1.0000000000000002, 0, -1, float("inf"), True],
)
def test_nearby_or_invalid_resolutions_are_refused(value):
    with pytest.raises(PreregistrationError):
        canonical_resolution(value)


def test_resolution_sequence_is_complete_and_not_rounded():
    assert set(canonical_resolution_sequence((2.0, 1.0, 0.5))) == {
        "2.0", "1.0", "0.5"
    }
    with pytest.raises(PreregistrationError):
        canonical_resolution_sequence((2.04, 1.04, 0.54))
    with pytest.raises(PreregistrationError):
        canonical_resolution_sequence((2.0, 1.0, 1.0))


def test_source_groups_must_be_nonempty_and_physically_disjoint(tmp_path):
    voiced, no_vocal = _files(tmp_path)
    result = source_manifest_groups(
        {"voiced": [voiced], "no_vocal": [no_vocal]}
    )
    assert set(result) == {"voiced", "no_vocal"}

    with pytest.raises(PreregistrationError, match="non-empty"):
        source_manifest_groups({"voiced": [], "no_vocal": [no_vocal]})

    alias = tmp_path / "no-vocal-hardlink.jsonl"
    os.link(voiced, alias)
    with pytest.raises(PreregistrationError, match="filesystem alias"):
        source_manifest_groups(
            {"voiced": [voiced], "no_vocal": [alias]}
        )


def test_symlinked_manifest_is_refused(tmp_path):
    voiced, no_vocal = _files(tmp_path)
    link = tmp_path / "voiced-link.jsonl"
    link.symlink_to(voiced)
    with pytest.raises(PreregistrationError, match="symlink"):
        source_manifest_groups(
            {"voiced": [link], "no_vocal": [no_vocal]}
        )


def test_stable_record_contains_physical_identity(tmp_path):
    voiced, _ = _files(tmp_path)
    record = stable_file_record(voiced)
    assert record.sha256 == sha256_bytes(voiced.read_bytes())
    assert record.size == voiced.stat().st_size
    assert record.device == voiced.stat().st_dev
    assert record.inode == voiced.stat().st_ino


def test_compiled_policy_and_digest_are_not_caller_selected():
    policy = canonical_policy()
    assert policy["task_id"] == TASK_ID
    assert policy["selected_method"] == SELECTED_METHOD
    assert policy["required_methods"] == list(REQUIRED_METHODS)
    assert policy["primary_resolution"] == PRIMARY_RESOLUTION
    assert policy["sensitivity_resolutions"] == list(
        SENSITIVITY_RESOLUTIONS
    )
    assert sha256_bytes(canonical_json(policy)) == (
        CANONICAL_POLICY_SEMANTIC_SHA256
    )


def test_write_once_replays_identically_and_refuses_different_record(tmp_path):
    document = _document(tmp_path)
    path = tmp_path / "run-input.json"
    first = write_once(path, document)
    second = write_once(path, document)
    assert first == second
    mutated = dict(document)
    mutated["experiment_id"] = "post-hoc"
    with pytest.raises(PreregistrationError, match="refusing to replace"):
        write_once(path, mutated)


def test_failed_witness_write_never_publishes_partial_path(tmp_path, monkeypatch):
    path = tmp_path / "run-input.json"
    monkeypatch.setattr(os, "write", lambda *_: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        write_once(path, _document(tmp_path))
    assert not path.exists()
    assert not list(tmp_path.glob(".run-input.json.*.tmp"))


def test_load_rejects_duplicate_keys_and_cross_group_record_alias(tmp_path):
    document = _document(tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        b'{"schema":"duplicate",' + canonical_json(document)[1:] + b"\n"
    )
    with pytest.raises(PreregistrationError, match="duplicate JSON object key"):
        load(duplicate)

    document["source_manifests"]["no_vocal"] = [
        dict(document["source_manifests"]["voiced"][0])
    ]
    aliased = tmp_path / "aliased.json"
    aliased.write_bytes(canonical_json(document) + b"\n")
    with pytest.raises(PreregistrationError, match="filesystem alias"):
        load(aliased)


def test_load_refuses_policy_or_resolution_mutation(tmp_path):
    document = _document(tmp_path)
    path = tmp_path / "run-input.json"
    write_once(path, document)

    changed = json.loads(path.read_text())
    changed["policy"]["selected_method"] = "O3_certified_convex"
    path.chmod(0o644)
    path.write_bytes(canonical_json(changed) + b"\n")
    with pytest.raises(PreregistrationError, match="policy differs"):
        load(path)


def test_report_must_bind_exact_prewritten_record(tmp_path):
    run = write_once(tmp_path / "run-input.json", _document(tmp_path))
    report = binding_fields(run)
    bind_report(report, run)
    for key in (
        "preregistration_container_sha256",
        "selected_method",
        "primary_resolution_seconds",
    ):
        broken = dict(report)
        broken[key] = "wrong"
        with pytest.raises(PreregistrationError, match=key):
            bind_report(broken, run)


def test_json_key_order_does_not_change_semantic_digest():
    left = {"a": 1, "b": {"x": 2, "y": 3}}
    right = {"b": {"y": 3, "x": 2}, "a": 1}
    assert canonical_json(left) == canonical_json(right)
    assert sha256_bytes(canonical_json(left)) == sha256_bytes(
        canonical_json(right)
    )


def test_binding_cli_refuses_post_hoc_resolution_before_output(tmp_path):
    document = _document(tmp_path)
    preregistration = write_once(tmp_path / "prereg.json", document)
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit, match="--resolution is forbidden"):
        certified_tool.main([
            "--preregistration", preregistration.path,
            "--truth-root", str(tmp_path / "truth"),
            "--output-root", str(output),
            "--resolution", "1.0",
        ])
    assert not output.exists()
