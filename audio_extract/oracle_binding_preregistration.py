"""Immutable preregistration contract for binding opera-routing experiments.

This module is intentionally independent of the routing optimizer and audio
metrics. It fixes the experiment identity *before* any held-out result exists:

* one task;
* one primary method;
* one exact primary resolution and two exact sensitivity resolutions;
* exactly two non-empty source-manifest groups;
* no symlink or hard-link alias across those groups;
* one write-once run-input record whose exact bytes and semantic content are
  committed into every later report.

A report that lacks the preregistration commitment is not binding evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os
import stat


SCHEMA = "audio-extract/oracle-routing-preregistration/v1"
TASK_ID = "soloist_vs_rest"
SELECTED_METHOD = "O2_global_medoid"
REQUIRED_METHODS = ("O2_global_medoid", "O3_certified_convex")
PRIMARY_RESOLUTION = "1.0"
SENSITIVITY_RESOLUTIONS = ("2.0", "0.5")
REQUIRED_RESOLUTIONS = (PRIMARY_RESOLUTION, *SENSITIVITY_RESOLUTIONS)
SOURCE_GROUPS = ("voiced", "no_vocal")


class PreregistrationError(RuntimeError):
    """The run-input record is mutable, ambiguous, or post-hoc configurable."""


@dataclass(frozen=True)
class StableFileRecord:
    path: str
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class PreregisteredRun:
    path: str
    container_sha256: str
    semantic_sha256: str
    document: Mapping[str, Any]


def canonical_json(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON and reject NaN/Infinity."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PreregistrationError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_policy() -> dict[str, Any]:
    """The compiled policy; callers cannot substitute methods or resolutions."""

    return {
        "schema": "audio-extract/oracle-routing-binding-policy/v1",
        "task_id": TASK_ID,
        "selected_method": SELECTED_METHOD,
        "required_methods": list(REQUIRED_METHODS),
        "primary_resolution_seconds": PRIMARY_RESOLUTION,
        "sensitivity_resolutions_seconds": list(SENSITIVITY_RESOLUTIONS),
        "required_resolutions_seconds": list(REQUIRED_RESOLUTIONS),
    }


CANONICAL_POLICY_SEMANTIC_SHA256 = sha256_bytes(canonical_json(canonical_policy()))


def canonical_resolution(value: Any) -> str:
    """Accept only one exact preregistered decimal value.

    Formatting a different float to one decimal place is forbidden. For
    example, 1.04 must not become the identity ``"1.0"``.
    """

    if isinstance(value, bool):
        raise PreregistrationError("boolean is not a routing resolution")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PreregistrationError(
            f"invalid routing resolution: {value!r}"
        ) from exc
    if not decimal.is_finite():
        raise PreregistrationError(
            f"routing resolution must be finite: {value!r}"
        )
    allowed = {Decimal(item): item for item in REQUIRED_RESOLUTIONS}
    if decimal not in allowed:
        raise PreregistrationError(
            f"resolution {value!r} is not exactly one of {REQUIRED_RESOLUTIONS}"
        )
    return allowed[decimal]


def canonical_resolution_sequence(values: Sequence[Any]) -> tuple[str, ...]:
    result = tuple(canonical_resolution(value) for value in values)
    if len(result) != len(REQUIRED_RESOLUTIONS):
        raise PreregistrationError(
            f"resolution count {len(result)} != {len(REQUIRED_RESOLUTIONS)}"
        )
    if len(set(result)) != len(result):
        raise PreregistrationError("routing resolutions contain duplicates")
    if set(result) != set(REQUIRED_RESOLUTIONS):
        raise PreregistrationError(
            f"routing resolution set {result} != {REQUIRED_RESOLUTIONS}"
        )
    return result


def _sha256_fd(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            break
        size += len(block)
        digest.update(block)
    return "sha256:" + digest.hexdigest(), size


def stable_file_record(path: Path) -> StableFileRecord:
    """Hash one ordinary file through a stable descriptor without following links."""

    path = Path(path)
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise PreregistrationError(f"cannot stat source manifest {path}: {exc}") from exc
    if stat.S_ISLNK(before_path.st_mode):
        raise PreregistrationError(f"source manifest may not be a symlink: {path}")
    if not stat.S_ISREG(before_path.st_mode):
        raise PreregistrationError(f"source manifest is not a regular file: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PreregistrationError(f"cannot open source manifest {path}: {exc}") from exc
    try:
        before_fd = os.fstat(fd)
        digest, size = _sha256_fd(fd)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)

    try:
        after_path = path.lstat()
    except OSError as exc:
        raise PreregistrationError(
            f"source manifest disappeared during hashing {path}: {exc}"
        ) from exc

    signatures = {
        (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_mode,
        )
        for item in (before_path, before_fd, after_fd, after_path)
    }
    if len(signatures) != 1:
        raise PreregistrationError(
            f"source manifest changed or path was replaced while hashing: {path}"
        )
    if size != before_fd.st_size:
        raise PreregistrationError(
            f"source manifest short read: {path}: {size} != {before_fd.st_size}"
        )
    return StableFileRecord(
        path=str(path.resolve(strict=True)),
        sha256=digest,
        size=size,
        device=int(before_fd.st_dev),
        inode=int(before_fd.st_ino),
        mtime_ns=int(before_fd.st_mtime_ns),
    )


def source_manifest_groups(
    groups: Mapping[str, Sequence[Path]],
) -> dict[str, list[dict[str, Any]]]:
    """Verify exactly two non-empty, physically disjoint manifest groups."""

    if not isinstance(groups, Mapping) or set(groups) != set(SOURCE_GROUPS):
        raise PreregistrationError(
            f"source manifest groups must be exactly {SOURCE_GROUPS}"
        )
    result: dict[str, list[dict[str, Any]]] = {}
    physical_owner: dict[tuple[int, int], tuple[str, str]] = {}
    resolved_owner: dict[str, str] = {}
    for group in SOURCE_GROUPS:
        paths = groups[group]
        if (
            not isinstance(paths, Sequence)
            or isinstance(paths, (str, bytes))
            or not paths
        ):
            raise PreregistrationError(
                f"source manifest group {group!r} must be non-empty"
            )
        rows = []
        seen_group: set[tuple[int, int]] = set()
        for value in paths:
            record = stable_file_record(Path(value))
            physical = (record.device, record.inode)
            if physical in seen_group:
                raise PreregistrationError(
                    f"duplicate physical source manifest within {group}: {record.path}"
                )
            seen_group.add(physical)
            previous = physical_owner.get(physical)
            if previous is not None:
                raise PreregistrationError(
                    "source manifest filesystem alias crosses groups: "
                    f"{previous[0]}:{previous[1]} and {group}:{record.path}"
                )
            physical_owner[physical] = (group, record.path)
            if record.path in resolved_owner:
                raise PreregistrationError(
                    f"source manifest path reused across groups: {record.path}"
                )
            resolved_owner[record.path] = group
            rows.append(record.to_dict())
        result[group] = rows
    return result


def _require_sha(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise PreregistrationError(f"{label} must be sha256:<64 hex>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise PreregistrationError(f"{label} must be sha256:<64 hex>") from exc
    return value.lower()


def build_document(
    *,
    experiment_id: str,
    source_groups: Mapping[str, Sequence[Path]],
    source_commit: str,
    truth_manifest_sha256: str,
    basis_audit_sha256: str,
    routing_config_sha256: str,
    resolutions: Sequence[Any] = REQUIRED_RESOLUTIONS,
) -> dict[str, Any]:
    """Build the complete immutable run-input document before execution."""

    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise PreregistrationError("experiment_id must be non-empty")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) not in (40, 64)
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in source_commit
        )
    ):
        raise PreregistrationError(
            "source_commit must be a 40- or 64-hex Git object"
        )
    policy = canonical_policy()
    if sha256_bytes(canonical_json(policy)) != CANONICAL_POLICY_SEMANTIC_SHA256:
        raise PreregistrationError("compiled canonical policy digest changed")
    resolution_values = canonical_resolution_sequence(resolutions)
    return {
        "schema": SCHEMA,
        "experiment_id": experiment_id.strip(),
        "task_id": TASK_ID,
        "source_commit": source_commit.lower(),
        "policy": policy,
        "policy_semantic_sha256": CANONICAL_POLICY_SEMANTIC_SHA256,
        "resolutions_seconds": list(resolution_values),
        "source_manifests": source_manifest_groups(source_groups),
        "truth_manifest_sha256": _require_sha(
            truth_manifest_sha256, "truth_manifest_sha256"
        ),
        "basis_audit_sha256": _require_sha(
            basis_audit_sha256, "basis_audit_sha256"
        ),
        "routing_config_sha256": _require_sha(
            routing_config_sha256, "routing_config_sha256"
        ),
    }


def write_once(path: Path, document: Mapping[str, Any]) -> PreregisteredRun:
    """Atomically create one exact preregistration witness, never overwrite it."""

    path = Path(path)
    payload = canonical_json(document) + b"\n"
    semantic = sha256_bytes(canonical_json(document))
    container = sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o444)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != payload:
            raise PreregistrationError(
                f"refusing to replace differing preregistration: {path}"
            )
    else:
        try:
            written = 0
            while written < len(payload):
                count = os.write(fd, payload[written:])
                if count <= 0:
                    raise PreregistrationError(
                        f"short write creating preregistration: {path}"
                    )
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
    loaded = load(path)
    if (
        loaded.container_sha256 != container
        or loaded.semantic_sha256 != semantic
        or dict(loaded.document) != dict(document)
    ):
        raise PreregistrationError("preregistration replay verification failed")
    return loaded


def load(path: Path) -> PreregisteredRun:
    """Load and validate an existing preregistration witness."""

    record = stable_file_record(Path(path))
    payload = Path(record.path).read_bytes()
    if sha256_bytes(payload) != record.sha256:
        raise PreregistrationError(
            "preregistration bytes changed after stable hash"
        )
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreregistrationError(
            f"invalid preregistration JSON: {exc}"
        ) from exc
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise PreregistrationError("wrong preregistration schema")
    if document.get("task_id") != TASK_ID:
        raise PreregistrationError(
            "preregistration task differs from compiled task"
        )
    policy = document.get("policy")
    if policy != canonical_policy():
        raise PreregistrationError(
            "preregistration policy differs from compiled policy"
        )
    if (
        document.get("policy_semantic_sha256")
        != CANONICAL_POLICY_SEMANTIC_SHA256
    ):
        raise PreregistrationError("preregistration policy digest differs")
    canonical_resolution_sequence(document.get("resolutions_seconds") or ())
    manifests = document.get("source_manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != set(SOURCE_GROUPS):
        raise PreregistrationError(
            "preregistration source-manifest groups differ"
        )
    for group in SOURCE_GROUPS:
        rows = manifests[group]
        if not isinstance(rows, list) or not rows:
            raise PreregistrationError(
                f"preregistration source-manifest group {group!r} is empty"
            )
    semantic = sha256_bytes(canonical_json(document))
    return PreregisteredRun(
        path=record.path,
        container_sha256=record.sha256,
        semantic_sha256=semantic,
        document=document,
    )


def bind_report(
    report: Mapping[str, Any],
    preregistration: PreregisteredRun,
) -> None:
    """Require a later result to commit to the prewritten run-input witness."""

    if not isinstance(report, Mapping):
        raise PreregistrationError("binding report must be an object")
    expected = {
        "preregistration_path": preregistration.path,
        "preregistration_container_sha256": preregistration.container_sha256,
        "preregistration_semantic_sha256": preregistration.semantic_sha256,
        "experiment_id": preregistration.document["experiment_id"],
        "task_id": TASK_ID,
        "policy_semantic_sha256": CANONICAL_POLICY_SEMANTIC_SHA256,
        "selected_method": SELECTED_METHOD,
        "primary_resolution_seconds": PRIMARY_RESOLUTION,
        "sensitivity_resolutions_seconds": list(SENSITIVITY_RESOLUTIONS),
        "required_methods": list(REQUIRED_METHODS),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise PreregistrationError(
                f"report/preregistration binding mismatch for {key}: "
                f"{report.get(key)!r} != {value!r}"
            )
