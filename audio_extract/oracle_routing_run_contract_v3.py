"""Pre-execution contract for the binding certified-routing experiment.

The contract closes three provenance loopholes:

* tile durations are exact preregistered values, not rounded report labels;
* voiced and no-vocal source-manifest groups must be physically distinct files;
* a runner report must bind a validated run-input document that existed before
  any routing output was written.

This module does not optimize or render audio. It is deliberately independent of
separator and metric code so the same checks can be used by the runner and by an
independent verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .oracle_routing_work_contract_v3 import (
    REQUIRED_WORKS,
    WORK_CONTRACT_SHA256,
)
from .oracle_routing_work_contract_v3 import (
    identity_dict as work_contract_identity,
)
from .oracle_routing_work_contract_v3 import (
    validate_identity as validate_work_contract,
)

RUN_INPUT_SCHEMA = "audio-extract/oracle-routing-run-input/v3"
RUN_CLAIM_SCHEMA = "audio-extract/oracle-routing-run-claim/v1"
REPORT_BINDING_SCHEMA = "audio-extract/oracle-routing-run-binding/v3"
TASK_ID = "soloist_vs_rest"
SELECTED_METHOD = "O2_global_medoid"
REQUIRED_METHODS = ("O2_global_medoid", "O3_certified_convex")
SOURCE_GROUPS = ("voiced", "no_vocal")


@dataclass(frozen=True)
class ResolutionSpec:
    seconds_decimal: str
    tile_frames: int
    role: str

    @property
    def seconds(self) -> float:
        return float(Decimal(self.seconds_decimal))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seconds_decimal": self.seconds_decimal,
            "tile_frames": self.tile_frames,
            "role": self.role,
        }


CANONICAL_RESOLUTIONS = (
    ResolutionSpec("2.0", 88_200, "sensitivity"),
    ResolutionSpec("1.0", 44_100, "primary"),
    ResolutionSpec("0.5", 22_050, "sensitivity"),
)

FROZEN_POLICY = {
    "schema": "audio-extract/oracle-routing-policy/v3",
    "task_id": TASK_ID,
    "selected_method": SELECTED_METHOD,
    "required_methods": list(REQUIRED_METHODS),
    "required_works": list(REQUIRED_WORKS),
    "work_contract_sha256": WORK_CONTRACT_SHA256,
    "resolutions": [item.to_dict() for item in CANONICAL_RESOLUTIONS],
}

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")


class RunContractError(RuntimeError):
    """The routing run is not the preregistered experiment it claims to be."""


@dataclass(frozen=True)
class StableFileRecord:
    path: str
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ValidatedRunInput:
    path: str
    sha256: str
    document: Mapping[str, Any]
    source_records: Mapping[str, tuple[StableFileRecord, ...]]


@dataclass(frozen=True)
class RunPreflight:
    run_input: ValidatedRunInput
    claim: Mapping[str, Any]
    binding: Mapping[str, Any]


_BASIC_RECORD_KEYS = {"path", "sha256"}
_BOUND_RECORD_KEYS = {
    *_BASIC_RECORD_KEYS,
    "device",
    "inode",
    "size",
    "mtime_ns",
    "ctime_ns",
    "mode",
}
_ANCHOR_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"
    r"issues/[1-9][0-9]*#issuecomment-[1-9][0-9]*$"
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunContractError(f"value is not canonical JSON: {exc}") from exc


def _sha_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def policy_sha256() -> str:
    return _sha_bytes(_canonical_bytes(FROZEN_POLICY))


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise RunContractError(f"{label} must be sha256:<64 lowercase hex>")
    return result


def _require_git(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if not _GIT_RE.fullmatch(result):
        raise RunContractError(f"{label} must be a 40- or 64-hex Git object ID")
    return result


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _require_utc(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result.endswith("Z"):
        raise RunContractError(f"{label} must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        raise RunContractError(f"{label} is not an ISO-8601 timestamp") from exc
    return result


def _utc_datetime(value: Any, label: str) -> datetime:
    text = _require_utc(value, label)
    return datetime.fromisoformat(text[:-1] + "+00:00")


def _require_output_root_absent(document: Mapping[str, Any], label: str) -> Path:
    root = Path(str(document.get("output_root") or ""))
    if not root.is_absolute():
        raise RunContractError("output_root must be absolute")
    if str(root.resolve(strict=False)) != str(root):
        raise RunContractError("output_root must be normalized and resolved")
    if os.path.lexists(root):
        raise RunContractError(f"{label}: output_root already exists: {root}")
    return root


def _require_outside_output_root(
    path: Path, document: Mapping[str, Any], label: str
) -> Path:
    resolved = path.resolve(strict=False)
    root = Path(str(document["output_root"]))
    if resolved == root or resolved.is_relative_to(root):
        raise RunContractError(f"{label} must be outside output_root")
    return resolved


def _resolution_decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise RunContractError("resolution values may not be booleans")
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RunContractError("resolution values must be finite")
        text = repr(value)
    else:
        raise RunContractError(
            f"resolution value has unsupported type {type(value).__name__}"
        )
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise RunContractError(f"invalid resolution value {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise RunContractError("resolution values must be finite and positive")
    return result


def require_canonical_resolutions(values: Sequence[Any]) -> tuple[float, ...]:
    """Refuse formatted aliases such as 1.04 -> report key '1.0'."""

    supplied = tuple(values)
    if len(supplied) != len(CANONICAL_RESOLUTIONS):
        raise RunContractError(
            f"resolution count {len(supplied)} != {len(CANONICAL_RESOLUTIONS)}"
        )
    for index, (value, expected) in enumerate(
        zip(supplied, CANONICAL_RESOLUTIONS, strict=True)
    ):
        actual_decimal = _resolution_decimal(value)
        expected_decimal = Decimal(expected.seconds_decimal)
        if actual_decimal != expected_decimal:
            raise RunContractError(
                f"resolution[{index}]={actual_decimal} is not the exact "
                f"preregistered {expected.seconds_decimal} seconds"
            )
        frames = actual_decimal * Decimal(44_100)
        if frames != frames.to_integral_value() or int(frames) != expected.tile_frames:
            raise RunContractError(
                f"resolution[{index}] does not map to canonical tile frames"
            )
    return tuple(item.seconds for item in CANONICAL_RESOLUTIONS)


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def _stable_bytes(path_value: Any, label: str) -> tuple[StableFileRecord, bytes]:
    raw = Path(str(path_value or ""))
    if not raw.is_absolute():
        raise RunContractError(f"{label}.path must be absolute")
    try:
        if raw.is_symlink():
            raise RunContractError(f"{label}.path may not be a symlink: {raw}")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(raw, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RunContractError(f"{label}.path is not a regular file")
            digest = hashlib.sha256()
            payload = bytearray()
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                digest.update(block)
                payload.extend(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = os.stat(raw, follow_symlinks=False)
    except RunContractError:
        raise
    except OSError as exc:
        raise RunContractError(f"cannot stably read {label}: {raw}: {exc}") from exc
    if _file_signature(before) != _file_signature(after):
        raise RunContractError(f"{label} changed while being read")
    if _file_signature(before) != _file_signature(current):
        raise RunContractError(f"{label} path was replaced while being read")
    return (
        StableFileRecord(
            path=str(raw.resolve(strict=True)),
            sha256="sha256:" + digest.hexdigest(),
            device=int(before.st_dev),
            inode=int(before.st_ino),
            size=int(before.st_size),
            mtime_ns=int(before.st_mtime_ns),
            ctime_ns=int(before.st_ctime_ns),
            mode=stat.S_IMODE(before.st_mode),
        ),
        bytes(payload),
    )


def _stable_file(path_value: Any, declared_sha: Any, label: str) -> StableFileRecord:
    declared = _require_sha(declared_sha, f"{label}.sha256")
    record, _ = _stable_bytes(path_value, label)
    if record.sha256 != declared:
        raise RunContractError(f"{label} hash mismatch: {record.sha256} != {declared}")
    return record


def _verify_bound_record(
    value: Mapping[str, Any], record: StableFileRecord, label: str
) -> None:
    for field in ("device", "inode", "size", "mtime_ns", "ctime_ns", "mode"):
        declared = value.get(field)
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise RunContractError(f"{label}.{field} must be an integer")
        if declared != getattr(record, field):
            raise RunContractError(
                f"{label}.{field} changed: {getattr(record, field)} != {declared}"
            )


def _records(
    values: Any,
    label: str,
    *,
    require_bound: bool = False,
) -> tuple[StableFileRecord, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RunContractError(f"{label} must be a non-empty array")
    if not values:
        raise RunContractError(f"{label} may not be empty")
    result = []
    for index, value in enumerate(values):
        expected_keys = _BOUND_RECORD_KEYS if require_bound else _BASIC_RECORD_KEYS
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise RunContractError(
                f"{label}[{index}] keys differ from {sorted(expected_keys)}"
            )
        record = _stable_file(value["path"], value["sha256"], f"{label}[{index}]")
        if require_bound:
            _verify_bound_record(value, record, f"{label}[{index}]")
        result.append(record)
    paths = [item.path for item in result]
    inodes = [(item.device, item.inode) for item in result]
    if len(set(paths)) != len(paths) or len(set(inodes)) != len(inodes):
        raise RunContractError(f"{label} contains duplicate physical files")
    return tuple(result)


def _source_groups(
    value: Any, *, require_bound: bool = False
) -> dict[str, tuple[StableFileRecord, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_GROUPS):
        raise RunContractError(
            f"source_manifests must contain exactly {list(SOURCE_GROUPS)}"
        )
    result = {
        group: _records(
            value[group],
            f"source_manifests.{group}",
            require_bound=require_bound,
        )
        for group in SOURCE_GROUPS
    }
    seen_path: dict[str, str] = {}
    seen_inode: dict[tuple[int, int], str] = {}
    seen_sha: dict[str, str] = {}
    for group in SOURCE_GROUPS:
        for record in result[group]:
            previous_path = seen_path.setdefault(record.path, group)
            if previous_path != group:
                raise RunContractError(
                    "one resolved source manifest is reused across voiced and "
                    "no_vocal groups"
                )
            identity = (record.device, record.inode)
            previous_inode = seen_inode.setdefault(identity, group)
            if previous_inode != group:
                raise RunContractError(
                    "one physical source-manifest inode is reused across voiced "
                    "and no_vocal groups"
                )
            previous_sha = seen_sha.setdefault(record.sha256, group)
            if previous_sha != group:
                raise RunContractError(
                    "byte-identical source manifests are reused across voiced "
                    "and no_vocal groups"
                )
    return result


def _record_mapping(records: Iterable[StableFileRecord]) -> list[dict[str, Any]]:
    return [record.public_dict() for record in records]


def build_run_input(
    *,
    source_commit: str,
    source_manifests: Mapping[str, Sequence[Mapping[str, str]]],
    truth_manifest: Mapping[str, str],
    basis_audit: Mapping[str, str],
    routing_config: Mapping[str, str],
    legacy_preregistration: Mapping[str, str],
    candidate_manifests: Sequence[Mapping[str, str]],
    output_root: str,
    prepared_nonce: str | None = None,
    prepared_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate one closed pre-execution input document."""

    commit = _require_git(source_commit, "source_commit")
    groups = _source_groups(source_manifests)
    truth = _records([truth_manifest], "truth_manifest")[0]
    basis = _records([basis_audit], "basis_audit")[0]
    routing = _records([routing_config], "routing_config")[0]
    legacy = _records([legacy_preregistration], "legacy_preregistration")[0]
    candidates = _records(candidate_manifests, "candidate_manifests")
    supplied_root = Path(str(output_root or ""))
    if not supplied_root.is_absolute():
        raise RunContractError("output_root must be absolute")
    root = supplied_root.resolve(strict=False)
    if os.path.lexists(root):
        raise RunContractError(f"output_root already exists: {root}")
    nonce = prepared_nonce or secrets.token_hex(32)
    if not _NONCE_RE.fullmatch(nonce):
        raise RunContractError("prepared_nonce must be 64 lowercase hex characters")
    prepared = _require_utc(prepared_at_utc or _utc_now(), "prepared_at_utc")
    return {
        "schema": RUN_INPUT_SCHEMA,
        "task_id": TASK_ID,
        "source_commit": commit,
        "frozen_policy": FROZEN_POLICY,
        "frozen_policy_sha256": policy_sha256(),
        "required_methods": list(REQUIRED_METHODS),
        "required_works": list(REQUIRED_WORKS),
        "work_contract": work_contract_identity(),
        "work_contract_sha256": WORK_CONTRACT_SHA256,
        "resolutions": [item.to_dict() for item in CANONICAL_RESOLUTIONS],
        "source_manifests": {
            group: _record_mapping(groups[group]) for group in SOURCE_GROUPS
        },
        "truth_manifest": truth.public_dict(),
        "basis_audit": basis.public_dict(),
        "routing_config": routing.public_dict(),
        "legacy_preregistration": legacy.public_dict(),
        "candidate_manifests": _record_mapping(candidates),
        "output_root": str(root),
        "prepared_nonce": nonce,
        "prepared_at_utc": prepared,
    }


def _validate_semantics(document: Mapping[str, Any]) -> None:
    required_keys = {
        "schema",
        "task_id",
        "source_commit",
        "frozen_policy",
        "frozen_policy_sha256",
        "required_methods",
        "required_works",
        "work_contract",
        "work_contract_sha256",
        "resolutions",
        "source_manifests",
        "truth_manifest",
        "basis_audit",
        "routing_config",
        "legacy_preregistration",
        "candidate_manifests",
        "output_root",
        "prepared_nonce",
        "prepared_at_utc",
    }
    if set(document) != required_keys:
        raise RunContractError(
            f"run-input keys differ: missing={sorted(required_keys - set(document))}, "
            f"extra={sorted(set(document) - required_keys)}"
        )
    if document.get("schema") != RUN_INPUT_SCHEMA:
        raise RunContractError("run-input schema mismatch")
    if document.get("task_id") != TASK_ID:
        raise RunContractError("run-input task is not the frozen task")
    _require_git(document.get("source_commit"), "source_commit")
    if document.get("frozen_policy") != FROZEN_POLICY:
        raise RunContractError("run-input frozen policy differs from compiled policy")
    if (
        _require_sha(document.get("frozen_policy_sha256"), "frozen_policy_sha256")
        != policy_sha256()
    ):
        raise RunContractError("run-input frozen policy digest differs")
    if tuple(document.get("required_methods") or ()) != REQUIRED_METHODS:
        raise RunContractError("run-input method set/order differs")
    if tuple(document.get("required_works") or ()) != REQUIRED_WORKS:
        raise RunContractError("run-input work set/order differs")
    work_contract = document.get("work_contract")
    if not isinstance(work_contract, Mapping):
        raise RunContractError("run-input work contract must be an object")
    try:
        validate_work_contract(work_contract)
    except ValueError as exc:
        raise RunContractError(f"run-input work contract differs: {exc}") from exc
    if (
        _require_sha(document.get("work_contract_sha256"), "work_contract_sha256")
        != WORK_CONTRACT_SHA256
    ):
        raise RunContractError("run-input work contract digest differs")
    if document.get("resolutions") != [
        item.to_dict() for item in CANONICAL_RESOLUTIONS
    ]:
        raise RunContractError("run-input resolution identity differs")
    require_canonical_resolutions(
        [item["seconds_decimal"] for item in document["resolutions"]]
    )
    nonce = str(document.get("prepared_nonce") or "")
    if not _NONCE_RE.fullmatch(nonce):
        raise RunContractError("prepared_nonce is invalid")
    _require_utc(document.get("prepared_at_utc"), "prepared_at_utc")
    root = Path(str(document.get("output_root") or ""))
    if not root.is_absolute():
        raise RunContractError("output_root must be absolute")
    if str(root.resolve(strict=False)) != str(root):
        raise RunContractError("output_root must be normalized and resolved")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except RunContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunContractError(f"cannot parse {label}: {exc}") from exc


def _read_bound_json_record(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BOUND_RECORD_KEYS:
        raise RunContractError(f"{label} is not a bound file record")
    record, payload = _stable_bytes(value["path"], label)
    if record.sha256 != _require_sha(value["sha256"], f"{label}.sha256"):
        raise RunContractError(f"{label} hash mismatch")
    _verify_bound_record(value, record, label)
    document = _parse_json(payload, label)
    if not isinstance(document, Mapping):
        raise RunContractError(f"{label} must contain a JSON object")
    return document


def load_bound_json_artifact(
    validated: ValidatedRunInput, field: str
) -> Mapping[str, Any]:
    if field not in {
        "truth_manifest",
        "basis_audit",
        "routing_config",
        "legacy_preregistration",
    }:
        raise RunContractError(f"unsupported bound JSON field: {field}")
    return _read_bound_json_record(validated.document[field], field)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _verify_digest_sidecar(path: Path, expected: str, label: str) -> None:
    _, payload = _stable_bytes(_sidecar_path(path), f"{label}_sidecar")
    try:
        supplied = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RunContractError(f"{label} sidecar is not ASCII") from exc
    if supplied != expected:
        raise RunContractError(f"{label} sidecar mismatch")


def validate_run_input(path: Path) -> ValidatedRunInput:
    """Reopen, validate, and rehash a prewritten run input and every dependency."""

    record, payload = _stable_bytes(path, "run_input")
    document = _parse_json(payload, "run input")
    if not isinstance(document, Mapping):
        raise RunContractError("run input must be a JSON object")
    _validate_semantics(document)
    groups = _source_groups(document["source_manifests"], require_bound=True)
    _records([document["truth_manifest"]], "truth_manifest", require_bound=True)
    _records([document["basis_audit"]], "basis_audit", require_bound=True)
    _records([document["routing_config"]], "routing_config", require_bound=True)
    _records(
        [document["legacy_preregistration"]],
        "legacy_preregistration",
        require_bound=True,
    )
    _records(
        document["candidate_manifests"],
        "candidate_manifests",
        require_bound=True,
    )
    for field in (
        "truth_manifest",
        "basis_audit",
        "routing_config",
        "legacy_preregistration",
    ):
        _read_bound_json_record(document[field], field)
    canonical = _canonical_bytes(document)
    digest = _sha_bytes(canonical)
    _verify_digest_sidecar(Path(record.path), digest, "run-input")
    return ValidatedRunInput(
        path=record.path,
        sha256=digest,
        document=document,
        source_records=groups,
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without ever exposing a partial final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        _, existing = _stable_bytes(path, f"existing_{path.name}")
        if existing == payload:
            return
        raise FileExistsError(f"immutable destination already differs: {path}")
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-length write while publishing immutable file")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _, existing = _stable_bytes(path, f"existing_{path.name}")
            if existing != payload:
                raise
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_immutable_json_artifact(
    path: Path, value: Mapping[str, Any]
) -> dict[str, str]:
    """Publish one canonical preregistration dependency and return its record."""

    resolved = path.resolve(strict=False)
    payload = _canonical_bytes(value) + b"\n"
    _publish_immutable(resolved, payload)
    record, reopened = _stable_bytes(resolved, "immutable_json_artifact")
    if reopened != payload:
        raise RunContractError("immutable JSON artifact changed after publication")
    return {"path": record.path, "sha256": record.sha256}


def _validate_dependency_bindings(document: Mapping[str, Any]) -> None:
    _source_groups(document["source_manifests"], require_bound=True)
    _records([document["truth_manifest"]], "truth_manifest", require_bound=True)
    _records([document["basis_audit"]], "basis_audit", require_bound=True)
    _records([document["routing_config"]], "routing_config", require_bound=True)
    _records(
        [document["legacy_preregistration"]],
        "legacy_preregistration",
        require_bound=True,
    )
    _records(
        document["candidate_manifests"],
        "candidate_manifests",
        require_bound=True,
    )
    for field in (
        "truth_manifest",
        "basis_audit",
        "routing_config",
        "legacy_preregistration",
    ):
        _read_bound_json_record(document[field], field)


def write_run_input(path: Path, document: Mapping[str, Any]) -> str:
    """Persist a canonical immutable run input; differing replay is impossible."""

    _validate_semantics(document)
    _validate_dependency_bindings(document)
    _require_output_root_absent(document, "run-input publication")
    path = _require_outside_output_root(path, document, "run-input path")
    payload = _canonical_bytes(document) + b"\n"
    digest = _sha_bytes(_canonical_bytes(document))
    _publish_immutable(_sidecar_path(path), (digest + "\n").encode())
    _publish_immutable(path, payload)
    validated = validate_run_input(path)
    if validated.sha256 != digest:
        raise RunContractError("run-input digest changed after publication")
    return digest


def claim_run_input(
    run_input_path: Path,
    claim_path: Path,
    *,
    external_anchor: str,
) -> dict[str, Any]:
    """Create an immutable pre-run claim before any output calculation starts."""

    validated = validate_run_input(run_input_path)
    _require_output_root_absent(validated.document, "run-input claim")
    anchor = str(external_anchor or "").strip()
    if not _ANCHOR_RE.fullmatch(anchor):
        raise RunContractError(
            "external_anchor must be an exact GitHub issue-comment URL"
        )
    resolved_claim_path = _require_outside_output_root(
        claim_path, validated.document, "run-claim path"
    )
    claim = {
        "schema": RUN_CLAIM_SCHEMA,
        "run_input_path": validated.path,
        "run_input_sha256": validated.sha256,
        "frozen_policy_sha256": policy_sha256(),
        "external_anchor": anchor,
        "claimed_at_utc": _utc_now(),
        "claim_nonce": secrets.token_hex(32),
        "pid": os.getpid(),
    }
    payload = _canonical_bytes(claim) + b"\n"
    claim_sha = _sha_bytes(_canonical_bytes(claim))
    _publish_immutable(
        _sidecar_path(resolved_claim_path),
        (claim_sha + "\n").encode(),
    )
    _publish_immutable(resolved_claim_path, payload)
    _require_output_root_absent(validated.document, "run-input claim publication")
    loaded = _load_claim(resolved_claim_path, validated)
    if loaded["claim_sha256"] != claim_sha:
        raise RunContractError("run claim changed after publication")
    return loaded


def _load_claim(
    claim_path: Path,
    validated: ValidatedRunInput,
) -> dict[str, Any]:
    record, payload = _stable_bytes(claim_path, "run_claim")
    claim = _parse_json(payload, "run claim")
    required = {
        "schema",
        "run_input_path",
        "run_input_sha256",
        "frozen_policy_sha256",
        "external_anchor",
        "claimed_at_utc",
        "claim_nonce",
        "pid",
    }
    if not isinstance(claim, Mapping) or set(claim) != required:
        raise RunContractError("run claim keys differ from the frozen schema")
    if claim.get("schema") != RUN_CLAIM_SCHEMA:
        raise RunContractError("run claim schema mismatch")
    if claim.get("run_input_path") != validated.path:
        raise RunContractError("run claim path does not bind the run input")
    if (
        _require_sha(claim.get("run_input_sha256"), "run_input_sha256")
        != validated.sha256
    ):
        raise RunContractError("run claim digest does not bind the run input")
    if (
        _require_sha(claim.get("frozen_policy_sha256"), "frozen_policy_sha256")
        != policy_sha256()
    ):
        raise RunContractError("run claim policy digest differs")
    anchor = str(claim.get("external_anchor") or "")
    if not _ANCHOR_RE.fullmatch(anchor):
        raise RunContractError("run claim external anchor is invalid")
    claimed = _utc_datetime(claim.get("claimed_at_utc"), "claimed_at_utc")
    prepared = _utc_datetime(validated.document["prepared_at_utc"], "prepared_at_utc")
    if claimed < prepared:
        raise RunContractError("run claim predates the prepared run input")
    if not _NONCE_RE.fullmatch(str(claim.get("claim_nonce") or "")):
        raise RunContractError("run claim nonce is invalid")
    pid = claim.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RunContractError("run claim pid is invalid")
    claim_sha = _sha_bytes(_canonical_bytes(claim))
    _verify_digest_sidecar(Path(record.path), claim_sha, "run-claim")
    return {
        **claim,
        "claim_sha256": claim_sha,
        "claim_path": record.path,
    }


def report_binding(
    validated: ValidatedRunInput,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    if claim.get("schema") != RUN_CLAIM_SCHEMA:
        raise RunContractError("claim schema mismatch")
    if claim.get("run_input_sha256") != validated.sha256:
        raise RunContractError("claim does not bind the validated run input")
    claim_sha = _require_sha(claim.get("claim_sha256"), "claim_sha256")
    return {
        "schema": REPORT_BINDING_SCHEMA,
        "run_input_path": validated.path,
        "run_input_sha256": validated.sha256,
        "output_root": validated.document["output_root"],
        "prepared_at_utc": validated.document["prepared_at_utc"],
        "prepared_nonce": validated.document["prepared_nonce"],
        "claim_path": str(claim["claim_path"]),
        "claim_sha256": claim_sha,
        "claimed_at_utc": claim["claimed_at_utc"],
        "claim_nonce": claim["claim_nonce"],
        "frozen_policy_sha256": policy_sha256(),
        "work_contract_sha256": WORK_CONTRACT_SHA256,
        "source_commit": validated.document["source_commit"],
        "external_anchor": claim["external_anchor"],
    }


def preflight_run(run_input_path: Path, claim_path: Path) -> RunPreflight:
    """Fail closed before audio opens, output creation, or optimizer setup."""

    validated = validate_run_input(run_input_path)
    _require_output_root_absent(validated.document, "runner preflight")
    claim = _load_claim(claim_path, validated)
    _require_output_root_absent(validated.document, "runner preflight completion")
    return RunPreflight(
        run_input=validated,
        claim=claim,
        binding=report_binding(validated, claim),
    )


def verify_report_binding(
    report: Mapping[str, Any],
    run_input_path: Path,
    claim_path: Path,
) -> ValidatedRunInput:
    """Require the report to reference the exact prewritten input and claim."""

    validated = validate_run_input(run_input_path)
    claim = _load_claim(claim_path, validated)
    supplied = report.get("run_input_binding")
    if not isinstance(supplied, Mapping):
        raise RunContractError("report lacks run_input_binding")
    expected = report_binding(
        validated,
        claim,
    )
    if supplied != expected:
        raise RunContractError("report/run-input precommit binding mismatch")
    return validated
