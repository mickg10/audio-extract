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

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
import re
import secrets
import stat


RUN_INPUT_SCHEMA = "audio-extract/oracle-routing-run-input/v3"
RUN_CLAIM_SCHEMA = "audio-extract/oracle-routing-run-claim/v1"
REPORT_BINDING_SCHEMA = "audio-extract/oracle-routing-run-binding/v3"
TASK_ID = "soloist_vs_rest"
SELECTED_METHOD = "O2_global_medoid"
REQUIRED_METHODS = ("O2_global_medoid", "O3_certified_convex")
REQUIRED_WORKS = (
    "bologna_verdi",
    "bologna_donizetti",
    "bologna_puccini",
    "aalto_mozart_dry",
)
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

    def public_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ValidatedRunInput:
    path: str
    sha256: str
    document: Mapping[str, Any]
    source_records: Mapping[str, tuple[StableFileRecord, ...]]


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
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
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


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _stable_file(path_value: Any, declared_sha: Any, label: str) -> StableFileRecord:
    declared = _require_sha(declared_sha, f"{label}.sha256")
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
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                digest.update(block)
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
    actual = "sha256:" + digest.hexdigest()
    if actual != declared:
        raise RunContractError(f"{label} hash mismatch: {actual} != {declared}")
    return StableFileRecord(
        path=str(raw.resolve(strict=True)),
        sha256=actual,
        device=int(before.st_dev),
        inode=int(before.st_ino),
        size=int(before.st_size),
    )


def _records(
    values: Any,
    label: str,
) -> tuple[StableFileRecord, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RunContractError(f"{label} must be a non-empty array")
    if not values:
        raise RunContractError(f"{label} may not be empty")
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise RunContractError(
                f"{label}[{index}] must contain exactly path and sha256"
            )
        result.append(
            _stable_file(
                value["path"], value["sha256"], f"{label}[{index}]"
            )
        )
    paths = [item.path for item in result]
    inodes = [(item.device, item.inode) for item in result]
    if len(set(paths)) != len(paths) or len(set(inodes)) != len(inodes):
        raise RunContractError(f"{label} contains duplicate physical files")
    return tuple(result)


def _source_groups(value: Any) -> dict[str, tuple[StableFileRecord, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_GROUPS):
        raise RunContractError(
            f"source_manifests must contain exactly {list(SOURCE_GROUPS)}"
        )
    result = {
        group: _records(value[group], f"source_manifests.{group}")
        for group in SOURCE_GROUPS
    }
    seen_path: dict[str, str] = {}
    seen_inode: dict[tuple[int, int], str] = {}
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
    return result


def _record_mapping(records: Iterable[StableFileRecord]) -> list[dict[str, str]]:
    return [record.public_dict() for record in records]


def build_run_input(
    *,
    source_commit: str,
    source_manifests: Mapping[str, Sequence[Mapping[str, str]]],
    truth_manifest: Mapping[str, str],
    basis_audit: Mapping[str, str],
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
    candidates = _records(candidate_manifests, "candidate_manifests")
    root = Path(str(output_root or ""))
    if not root.is_absolute():
        raise RunContractError("output_root must be absolute")
    nonce = prepared_nonce or secrets.token_hex(32)
    if not _NONCE_RE.fullmatch(nonce):
        raise RunContractError("prepared_nonce must be 64 lowercase hex characters")
    prepared = _require_utc(
        prepared_at_utc or _utc_now(), "prepared_at_utc"
    )
    return {
        "schema": RUN_INPUT_SCHEMA,
        "task_id": TASK_ID,
        "source_commit": commit,
        "frozen_policy": FROZEN_POLICY,
        "frozen_policy_sha256": policy_sha256(),
        "required_methods": list(REQUIRED_METHODS),
        "required_works": list(REQUIRED_WORKS),
        "resolutions": [item.to_dict() for item in CANONICAL_RESOLUTIONS],
        "source_manifests": {
            group: _record_mapping(groups[group]) for group in SOURCE_GROUPS
        },
        "truth_manifest": truth.public_dict(),
        "basis_audit": basis.public_dict(),
        "candidate_manifests": _record_mapping(candidates),
        "output_root": str(root),
        "prepared_nonce": nonce,
        "prepared_at_utc": prepared,
    }


def _validate_semantics(document: Mapping[str, Any]) -> None:
    required_keys = {
        "schema", "task_id", "source_commit", "frozen_policy",
        "frozen_policy_sha256", "required_methods", "required_works",
        "resolutions", "source_manifests", "truth_manifest", "basis_audit",
        "candidate_manifests", "output_root", "prepared_nonce",
        "prepared_at_utc",
    }
    if set(document) != required_keys:
        raise RunContractError(
            f"run-input keys differ: missing={sorted(required_keys-set(document))}, "
            f"extra={sorted(set(document)-required_keys)}"
        )
    if document.get("schema") != RUN_INPUT_SCHEMA:
        raise RunContractError("run-input schema mismatch")
    if document.get("task_id") != TASK_ID:
        raise RunContractError("run-input task is not the frozen task")
    _require_git(document.get("source_commit"), "source_commit")
    if document.get("frozen_policy") != FROZEN_POLICY:
        raise RunContractError("run-input frozen policy differs from compiled policy")
    if _require_sha(
        document.get("frozen_policy_sha256"), "frozen_policy_sha256"
    ) != policy_sha256():
        raise RunContractError("run-input frozen policy digest differs")
    if tuple(document.get("required_methods") or ()) != REQUIRED_METHODS:
        raise RunContractError("run-input method set/order differs")
    if tuple(document.get("required_works") or ()) != REQUIRED_WORKS:
        raise RunContractError("run-input work set/order differs")
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
    if not Path(str(document.get("output_root") or "")).is_absolute():
        raise RunContractError("output_root must be absolute")


def validate_run_input(path: Path) -> ValidatedRunInput:
    """Reopen, validate, and rehash a prewritten run input and every dependency."""

    record = _stable_file(
        path, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_input",
    )
    try:
        document = json.loads(Path(record.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunContractError(f"cannot parse run input: {exc}") from exc
    if not isinstance(document, Mapping):
        raise RunContractError("run input must be a JSON object")
    _validate_semantics(document)
    groups = _source_groups(document["source_manifests"])
    _records([document["truth_manifest"]], "truth_manifest")
    _records([document["basis_audit"]], "basis_audit")
    _records(document["candidate_manifests"], "candidate_manifests")
    canonical = _canonical_bytes(document)
    digest = _sha_bytes(canonical)
    return ValidatedRunInput(
        path=record.path,
        sha256=digest,
        document=document,
        source_records=groups,
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_run_input(path: Path, document: Mapping[str, Any]) -> str:
    """Persist a canonical immutable run input; differing replay is impossible."""

    _validate_semantics(document)
    payload = _canonical_bytes(document) + b"\n"
    _write_exclusive(path, payload)
    digest = _sha_bytes(_canonical_bytes(document))
    _write_exclusive(path.with_suffix(path.suffix + ".sha256"), (digest + "\n").encode())
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
    anchor = str(external_anchor or "").strip()
    if not anchor:
        raise RunContractError("external_anchor must be non-empty")
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
    _write_exclusive(claim_path, payload)
    claim_sha = _sha_bytes(_canonical_bytes(claim))
    _write_exclusive(
        claim_path.with_suffix(claim_path.suffix + ".sha256"),
        (claim_sha + "\n").encode(),
    )
    return {**claim, "claim_sha256": claim_sha, "claim_path": str(claim_path.resolve())}


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
        "run_input_sha256": validated.sha256,
        "claim_sha256": claim_sha,
        "frozen_policy_sha256": policy_sha256(),
        "source_commit": validated.document["source_commit"],
        "prepared_nonce": validated.document["prepared_nonce"],
        "external_anchor": claim["external_anchor"],
    }


def verify_report_binding(
    report: Mapping[str, Any],
    run_input_path: Path,
    claim_path: Path,
) -> ValidatedRunInput:
    """Require the report to reference the exact prewritten input and claim."""

    validated = validate_run_input(run_input_path)
    try:
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunContractError(f"cannot parse run claim: {exc}") from exc
    if not isinstance(claim, Mapping) or claim.get("schema") != RUN_CLAIM_SCHEMA:
        raise RunContractError("run claim is invalid")
    claim_sha = _sha_bytes(_canonical_bytes(claim))
    sidecar = claim_path.with_suffix(claim_path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != claim_sha:
        raise RunContractError("run-claim sidecar mismatch")
    supplied = report.get("run_input_binding")
    if not isinstance(supplied, Mapping):
        raise RunContractError("report lacks run_input_binding")
    expected = report_binding(
        validated,
        {**claim, "claim_sha256": claim_sha},
    )
    if supplied != expected:
        raise RunContractError("report/run-input precommit binding mismatch")
    return validated
