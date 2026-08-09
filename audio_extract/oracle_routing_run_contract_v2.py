"""Immutable pre-run and exact-truth contract for binding oracle routing.

This module is independent of the routing optimizer. It closes provenance
ambiguities before expensive audio work begins:

* resolution identity is the exact numeric tuple ``(2.0, 1.0, 0.5)`` and is
  never inferred from formatted labels;
* voiced and no-vocal source manifests must be physically distinct files, not
  merely different path strings;
* every exact M/A/V reference and the complete run-input record are frozen and
  content-bound before the first route is computed.

A completed routing report must carry the SHA-256 of the prewritten run-input
record. An independent verifier can then reject a report paired with a record
created or changed after the run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import identity
from .oracle_routing_binding_policy_v2 import CANONICAL_POLICY_SHA256

RUN_INPUT_SCHEMA = "audio-extract/oracle-routing-run-inputs/v2"
TRUTH_MANIFEST_SCHEMA = "audio-extract/oracle-routing-truth-manifest/v2"
RUN_ANCHOR_SCHEMA = "audio-extract/oracle-routing-run-anchor/v2"
SOURCE_MANIFEST_GROUPS = ("voiced", "no_vocal")
CANONICAL_RESOLUTIONS = (2.0, 1.0, 0.5)
RESOLUTION_KEYS = {2.0: "2.0", 1.0: "1.0", 0.5: "0.5"}
TRUTH_FILENAMES = {
    "mixture": "mix_with_voice.wav",
    "accompaniment": "orchestra_only.wav",
    "vocal": "voice_ref.wav",
}


class RunContractError(RuntimeError):
    """The pre-run inputs or exact references are not immutable/coherent."""


@dataclass(frozen=True)
class StableFile:
    path: str
    container_sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class RunInputAnchor:
    schema: str
    path: str
    sha256: str
    code_commit: str
    works: tuple[str, ...]
    resolutions_seconds: tuple[float, ...]
    truth_manifest_path: str
    truth_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "sha256": self.sha256,
            "code_commit": self.code_commit,
            "works": list(self.works),
            "resolutions_seconds": list(self.resolutions_seconds),
            "truth_manifest_path": self.truth_manifest_path,
            "truth_manifest_sha256": self.truth_manifest_sha256,
        }


def _sha_identity(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not result.startswith("sha256:") or len(result) != 71:
        raise RunContractError(f"{label} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise RunContractError(f"{label} must be sha256:<64 hex>") from exc
    return result


def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def stable_file(path: Path) -> tuple[bytes, StableFile]:
    """Read one non-symlink regular file and bind the path to its opened inode."""

    try:
        if path.is_symlink():
            raise RunContractError(f"refusing symlinked input: {path}")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RunContractError(f"input is not a regular file: {path}")
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except RunContractError:
        raise
    except OSError as exc:
        raise RunContractError(f"cannot read {path}: {exc}") from exc
    if (
        _signature(before) != _signature(after)
        or _signature(before) != _signature(current)
    ):
        raise RunContractError(f"input changed or path was replaced: {path}")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return payload, StableFile(
        path=str(path.resolve(strict=True)),
        container_sha256=digest,
        device=int(before.st_dev),
        inode=int(before.st_ino),
        size=int(before.st_size),
        mtime_ns=int(before.st_mtime_ns),
    )


def _json_object(payload: bytes, label: str) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RunContractError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunContractError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RunContractError(f"{label} must be a JSON object")
    _reject_nonfinite(value, label)
    return value


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunContractError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{index}]")
        return
    raise RunContractError(
        f"{label} contains unsupported JSON value {type(value)!r}"
    )


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Canonical semantic JSON for equality/hashing, with tuples as arrays."""

    _reject_nonfinite(value, "canonical JSON")
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _same_json(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_json(left) == canonical_json(right)


def validate_resolutions(values: Sequence[Any]) -> tuple[float, ...]:
    """Require the exact ordered preregistered numeric tuple."""

    if isinstance(values, (str, bytes, bytearray)):
        raise RunContractError("resolutions_seconds must be an array")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RunContractError(
                "routing resolutions must be numeric, not bool/string"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RunContractError("routing resolutions must be finite")
        result.append(numeric)
    resolved = tuple(result)
    if resolved != CANONICAL_RESOLUTIONS:
        raise RunContractError(
            f"routing resolutions differ from preregistration: {resolved} != "
            f"{CANONICAL_RESOLUTIONS}"
        )
    return resolved


def resolution_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunContractError(
            "resolution must be a numeric preregistered value"
        )
    numeric = float(value)
    try:
        return RESOLUTION_KEYS[numeric]
    except KeyError as exc:
        raise RunContractError(
            f"resolution is outside preregistration: {numeric}"
        ) from exc


def verify_disjoint_manifest_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[tuple[str, StableFile, str], ...]:
    """Verify exact source-manifest groups and reject hard-link aliases."""

    if not isinstance(groups, Mapping) or set(groups) != set(
        SOURCE_MANIFEST_GROUPS
    ):
        raise RunContractError(
            "source manifest groups must contain exactly voiced and no_vocal"
        )
    result: list[tuple[str, StableFile, str]] = []
    seen_physical: dict[tuple[int, int], tuple[str, str]] = {}
    for group in SOURCE_MANIFEST_GROUPS:
        records = groups[group]
        if not isinstance(records, Sequence) or isinstance(
            records, (str, bytes, bytearray)
        ) or not records:
            raise RunContractError(
                f"source manifest group {group!r} must be a non-empty array"
            )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "sha256",
            }:
                raise RunContractError(
                    f"{group} source record {index} must contain exactly "
                    "path and sha256"
                )
            path_text = str(record.get("path") or "")
            if not path_text:
                raise RunContractError(
                    f"{group} source record {index} has no path"
                )
            path = Path(path_text)
            payload, file_record = stable_file(path)
            expected = _sha_identity(
                record.get("sha256"), f"{group} source SHA"
            )
            if file_record.container_sha256 != expected:
                raise RunContractError(
                    f"{group} source manifest changed: {path}"
                )
            physical = (file_record.device, file_record.inode)
            previous = seen_physical.get(physical)
            if previous is not None:
                raise RunContractError(
                    "source manifests reuse one physical file: "
                    f"{previous} and {(group, file_record.path)}"
                )
            seen_physical[physical] = (group, file_record.path)
            if len(payload) == 0:
                raise RunContractError(f"source manifest is empty: {path}")
            result.append((group, file_record, expected))
    return tuple(result)


def _require_stereo_wave(handle, path: Path, info: sf.SoundFile) -> None:
    if info.format not in {"WAV", "WAVEX"}:
        raise RunContractError(f"truth audio is not a WAV container: {path}")
    handle.seek(0)
    header = handle.read(12)
    if len(header) != 12 or header[:4] not in {b"RIFF", b"RF64"} or header[8:] != b"WAVE":
        raise RunContractError(f"truth audio has an invalid WAV header: {path}")
    extensible_mask = None
    while True:
        chunk_header = handle.read(8)
        if len(chunk_header) != 8:
            break
        chunk, size = struct.unpack("<4sI", chunk_header)
        payload = handle.read(size)
        if len(payload) != size:
            raise RunContractError(f"truncated WAV chunk in truth audio: {path}")
        if size & 1:
            handle.read(1)
        if chunk == b"fmt ":
            if len(payload) < 16:
                raise RunContractError(f"invalid WAV fmt chunk: {path}")
            format_tag, channels = struct.unpack_from("<HH", payload)
            if channels != 2:
                raise RunContractError(f"truth WAV is not two-channel: {path}")
            if format_tag == 0xFFFE:
                if len(payload) < 40:
                    raise RunContractError(
                        f"truncated WAVE_FORMAT_EXTENSIBLE chunk: {path}"
                    )
                extensible_mask = struct.unpack_from("<I", payload, 20)[0]
            break
    if extensible_mask is not None and extensible_mask != 0x3:
        raise RunContractError(
            f"truth WAV channel mask is not FL/FR stereo: {path}: "
            f"0x{extensible_mask:x}"
        )


def _read_audio_record(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any], tuple[int, int]]:
    """Hash/decode one exact FLOAT file through one stable descriptor."""

    try:
        if path.is_symlink():
            raise RunContractError(f"refusing symlinked truth audio: {path}")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RunContractError(
                    f"truth audio is not a regular file: {path}"
                )
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            handle.seek(0)
            info = sf.info(handle)
            if (
                info.samplerate != 44_100
                or info.channels != 2
                or info.subtype != "FLOAT"
            ):
                raise RunContractError(
                    f"expected 44.1-kHz stereo FLOAT truth: {path}: {info}"
                )
            _require_stereo_wave(handle, path, info)
            handle.seek(0)
            audio, sample_rate = sf.read(
                handle, dtype="float32", always_2d=True
            )
            after = os.fstat(handle.fileno())
        current = path.stat()
    except RunContractError:
        raise
    except (OSError, RuntimeError) as exc:
        raise RunContractError(
            f"cannot read truth audio {path}: {exc}"
        ) from exc
    if (
        _signature(before) != _signature(after)
        or _signature(before) != _signature(current)
    ):
        raise RunContractError(
            f"truth audio changed or path was replaced: {path}"
        )
    if audio.shape != (info.frames, 2) or not np.all(np.isfinite(audio)):
        raise RunContractError(f"truth audio is invalid: {path}")
    record = {
        "path": str(path.resolve(strict=True)),
        "container_sha256": "sha256:" + digest.hexdigest(),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            audio, int(sample_rate), ["FL", "FR"], len(audio)
        ),
        "frames": len(audio),
        "sample_rate_hz": int(sample_rate),
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
    }
    return audio, record, (int(before.st_dev), int(before.st_ino))


def build_truth_manifest(
    truth_root: Path,
    works: Sequence[str],
    *,
    identity_tolerance: float = 2e-5,
) -> dict[str, Any]:
    """Freeze every exact M/A/V artifact and reprove ``M=A+V``."""

    if (
        not math.isfinite(float(identity_tolerance))
        or identity_tolerance < 0
    ):
        raise RunContractError(
            "truth identity tolerance must be finite/non-negative"
        )
    work_ids = tuple(str(work) for work in works)
    if (
        not work_ids
        or any(not work for work in work_ids)
        or len(set(work_ids)) != len(work_ids)
    ):
        raise RunContractError("truth work IDs must be unique and non-empty")
    records: dict[str, Any] = {}
    for work in work_ids:
        audio: dict[str, np.ndarray] = {}
        roles: dict[str, Any] = {}
        physical: set[tuple[int, int]] = set()
        for role, filename in TRUTH_FILENAMES.items():
            value, record, file_identity = _read_audio_record(
                truth_root / work / filename
            )
            if file_identity in physical:
                raise RunContractError(
                    f"truth roles reuse one physical file: {work}/{role}"
                )
            physical.add(file_identity)
            audio[role] = value
            record["path"] = str(Path(work) / filename)
            roles[role] = record
        grids = {
            (
                record["frames"],
                record["sample_rate_hz"],
                tuple(record["channels"]),
                record["subtype"],
            )
            for record in roles.values()
        }
        if len(grids) != 1:
            raise RunContractError(
                f"truth grids differ for work {work}: {grids}"
            )
        residual = (
            audio["mixture"].astype(np.float64)
            - audio["accompaniment"].astype(np.float64)
            - audio["vocal"].astype(np.float64)
        )
        maximum = float(np.max(np.abs(residual)))
        if maximum > float(identity_tolerance):
            raise RunContractError(
                f"truth identity failed for {work}: {maximum} > "
                f"{identity_tolerance}"
            )
        records[work] = {
            "roles": roles,
            "mixture_identity_max_abs": maximum,
        }
    return {
        "schema": TRUTH_MANIFEST_SCHEMA,
        "works": list(work_ids),
        "identity_tolerance": float(identity_tolerance),
        "records": records,
    }


def _max_fact_matches(current: float, declared: Any) -> bool:
    if (
        isinstance(declared, bool)
        or not isinstance(declared, (int, float))
        or not math.isfinite(float(declared))
        or float(declared) < 0
    ):
        return False
    expected = float(declared)
    allowance = (
        8.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(current), abs(expected))
    )
    return math.isclose(
        current, expected, rel_tol=0.0, abs_tol=allowance
    )


def verify_truth_manifest(
    value: Mapping[str, Any], *, truth_root: Path
) -> str:
    """Reopen every frozen truth file and compare all declared identities."""

    if value.get("schema") != TRUTH_MANIFEST_SCHEMA:
        raise RunContractError("wrong truth-manifest schema")
    works = value.get("works")
    records = value.get("records")
    tolerance = value.get("identity_tolerance")
    if (
        not isinstance(works, list)
        or not works
        or not isinstance(records, Mapping)
    ):
        raise RunContractError("truth manifest lacks works/records")
    if (
        any(not isinstance(work, str) or not work for work in works)
        or set(records) != set(works)
        or len(set(works)) != len(works)
    ):
        raise RunContractError("truth manifest work set is incoherent")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0
    ):
        raise RunContractError("truth manifest tolerance is invalid")

    for work in works:
        work_record = records[work]
        if not isinstance(work_record, Mapping) or set(work_record) != {
            "roles",
            "mixture_identity_max_abs",
        }:
            raise RunContractError(
                f"truth manifest work record is invalid: {work}"
            )
        declared_roles = work_record["roles"]
        if (
            not isinstance(declared_roles, Mapping)
            or set(declared_roles) != set(TRUTH_FILENAMES)
        ):
            raise RunContractError(
                f"truth role set is incomplete: {work}"
            )
        audio: dict[str, np.ndarray] = {}
        physical: set[tuple[int, int]] = set()
        for role in TRUTH_FILENAMES:
            declared = declared_roles[role]
            if not isinstance(declared, Mapping):
                raise RunContractError(
                    f"truth role record is invalid: {work}/{role}"
                )
            path_text = str(declared.get("path") or "")
            if not path_text:
                raise RunContractError(
                    f"truth role path is missing: {work}/{role}"
                )
            logical = Path(path_text)
            if logical.is_absolute() or ".." in logical.parts:
                raise RunContractError(
                    f"truth role path is not bundle-relative: {work}/{role}"
                )
            source_path = truth_root / logical
            current_audio, current, file_identity = _read_audio_record(source_path)
            current["path"] = path_text
            if file_identity in physical:
                raise RunContractError(
                    f"truth roles reuse one physical file: {work}/{role}"
                )
            physical.add(file_identity)
            if current != dict(declared):
                raise RunContractError(
                    f"truth artifact changed: {work}/{role}"
                )
            audio[role] = current_audio

        residual = (
            audio["mixture"].astype(np.float64)
            - audio["accompaniment"].astype(np.float64)
            - audio["vocal"].astype(np.float64)
        )
        maximum = float(np.max(np.abs(residual)))
        if maximum > float(tolerance):
            raise RunContractError(
                f"truth identity no longer holds: {work}"
            )
        if not _max_fact_matches(
            maximum, work_record["mixture_identity_max_abs"]
        ):
            raise RunContractError(
                f"truth identity fact changed: {work}"
            )
    return semantic_sha256(value)


def load_run_input_anchor(
    path: Path,
    *,
    expected_code_commit: str,
    expected_works: Sequence[str],
    expected_run_config: Mapping[str, Any],
    expected_truth_manifest_sha256: str,
    expected_binding_policy_sha256: str = CANONICAL_POLICY_SHA256,
) -> RunInputAnchor:
    """Verify and bind a record that must exist before routing starts."""

    payload, file_record = stable_file(path)
    value = _json_object(payload, "run inputs")
    if value.get("schema") != RUN_INPUT_SCHEMA:
        raise RunContractError("wrong run-input schema")
    if value.get("code_commit") != expected_code_commit:
        raise RunContractError(
            "run-input code commit differs from execution"
        )
    works = tuple(value.get("works") or ())
    if works != tuple(expected_works):
        raise RunContractError(
            "run-input work order differs from execution"
        )
    run_config = value.get("run_config")
    if not isinstance(run_config, Mapping) or not _same_json(
        run_config, expected_run_config
    ):
        raise RunContractError(
            "run-input configuration differs from execution"
        )
    resolutions = validate_resolutions(
        run_config.get("resolutions_seconds") or ()
    )
    verify_disjoint_manifest_groups(
        value.get("source_manifest_groups") or {}
    )

    expected_policy = _sha_identity(
        expected_binding_policy_sha256,
        "expected binding policy SHA",
    )
    declared_policy = _sha_identity(
        value.get("binding_policy_sha256"),
        "run-input binding policy SHA",
    )
    if declared_policy != expected_policy:
        raise RunContractError(
            "run-input binding-policy SHA differs from execution"
        )

    truth_record = value.get("truth_manifest")
    if not isinstance(truth_record, Mapping) or set(truth_record) != {
        "path",
        "sha256",
    }:
        raise RunContractError(
            "run inputs lack the exact truth-manifest record"
        )
    expected_truth_sha = _sha_identity(
        expected_truth_manifest_sha256,
        "expected truth manifest SHA",
    )
    declared_truth_sha = _sha_identity(
        truth_record.get("sha256"),
        "run-input truth manifest SHA",
    )
    if declared_truth_sha != expected_truth_sha:
        raise RunContractError(
            "run-input truth-manifest SHA differs from execution"
        )
    truth_path_text = str(truth_record.get("path") or "")
    if not truth_path_text:
        raise RunContractError("run-input truth manifest has no path")
    truth_payload, truth_file = stable_file(Path(truth_path_text))
    if truth_file.container_sha256 != declared_truth_sha:
        raise RunContractError("truth-manifest file bytes changed")
    truth_value = _json_object(truth_payload, "truth manifest")
    if tuple(truth_value.get("works") or ()) != works:
        raise RunContractError(
            "truth-manifest work order differs from run inputs"
        )
    truth_root_text = str(value.get("truth_root") or "")
    if not truth_root_text:
        raise RunContractError("run inputs lack a runtime truth_root")
    verify_truth_manifest(truth_value, truth_root=Path(truth_root_text))

    return RunInputAnchor(
        schema=RUN_ANCHOR_SCHEMA,
        path=file_record.path,
        sha256=file_record.container_sha256,
        code_commit=expected_code_commit,
        works=works,
        resolutions_seconds=resolutions,
        truth_manifest_path=truth_file.path,
        truth_manifest_sha256=declared_truth_sha,
    )
