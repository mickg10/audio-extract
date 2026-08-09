#!/usr/bin/env python3
"""Independently verify a packaged exact-classical release.

This verifier is intentionally separate from the packager. It reopens every
source, candidate, target, and packaged WAV; recomputes all container/decoded-PCM
identities; verifies exact grids; checks that release copies do not share inodes
with immutable sources; proves ``removed_vocal = mixture - primary``; and checks
the content-addressed removed-vocal lineage node.

The command fails closed and writes no audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_extract import identity


class VerificationError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _read_float(path: Path, *, frames: int | None = None,
                sample_rate: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    info = sf.info(path)
    if info.subtype != "FLOAT" or info.channels != 2:
        raise VerificationError(f"expected stereo FLOAT WAV: {path} ({info})")
    if frames is not None and info.frames != frames:
        raise VerificationError(f"frame mismatch: {path} {info.frames} != {frames}")
    if sample_rate is not None and info.samplerate != sample_rate:
        raise VerificationError(
            f"sample-rate mismatch: {path} {info.samplerate} != {sample_rate}"
        )
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.all(np.isfinite(audio)):
        raise VerificationError(f"non-finite audio: {path}")
    record = {
        "path": str(path),
        "container_sha256": _sha_file(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            audio, int(sr), ["FL", "FR"], len(audio)
        ),
        "frames": len(audio),
        "sample_rate_hz": int(sr),
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
    }
    return audio, record


def _assert_record(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in (
        "container_sha256", "artifact_pcm_sha256", "frames",
        "sample_rate_hz", "channels", "subtype",
    ):
        if actual.get(key) != expected.get(key):
            raise VerificationError(
                f"{label} {key} mismatch: {actual.get(key)} != {expected.get(key)}"
            )


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if ":" in path_value:
        _, suffix = path_value.split(":", 1)
        candidate = Path(suffix)
        if candidate.is_absolute() and candidate.exists():
            return candidate
    raise FileNotFoundError(path_value)


def _candidate_by_name(work: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in work["candidates"] if row["candidate"] == name]
    if len(rows) != 1:
        raise VerificationError(f"candidate lookup failed: {name}")
    return rows[0]


def _verify_truth(work: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    truth = work.get("truth", {})
    records = truth.get("records")
    if not isinstance(records, dict) or set(records) != {"mixture", "accompaniment", "vocal"}:
        raise VerificationError("release report lacks frozen mixture/A/V truth records")
    arrays: dict[str, np.ndarray] = {}
    actual_records: dict[str, dict] = {}
    for role in ("mixture", "accompaniment", "vocal"):
        expected = records[role]
        path = _resolve(expected["path"])
        audio, actual = _read_float(
            path, frames=int(expected["frames"]),
            sample_rate=int(expected["sample_rate_hz"]),
        )
        _assert_record(actual, expected, f"truth/{role}")
        arrays[role] = audio
        actual_records[role] = actual
    identity_error = (
        arrays["mixture"].astype(np.float64)
        - arrays["accompaniment"].astype(np.float64)
        - arrays["vocal"].astype(np.float64)
    )
    maximum = float(np.max(np.abs(identity_error)))
    if maximum > 2e-5:
        raise VerificationError(f"frozen truth no longer satisfies M=A+V: {maximum}")
    return arrays, actual_records


def _verify_source_candidate(row: dict[str, Any], truth_record: dict[str, Any]) -> tuple[Path, np.ndarray, dict]:
    path = _resolve(row["path"])
    audio, actual = _read_float(
        path, frames=int(truth_record["frames"]),
        sample_rate=int(truth_record["sample_rate_hz"]),
    )
    expected = {
        "container_sha256": row["container_sha256"],
        "artifact_pcm_sha256": row["artifact_pcm_sha256"],
        "frames": int(row["frames"]),
        "sample_rate_hz": int(row["sample_rate_hz"]),
        "channels": list(row["channels"]),
        "subtype": row["subtype"],
    }
    _assert_record(actual, expected, f"candidate/{row['candidate']}")
    if not str(row["recipe_id"]).startswith("sha256:"):
        raise VerificationError(f"noncanonical candidate recipe ID: {row['recipe_id']}")
    return path, audio, actual


def _verify_packaged_copy(
    package_path: Path, source_path: Path, expected: dict[str, Any], label: str
) -> dict[str, Any]:
    _, actual = _read_float(
        package_path, frames=int(expected["frames"]),
        sample_rate=int(expected["sample_rate_hz"]),
    )
    _assert_record(actual, expected, label)
    if os.path.samefile(package_path, source_path):
        raise VerificationError(f"{label} shares an inode with immutable source")
    return actual


def _verify_removed_lineage(
    work_dir: Path, work: dict[str, Any], manifest: dict[str, Any],
    mixture: np.ndarray, primary_audio: np.ndarray,
    mixture_record: dict[str, Any], primary: dict[str, Any],
) -> dict[str, Any]:
    packaged = manifest["packaged_artifacts"]["removed_vocal_primary"]
    path = work_dir / packaged["path"]
    removed, actual = _read_float(
        path, frames=int(mixture_record["frames"]),
        sample_rate=int(mixture_record["sample_rate_hz"]),
    )
    _assert_record(actual, packaged, "removed_vocal_primary")
    expected_audio = mixture.astype(np.float32) - primary_audio.astype(np.float32)
    if not np.array_equal(removed, expected_audio):
        maximum = float(np.max(np.abs(removed.astype(np.float64) - expected_audio.astype(np.float64))))
        raise VerificationError(f"removed vocal is not exact M-primary: {maximum}")

    lineage_root = work_dir / "lineage" / "removed-vocal.primary"
    recipe_path = lineage_root / "recipe.json"
    execution_path = lineage_root / "execution.json"
    complete_path = lineage_root / "COMPLETE"
    for required in (recipe_path, execution_path, complete_path):
        if not required.exists():
            raise VerificationError(f"missing removed-vocal lineage artifact: {required}")
    recipe = json.loads(recipe_path.read_text())
    execution = json.loads(execution_path.read_text())
    rid = identity.recipe_id(recipe)
    if rid != manifest.get("removed_vocal_recipe_id"):
        raise VerificationError(
            f"removed-vocal recipe ID mismatch: {rid} != {manifest.get('removed_vocal_recipe_id')}"
        )
    if recipe["operation"].get("type") != "mixture_minus_source":
        raise VerificationError("removed-vocal recipe operation is not mixture_minus_source")
    if recipe["input_pcm"].get("input_pcm_sha256") != mixture_record["artifact_pcm_sha256"]:
        raise VerificationError("removed-vocal recipe mixture parent mismatch")
    members = recipe.get("model", {}).get("members")
    if members != [primary["recipe_id"]]:
        raise VerificationError("removed-vocal recipe primary recipe parent mismatch")
    effective = recipe.get("effective_config", {})
    if effective.get("parent_artifact_pcm_sha256") != primary["artifact_pcm_sha256"]:
        raise VerificationError("removed-vocal recipe primary PCM parent mismatch")
    if effective.get("alignment") != "source-grid-exact":
        raise VerificationError("removed-vocal recipe does not declare exact grid")
    if packaged.get("recipe_id") and packaged["recipe_id"] != rid:
        raise VerificationError("packaged removed-vocal record recipe mismatch")
    parents = execution.get("parents")
    if not isinstance(parents, list) or primary["recipe_id"] not in json.dumps(parents):
        raise VerificationError("removed-vocal execution lacks explicit primary parent")
    return {
        "recipe_id": rid,
        "recipe_json_sha256": _sha_file(recipe_path),
        "execution_json_sha256": _sha_file(execution_path),
        "artifact": actual,
    }


def verify_work(report_work: dict[str, Any], package_root: Path) -> dict[str, Any]:
    work_id = report_work["work_id"]
    work_dir = package_root / work_id
    manifest_path = work_dir / "manifest.json"
    complete = work_dir / "COMPLETE"
    if not manifest_path.exists() or not complete.is_file():
        raise VerificationError(f"incomplete package for {work_id}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("work_id") != work_id:
        raise VerificationError("manifest work ID mismatch")
    if manifest.get("status") not in {"final", "needs_human_ab"}:
        raise VerificationError(f"noncanonical terminal status: {manifest.get('status')}")
    if manifest.get("population_risk_claim") is not None:
        raise VerificationError("exact engineering release must not make population-risk claim")
    if manifest.get("primary", {}).get("candidate") != report_work["primary"]:
        raise VerificationError("manifest/report primary mismatch")
    if manifest.get("alternate", {}).get("candidate") != report_work["alternate"]:
        raise VerificationError("manifest/report alternate mismatch")

    truth_arrays, truth_records = _verify_truth(report_work)
    primary = _candidate_by_name(report_work, report_work["primary"])
    alternate = _candidate_by_name(report_work, report_work["alternate"])
    primary_source, primary_audio, primary_record = _verify_source_candidate(
        primary, truth_records["mixture"]
    )
    alternate_source, _, alternate_record = _verify_source_candidate(
        alternate, truth_records["mixture"]
    )
    if primary_record["artifact_pcm_sha256"] == alternate_record["artifact_pcm_sha256"]:
        raise VerificationError("primary and alternate are byte-identical aliases")

    packaged = manifest.get("packaged_artifacts", {})
    required = {
        "accompaniment_primary", "accompaniment_alternate",
        "removed_vocal_primary", "exact_orchestra_target", "exact_voice_target",
    }
    if set(packaged) != required:
        raise VerificationError(
            f"package inventory mismatch: {set(packaged)} != {required}"
        )
    actual_packaged = {
        "accompaniment_primary": _verify_packaged_copy(
            work_dir / packaged["accompaniment_primary"]["path"],
            primary_source, primary_record, "accompaniment_primary",
        ),
        "accompaniment_alternate": _verify_packaged_copy(
            work_dir / packaged["accompaniment_alternate"]["path"],
            alternate_source, alternate_record, "accompaniment_alternate",
        ),
        "exact_orchestra_target": _verify_packaged_copy(
            work_dir / packaged["exact_orchestra_target"]["path"],
            _resolve(truth_records["accompaniment"]["path"]),
            truth_records["accompaniment"], "exact_orchestra_target",
        ),
        "exact_voice_target": _verify_packaged_copy(
            work_dir / packaged["exact_voice_target"]["path"],
            _resolve(truth_records["vocal"]["path"]),
            truth_records["vocal"], "exact_voice_target",
        ),
    }
    removed = _verify_removed_lineage(
        work_dir, report_work, manifest,
        truth_arrays["mixture"], primary_audio,
        truth_records["mixture"], primary,
    )
    actual_packaged["removed_vocal_primary"] = removed["artifact"]
    return {
        "work_id": work_id,
        "status": manifest["status"],
        "release_scope": manifest.get("release_scope"),
        "primary": primary["candidate"],
        "alternate": alternate["candidate"],
        "manifest_sha256": _sha_file(manifest_path),
        "report_json_sha256": _sha_file(work_dir / "report.json"),
        "report_md_sha256": _sha_file(work_dir / "report.md"),
        "removed_vocal_lineage": removed,
        "packaged_artifacts": actual_packaged,
    }


def run(report_path: Path, package_root: Path, output_path: Path | None) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    if report.get("status") not in {"final", "needs_human_ab"}:
        raise VerificationError(f"noncanonical release report status: {report.get('status')}")
    works = report.get("works")
    if not isinstance(works, list) or not works:
        raise VerificationError("release report has no works")
    results = [verify_work(work, package_root) for work in works]
    overall = "final" if all(item["status"] == "final" for item in results) else "needs_human_ab"
    if overall != report["status"]:
        raise VerificationError(
            f"top-level/per-work status mismatch: {report['status']} != {overall}"
        )
    verification = {
        "schema": "audio-extract/classical-release-verification/v1",
        "status": "verified",
        "release_status": overall,
        "release_report": str(report_path),
        "release_report_sha256": _sha_file(report_path),
        "package_root": str(package_root),
        "works": results,
    }
    payload = json.dumps(verification, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.read_text() != payload:
            raise VerificationError(f"refusing to rewrite differing verification: {output_path}")
        if not output_path.exists():
            output_path.write_text(payload)
    return verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("verify-classical-release")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.report, args.package_root, args.output)
    print(json.dumps({
        "status": result["status"],
        "release_status": result["release_status"],
        "works": len(result["works"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
