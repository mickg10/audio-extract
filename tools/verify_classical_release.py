#!/usr/bin/env python3
"""Read-only v2 verifier for exact-classical release packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
from audio_extract import identity


class VerificationError(RuntimeError):
    pass


_KEYS = (
    "container_sha256", "artifact_pcm_sha256", "frames",
    "sample_rate_hz", "channels", "subtype",
)


def _sig(st: os.stat_result) -> tuple[int, int, int, int]:
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _stable_bytes(path: Path) -> tuple[bytes, str]:
    try:
        if path.is_symlink():
            raise VerificationError(f"refusing symlinked metadata file: {path}")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except VerificationError:
        raise
    except OSError as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if _sig(before) != _sig(after) or _sig(before) != _sig(current):
        raise VerificationError(f"file changed or path was replaced: {path}")
    return payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def _text(path: Path) -> tuple[str, str]:
    payload, digest = _stable_bytes(path)
    try:
        return payload.decode("utf-8"), digest
    except UnicodeDecodeError as exc:
        raise VerificationError(f"metadata is not UTF-8: {path}") from exc


def _json(path: Path, label: str) -> tuple[Any, str]:
    text, digest = _text(path)
    try:
        return json.loads(text), digest
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON in {label}: {path}: {exc}") from exc


def _empty(path: Path, label: str) -> None:
    payload, _ = _stable_bytes(path)
    if payload:
        raise VerificationError(f"{label} marker is not empty: {path}")


def _audio(path: Path, *, frames: int | None = None,
           sample_rate: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        if path.is_symlink():
            raise VerificationError(f"refusing symlinked audio: {path}")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            handle.seek(0)
            info = sf.info(handle)
            if info.subtype != "FLOAT" or info.channels != 2:
                raise VerificationError(f"expected stereo FLOAT WAV: {path} ({info})")
            if frames is not None and info.frames != frames:
                raise VerificationError(f"frame mismatch: {path} {info.frames} != {frames}")
            if sample_rate is not None and info.samplerate != sample_rate:
                raise VerificationError(
                    f"sample-rate mismatch: {path} {info.samplerate} != {sample_rate}"
                )
            handle.seek(0)
            value, sr = sf.read(handle, dtype="float32", always_2d=True)
            after = os.fstat(handle.fileno())
        current = path.stat()
    except VerificationError:
        raise
    except (OSError, RuntimeError) as exc:
        raise VerificationError(f"cannot read audio {path}: {exc}") from exc
    if _sig(before) != _sig(after) or _sig(before) != _sig(current):
        raise VerificationError(f"audio changed or path was replaced: {path}")
    if not np.all(np.isfinite(value)):
        raise VerificationError(f"non-finite audio: {path}")
    return value, {
        "path": str(path),
        "container_sha256": "sha256:" + digest.hexdigest(),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            value, int(sr), ["FL", "FR"], len(value)
        ),
        "frames": len(value), "sample_rate_hz": int(sr),
        "channels": ["FL", "FR"], "subtype": "FLOAT",
    }


def _record(actual: dict, expected: dict, label: str) -> None:
    if not isinstance(expected, dict):
        raise VerificationError(f"{label} record is not an object")
    for key in _KEYS:
        if actual.get(key) != expected.get(key):
            raise VerificationError(
                f"{label} {key} mismatch: {actual.get(key)} != {expected.get(key)}"
            )


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise VerificationError(f"{label} mismatch")


def _source(value: str) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    if ":" in str(value):
        candidate = Path(str(value).split(":", 1)[1])
        if candidate.is_absolute() and candidate.exists():
            return candidate
    raise VerificationError(f"cannot resolve source path: {value}")


def _work(root: Path, work_id: str) -> Path:
    rel = Path(str(work_id))
    if not str(work_id) or rel.is_absolute() or len(rel.parts) != 1 or str(work_id) in {".", ".."}:
        raise VerificationError(f"unsafe work ID: {work_id!r}")
    try:
        base = root.resolve(strict=True)
        candidate = base / rel
        if candidate.is_symlink():
            raise VerificationError(f"work directory is a symlink: {candidate}")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base)
    except VerificationError:
        raise
    except (OSError, ValueError) as exc:
        raise VerificationError(f"work package escapes or is unavailable: {candidate}") from exc
    if not resolved.is_dir():
        raise VerificationError(f"work package is not a directory: {resolved}")
    return resolved


def _inside(work_dir: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} path is missing")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise VerificationError(f"{label} path escapes work directory: {value!r}")
    base = work_dir.resolve(strict=True)
    cursor = base
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise VerificationError(f"{label} path contains a symlink: {value!r}")
    try:
        path = (base / rel).resolve(strict=True)
        path.relative_to(base)
    except (OSError, ValueError) as exc:
        raise VerificationError(f"{label} path escapes or is unavailable: {value!r}") from exc
    if not path.is_file():
        raise VerificationError(f"{label} is not a file: {path}")
    return path


def _artifact(work_dir: Path, row: dict, label: str) -> Path:
    if not isinstance(row, dict):
        raise VerificationError(f"{label} manifest record is not an object")
    return _inside(work_dir, row.get("path"), label)


def _candidate(work: dict, name: str) -> dict:
    rows = [row for row in work.get("candidates", []) if row.get("candidate") == name]
    if len(rows) != 1:
        raise VerificationError(f"candidate lookup failed: {name}")
    return rows[0]


def _truth(work: dict) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    records = work.get("truth", {}).get("records")
    if not isinstance(records, dict) or set(records) != {"mixture", "accompaniment", "vocal"}:
        raise VerificationError("release report lacks frozen mixture/A/V truth records")
    arrays, actual = {}, {}
    for role in ("mixture", "accompaniment", "vocal"):
        expected = records[role]
        arrays[role], actual[role] = _audio(
            _source(expected["path"]), frames=int(expected["frames"]),
            sample_rate=int(expected["sample_rate_hz"]),
        )
        _record(actual[role], expected, f"truth/{role}")
    grids = {
        (row["frames"], row["sample_rate_hz"], tuple(row["channels"]), row["subtype"])
        for row in actual.values()
    }
    if len(grids) != 1:
        raise VerificationError(f"truth source grids differ: {grids}")
    delta = (arrays["mixture"].astype(np.float64)
             - arrays["accompaniment"].astype(np.float64)
             - arrays["vocal"].astype(np.float64))
    maximum = float(np.max(np.abs(delta)))
    if maximum > 2e-5:
        raise VerificationError(f"frozen truth no longer satisfies M=A+V: {maximum}")
    return arrays, actual


def _source_candidate(row: dict, truth: dict) -> tuple[Path, np.ndarray, dict]:
    path = _source(row["path"])
    value, actual = _audio(
        path, frames=int(truth["frames"]), sample_rate=int(truth["sample_rate_hz"])
    )
    expected = {key: row.get(key) for key in _KEYS}
    _record(actual, expected, f"candidate/{row.get('candidate')}")
    rid = str(row.get("recipe_id", ""))
    if not rid.startswith("sha256:") or len(rid) != 71:
        raise VerificationError(f"noncanonical candidate recipe ID: {rid}")
    return path, value, actual


def _copy(path: Path, source: Path, source_record: dict,
          manifest_record: dict, label: str) -> dict:
    _, actual = _audio(
        path, frames=int(source_record["frames"]),
        sample_rate=int(source_record["sample_rate_hz"]),
    )
    _record(actual, source_record, f"{label}/source")
    _record(actual, manifest_record, f"{label}/manifest")
    try:
        if os.path.samefile(path, source):
            raise VerificationError(f"{label} shares an inode with immutable source")
    except OSError as exc:
        raise VerificationError(f"cannot compare package/source inode for {label}: {exc}") from exc
    return actual


def _reports(work_dir: Path, work: dict) -> tuple[str, str]:
    report_path = _inside(work_dir, "report.json", "packaged report.json")
    markdown_path = _inside(work_dir, "report.md", "packaged report.md")
    packaged, report_digest = _json(report_path, "packaged report.json")
    _same(packaged, work, "packaged report.json")
    markdown, markdown_digest = _text(markdown_path)
    required = (
        "# Exact classical baseline release report", f"### {work['work_id']}",
        f"Terminal status: `{work['status']}`", f"Primary: `{work['primary']}`",
        f"Alternate: `{work['alternate']}`",
    )
    missing = [item for item in required if item not in markdown]
    if missing:
        raise VerificationError(f"packaged report.md lacks required facts: {missing}")
    return report_digest, markdown_digest


def _removed(work_dir: Path, manifest: dict, mixture: np.ndarray,
             primary_audio: np.ndarray, mixture_record: dict,
             primary: dict) -> dict:
    packaged = manifest["packaged_artifacts"]["removed_vocal_primary"]
    output_path = _artifact(work_dir, packaged, "removed_vocal_primary")
    output, actual = _audio(
        output_path, frames=int(mixture_record["frames"]),
        sample_rate=int(mixture_record["sample_rate_hz"]),
    )
    _record(actual, packaged, "removed_vocal_primary/manifest")
    expected = mixture.astype(np.float32) - primary_audio.astype(np.float32)
    if not np.array_equal(output, expected):
        maximum = float(np.max(np.abs(output.astype(np.float64)-expected.astype(np.float64))))
        raise VerificationError(f"removed vocal is not exact M-primary: {maximum}")

    recipe_path = _inside(work_dir, "lineage/removed-vocal.primary/recipe.json", "removed-vocal recipe")
    execution_path = _inside(work_dir, "lineage/removed-vocal.primary/execution.json", "removed-vocal execution")
    pcm_path = _inside(work_dir, "lineage/removed-vocal.primary/output.pcm.sha256", "removed-vocal PCM marker")
    complete_path = _inside(work_dir, "lineage/removed-vocal.primary/COMPLETE", "removed-vocal COMPLETE marker")
    _empty(complete_path, "removed-vocal lineage COMPLETE")
    recipe, recipe_digest = _json(recipe_path, "removed-vocal recipe")
    execution, execution_digest = _json(execution_path, "removed-vocal execution")
    rid = identity.recipe_id(recipe)
    if rid != manifest.get("removed_vocal_recipe_id"):
        raise VerificationError(f"removed-vocal recipe ID mismatch: {rid} != {manifest.get('removed_vocal_recipe_id')}")
    _same(manifest.get("removed_vocal_recipe"), recipe, "manifest/removed_vocal_recipe")
    if recipe.get("operation") != {
        "type": "mixture_minus_source", "target": "vocals",
        "construction": "mixture_minus_source",
    }:
        raise VerificationError("removed-vocal recipe operation contract mismatch")
    _same(recipe.get("input_pcm"), {
        "sha256": mixture_record["artifact_pcm_sha256"],
        "sample_rate_hz": int(mixture_record["sample_rate_hz"]),
        "channel_layout": list(mixture_record["channels"]),
        "frames": int(mixture_record["frames"]),
        "sample_format": "float32-le-interleaved",
    }, "removed-vocal mixture parent")
    parent = primary["recipe_id"]
    model = recipe.get("model", {})
    if model.get("members") != [parent]:
        raise VerificationError("removed-vocal recipe primary recipe parent mismatch")
    if model.get("weights_sha256") != hashlib.sha256(parent.encode()).hexdigest():
        raise VerificationError("removed-vocal deterministic parent hash mismatch")
    if (model.get("model_id"), model.get("adapter"), model.get("adapter_revision")) != (
        "deterministic-complement-residual",
        "audio_extract.classical_release.package_work", "exact-grid-complement/v1",
    ):
        raise VerificationError("removed-vocal deterministic adapter identity mismatch")
    _same(recipe.get("effective_config"), {
        "parent_recipe_ids": [parent],
        "parent_artifact_pcm_sha256": primary["artifact_pcm_sha256"],
        "alignment": "source-grid-exact",
    }, "removed-vocal effective configuration")
    if recipe.get("software", {}).get("audio_extract_commit") != manifest.get("code_commit"):
        raise VerificationError("removed-vocal recipe/manifest code commit mismatch")
    if packaged.get("recipe_id") not in {None, rid}:
        raise VerificationError("packaged removed-vocal record recipe mismatch")
    if execution.get("parents") != [parent]:
        raise VerificationError("removed-vocal execution parents are not exact")
    _same(execution.get("output"), packaged, "removed-vocal execution output")
    pcm, pcm_digest = _text(pcm_path)
    if pcm.strip() != actual["artifact_pcm_sha256"]:
        raise VerificationError("removed-vocal lineage PCM marker mismatch")
    return {
        "recipe_id": rid, "recipe_json_sha256": recipe_digest,
        "execution_json_sha256": execution_digest,
        "output_pcm_marker_sha256": pcm_digest, "artifact": actual,
    }


def verify_work(report_work: dict, package_root: Path, *,
                expected_code_commit: str | None = None) -> dict[str, Any]:
    work_id = str(report_work.get("work_id", ""))
    work_dir = _work(package_root, work_id)
    manifest_path = _inside(work_dir, "manifest.json", "package manifest")
    complete_path = _inside(work_dir, "COMPLETE", "package COMPLETE marker")
    _empty(complete_path, f"package COMPLETE/{work_id}")
    manifest, manifest_digest = _json(manifest_path, "package manifest")
    if manifest.get("schema") != "audio-extract/classical-exact-release/v1":
        raise VerificationError("manifest schema mismatch")
    if manifest.get("work_id") != work_id:
        raise VerificationError("manifest work ID mismatch")
    if manifest.get("status") not in {"final", "needs_human_ab"}:
        raise VerificationError(f"noncanonical terminal status: {manifest.get('status')}")
    for key in ("status", "release_scope", "policy_id"):
        if manifest.get(key) != report_work.get(key):
            raise VerificationError(f"manifest/report {key} mismatch")
    if manifest.get("population_risk_claim") is not None:
        raise VerificationError("exact engineering release must not make population-risk claim")
    if expected_code_commit is not None and manifest.get("code_commit") != expected_code_commit:
        raise VerificationError("manifest/release-report code commit mismatch")
    report_json_digest, report_md_digest = _reports(work_dir, report_work)

    truth_audio, truth_records = _truth(report_work)
    primary = _candidate(report_work, report_work["primary"])
    alternate = _candidate(report_work, report_work["alternate"])
    _same(manifest.get("primary"), primary, "manifest/report primary candidate")
    _same(manifest.get("alternate"), alternate, "manifest/report alternate candidate")
    _same(manifest.get("pareto_front"), report_work.get("pareto_front"), "manifest/report Pareto front")
    _same(manifest.get("artifact_aliases"), report_work.get("artifact_aliases"), "manifest/report artifact aliases")
    _same(manifest.get("truth_source_paths"), report_work.get("truth"), "manifest/report truth paths")
    primary_source, primary_audio, primary_record = _source_candidate(primary, truth_records["mixture"])
    alternate_source, _, alternate_record = _source_candidate(alternate, truth_records["mixture"])
    if primary_record["artifact_pcm_sha256"] == alternate_record["artifact_pcm_sha256"]:
        raise VerificationError("primary and alternate are byte-identical aliases")

    packaged = manifest.get("packaged_artifacts", {})
    required = {"accompaniment_primary", "accompaniment_alternate", "removed_vocal_primary", "exact_orchestra_target", "exact_voice_target"}
    if not isinstance(packaged, dict) or set(packaged) != required:
        raise VerificationError(f"package inventory mismatch: {set(packaged) if isinstance(packaged, dict) else packaged} != {required}")
    actual = {
        "accompaniment_primary": _copy(
            _artifact(work_dir, packaged["accompaniment_primary"], "accompaniment_primary"),
            primary_source, primary_record, packaged["accompaniment_primary"], "accompaniment_primary"),
        "accompaniment_alternate": _copy(
            _artifact(work_dir, packaged["accompaniment_alternate"], "accompaniment_alternate"),
            alternate_source, alternate_record, packaged["accompaniment_alternate"], "accompaniment_alternate"),
        "exact_orchestra_target": _copy(
            _artifact(work_dir, packaged["exact_orchestra_target"], "exact_orchestra_target"),
            _source(truth_records["accompaniment"]["path"]), truth_records["accompaniment"],
            packaged["exact_orchestra_target"], "exact_orchestra_target"),
        "exact_voice_target": _copy(
            _artifact(work_dir, packaged["exact_voice_target"], "exact_voice_target"),
            _source(truth_records["vocal"]["path"]), truth_records["vocal"],
            packaged["exact_voice_target"], "exact_voice_target"),
    }
    removed = _removed(
        work_dir, manifest, truth_audio["mixture"], primary_audio,
        truth_records["mixture"], primary,
    )
    actual["removed_vocal_primary"] = removed["artifact"]
    return {
        "work_id": work_id, "status": manifest["status"],
        "release_scope": manifest.get("release_scope"),
        "primary": primary["candidate"], "alternate": alternate["candidate"],
        "manifest_sha256": manifest_digest,
        "report_json_sha256": report_json_digest,
        "report_md_sha256": report_md_digest,
        "removed_vocal_lineage": removed, "packaged_artifacts": actual,
    }


def run(report_path: Path, package_root: Path,
        output_path: Path | None) -> dict[str, Any]:
    report, report_digest = _json(report_path, "release report")
    if report.get("schema") != "audio-extract/classical-exact-release-report/v1":
        raise VerificationError("release report schema mismatch")
    if report.get("status") not in {"final", "needs_human_ab"}:
        raise VerificationError(f"noncanonical release report status: {report.get('status')}")
    if report.get("population_risk_claim") is not None:
        raise VerificationError("release report must not make a population-risk claim")
    code_commit = report.get("code_commit")
    if not isinstance(code_commit, str) or not code_commit.strip():
        raise VerificationError("release report code_commit is missing")
    works = report.get("works")
    if not isinstance(works, list) or not works:
        raise VerificationError("release report has no works")
    ids = [str(work.get("work_id", "")) for work in works]
    if len(ids) != len(set(ids)):
        raise VerificationError(f"release report has duplicate work IDs: {ids}")
    results = [
        verify_work(work, package_root, expected_code_commit=code_commit)
        for work in works
    ]
    overall = "final" if all(row["status"] == "final" for row in results) else "needs_human_ab"
    if overall != report["status"]:
        raise VerificationError(f"top-level/per-work status mismatch: {report['status']} != {overall}")
    result = {
        "schema": "audio-extract/classical-release-verification/v2",
        "status": "verified", "release_status": overall,
        "release_report": str(report_path),
        "release_report_sha256": report_digest,
        "package_root": str(package_root.resolve(strict=True)), "works": results,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.read_text() != payload:
            raise VerificationError(f"refusing to rewrite differing verification: {output_path}")
        if not output_path.exists():
            output_path.write_text(payload)
    return result


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
        "status": result["status"], "release_status": result["release_status"],
        "works": len(result["works"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
