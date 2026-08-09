"""Score and package the best currently available exact classical candidates.

This is the immediate delivery lane for the audited ``candidates-v2`` FLOAT
artifacts.  It reopens and rehashes every candidate, measures it against exact
accompaniment/featured-soloist references, applies an explicit *engineering*
screening policy, and emits one primary plus one byte-distinct alternate.

The tool makes no population-risk or arbitrary-master claim.  Its only terminal
statuses follow the repository contract:

``final``
    At least one candidate passes every declared exact-reference screen.

``needs_human_ab``
    No candidate passes, or required exact evidence is unavailable.  The tool may
    still package a clearly labelled engineering preview and alternate; it never
    calls that preview certified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import identity
from .judge_labels import local_source_coordinate_labels
from .judge_train import label_targets
from .metrics_v2 import brightness_v2, fullness_v2, stereo_v2, transient_v2

_EPS = 1e-12


class ClassicalReleaseError(RuntimeError):
    """The release inputs or immutable output contract are invalid."""


@dataclass(frozen=True)
class ScreeningPolicy:
    policy_id: str
    retained_voice_coef_p90_max: float
    retained_voice_db_p90_max: float
    event_hole_db_p90_max: float
    event_hole_db_max_max: float
    alpha_error_p90_max: float
    stereo_width_dev_db_max: float
    coherence_dev_max: float
    min_identifiable_tiles: int
    min_identifiable_fraction: float
    artifact_ratio_p90_max: float | None = None

    @classmethod
    def from_mapping(cls, value: dict) -> "ScreeningPolicy":
        policy = cls(**value)
        if not str(policy.policy_id).strip():
            raise ValueError("policy_id must be non-empty")
        positive = (
            "retained_voice_coef_p90_max", "event_hole_db_p90_max",
            "event_hole_db_max_max", "alpha_error_p90_max",
            "stereo_width_dev_db_max", "coherence_dev_max",
        )
        for name in positive:
            item = float(getattr(policy, name))
            if not math.isfinite(item) or item <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(float(policy.retained_voice_db_p90_max)):
            raise ValueError("retained_voice_db_p90_max must be finite")
        if policy.artifact_ratio_p90_max is not None:
            value = float(policy.artifact_ratio_p90_max)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("artifact_ratio_p90_max must be positive when enabled")
        if int(policy.min_identifiable_tiles) < 1:
            raise ValueError("min_identifiable_tiles must be at least one")
        fraction = float(policy.min_identifiable_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("min_identifiable_fraction must lie in (0,1]")
        return policy


@dataclass(frozen=True)
class CandidateRow:
    work_id: str
    candidate: str
    recipe_id: str
    path: str
    artifact_pcm_sha256: str
    container_sha256: str
    sr_hz: int
    frames: int
    channels: tuple[str, ...]
    subtype: str


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _resolve_host_path(value: str) -> Path:
    direct = Path(value)
    if direct.exists():
        return direct
    if ":" in value:
        _, suffix = value.split(":", 1)
        candidate = Path(suffix)
        if candidate.is_absolute() and candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def load_candidate_manifest(path: Path) -> list[CandidateRow]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row = CandidateRow(
            work_id=str(raw["work_id"]),
            candidate=str(raw["candidate"]),
            recipe_id=str(raw["recipe_id"]),
            path=str(raw["path"]),
            artifact_pcm_sha256=str(raw["artifact_pcm_sha256"]),
            container_sha256=str(raw["container_sha256"]),
            sr_hz=int(raw["sr_hz"]),
            frames=int(raw["frames"]),
            channels=tuple(raw["channels"]),
            subtype=str(raw["subtype"]),
        )
        if not row.recipe_id.startswith("sha256:"):
            raise ClassicalReleaseError(f"noncanonical recipe identity: {row.recipe_id}")
        if row.subtype != "FLOAT" or row.channels != ("FL", "FR"):
            raise ClassicalReleaseError(f"non-FLOAT/stereo analysis row: {row}")
        if row.frames <= 0 or row.sr_hz <= 0:
            raise ClassicalReleaseError(f"invalid candidate grid: {row}")
        rows.append(row)
    if not rows:
        raise ClassicalReleaseError(f"empty candidate manifest: {path}")
    keys = [(row.work_id, row.candidate) for row in rows]
    if len(keys) != len(set(keys)):
        raise ClassicalReleaseError("duplicate work/candidate rows")
    return rows


def _read_float(path: Path, *, sr: int | None = None,
                frames: int | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    if info.subtype != "FLOAT" or info.channels != 2:
        raise ClassicalReleaseError(f"expected stereo FLOAT: {path} ({info})")
    if sr is not None and info.samplerate != sr:
        raise ClassicalReleaseError(f"sample-rate mismatch: {path}")
    if frames is not None and info.frames != frames:
        raise ClassicalReleaseError(f"frame mismatch: {path}")
    audio, actual_sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.all(np.isfinite(audio)):
        raise ClassicalReleaseError(f"non-finite audio: {path}")
    return audio, int(actual_sr)


def _audio_record(path: Path, *, expected_sr: int | None = None,
                  expected_frames: int | None = None) -> dict:
    audio, sr = _read_float(path, sr=expected_sr, frames=expected_frames)
    return {
        "path": str(path),
        "container_sha256": _sha_file(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            audio, sr, ["FL", "FR"], len(audio)
        ),
        "frames": len(audio),
        "sample_rate_hz": sr,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
    }


def verify_candidate(row: CandidateRow, *, expected_sr: int,
                     expected_frames: int) -> tuple[Path, np.ndarray]:
    if row.sr_hz != expected_sr or row.frames != expected_frames:
        raise ClassicalReleaseError(
            f"truth-grid mismatch for {row.work_id}/{row.candidate}: "
            f"candidate=({row.frames},{row.sr_hz}) truth=({expected_frames},{expected_sr})"
        )
    path = _resolve_host_path(row.path)
    record = _audio_record(path, expected_sr=expected_sr, expected_frames=expected_frames)
    if record["container_sha256"] != row.container_sha256:
        raise ClassicalReleaseError(f"container hash mismatch: {row.work_id}/{row.candidate}")
    if record["artifact_pcm_sha256"] != row.artifact_pcm_sha256:
        raise ClassicalReleaseError(f"PCM hash mismatch: {row.work_id}/{row.candidate}")
    audio, _ = _read_float(path, sr=expected_sr, frames=expected_frames)
    return path, audio


def _finite_metric(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ClassicalReleaseError(f"required metric unavailable/non-finite: {name}={value!r}")
    return float(value)


def exact_metrics(candidate: np.ndarray, accompaniment: np.ndarray,
                  vocal: np.ndarray, sr: int) -> dict:
    if candidate.shape != accompaniment.shape or candidate.shape != vocal.shape:
        raise ClassicalReleaseError("candidate and exact sources must share one grid")
    labels = local_source_coordinate_labels(
        candidate, accompaniment, vocal,
        tile_frames=max(256, round(0.5 * sr)),
        hop_frames=max(128, round(0.25 * sr)),
    )
    metrics = label_targets(labels)
    available = int(metrics.get("_available_tiles", 0))
    total = int(metrics.get("_total_tiles", 0))
    if total <= 0 or not 0 <= available <= total:
        raise ClassicalReleaseError(
            f"invalid exact-label coverage: available={available}, total={total}"
        )
    metrics["identifiable_fraction"] = available / total
    required = (
        "retained_voice_db_p90", "retained_voice_coef_p90",
        "event_hole_db_p90", "event_hole_db_max", "alpha_error_p90",
        "artifact_ratio_p90",
    )
    for name in required:
        _finite_metric(metrics, name)
    if metrics["event_hole_db_max"] + 1e-9 < metrics["event_hole_db_p90"]:
        raise ClassicalReleaseError("incoherent exact hole labels: max < p90")

    error = candidate.astype(np.float64) - accompaniment.astype(np.float64)
    signal_energy = float(np.square(accompaniment.astype(np.float64)).sum())
    error_energy = float(np.square(error).sum())
    metrics["scale_dependent_sdr_db"] = float(
        10.0 * np.log10((signal_energy + _EPS) / (error_energy + _EPS))
    )
    observations = (
        fullness_v2(candidate, accompaniment, sr)
        + brightness_v2(candidate, accompaniment, sr)
        + transient_v2(candidate, accompaniment, sr)
        + stereo_v2(candidate, accompaniment)
    )
    for observation in observations:
        if not observation["available"] or observation["value"] is None:
            raise ClassicalReleaseError(
                f"required fidelity observation unavailable: {observation['metric']}"
            )
        metrics[observation["metric"]] = float(observation["value"])
    for name in (
        "scale_dependent_sdr_db", "band_deficit_db/v2", "contiguous_hole_db/v2",
        "erb_envelope_dist_db/v2", "transient_loss/v2", "transient_excess/v2",
        "stereo_width_dev_db/v2", "interchannel_coherence_dev/v2",
    ):
        _finite_metric(metrics, name)
    return metrics


def screening(metrics: dict, policy: ScreeningPolicy) -> dict:
    checks: list[dict] = []

    def upper(name: str, limit: float, *, severity: float | None = None) -> None:
        value = _finite_metric(metrics, name)
        checks.append({
            "metric": name, "value": value, "limit": float(limit),
            "comparator": "<=", "passed": value <= limit,
            "normalized_severity": float(value / limit if severity is None else severity),
        })

    upper("retained_voice_coef_p90", policy.retained_voice_coef_p90_max)
    voice_db = _finite_metric(metrics, "retained_voice_db_p90")
    checks.append({
        "metric": "retained_voice_db_p90", "value": voice_db,
        "limit": float(policy.retained_voice_db_p90_max), "comparator": "<=",
        "passed": voice_db <= policy.retained_voice_db_p90_max,
        "normalized_severity": float(
            10.0 ** ((voice_db - policy.retained_voice_db_p90_max) / 20.0)
        ),
    })
    upper("event_hole_db_p90", policy.event_hole_db_p90_max)
    upper("event_hole_db_max", policy.event_hole_db_max_max)
    upper("alpha_error_p90", policy.alpha_error_p90_max)
    upper("stereo_width_dev_db/v2", policy.stereo_width_dev_db_max)
    upper("interchannel_coherence_dev/v2", policy.coherence_dev_max)
    if policy.artifact_ratio_p90_max is not None:
        upper("artifact_ratio_p90", policy.artifact_ratio_p90_max)

    available = int(_finite_metric(metrics, "_available_tiles"))
    total = int(_finite_metric(metrics, "_total_tiles"))
    fraction = _finite_metric(metrics, "identifiable_fraction")
    checks.extend([
        {
            "metric": "_available_tiles", "value": available,
            "limit": int(policy.min_identifiable_tiles), "comparator": ">=",
            "passed": available >= policy.min_identifiable_tiles,
            "normalized_severity": float(
                policy.min_identifiable_tiles / max(available, _EPS)
            ),
        },
        {
            "metric": "identifiable_fraction", "value": fraction,
            "limit": float(policy.min_identifiable_fraction), "comparator": ">=",
            "passed": fraction >= policy.min_identifiable_fraction,
            "normalized_severity": float(
                policy.min_identifiable_fraction / max(fraction, _EPS)
            ),
            "total_tiles": total,
        },
    ])
    failures = [check for check in checks if not check["passed"]]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "critical_max": max(float(check["normalized_severity"]) for check in checks),
    }


def _dominates(left: dict, right: dict) -> bool:
    names = (
        "retained_voice_coef_p90", "retained_voice_db_p90",
        "event_hole_db_p90", "event_hole_db_max", "alpha_error_p90",
        "artifact_ratio_p90", "stereo_width_dev_db/v2",
        "interchannel_coherence_dev/v2",
    )
    lv = [_finite_metric(left, name) for name in names]
    rv = [_finite_metric(right, name) for name in names]
    return all(a <= b for a, b in zip(lv, rv)) and any(a < b for a, b in zip(lv, rv))


def pareto_front(rows: list[dict]) -> list[str]:
    return [
        row["candidate"] for row in rows
        if not any(
            other is not row and _dominates(other["metrics"], row["metrics"])
            for other in rows
        )
    ]


def rank_rows(rows: list[dict]) -> list[dict]:
    """Feasibility first, then worst normalized critical risk and fidelity."""
    return sorted(
        rows,
        key=lambda row: (
            not row["screening"]["passed"],
            row["screening"]["critical_max"],
            _finite_metric(row["metrics"], "artifact_ratio_p90"),
            -_finite_metric(row["metrics"], "scale_dependent_sdr_db"),
            _finite_metric(row["metrics"], "erb_envelope_dist_db/v2"),
            row["candidate"],
        ),
    )


def _unique_artifacts(scored: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    by_pcm: dict[str, list[dict]] = {}
    for row in scored:
        by_pcm.setdefault(row["artifact_pcm_sha256"], []).append(row)
    unique = []
    aliases: dict[str, list[str]] = {}
    for rows in by_pcm.values():
        ranked = rank_rows(rows)
        representative = ranked[0]
        unique.append(representative)
        aliases[representative["candidate"]] = sorted(
            row["candidate"] for row in rows if row is not representative
        )
    return unique, aliases


def score_work(rows: list[CandidateRow], truth_root: Path,
               policy: ScreeningPolicy, work_id: str) -> dict:
    selected = [row for row in rows if row.work_id == work_id]
    if len(selected) < 2:
        raise ClassicalReleaseError(f"{work_id}: fewer than two candidates")
    truth = truth_root / work_id
    mixture, sr = _read_float(truth / "mix_with_voice.wav")
    accompaniment, _ = _read_float(
        truth / "orchestra_only.wav", sr=sr, frames=len(mixture)
    )
    vocal, _ = _read_float(
        truth / "voice_ref.wav", sr=sr, frames=len(mixture)
    )
    identity_error = (
        mixture.astype(np.float64)
        - accompaniment.astype(np.float64)
        - vocal.astype(np.float64)
    )
    if float(np.max(np.abs(identity_error))) > 2e-5:
        raise ClassicalReleaseError(f"{work_id}: M=A+V identity failed")
    scored = []
    for row in selected:
        path, audio = verify_candidate(
            row, expected_sr=sr, expected_frames=len(mixture)
        )
        metrics = exact_metrics(audio, accompaniment, vocal, sr)
        scored.append({
            "work_id": work_id,
            "candidate": row.candidate,
            "recipe_id": row.recipe_id,
            "path": str(path),
            "artifact_pcm_sha256": row.artifact_pcm_sha256,
            "container_sha256": row.container_sha256,
            "frames": row.frames,
            "sample_rate_hz": row.sr_hz,
            "channels": list(row.channels),
            "subtype": row.subtype,
            "metrics": metrics,
            "screening": screening(metrics, policy),
        })
    unique, aliases = _unique_artifacts(scored)
    ranked = rank_rows(unique)
    if len(ranked) < 2:
        raise ClassicalReleaseError(f"{work_id}: fewer than two byte-distinct candidates")
    status = "final" if ranked[0]["screening"]["passed"] else "needs_human_ab"
    release_scope = (
        "exact_benchmark_qualified" if status == "final" else "engineering_preview"
    )
    return {
        "work_id": work_id,
        "policy_id": policy.policy_id,
        "status": status,
        "release_scope": release_scope,
        "primary": ranked[0]["candidate"],
        "alternate": ranked[1]["candidate"],
        "pareto_front": pareto_front(unique),
        "artifact_aliases": aliases,
        "candidates": scored,
        "ranking": [row["candidate"] for row in ranked],
        "truth": {
            "mixture": str(truth / "mix_with_voice.wav"),
            "accompaniment": str(truth / "orchestra_only.wav"),
            "vocal": str(truth / "voice_ref.wav"),
            "frames": len(mixture),
            "sample_rate_hz": sr,
        },
    }


def _copy_verified(source: Path, destination: Path, *, expected: dict) -> dict:
    """Copy to a distinct inode and verify both source and destination identities."""
    source_record = _audio_record(
        source, expected_sr=int(expected["sample_rate_hz"]),
        expected_frames=int(expected["frames"]),
    )
    for name in ("container_sha256", "artifact_pcm_sha256"):
        if source_record[name] != expected[name]:
            raise ClassicalReleaseError(
                f"source changed since scoring: {source} {name} "
                f"{source_record[name]} != {expected[name]}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(fd)
        temporary = Path(name)
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    destination_record = _audio_record(
        destination, expected_sr=int(expected["sample_rate_hz"]),
        expected_frames=int(expected["frames"]),
    )
    for name in ("container_sha256", "artifact_pcm_sha256"):
        if destination_record[name] != expected[name]:
            raise ClassicalReleaseError(
                f"packaged artifact mismatch: {destination} {name}"
            )
    try:
        if os.path.samefile(source, destination):
            raise ClassicalReleaseError(
                f"release artifact shares an inode with immutable source: {destination}"
            )
    except FileNotFoundError:  # pragma: no cover - already checked above
        raise ClassicalReleaseError(f"missing package artifact: {destination}")
    destination_record["path"] = destination.name
    return destination_record


def _write_float(path: Path, audio: np.ndarray, sr: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(fd)
        temporary = Path(name)
        try:
            sf.write(temporary, audio.astype(np.float32), sr,
                     subtype="FLOAT", format="WAV")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    reopened, reopened_sr = _read_float(path, sr=sr, frames=len(audio))
    if not np.array_equal(reopened, audio.astype(np.float32)):
        raise ClassicalReleaseError(f"FLOAT reopen mismatch: {path}")
    record = _audio_record(path, expected_sr=sr, expected_frames=len(audio))
    record["path"] = path.name
    return record


def _candidate_expected(row: dict) -> dict:
    return {
        "container_sha256": row["container_sha256"],
        "artifact_pcm_sha256": row["artifact_pcm_sha256"],
        "frames": row["frames"],
        "sample_rate_hz": row["sample_rate_hz"],
    }


def _markdown(work_reports: list[dict]) -> str:
    lines = [
        "# Exact classical baseline release report",
        "",
        "> Engineering selection on exact linear references; no population-risk or arbitrary-master claim.",
        "",
    ]
    columns = [
        "work", "candidate", "screen", "voice coef p90", "voice dB p90",
        "hole p90", "hole max", "alpha err", "artifact", "SDR", "stereo dev",
        "coverage",
    ]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for report in work_reports:
        for row in rank_rows(report["candidates"]):
            metrics = row["metrics"]
            lines.append("| " + " | ".join([
                report["work_id"], row["candidate"],
                "pass" if row["screening"]["passed"] else "fail",
                f"{metrics['retained_voice_coef_p90']:.4f}",
                f"{metrics['retained_voice_db_p90']:.2f}",
                f"{metrics['event_hole_db_p90']:.2f}",
                f"{metrics['event_hole_db_max']:.2f}",
                f"{metrics['alpha_error_p90']:.4f}",
                f"{metrics['artifact_ratio_p90']:.4f}",
                f"{metrics['scale_dependent_sdr_db']:.2f}",
                f"{metrics['stereo_width_dev_db/v2']:.3f}",
                f"{metrics['_available_tiles']}/{metrics['_total_tiles']}",
            ]) + " |")
    lines += ["", "## Selected outputs", ""]
    for report in work_reports:
        lines += [
            f"### {report['work_id']}", "",
            f"- Terminal status: `{report['status']}`",
            f"- Release scope: `{report['release_scope']}`",
            f"- Primary: `{report['primary']}`",
            f"- Alternate: `{report['alternate']}`",
            f"- Pareto front: `{', '.join(report['pareto_front'])}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def package_work(report: dict, output_root: Path) -> dict:
    work_dir = output_root / report["work_id"]
    rows = {row["candidate"]: row for row in report["candidates"]}
    primary = rows[report["primary"]]
    alternate = rows[report["alternate"]]
    truth = report["truth"]
    mixture, sr = _read_float(Path(truth["mixture"]))
    if sr != truth["sample_rate_hz"] or len(mixture) != truth["frames"]:
        raise ClassicalReleaseError("truth grid changed before packaging")

    truth_records = {
        "exact_orchestra_target": _audio_record(
            Path(truth["accompaniment"]), expected_sr=sr,
            expected_frames=len(mixture),
        ),
        "exact_voice_target": _audio_record(
            Path(truth["vocal"]), expected_sr=sr, expected_frames=len(mixture),
        ),
    }
    packaged = {
        "accompaniment_primary": _copy_verified(
            Path(primary["path"]), work_dir / "accompaniment.primary.f32.wav",
            expected=_candidate_expected(primary),
        ),
        "accompaniment_alternate": _copy_verified(
            Path(alternate["path"]), work_dir / "accompaniment.alternate.f32.wav",
            expected=_candidate_expected(alternate),
        ),
        "exact_orchestra_target": _copy_verified(
            Path(truth["accompaniment"]), work_dir / "exact-orchestra-target.f32.wav",
            expected=truth_records["exact_orchestra_target"],
        ),
        "exact_voice_target": _copy_verified(
            Path(truth["vocal"]), work_dir / "exact-voice-target.f32.wav",
            expected=truth_records["exact_voice_target"],
        ),
    }
    primary_audio, _ = _read_float(
        Path(primary["path"]), sr=sr, frames=len(mixture)
    )
    packaged["removed_vocal_primary"] = _write_float(
        work_dir / "removed-vocal.primary.f32.wav",
        mixture.astype(np.float32) - primary_audio.astype(np.float32), sr,
    )
    manifest = {
        "schema": "audio-extract/classical-exact-release/v1",
        "status": report["status"],
        "release_scope": report["release_scope"],
        "population_risk_claim": None,
        "work_id": report["work_id"],
        "policy_id": report["policy_id"],
        "primary": primary,
        "alternate": alternate,
        "pareto_front": report["pareto_front"],
        "artifact_aliases": report["artifact_aliases"],
        "packaged_artifacts": packaged,
        "truth_source_paths": truth,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path = work_dir / "manifest.json"
    if path.exists() and path.read_text() != payload:
        raise ClassicalReleaseError(f"refusing to replace differing manifest: {path}")
    if not path.exists():
        path.write_text(payload)
    return manifest


def run(*, candidate_manifest: Path, truth_root: Path, policy_path: Path,
        report_json: Path, report_md: Path, package_root: Path | None) -> dict:
    rows = load_candidate_manifest(candidate_manifest)
    policy = ScreeningPolicy.from_mapping(yaml.safe_load(policy_path.read_text()))
    works = sorted({row.work_id for row in rows})
    reports = [score_work(rows, truth_root, policy, work) for work in works]
    status = "final" if all(report["status"] == "final" for report in reports) else "needs_human_ab"
    result = {
        "schema": "audio-extract/classical-exact-release-report/v1",
        "status": status,
        "release_scope": "exact_linear_reference_engineering_release",
        "population_risk_claim": None,
        "policy": asdict(policy),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha_file(candidate_manifest),
        "works": reports,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    if report_json.exists() and report_json.read_text() != payload:
        raise ClassicalReleaseError(f"refusing to replace differing report: {report_json}")
    if not report_json.exists():
        report_json.write_text(payload)
    markdown = _markdown(reports)
    if report_md.exists() and report_md.read_text() != markdown:
        raise ClassicalReleaseError(f"refusing to replace differing report: {report_md}")
    if not report_md.exists():
        report_md.write_text(markdown)
    packages = []
    if package_root is not None:
        packages = [package_work(report, package_root) for report in reports]
    result["packages"] = len(packages)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("audio-extract classical-release")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        candidate_manifest=args.candidate_manifest,
        truth_root=args.truth_root,
        policy_path=args.policy,
        report_json=args.report_json,
        report_md=args.report_md,
        package_root=args.package_root,
    )
    print(json.dumps({
        "status": result["status"], "works": len(result["works"]),
        "packages": result["packages"], "report": str(args.report_json),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
