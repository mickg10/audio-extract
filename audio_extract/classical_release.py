"""Score and package the best currently available exact classical candidates.

This module is the delivery lane for the audited ``candidates-v2`` artifacts. It
uses exact accompaniment and featured-soloist references, writes a complete
metric table, applies an explicit engineering screening policy, and packages a
primary plus alternate without making a population-risk or arbitrary-master
claim.

The policy is deliberately separate from the measurements. A candidate that
fails every screening policy can still be packaged as an ``engineering_preview``
with the failure reasons and alternate visible; it is never silently called
certified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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
    event_hole_db_p90_max: float
    event_hole_db_max_max: float
    alpha_error_p90_max: float
    stereo_width_dev_db_max: float
    coherence_dev_max: float
    artifact_ratio_p90_max: float | None = None

    @classmethod
    def from_mapping(cls, value: dict) -> "ScreeningPolicy":
        policy = cls(**value)
        for name, item in asdict(policy).items():
            if name == "policy_id":
                if not str(item).strip():
                    raise ValueError("policy_id must be non-empty")
            elif item is not None and (not math.isfinite(float(item)) or float(item) <= 0):
                raise ValueError(f"{name} must be positive and finite")
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
        if row.subtype != "FLOAT" or row.channels != ("FL", "FR"):
            raise ClassicalReleaseError(f"non-FLOAT/stereo analysis row: {row}")
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


def verify_candidate(row: CandidateRow) -> tuple[Path, np.ndarray]:
    path = _resolve_host_path(row.path)
    if _sha_file(path) != row.container_sha256:
        raise ClassicalReleaseError(f"container hash mismatch: {row.work_id}/{row.candidate}")
    audio, sr = _read_float(path, sr=row.sr_hz, frames=row.frames)
    pcm = identity.artifact_pcm_sha256(audio, sr, list(row.channels), len(audio))
    if pcm != row.artifact_pcm_sha256:
        raise ClassicalReleaseError(f"PCM hash mismatch: {row.work_id}/{row.candidate}")
    return path, audio


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
    if metrics.get("_available_tiles", 0) <= 0:
        raise ClassicalReleaseError("no identifiable exact-label tiles")
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
        metrics[observation["metric"]] = (
            observation["value"] if observation["available"] else None
        )
    return metrics


def screening(metrics: dict, policy: ScreeningPolicy) -> dict:
    checks = {
        "retained_voice_coef_p90": (
            float(metrics["retained_voice_coef_p90"]),
            policy.retained_voice_coef_p90_max,
        ),
        "event_hole_db_p90": (
            float(metrics["event_hole_db_p90"]), policy.event_hole_db_p90_max,
        ),
        "event_hole_db_max": (
            float(metrics["event_hole_db_max"]), policy.event_hole_db_max_max,
        ),
        "alpha_error_p90": (
            float(metrics["alpha_error_p90"]), policy.alpha_error_p90_max,
        ),
        "stereo_width_dev_db/v2": (
            float(metrics["stereo_width_dev_db/v2"]),
            policy.stereo_width_dev_db_max,
        ),
        "interchannel_coherence_dev/v2": (
            float(metrics["interchannel_coherence_dev/v2"]),
            policy.coherence_dev_max,
        ),
    }
    if policy.artifact_ratio_p90_max is not None:
        checks["artifact_ratio_p90"] = (
            float(metrics["artifact_ratio_p90"]), policy.artifact_ratio_p90_max,
        )
    failures = [
        {"metric": name, "value": value, "limit": limit}
        for name, (value, limit) in checks.items()
        if not math.isfinite(value) or value > limit
    ]
    normalized = {
        name: value / limit for name, (value, limit) in checks.items()
    }
    return {
        "passed": not failures,
        "failures": failures,
        "normalized": normalized,
        "critical_max": max(normalized.values()),
    }


def _dominates(left: dict, right: dict) -> bool:
    names = (
        "retained_voice_coef_p90", "event_hole_db_p90",
        "event_hole_db_max", "alpha_error_p90", "artifact_ratio_p90",
        "stereo_width_dev_db/v2", "interchannel_coherence_dev/v2",
    )
    lv = [float(left[name]) for name in names]
    rv = [float(right[name]) for name in names]
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
            float(row["metrics"]["artifact_ratio_p90"]),
            -float(row["metrics"]["scale_dependent_sdr_db"]),
            float(row["metrics"].get("erb_envelope_dist_db/v2") or math.inf),
            row["candidate"],
        ),
    )


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
        path, audio = verify_candidate(row)
        metrics = exact_metrics(audio, accompaniment, vocal, sr)
        scored.append({
            "work_id": work_id,
            "candidate": row.candidate,
            "recipe_id": row.recipe_id,
            "path": str(path),
            "artifact_pcm_sha256": row.artifact_pcm_sha256,
            "container_sha256": row.container_sha256,
            "metrics": metrics,
            "screening": screening(metrics, policy),
        })
    ranked = rank_rows(scored)
    status = "exact_benchmark_qualified" if ranked[0]["screening"]["passed"] else "engineering_preview"
    return {
        "work_id": work_id,
        "policy_id": policy.policy_id,
        "status": status,
        "primary": ranked[0]["candidate"],
        "alternate": ranked[1]["candidate"],
        "pareto_front": pareto_front(scored),
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


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha_file(destination) != _sha_file(source):
            raise ClassicalReleaseError(f"refusing to replace differing file: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if _sha_file(destination) != _sha_file(source):
        raise ClassicalReleaseError(f"packaged file hash mismatch: {destination}")


def _write_float(path: Path, audio: np.ndarray, sr: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        reopened, reopened_sr = _read_float(path, sr=sr, frames=len(audio))
        if not np.array_equal(reopened, audio.astype(np.float32)):
            raise ClassicalReleaseError(f"refusing to replace differing FLOAT: {path}")
    else:
        sf.write(path, audio.astype(np.float32), sr, subtype="FLOAT")
        reopened, reopened_sr = _read_float(path, sr=sr, frames=len(audio))
        if not np.array_equal(reopened, audio.astype(np.float32)):
            raise ClassicalReleaseError(f"FLOAT reopen mismatch: {path}")
    return {
        "path": str(path),
        "container_sha256": _sha_file(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            reopened, reopened_sr, ["FL", "FR"], len(reopened)
        ),
    }


def _markdown(work_reports: list[dict]) -> str:
    lines = [
        "# Exact classical baseline release report",
        "",
        "> Engineering selection on exact linear references; no population-risk or arbitrary-master claim.",
        "",
    ]
    columns = [
        "work", "candidate", "status", "voice coef p90", "voice dB p90",
        "hole p90", "hole max", "alpha err", "artifact", "SDR", "stereo dev",
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
            ]) + " |")
    lines += ["", "## Selected outputs", ""]
    for report in work_reports:
        lines += [
            f"### {report['work_id']}",
            "",
            f"- Status: `{report['status']}`",
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
    _link_or_copy(Path(primary["path"]), work_dir / "accompaniment.primary.f32.wav")
    _link_or_copy(Path(alternate["path"]), work_dir / "accompaniment.alternate.f32.wav")
    truth = report["truth"]
    _link_or_copy(Path(truth["accompaniment"]), work_dir / "exact-orchestra-target.f32.wav")
    _link_or_copy(Path(truth["vocal"]), work_dir / "exact-voice-target.f32.wav")
    mixture, sr = _read_float(Path(truth["mixture"]))
    primary_audio, _ = _read_float(
        Path(primary["path"]), sr=sr, frames=len(mixture)
    )
    removed = _write_float(
        work_dir / "removed-vocal.primary.f32.wav",
        mixture.astype(np.float32) - primary_audio.astype(np.float32), sr,
    )
    manifest = {
        "schema": "audio-extract/classical-exact-release/v1",
        "scope": "exact_linear_reference_engineering_release",
        "population_risk_claim": None,
        "work_id": report["work_id"],
        "status": report["status"],
        "policy_id": report["policy_id"],
        "primary": primary,
        "alternate": alternate,
        "pareto_front": report["pareto_front"],
        "removed_vocal_primary": removed,
        "truth": truth,
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
    result = {
        "schema": "audio-extract/classical-exact-release-report/v1",
        "scope": "exact_linear_reference_engineering_release",
        "population_risk_claim": None,
        "policy": asdict(policy),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha_file(candidate_manifest),
        "works": reports,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report_json.parent.mkdir(parents=True, exist_ok=True)
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
        "status": "complete", "works": len(result["works"]),
        "packages": result["packages"], "report": str(args.report_json),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
