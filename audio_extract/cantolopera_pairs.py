"""Audit lossless Cantolopera mixture/orchestra pairs for grouped training.

The acquired catalogue contains several role-omission variants of the same
underlying operatic number.  This module keeps those derivatives in one stable
take group and measures pair integrity without editing or materializing audio.
It deliberately reports alignment/null evidence instead of declaring every
catalogue pair ``linear_exact`` merely because both files exist.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from . import alignment, canon, identity


INVENTORY_SCHEMA = "audio-extract/cantolopera-audio-inventory/v1"
PAIR_SCHEMA = "audio-extract/cantolopera-pair-audit/v1"
SUMMARY_SCHEMA = "audio-extract/cantolopera-pair-audit-summary/v1"
EXPECTED_SAMPLE_RATE = 48_000
EXPECTED_CHANNELS = 2
EXPECTED_SUBTYPE = "FLOAT"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _ascii_words(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _without_senza_variant(value: str, *, opera: bool) -> str:
    value = re.sub(r"\([^)]*\bsenza\b[^)]*\)", " ", str(value), flags=re.I)
    if opera:
        value = re.sub(r"\bcompleta\b", " ", value, flags=re.I)
        value = re.split(r"\bsenza\b", value, maxsplit=1, flags=re.I)[0]
    return _ascii_words(value)


def take_group_facts(catalog: dict[str, Any]) -> dict[str, Any]:
    """Return a stable group shared by role-omission variants of one number."""

    key = {
        "composer": _ascii_words(catalog.get("composer", "")),
        "opera": _without_senza_variant(catalog.get("opera", ""), opera=True),
        "title": _without_senza_variant(catalog.get("title", ""), opera=False),
    }
    if not all(key.values()):
        raise ValueError(f"catalogue row lacks groupable composer/opera/title: {catalog}")
    digest = identity.blob_sha256(canon.canonicalize(key)).removeprefix("sha256:")
    return {"take_group_id": f"cantolopera:take:{digest[:24]}", "take_group_key": key}


def _resolved_audio_path(audio_root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe inventory audio path: {relative!r}")
    if rel.parts and rel.parts[0] in {audio_root.name, "full_48khz_f32"}:
        rel = Path(*rel.parts[1:])
    result = (audio_root / rel).resolve()
    root = audio_root.resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"inventory path escapes audio root: {relative!r}")
    return result


def load_completed_pairs(inventory_path: Path, audio_root: Path) -> list[dict[str, Any]]:
    """Load exactly one completed voice/orchestra row for every saved pair."""

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for number, line in enumerate(inventory_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != INVENTORY_SCHEMA:
            raise ValueError(f"inventory line {number} has wrong schema")
        if row.get("storage_state") != "complete":
            continue
        pair_id = row.get("pair_id")
        version = row.get("version")
        if not isinstance(pair_id, str) or version not in {"voice", "orchestra"}:
            raise ValueError(f"invalid completed inventory row at line {number}")
        versions = grouped.setdefault(pair_id, {})
        if version in versions:
            raise ValueError(f"duplicate {pair_id}/{version} inventory row")
        versions[version] = row

    pairs = []
    for pair_id, versions in sorted(grouped.items()):
        if set(versions) != {"voice", "orchestra"}:
            raise ValueError(f"incomplete completed pair {pair_id}: {sorted(versions)}")
        voice, orchestra = versions["voice"], versions["orchestra"]
        if voice.get("catalog") != orchestra.get("catalog"):
            raise ValueError(f"catalogue facts differ inside pair {pair_id}")
        catalog = voice["catalog"]
        pairs.append(
            {
                "pair_id": pair_id,
                "catalog": catalog,
                **take_group_facts(catalog),
                "files": {
                    version: {
                        "path": str(
                            _resolved_audio_path(
                                audio_root, versions[version]["file"]["relative_path"]
                            )
                        ),
                        "container_sha256": versions[version]["file"]["sha256"],
                        "inventory_audio": versions[version].get("audio"),
                    }
                    for version in ("voice", "orchestra")
                },
            }
        )
    return pairs


def _probe_starts(frames: int, probe_frames: int, count: int) -> list[int]:
    if frames <= probe_frames:
        return [0]
    return sorted(
        {
            int(round(value))
            for value in np.linspace(0, frames - probe_frames, max(2, count))
        }
    )


def _read_probe(path: Path, start: int, frames: int) -> np.ndarray:
    audio, rate = sf.read(
        path, start=start, frames=frames, dtype="float32", always_2d=True
    )
    if rate != EXPECTED_SAMPLE_RATE or len(audio) != frames:
        raise ValueError(f"short or off-rate probe read: {path} at {start}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"non-finite samples in {path}")
    return np.asarray(audio, dtype=np.float32)


def _null_blocks(
    mixture: np.ndarray, orchestra: np.ndarray, block_frames: int
) -> list[dict[str, float]]:
    rows = []
    for start in range(0, len(mixture) - block_frames + 1, block_frames):
        m = mixture[start : start + block_frames].astype(np.float64)
        a = orchestra[start : start + block_frames].astype(np.float64)
        m -= np.mean(m, axis=0, keepdims=True)
        a -= np.mean(a, axis=0, keepdims=True)
        denominator = float(np.sum(np.square(a)))
        mixture_energy = float(np.sum(np.square(m)))
        if denominator <= 1e-15 or mixture_energy <= 1e-15:
            continue
        gain = float(np.sum(a * m) / denominator)
        residual = m - gain * a
        ratio = float(np.sum(np.square(residual)) / mixture_energy)
        correlation = float(
            np.sum(a * m)
            / math.sqrt(denominator * mixture_energy)
        )
        rows.append(
            {
                "gain": gain,
                "null_db": 10.0 * math.log10(max(ratio, 1e-30)),
                "correlation": correlation,
            }
        )
    return rows


def _percentile(values: Iterable[float], value: float) -> float | None:
    data = list(values)
    if not data:
        return None
    return float(np.percentile(np.asarray(data, dtype=np.float64), value))


def audit_pair(
    pair: dict[str, Any],
    *,
    verify_container_hashes: bool = True,
    probe_seconds: float = 4.0,
    probe_count: int = 5,
    max_shift_samples: int = 8192,
    null_block_seconds: float = 0.5,
) -> dict[str, Any]:
    """Return objective grid, alignment, and low-residual evidence for one pair."""

    paths = {name: Path(facts["path"]) for name, facts in pair["files"].items()}
    result: dict[str, Any] = {
        "schema": PAIR_SCHEMA,
        "pair_id": pair["pair_id"],
        "take_group_id": pair["take_group_id"],
        "take_group_key": pair["take_group_key"],
        "catalog": pair["catalog"],
        "files": pair["files"],
    }
    errors = []
    infos = {}
    for version, path in paths.items():
        if not path.is_file():
            errors.append(f"missing {version} file: {path}")
            continue
        info = sf.info(path)
        infos[version] = {
            "frames": int(info.frames),
            "sample_rate_hz": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": info.subtype,
        }
        expected = pair["files"][version].get("inventory_audio") or {}
        for key, observed in (
            ("frames", int(info.frames)),
            ("sample_rate_hz", int(info.samplerate)),
            ("channels", int(info.channels)),
            ("subtype", info.subtype),
        ):
            if expected.get(key) != observed:
                errors.append(
                    f"{version} {key} differs from inventory: "
                    f"{observed!r} != {expected.get(key)!r}"
                )
        if verify_container_hashes:
            actual = _file_sha256(path)
            if actual != pair["files"][version]["container_sha256"]:
                errors.append(f"{version} container SHA-256 mismatch")

    result["audio"] = infos
    required_grid = {
        "sample_rate_hz": EXPECTED_SAMPLE_RATE,
        "channels": EXPECTED_CHANNELS,
        "subtype": EXPECTED_SUBTYPE,
    }
    grid_valid = len(infos) == 2 and all(
        info.get("frames", 0) > 0
        and all(info.get(key) == value for key, value in required_grid.items())
        for info in infos.values()
    )
    frame_match = grid_valid and infos["voice"]["frames"] == infos["orchestra"]["frames"]
    result["grid"] = {
        **required_grid,
        "valid": grid_valid,
        "frame_count_match": frame_match,
        "frames": infos.get("voice", {}).get("frames") if frame_match else None,
    }
    if errors or not frame_match:
        result.update(
            {
                "status": "invalid_pair_evidence",
                "alignment": None,
                "null_evidence": None,
                "errors": errors + ([] if frame_match else ["pair sample grids differ"]),
            }
        )
        return result

    frames = infos["voice"]["frames"]
    probe_frames = min(frames, max(32, round(probe_seconds * EXPECTED_SAMPLE_RATE)))
    block_frames = min(
        probe_frames, max(16, round(null_block_seconds * EXPECTED_SAMPLE_RATE))
    )
    alignment_rows = []
    null_rows = []
    for start in _probe_starts(frames, probe_frames, probe_count):
        mixture = _read_probe(paths["voice"], start, probe_frames)
        orchestra = _read_probe(paths["orchestra"], start, probe_frames)
        aligned = alignment.estimate_alignment(
            orchestra,
            mixture,
            max_shift=min(max_shift_samples, probe_frames - 1),
            use_phat=False,
        )
        alignment_rows.append(
            {
                "start_frame": start,
                "delay_samples": aligned.delay_samples,
                "fractional_samples": aligned.fractional,
                "polarity": aligned.polarity,
                "channel_swap": aligned.channel_swap,
                "confidence": aligned.confidence,
            }
        )
        aligned_mixture = alignment.apply_alignment(
            mixture, aligned, target_len=len(orchestra)
        )
        margin = min(max_shift_samples + 2, max(0, len(orchestra) // 4))
        if margin and len(orchestra) > 2 * margin + block_frames:
            aligned_mixture = aligned_mixture[margin:-margin]
            orchestra_for_null = orchestra[margin:-margin]
        else:
            orchestra_for_null = orchestra
        null_rows.extend(
            _null_blocks(aligned_mixture, orchestra_for_null, block_frames)
        )

    delays = [row["delay_samples"] for row in alignment_rows]
    fractions = [row["fractional_samples"] for row in alignment_rows]
    median_delay = int(round(float(np.median(delays))))
    zero_lag_votes = sum(
        abs(row["delay_samples"]) <= 1
        and row["polarity"] == 1
        and row["channel_swap"] is False
        for row in alignment_rows
    )
    zero_lag_supported = (
        abs(median_delay) <= 1
        and zero_lag_votes >= math.ceil(0.6 * len(alignment_rows))
    )
    result["alignment"] = {
        "probe_count": len(alignment_rows),
        "probes": alignment_rows,
        "median_delay_samples": median_delay,
        "median_fractional_samples": float(np.median(fractions)),
        "minimum_confidence": min(row["confidence"] for row in alignment_rows),
        "confidence_p10": _percentile(
            (row["confidence"] for row in alignment_rows), 10
        ),
        "confidence_median": _percentile(
            (row["confidence"] for row in alignment_rows), 50
        ),
        "zero_lag_positive_polarity_no_swap_votes": zero_lag_votes,
        "zero_lag_positive_polarity_no_swap_vote_fraction": (
            zero_lag_votes / len(alignment_rows)
        ),
        "zero_lag_positive_polarity_no_swap_supported": zero_lag_supported,
    }

    ordered = sorted(null_rows, key=lambda row: row["null_db"])
    best_count = max(1, math.ceil(0.2 * len(ordered))) if ordered else 0
    best = ordered[:best_count]
    result["null_evidence"] = {
        "block_seconds": null_block_seconds,
        "blocks": len(null_rows),
        "null_db_p10": _percentile((row["null_db"] for row in null_rows), 10),
        "null_db_p50": _percentile((row["null_db"] for row in null_rows), 50),
        "best_quintile_gain_median": _percentile(
            (row["gain"] for row in best), 50
        ),
        "best_quintile_gain_p10": _percentile((row["gain"] for row in best), 10),
        "best_quintile_gain_p90": _percentile((row["gain"] for row in best), 90),
        "best_quintile_correlation_p10": _percentile(
            (row["correlation"] for row in best), 10
        ),
    }
    null_p10 = result["null_evidence"]["null_db_p10"]
    if not zero_lag_supported:
        grade = "needs_alignment_review"
    elif null_p10 is not None and null_p10 <= -20.0:
        grade = "strong_same_take_evidence"
    elif null_p10 is not None and null_p10 <= -10.0:
        grade = "weak_same_take_evidence"
    else:
        grade = "insufficient_same_take_evidence"
    result.update({"status": "audited", "suggested_grade": grade, "errors": []})
    return result


def audit_inventory(
    inventory_path: Path,
    audio_root: Path,
    *,
    verify_container_hashes: bool = True,
    probe_seconds: float = 4.0,
    probe_count: int = 5,
    max_shift_samples: int = 8192,
    null_block_seconds: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = load_completed_pairs(inventory_path, audio_root)
    rows = [
        audit_pair(
            pair,
            verify_container_hashes=verify_container_hashes,
            probe_seconds=probe_seconds,
            probe_count=probe_count,
            max_shift_samples=max_shift_samples,
            null_block_seconds=null_block_seconds,
        )
        for pair in pairs
    ]
    grades = Counter(row.get("suggested_grade", "invalid_pair_evidence") for row in rows)
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["take_group_id"], []).append(row["pair_id"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _file_sha256(inventory_path),
        "audio_root": str(audio_root.resolve()),
        "container_hashes_verified": verify_container_hashes,
        "pairs": len(rows),
        "take_groups": len(groups),
        "multi_variant_take_groups": sum(len(items) > 1 for items in groups.values()),
        "pairs_in_multi_variant_groups": sum(
            len(items) for items in groups.values() if len(items) > 1
        ),
        "grades": dict(sorted(grades.items())),
        "invalid_pairs": [
            {"pair_id": row["pair_id"], "errors": row["errors"]}
            for row in rows
            if row["status"] != "audited"
        ],
    }
    return rows, summary


def write_audit(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_jsonl: Path,
    output_summary: Path,
) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    summary_payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for path, content in ((output_jsonl, payload), (output_summary, summary_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"refusing to rewrite differing pair audit: {path}")
        if not path.exists():
            path.write_text(content)
