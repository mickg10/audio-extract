"""Build one machine-readable catalog for the project's distributed audio data.

The catalog is metadata-only: it reads directory entries and WAV headers, not
audio payloads.  Existing authoritative manifests are embedded or referenced so
the generated file is useful without pretending that mtimes are content hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NAS_HOST = "mickg@nas642tail"
NAS_ROOT = "/tanksmall/MICKG2/mickg/cantolopera/full_48khz_f32"
PRODUCTION_HOST = "mickg10@10.0.27.98"
PRODUCTION_ROOT = "/share/homes/mickg10/datasets/production-outputs/median-v2-48k-f32"
WORKS_HOST = "ttuser@100.91.242.69"
WORKS_ROOT = "/home/ttuser/datasets/works"
HELD_ROOTS = {
    "aalto": "/home/ttuser/datasets/aalto",
    "bologna": "/home/ttuser/datasets/bologna",
    "freidi": "/home/ttuser/datasets/freidi",
    "spheres": "/home/ttuser/datasets/spheres",
}


REMOTE_SCAN = r"""
import json, os, struct, sys

root = sys.argv[1]
wav_headers = sys.argv[2] == "1"

def wav_info(path):
    out = {}
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"RIFF":
                return {"header_error": "not RIFF"}
            fh.read(4)
            if fh.read(4) != b"WAVE":
                return {"header_error": "not WAVE"}
            while True:
                header = fh.read(8)
                if len(header) != 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", header)
                if chunk_id == b"fmt ":
                    data = fh.read(chunk_size)
                    if len(data) < 16:
                        return {"header_error": "short fmt chunk"}
                    fmt, channels, sr, byte_rate, align, bits = struct.unpack(
                        "<HHIIHH", data[:16]
                    )
                    out.update({
                        "format_code": fmt,
                        "channels": channels,
                        "sample_rate_hz": sr,
                        "byte_rate": byte_rate,
                        "block_align": align,
                        "bits_per_sample": bits,
                    })
                elif chunk_id == b"data":
                    data_offset = fh.tell()
                    out["data_offset_bytes"] = data_offset
                    out["declared_audio_data_bytes"] = chunk_size
                    # ffmpeg's pipe-friendly WAV writer uses UINT32_MAX when it
                    # cannot seek back to finalize RIFF/data sizes.  These files
                    # contain audio through EOF, so record the physical extent.
                    out["audio_data_bytes"] = (
                        os.fstat(fh.fileno()).st_size - data_offset
                        if chunk_size == 0xFFFFFFFF
                        else chunk_size
                    )
                    break
                else:
                    fh.seek(chunk_size + (chunk_size & 1), 1)
        if out.get("byte_rate") and out.get("audio_data_bytes") is not None:
            out["duration_seconds"] = round(
                out["audio_data_bytes"] / out["byte_rate"], 6
            )
        if out.get("format_code") == 3:
            out["encoding"] = "IEEE_FLOAT"
        elif out.get("format_code") == 1:
            out["encoding"] = "PCM_INTEGER"
    except Exception as exc:
        out = {"header_error": f"{type(exc).__name__}: {exc}"}
    return out

for dirpath, dirnames, filenames in os.walk(root):
    dirnames.sort()
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        st = os.stat(path, follow_symlinks=False)
        row = {
            "path": os.path.relpath(path, root),
            "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "kind": "symlink" if os.path.islink(path) else "file",
        }
        if wav_headers and name.lower().endswith(".wav") and not os.path.islink(path):
            row["audio"] = wav_info(path)
        print(json.dumps(row, sort_keys=True, separators=(",", ":")))
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _remote_scan(host: str, root: str, *, wav_headers: bool) -> list[dict[str, Any]]:
    command = "python3 -c {} {} {}".format(
        shlex.quote(REMOTE_SCAN), shlex.quote(root), "1" if wav_headers else "0"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def _remote_text(host: str, path: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "cat", "--", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _remote_symlinks(host: str, root: str) -> list[str]:
    command = "find {} -maxdepth 1 -type l -name '*.wav' -print".format(
        shlex.quote(root)
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = root.rstrip("/") + "/"
    return sorted(
        line.removeprefix(prefix)
        for line in result.stdout.splitlines()
        if line.strip()
    )


def _production_instrumentals() -> dict[str, Any]:
    catalogue_text = _remote_text(PRODUCTION_HOST, f"{PRODUCTION_ROOT}/outputs.jsonl")
    summary_text = _remote_text(PRODUCTION_HOST, f"{PRODUCTION_ROOT}/summary.json")
    rows = [
        json.loads(line)
        for line in catalogue_text.splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_text)
    catalogue_sha256 = "sha256:" + hashlib.sha256(catalogue_text.encode()).hexdigest()
    summary_sha256 = "sha256:" + hashlib.sha256(summary_text.encode()).hexdigest()
    if summary.get("catalogue_sha256") != catalogue_sha256:
        raise ValueError("production summary does not bind the exact output catalogue")
    if summary.get("runs") != len(rows) or summary.get("outputs") != 2 * len(rows):
        raise ValueError("production summary counts differ from the output catalogue")
    if any(
        row.get("schema") != "audio-extract/median-v2-production-output/v1"
        or set(row.get("variants", {})) != {"median", "mdx"}
        or not all(row.get("validation", {}).values())
        for row in rows
    ):
        raise ValueError("production catalogue contains an invalid delivery row")
    wav_links = _remote_symlinks(PRODUCTION_HOST, PRODUCTION_ROOT)
    if len(wav_links) != summary["outputs"]:
        raise ValueError("production output link count differs from the summary")
    return {
        "storage": {"host": PRODUCTION_HOST, "root": PRODUCTION_ROOT},
        "content_hash_status": "container_and_decoded_pcm_sha256_manifest",
        "summary": summary,
        "catalogue": {
            "path": "outputs.jsonl",
            "bytes": len(catalogue_text.encode()),
            "sha256": catalogue_sha256,
            "records": len(rows),
        },
        "summary_file": {
            "path": "summary.json",
            "bytes": len(summary_text.encode()),
            "sha256": summary_sha256,
        },
        "items": rows,
        "delivery_links": wav_links,
    }


def _cantolopera(files: list[dict[str, Any]]) -> dict[str, Any]:
    complete_files = [row for row in files if "/" not in row["path"]]
    partial_files = [row for row in files if row["path"].startswith(".rsync-partial/")]
    by_slug: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unexpected: list[dict[str, Any]] = []
    for row in complete_files:
        name = row["path"]
        role = None
        for candidate in ("voice", "orchestra"):
            suffix = f"_{candidate}.wav"
            if name.endswith(suffix):
                role = candidate
                slug = name[: -len(suffix)]
                break
        if role is None:
            unexpected.append(row)
        else:
            by_slug[slug][role] = row

    items = []
    for slug, roles in sorted(by_slug.items()):
        voice_duration = roles.get("voice", {}).get("audio", {}).get("duration_seconds")
        orchestra_duration = (
            roles.get("orchestra", {}).get("audio", {}).get("duration_seconds")
        )
        duration_delta = (
            abs(voice_duration - orchestra_duration)
            if voice_duration is not None and orchestra_duration is not None
            else None
        )
        items.append(
            {
                "item_id": slug,
                "complete_pair": set(roles) == {"voice", "orchestra"},
                "files": roles,
                "pair_bytes": sum(row["bytes"] for row in roles.values()),
                "duration_delta_seconds": duration_delta,
                "duration_match_exact": duration_delta == 0,
                "duration_compatible_50ms": (
                    duration_delta is not None and duration_delta <= 0.050
                ),
            }
        )
    complete_audio = [row["audio"] for row in complete_files if "audio" in row]
    expected = {
        "container": "RIFF/WAVE",
        "encoding": "IEEE_FLOAT",
        "sample_rate_hz": 48000,
        "channels": 2,
        "bits_per_sample": 32,
    }

    def conforms(audio: dict[str, Any]) -> bool:
        return (
            audio.get("encoding") == expected["encoding"]
            and audio.get("sample_rate_hz") == expected["sample_rate_hz"]
            and audio.get("channels") == expected["channels"]
            and audio.get("bits_per_sample") == expected["bits_per_sample"]
        )

    return {
        "storage": {"host": NAS_HOST, "root": NAS_ROOT},
        "catalog_scope": "regular files immediately below root; transfer fragments separated",
        "content_hash_status": "not_computed_metadata_catalog",
        "summary": {
            "complete_pairs": sum(item["complete_pair"] for item in items),
            "items": len(items),
            "complete_files": len(complete_files),
            "complete_bytes": sum(row["bytes"] for row in complete_files),
            "partial_transfer_files": len(partial_files),
            "partial_transfer_bytes": sum(row["bytes"] for row in partial_files),
            "unexpected_root_files": len(unexpected),
            "duration_exact_pairs": sum(item["duration_match_exact"] for item in items),
            "duration_compatible_50ms_pairs": sum(
                item["duration_compatible_50ms"] for item in items
            ),
            "audio_hours_one_copy_per_pair": round(
                sum(
                    item["files"]["voice"]["audio"]["duration_seconds"]
                    for item in items
                    if item["complete_pair"]
                )
                / 3600,
                3,
            ),
            "header_errors": sum("header_error" in audio for audio in complete_audio),
            "contract_conforming_files": sum(
                conforms(audio) for audio in complete_audio
            ),
        },
        "expected_audio_contract": expected,
        "items": items,
        "incomplete_transfer_fragments": partial_files,
        "unexpected_root_files": unexpected,
    }


def _works(files: list[dict[str, Any]], works_manifest: Path) -> dict[str, Any]:
    by_top: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    support_files = []
    for row in files:
        if "/" not in row["path"]:
            support_files.append(row)
            continue
        top = row["path"].split("/", 1)[0]
        by_top[top]["files"] += 1
        by_top[top]["bytes"] += row["bytes"]
    return {
        "storage": {"host": WORKS_HOST, "root": WORKS_ROOT},
        "content_hash_status": "not_computed_metadata_catalog",
        "summary": {
            "files": len(files),
            "bytes": sum(row["bytes"] for row in files),
            "top_level_corpora": len(by_top),
            "declared_work_records": len(_jsonl(works_manifest)),
        },
        "corpora": [
            {"corpus_id": name, **counts} for name, counts in sorted(by_top.items())
        ],
        "declared_works": _jsonl(works_manifest),
        "files": files,
        "top_level_support_files": support_files,
    }


def _held_corpora() -> dict[str, Any]:
    corpora = []
    for corpus_id, root in HELD_ROOTS.items():
        files = _remote_scan(WORKS_HOST, root, wav_headers=False)
        corpora.append(
            {
                "corpus_id": corpus_id,
                "storage": {"host": WORKS_HOST, "root": root},
                "summary": {
                    "files": len(files),
                    "bytes": sum(row["bytes"] for row in files),
                },
                "files": files,
            }
        )
    return {
        "content_hash_status": "not_computed_metadata_catalog",
        "summary": {
            "corpora": len(corpora),
            "files": sum(row["summary"]["files"] for row in corpora),
            "bytes": sum(row["summary"]["bytes"] for row in corpora),
        },
        "corpora": corpora,
    }


def build() -> dict[str, Any]:
    works_manifest = ROOT / "calibration" / "works_manifest.jsonl"
    candidate_manifest = ROOT / "datasets" / "candidates-v2.jsonl"
    cantolopera_manifest = ROOT / "calibration" / "cantolopera_full_manifest.jsonl"
    provenance_paths = [
        works_manifest,
        candidate_manifest,
        cantolopera_manifest,
        ROOT / "calibration" / "cantolopera_previews.json",
        ROOT / "calibration" / "manifest.json",
        ROOT / "calibration" / "novocal_result.json",
        ROOT / "datasets" / "classical-v1.jsonl",
        ROOT / "datasets" / "classical-v1-splits.json",
    ]
    nas_files = _remote_scan(NAS_HOST, NAS_ROOT, wav_headers=True)
    works_files = _remote_scan(WORKS_HOST, WORKS_ROOT, wav_headers=False)
    candidates = _jsonl(candidate_manifest)
    cantolopera = _cantolopera(nas_files)
    open_corpora = _works(works_files, works_manifest)
    held_corpora = _held_corpora()
    production_instrumentals = _production_instrumentals()
    result: dict[str, Any] = {
        "schema": "audio-extract/audio-treasure-catalog/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/build_audio_treasure_catalog.py",
        "scope": {
            "description": (
                "Source/reference corpora, immutable v2 candidate index, and "
                "audited production instrumentals"
            ),
            "metadata_only": True,
            "warning": "File size and mtime prove inventory presence, not content identity; use the immutable v2 manifests where hashes are present.",
        },
        "source_inventory_summary": {
            "files": (
                cantolopera["summary"]["complete_files"]
                + open_corpora["summary"]["files"]
                + held_corpora["summary"]["files"]
            ),
            "bytes": (
                cantolopera["summary"]["complete_bytes"]
                + open_corpora["summary"]["bytes"]
                + held_corpora["summary"]["bytes"]
            ),
            "note": (
                "Excludes Cantolopera .rsync-partial fragments, derived candidates, "
                "and production deliveries."
            ),
        },
        "cantolopera_full": cantolopera,
        "open_reference_corpora": open_corpora,
        "held_reference_corpora": held_corpora,
        "production_instrumentals": production_instrumentals,
        "v2_candidates": {
            "manifest": str(candidate_manifest.relative_to(ROOT)),
            "manifest_sha256": _sha256(candidate_manifest),
            "summary": {
                "candidate_records": len(candidates),
                "works": len({row["work_id"] for row in candidates}),
                "all_have_container_sha256": all(
                    bool(row.get("container_sha256")) for row in candidates
                ),
                "all_have_pcm_sha256": all(
                    bool(row.get("artifact_pcm_sha256")) for row in candidates
                ),
            },
            "items": candidates,
        },
        "authoritative_manifest_provenance": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in provenance_paths
        ],
    }
    identity_doc = json.loads(json.dumps(result))
    identity_doc.pop("generated_at")
    result["catalog_semantic_sha256"] = hashlib.sha256(
        json.dumps(identity_doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "audio-treasure-catalog-v1.json",
    )
    args = parser.parse_args()
    catalog = build()
    args.output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(catalog["catalog_semantic_sha256"])


if __name__ == "__main__":
    main()
