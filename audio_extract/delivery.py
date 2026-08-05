"""M6 delivery: float32 master → gain / dither / encode child DAG nodes (docs/v2 §7).

Delivery is a DAG of child nodes off a finalist candidate, never edits of it:
- `global_gain`  — one global gain to a target (float32 master, dynamics intact);
- `dither_quantize` — float master → integer PCM with TPDF dither, applied ONCE;
- `encode_delivery` — AAC encoded **from the float master**, never from a dithered
  intermediate (so we never dither twice).

Each node is content-addressed with its own id and records its parent, keeping the
source→delivery lineage auditable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import canon, identity

_DELIVERY_DOMAIN = b"audio-extract-delivery-v2\x00"


def delivery_node_id(operation: str, parents: list[str], params: dict, code_commit: str) -> str:
    obj = {"schema": "audio-extract/delivery/v1", "operation": operation,
           "parents": list(parents), "params": params, "code": code_commit}
    return "sha256:" + hashlib.sha256(_DELIVERY_DOMAIN + canon.canonicalize(obj)).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def global_gain(x: np.ndarray, target_dbfs: float = -5.0, mode: str = "peak"):
    """Return ``(gained_float32, applied_gain_db)``. Peak-normalize to target dBFS —
    a single global gain, dynamics untouched (docs/v2 §7.3)."""
    a = np.asarray(x, dtype=np.float64)
    if mode != "peak":
        raise ValueError(f"unsupported gain mode: {mode!r}")
    peak = float(np.max(np.abs(a))) or 1e-12
    g = (10 ** (target_dbfs / 20.0)) / peak
    return (a * g).astype(np.float32), float(20 * np.log10(g))


def tpdf_dither_float(x: np.ndarray, bits: int, seed: int = 0) -> np.ndarray:
    """Add TPDF dither of 2-LSB peak-to-peak, in the float domain, so the final
    quantize (by the WAV writer) realizes a properly dithered conversion. Clipped
    to [-1, 1]. Applied only here, once (docs/v2 §7.2)."""
    rng = np.random.default_rng(seed)
    lsb = 1.0 / (2 ** (bits - 1))
    d = (rng.random(x.shape) - rng.random(x.shape)) * lsb   # triangular PDF
    return np.clip(np.asarray(x, dtype=np.float64) + d, -1.0, 1.0)


def encode_aac(master_f32: np.ndarray, sr: int, out_path: Path, bitrate: str = "256k") -> None:
    """AAC from the float master via ffmpeg — no dithered intermediate."""
    import soundfile as sf

    fd, tmp = tempfile.mkstemp(suffix=".f32.wav")
    import os
    os.close(fd)
    try:
        sf.write(tmp, np.asarray(master_f32, dtype=np.float32), sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:a", "aac", "-b:a", bitrate, str(out_path)],
                       check=True, capture_output=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deliver(layout, source_record: dict, candidate_recipe_id: str, *, target_dbfs: float = -5.0,
            bitrate: str = "256k", code_commit: str = "", dither_seed: int = 0) -> dict:
    """Materialize the delivery DAG for a finalist candidate; return the repro report."""
    import soundfile as sf

    from .manifest import Manifest

    sr = int(source_record["sample_rate_hz"])
    cand_dir = layout.candidate_dir(candidate_recipe_id)
    master, _ = sf.read(str(cand_dir / "output.f32.wav"), dtype="float64", always_2d=True)
    layout_names = (source_record["channel_layout"]
                    if master.shape[1] == len(source_record["channel_layout"])
                    else [f"CH{i}" for i in range(master.shape[1])])
    rdir = layout.render_dir(candidate_recipe_id)
    rdir.mkdir(parents=True, exist_ok=True)

    # --- global_gain child (float32 master) ---
    gained, applied_db = global_gain(master, target_dbfs)
    gain_id = delivery_node_id("global_gain", [candidate_recipe_id],
                               {"target_micro_dbfs": int(round(target_dbfs * 1000)), "mode": "peak"}, code_commit)
    sf.write(str(rdir / "master_gain.f32.wav"), gained, sr, subtype="FLOAT")
    nodes = [{
        "recipe_id": gain_id, "operation": "global_gain", "parents": [candidate_recipe_id],
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(gained, sr, layout_names),
        "sample_format": "float32", "applied_gain_db": round(applied_db, 4),
        "file": "master_gain.f32.wav",
    }]

    # --- dither_quantize children (from the float master, TPDF, once) ---
    for bits, fname in ((24, "delivery_pcm24.wav"), (16, "delivery_pcm16.wav")):
        dithered = tpdf_dither_float(gained, bits, seed=dither_seed)
        sf.write(str(rdir / fname), dithered, sr, subtype=f"PCM_{bits}")
        did = delivery_node_id("dither_quantize", [gain_id],
                               {"bits": bits, "dither": "tpdf", "seed": dither_seed}, code_commit)
        nodes.append({"recipe_id": did, "operation": "dither_quantize", "parents": [gain_id],
                      "sample_format": f"pcm_s{bits}", "dither": "tpdf",
                      "container_sha256": _sha256_file(rdir / fname), "file": fname})

    # --- encode_delivery child (AAC from the float master) ---
    encode_ok = True
    try:
        encode_aac(gained, sr, rdir / "delivery.m4a", bitrate)
    except Exception:
        encode_ok = False
    if encode_ok:
        eid = delivery_node_id("encode_delivery", [gain_id], {"codec": "aac", "bitrate": bitrate}, code_commit)
        nodes.append({"recipe_id": eid, "operation": "encode_delivery", "parents": [gain_id],
                      "sample_format": "aac", "bitrate": bitrate,
                      "container_sha256": _sha256_file(rdir / "delivery.m4a"), "file": "delivery.m4a"})

    with Manifest(layout.manifest_sqlite) as man:
        for n in nodes:
            man.upsert_candidate({
                "recipe_id": n["recipe_id"], "operation": n["operation"], "parents": n["parents"],
                "artifact_pcm_sha256": n.get("artifact_pcm_sha256") or n.get("container_sha256"),
                "sample_rate_hz": sr, "channels": layout_names, "frames": int(gained.shape[0]),
                "sample_format": n["sample_format"], "status": "complete", "created_at": _now(),
            })
        man.set_state(layout.track_id, "COMPLETE")

    report = {
        "schema": "audio-extract/repro/v1",
        "track_id": layout.track_id,
        "source": {k: source_record.get(k) for k in
                   ("source_blob_sha256", "input_pcm_sha256", "sample_rate_hz", "channel_layout", "frames")},
        "finalist_candidate": candidate_recipe_id,
        "delivery_nodes": nodes,
        "software": identity.execution_fingerprint(),
        "created_at": _now(),
    }
    (rdir / "repro.json").write_text(json.dumps(report, indent=2))
    return report
