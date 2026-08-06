"""Separator adapter over ``audio-separator`` (docs/v2 §1.4, §7.4).

The stock library writes int16 via pydub and normalizes to a 0.9 peak by default.
This adapter takes §7.4 option 2: ``use_soundfile=True`` (float WAV writer),
``normalization_threshold=1.0`` and ``amplification_threshold=0.0`` (no gain
touching), then reads stems back as float64. Identity records the *actual*
checkpoint hash, not the filename.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import recipe as recipe_mod

DEFAULT_MODEL_DIR = Path.home() / "audio-extract" / "models"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


@dataclass
class SepOutput:
    stems: dict           # normalized_name -> float64 array (n, ch)
    sr: int
    model_filename: str
    model_sha256: str
    overlap: int


def _normalize_stem_name(stem: str) -> str:
    s = stem.lower()
    if "vocal" in s:
        return "vocals"
    if "instrument" in s or "_inst" in s or s.endswith("inst"):
        return "instrumental"
    return s


def _residual_primary(stems: dict):
    """Stem to subtract for a residual, preferring vocals. Uses explicit None
    checks — ``a or b`` on numpy arrays raises 'truth value ambiguous'."""
    prim = stems.get("vocals")
    if prim is None:
        prim = stems.get("instrumental")
    return prim


class Separator:
    """Thin wrapper: one loaded model, float-preserving output."""

    def __init__(self, model_filename: str, overlap: int = 8,
                 model_dir: str | Path = DEFAULT_MODEL_DIR, output_dir: str | Path | None = None):
        from audio_separator.separator import Separator as ASeparator

        self.model_filename = model_filename
        self.overlap = overlap
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="ae_sep_"))
        self._out.mkdir(parents=True, exist_ok=True)
        self._sep = ASeparator(
            model_file_dir=str(self.model_dir),
            output_dir=str(self._out),
            output_format="WAV",
            use_soundfile=True,           # float writer, not pydub int16 (§7.4)
            normalization_threshold=1.0,  # do not normalize output level
            amplification_threshold=0.0,  # do not amplify
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": 1, "overlap": overlap, "pitch_shift": 0},
        )
        self._sep.load_model(model_filename=model_filename)
        mp = self.model_dir / model_filename
        self.model_sha256 = _sha256_file(mp) if mp.exists() else "unknown"

    def separate_file(self, audio_path: str | Path) -> SepOutput:
        import soundfile as sf

        before = set(self._out.glob("*.wav"))
        self._sep.separate(str(audio_path))
        produced = [p for p in self._out.glob("*.wav") if p not in before]
        stems: dict[str, np.ndarray] = {}
        sr = 44100
        for p in produced:  # scan the dir so we capture every stem, not just the return value
            arr, sr = sf.read(str(p), dtype="float64", always_2d=True)
            stems[_normalize_stem_name(p.stem)] = arr
        return SepOutput(stems, int(sr), self.model_filename, self.model_sha256, self.overlap)


def provisional_vocal(canonical_wav: str | Path, model_filename: str = "Kim_Vocal_2.onnx",
                      overlap: int = 8, model_dir: str | Path = DEFAULT_MODEL_DIR):
    """Return ``(vocal[n,ch], accompaniment[n,ch], sr, SepOutput)`` for the miner.
    Provisional stem = kim_vocal_2 (decision recorded in issue #1)."""
    sep = Separator(model_filename, overlap=overlap, model_dir=model_dir)
    out = sep.separate_file(canonical_wav)
    vocal = out.stems.get("vocals")
    accomp = out.stems.get("instrumental")
    if accomp is None and vocal is not None:
        import soundfile as sf

        mix, _ = sf.read(str(canonical_wav), dtype="float64", always_2d=True)
        n = min(len(mix), len(vocal))
        accomp = mix[:n] - vocal[:n]
    return vocal, accomp, out.sr, out


def build_separate_recipe(source_record: dict, *, model_filename: str, model_sha256: str,
                          overlap: int, construction: str, target: str, code_commit: str) -> dict:
    """Assemble a v2 recipe object for a separation candidate (docs/v2 §1.3)."""
    return {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": {
            "sha256": source_record["input_pcm_sha256"],
            "sample_rate_hz": source_record["sample_rate_hz"],
            "channel_layout": source_record["channel_layout"],
            "frames": source_record["frames"],
            "sample_format": "float32-le-interleaved",
        },
        "operation": {"type": "separate", "target": target, "construction": construction},
        "model": {
            "model_id": model_filename,
            "weights_sha256": (model_sha256.split(":")[-1] if model_sha256 and model_sha256 != "unknown"
                               else "unknown"),
            "adapter": "audio-separator",
            "adapter_revision": "audio-separator+audio-extract-adapter-v1",
        },
        "effective_config": {
            "model_sample_rate_hz": source_record["sample_rate_hz"],
            "overlap_factor": overlap,
        },
        "software": {"audio_extract_commit": code_commit},
    }


def render_candidate(layout, source_record: dict, *, model_filename: str, target: str,
                     construction: str, overlap: int, code_commit: str,
                     model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict:
    """Run one separation, build the construction, and write an immutable candidate
    keyed by ``recipe_id`` (docs/v2 §1.4, §1.6). Returns the manifest record.

    Content-addressed: when the checkpoint is already present its hash gives the
    ``recipe_id`` up front, so a cached candidate is returned *without* re-running
    the separator. Completed candidate bytes are never rewritten (append-only).
    """
    import tempfile
    from datetime import datetime, timezone

    import soundfile as sf

    from . import identity
    from .manifest import Manifest
    from .storage import ImmutableWriteError

    canonical = Path(layout.source_dir) / "canonical.f32.wav"

    def _record_from_dir(rid: str) -> dict:
        cdir = layout.candidate_dir(rid)
        artifact = (cdir / "output.pcm.sha256").read_text().strip()
        info = sf.info(str(cdir / "output.f32.wav"))
        return {
            "recipe_id": rid,
            "operation": "separate" if construction.startswith("native") else "mixture_minus_source",
            "parents": [source_record["input_pcm_sha256"]],
            "artifact_pcm_sha256": artifact,
            "sample_rate_hz": int(info.samplerate),
            "channels": source_record["channel_layout"] if info.channels == len(source_record["channel_layout"])
            else [f"CH{i}" for i in range(info.channels)],
            "frames": int(info.frames),
            "sample_format": "float32",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cached": True,
        }

    # Fast path: checkpoint already downloaded -> compute recipe_id and short-circuit.
    ckpt = Path(model_dir) / model_filename
    if ckpt.exists():
        pre_hash = _sha256_file(ckpt)
        pre_recipe = build_separate_recipe(
            source_record, model_filename=model_filename, model_sha256=pre_hash,
            overlap=overlap, construction=construction, target=target, code_commit=code_commit,
        )
        pre_rid = identity.recipe_id(pre_recipe)
        if (layout.candidate_dir(pre_rid) / "output.f32.wav").exists():
            rec = _record_from_dir(pre_rid)
            with Manifest(layout.manifest_sqlite) as man:
                man.upsert_candidate(rec)
            return rec

    sep = Separator(model_filename, overlap=overlap, model_dir=model_dir)
    out = sep.separate_file(canonical)
    if out.model_sha256 == "unknown":  # an unidentified model cannot produce a trustworthy cache key
        raise RuntimeError(f"refusing to render candidate: model {model_filename!r} has no checkpoint hash")
    sr = out.sr
    mix, _ = sf.read(str(canonical), dtype="float64", always_2d=True)

    def _stem(name: str):
        return out.stems.get(name)

    if construction == "native_primary":
        arr = _stem(target)
    elif construction == "native_secondary":
        arr = _stem("instrumental" if target == "vocals" else "vocals")
    elif construction in ("mixture_minus_primary", "mixture_minus_source"):
        prim = _stem(target)
        if prim is not None:
            n = min(len(mix), len(prim))
            arr = mix[:n] - prim[:n]
        else:
            arr = None
    else:
        raise ValueError(f"unknown construction: {construction!r}")

    if arr is None:  # native stem absent -> residual fallback from whichever stem exists
        prim = _residual_primary(out.stems)
        if prim is None:
            raise RuntimeError("separator produced no usable stem")
        n = min(len(mix), len(prim))
        arr = mix[:n] - prim[:n]

    recipe = build_separate_recipe(
        source_record, model_filename=model_filename, model_sha256=out.model_sha256,
        overlap=overlap, construction=construction, target=target, code_commit=code_commit,
    )
    rid = identity.recipe_id(recipe)
    frames, channels = int(arr.shape[0]), int(arr.shape[1])
    ch_layout = (source_record["channel_layout"]
                 if channels == len(source_record["channel_layout"])
                 else [f"CH{i}" for i in range(channels)])
    artifact = identity.artifact_pcm_sha256(arr, sr, ch_layout, frames)

    cand_dir = layout.candidate_dir(rid)
    cached = (cand_dir / "output.f32.wav").exists()
    if not cached:
        fd, tmp = tempfile.mkstemp(suffix=".f32.wav")
        import os
        os.close(fd)
        sf.write(tmp, arr.astype("float32"), sr, subtype="FLOAT")
        try:
            layout.write_candidate(rid, recipe, identity.execution_fingerprint(), Path(tmp), artifact)
        except ImmutableWriteError:
            cached = True

    record = {
        "recipe_id": rid,
        "operation": "separate" if construction.startswith("native") else "mixture_minus_source",
        "parents": [source_record["input_pcm_sha256"]],
        "artifact_pcm_sha256": artifact,
        "sample_rate_hz": sr,
        "channels": ch_layout,
        "frames": frames,
        "sample_format": "float32",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with Manifest(layout.manifest_sqlite) as man:
        man.upsert_candidate(record)
    record["cached"] = cached
    return record
