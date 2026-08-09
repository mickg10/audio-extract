"""Separator adapter over ``audio-separator`` (docs/v2 §1.4, §7.4).

The stock library writes int16 via pydub and normalizes to a 0.9 peak by default.
This adapter takes §7.4 option 2: ``use_soundfile=True`` (float WAV writer),
``normalization_threshold=1.0`` and ``amplification_threshold=0.0`` (no gain
touching), then reads stems back as float64. Identity records the *actual*
checkpoint hash, not the filename.
"""

from __future__ import annotations

import hashlib
import json
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
    """audio-separator writes ``<input>_(Stem)_<model>.wav`` — the STEM lives in the
    parenthesized tag. Matching on the whole filename is a trap: model names like
    ``Kim_Vocal_2`` contain "vocal", which made BOTH stems normalize to "vocals"
    with glob order deciding which audio won (found by the real-opera GPU run:
    the provisional vocal was randomly the instrumental). Parse the tag first;
    fall back to the whole-name heuristic only when no tag exists."""
    import re

    m = re.search(r"\(([^)]+)\)", stem)
    s = (m.group(1) if m else stem).lower()
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
        self._owns_out = output_dir is None
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
        if self._owns_out:
            for p in produced:
                p.unlink(missing_ok=True)
            try:
                self._out.rmdir()
            except OSError:
                pass
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
                          overlap: int, construction: str, target: str, code_commit: str,
                          executed_bundle_id: str | None = None) -> dict:
    """Assemble a v2 recipe object for a separation candidate (docs/v2 §1.3).
    When the model lock resolved this execution, ``executed_bundle_id`` pins the
    exact bundle (weights+config+adapter) into the identity (v2.1 §6)."""
    model_block_extra = {"executed_bundle_id": executed_bundle_id} if executed_bundle_id else {}
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
            **model_block_extra,
        },
        "effective_config": {
            "model_sample_rate_hz": source_record["sample_rate_hz"],
            "overlap_factor": overlap,
        },
        "software": {"audio_extract_commit": code_commit},
    }


def _stft_geometric_median(stack, *, n_fft: int = 2048, hop: int = 512,
                           max_iter: int = 40, tol: float = 1e-4):
    """Stereo-coherent geometric median in complex STFT coordinates.

    Each time-frequency observation is the real vector
    ``[Re(L), Re(R), Im(L), Im(R)]``.  The joint vector makes the construction
    invariant (within numerical tolerance) to an orthonormal L/R↔M/S change of
    basis, unlike independently taking component medians.
    """
    import librosa
    import numpy as np

    values = np.asarray(stack, dtype="float32")
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("STFT geometric median needs (members>=3, frames, channels)")
    if not np.all(np.isfinite(values)):
        raise ValueError("STFT geometric median input contains non-finite samples")
    specs = np.stack([
        np.stack([
            librosa.stft(member[:, channel], n_fft=n_fft, hop_length=hop,
                         win_length=n_fft, window="hann", center=True,
                         pad_mode="constant")
            for channel in range(values.shape[2])
        ], axis=0)
        for member in values
    ], axis=0)  # (K, C, F, T)
    spectral = specs.transpose(0, 2, 3, 1)
    points = np.concatenate((spectral.real, spectral.imag), axis=-1).astype("float64")
    median = points.mean(axis=0)
    eps = 1e-12
    for _ in range(max_iter):
        delta = points - median[None]
        distance = np.sqrt(np.sum(delta * delta, axis=-1)).clip(min=eps)
        weight = 1.0 / distance
        updated = np.einsum("kft,kftd->ftd", weight, points, optimize=True)
        updated /= weight.sum(axis=0)[..., None]
        relative = np.sqrt(np.sum((updated - median) ** 2, axis=-1))
        scale = np.sqrt(np.sum(median ** 2, axis=-1)).clip(min=eps)
        median = updated
        if float(np.max(relative / scale)) <= tol:
            break
    channels = values.shape[2]
    combined = median[..., :channels] + 1j * median[..., channels:]
    output = np.stack([
        librosa.istft(combined[..., channel], hop_length=hop, win_length=n_fft,
                      window="hann", center=True, length=values.shape[1])
        for channel in range(channels)
    ], axis=1)
    if output.shape != values.shape[1:] or not np.all(np.isfinite(output)):
        raise RuntimeError(f"invalid STFT geometric-median output: {output.shape}")
    return output.astype("float64")


def render_ensemble_candidate(layout, source_record: dict, *, member_recipe_ids: list[str],
                              algo: str = "median", weights: list[float] | None = None,
                              code_commit: str = "") -> dict:
    """Combine explicitly parented vocal estimates and render ``M - V_hat``.

    Algorithms are component median, global convex mean, or stereo-coherent
    complex-STFT geometric median.  Recipe identity canonicalizes member/weight
    pairs together.  Completed candidates take a metadata-only cache fast path.
    """
    import tempfile
    from datetime import datetime, timezone

    import numpy as np
    import soundfile as sf

    from . import identity
    from .alignment import apply_alignment, estimate_alignment
    from .manifest import Manifest
    from .storage import ImmutableWriteError

    if algo not in {"median", "mean", "stft_geometric_median"}:
        raise ValueError(f"unknown ensemble algorithm: {algo}")
    if algo in {"median", "stft_geometric_median"} and len(member_recipe_ids) < 3:
        raise ValueError(f"{algo} ensemble needs >=3 members")
    if algo == "mean" and len(member_recipe_ids) < 2:
        raise ValueError("mean ensemble needs >=2 members")
    if len(set(member_recipe_ids)) != len(member_recipe_ids):
        raise ValueError("ensemble member recipe IDs must be unique")
    if weights is not None:
        if len(weights) != len(member_recipe_ids):
            raise ValueError("weights length must match members")
        w = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            raise ValueError("weights must be finite with positive sum")
        w = w / w.sum()
    else:
        w = np.full(len(member_recipe_ids), 1.0 / len(member_recipe_ids))

    pairs = sorted(zip(member_recipe_ids, w.tolist()), key=lambda pair: pair[0])
    member_recipe_ids = [pair[0] for pair in pairs]
    w = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    weights_ppm = [int(round(value * 1_000_000)) for value in w]

    canonical = Path(layout.source_dir) / "canonical.f32.wav"
    mix, sr = sf.read(str(canonical), dtype="float64", always_2d=True)
    sr = int(sr)
    expected_grid = (
        int(source_record["frames"]), int(source_record["sample_rate_hz"]),
        len(source_record["channel_layout"]),
    )
    if (len(mix), sr, mix.shape[1]) != expected_grid:
        raise ValueError(
            f"canonical/source-record grid mismatch: {(len(mix), sr, mix.shape[1])} "
            f"!= {expected_grid}"
        )

    effective_config = {"model_sample_rate_hz": sr,
                        "alignment": "gcc_phat+fractional/v1"}
    construction = "waveform_ensemble"
    if algo == "stft_geometric_median":
        construction = "spectral_ensemble"
        effective_config.update({
            "domain": "complex_stft_stereo_vector",
            "n_fft": 2048,
            "hop_length": 512,
            "window": "hann",
            "center": True,
            "pad_mode": "constant",
            "whitening": "identity",
            "solver": "weiszfeld",
            "solver_max_iter": 40,
            "solver_tolerance": "0.0001",
        })
    identity_payload = json.dumps({
        "algo": algo,
        "member_weight_pairs": list(zip(member_recipe_ids, weights_ppm)),
        "effective_config": effective_config,
    }, sort_keys=True, separators=(",", ":"))
    recipe = {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": {
            "sha256": source_record["input_pcm_sha256"],
            "sample_rate_hz": source_record["sample_rate_hz"],
            "channel_layout": source_record["channel_layout"],
            "frames": source_record["frames"],
            "sample_format": "float32-le-interleaved",
        },
        "operation": {"type": "mixture_minus_source", "target": "instrumental",
                      "construction": construction},
        "model": {"model_id": f"vocal-ensemble-{algo}",
                  "weights_sha256": hashlib.sha256(identity_payload.encode()).hexdigest(),
                  "adapter": "audio-extract-ensemble",
                  "adapter_revision": "ensemble-exact-grid-v2",
                  "members": member_recipe_ids,
                  "algo": algo,
                  "member_weights_ppm": weights_ppm},
        "effective_config": effective_config,
        "software": {"audio_extract_commit": code_commit},
    }
    rid = identity.recipe_id(recipe)

    # Validate parent roles even for a cache hit, but do not reload/alignment-process
    # their full audio unless this recipe must actually be rendered.
    for parent_id in member_recipe_ids:
        parent_dir = layout.candidate_dir(parent_id)
        parent_recipe_path = parent_dir / "recipe.json"
        if not parent_recipe_path.exists() or not (parent_dir / "output.f32.wav").exists():
            raise ValueError(f"ensemble member {parent_id} not in the store")
        parent_op = json.loads(parent_recipe_path.read_text()).get("operation", {})
        if not (parent_op.get("type") == "separate"
                and parent_op.get("construction") == "native_primary"
                and parent_op.get("target") == "vocals"):
            raise ValueError(f"ensemble member {parent_id} is not a native vocal estimate")

    completed = layout.candidate_dir(rid)
    if (completed / "output.f32.wav").exists():
        info = sf.info(completed / "output.f32.wav")
        if (info.frames, info.samplerate, info.channels, info.subtype) != (*expected_grid, "FLOAT"):
            raise ValueError(f"cached ensemble grid mismatch: {info}")
        artifact = (completed / "output.pcm.sha256").read_text().strip()
        execution = json.loads((completed / "execution.json").read_text())
        record = {
            "recipe_id": rid, "operation": "mixture_minus_source",
            "parents": member_recipe_ids, "artifact_pcm_sha256": artifact,
            "sample_rate_hz": sr, "channels": source_record["channel_layout"],
            "frames": info.frames, "sample_format": "float32", "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(), "cached": True,
            "alignments": execution.get("alignments", []),
        }
        with Manifest(layout.manifest_sqlite) as man:
            man.upsert_candidate(record)
        return record

    # load member VOCAL stems; refuse non-vocal members (role discipline)
    members, alignments = [], []
    for parent_id in member_recipe_ids:
        cdir = layout.candidate_dir(parent_id)
        rj = cdir / "recipe.json"
        arr, msr = sf.read(str(cdir / "output.f32.wav"), dtype="float64", always_2d=True)
        if (len(arr), int(msr), arr.shape[1]) != expected_grid:
            raise ValueError(
                f"member {parent_id} grid {(len(arr), int(msr), arr.shape[1])} != {expected_grid}"
            )
        al = estimate_alignment(mix, arr)     # align each member to the mixture grid
        members.append(apply_alignment(arr, al, target_len=mix.shape[0]))
        alignments.append({"recipe_id": parent_id, "delay": al.delay_samples,
                           "polarity": al.polarity, "confidence": round(al.confidence, 4)})

    stack = np.stack(members, axis=0)
    if algo == "median":
        v_ens = np.median(stack, axis=0)
    elif algo == "stft_geometric_median":
        v_ens = _stft_geometric_median(stack)
    else:
        v_ens = np.tensordot(w, stack, axes=(0, 0))
    accomp = mix - v_ens
    if accomp.shape != mix.shape or not np.all(np.isfinite(accomp)):
        raise RuntimeError(f"invalid ensemble output grid/content: {accomp.shape}")
    frames, channels = int(accomp.shape[0]), int(accomp.shape[1])
    ch_layout = (source_record["channel_layout"]
                 if channels == len(source_record["channel_layout"])
                 else [f"CH{i}" for i in range(channels)])
    artifact = identity.artifact_pcm_sha256(accomp.astype("float32"), sr, ch_layout, frames)

    cached = False
    fd, tmp = tempfile.mkstemp(suffix=".f32.wav")
    import os
    os.close(fd)
    sf.write(tmp, accomp.astype("float32"), sr, subtype="FLOAT")
    try:
        layout.write_candidate(rid, recipe,
                               {**identity.execution_fingerprint(),
                                "alignments": alignments}, Path(tmp), artifact)
    except ImmutableWriteError:
        cached = True

    record = {
        "recipe_id": rid, "operation": "mixture_minus_source",
        "parents": member_recipe_ids,
        "artifact_pcm_sha256": artifact, "sample_rate_hz": sr,
        "channels": ch_layout, "frames": frames, "sample_format": "float32",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cached": cached, "alignments": alignments,
    }
    with Manifest(layout.manifest_sqlite) as man:
        man.upsert_candidate(record)   # upsert maps known columns; extras ignored
    return record


def render_residual_candidate(layout, source_record: dict, *, vocal_recipe_id: str,
                              code_commit: str = "") -> dict:
    """Render the exact-grid ``M - V_hat`` child of one immutable vocal candidate.

    The vocal candidate is an explicit recipe parent.  Alignment is therefore a
    recorded part of this child recipe rather than hidden inside a second model
    execution.  Repeated calls return the completed immutable artifact.
    """
    import os
    import tempfile
    from datetime import datetime, timezone

    import soundfile as sf

    from . import identity
    from .alignment import apply_alignment, estimate_alignment
    from .manifest import Manifest
    from .storage import ImmutableWriteError

    canonical = Path(layout.source_dir) / "canonical.f32.wav"
    mix, sr = sf.read(str(canonical), dtype="float64", always_2d=True)
    sr = int(sr)
    expected_grid = (
        int(source_record["frames"]), int(source_record["sample_rate_hz"]),
        len(source_record["channel_layout"]),
    )
    if (len(mix), sr, mix.shape[1]) != expected_grid:
        raise ValueError(
            f"canonical/source-record grid mismatch: {(len(mix), sr, mix.shape[1])} "
            f"!= {expected_grid}"
        )

    parent_dir = layout.candidate_dir(vocal_recipe_id)
    parent_recipe_path = parent_dir / "recipe.json"
    parent_output = parent_dir / "output.f32.wav"
    if not parent_recipe_path.exists() or not parent_output.exists():
        raise ValueError(f"vocal parent {vocal_recipe_id} is not complete")
    parent_recipe = json.loads(parent_recipe_path.read_text())
    parent_op = parent_recipe.get("operation", {})
    if not (
        parent_op.get("type") == "separate"
        and parent_op.get("construction") == "native_primary"
        and parent_op.get("target") == "vocals"
    ):
        raise ValueError(f"candidate {vocal_recipe_id} is not a native vocal parent")
    vocal, vocal_sr = sf.read(str(parent_output), dtype="float64", always_2d=True)
    if (len(vocal), int(vocal_sr), vocal.shape[1]) != expected_grid:
        raise ValueError(
            f"vocal parent grid mismatch: {(len(vocal), int(vocal_sr), vocal.shape[1])} "
            f"!= {expected_grid}"
        )
    alignment = estimate_alignment(mix, vocal)
    aligned = apply_alignment(vocal, alignment, target_len=len(mix))

    recipe = {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": {
            "sha256": source_record["input_pcm_sha256"],
            "sample_rate_hz": sr,
            "channel_layout": source_record["channel_layout"],
            "frames": len(mix),
            "sample_format": "float32-le-interleaved",
        },
        "operation": {"type": "mixture_minus_source", "target": "instrumental",
                      "construction": "mixture_minus_source"},
        "model": {
            "model_id": "single-vocal-residual",
            "weights_sha256": hashlib.sha256(vocal_recipe_id.encode()).hexdigest(),
            "adapter": "audio-extract-residual",
            "adapter_revision": "residual-exact-grid-v1",
            "members": [vocal_recipe_id],
        },
        "effective_config": {
            "model_sample_rate_hz": sr,
            "alignment": "gcc_phat+fractional/v1",
        },
        "software": {"audio_extract_commit": code_commit},
    }
    rid = identity.recipe_id(recipe)
    cdir = layout.candidate_dir(rid)
    cached = (cdir / "output.f32.wav").exists()
    if cached:
        output_info = sf.info(cdir / "output.f32.wav")
        artifact = (cdir / "output.pcm.sha256").read_text().strip()
        if (output_info.frames, output_info.samplerate, output_info.channels,
                output_info.subtype) != (*expected_grid, "FLOAT"):
            raise ValueError(f"cached residual grid mismatch: {output_info}")
    else:
        accompaniment = (mix - aligned).astype("float32")
        artifact = identity.artifact_pcm_sha256(
            accompaniment, sr, source_record["channel_layout"], len(accompaniment)
        )
        fd, tmp_name = tempfile.mkstemp(suffix=".f32.wav")
        os.close(fd)
        sf.write(tmp_name, accompaniment, sr, subtype="FLOAT")
        try:
            layout.write_candidate(
                rid, recipe,
                {**identity.execution_fingerprint(), "alignment": {
                    "recipe_id": vocal_recipe_id,
                    "delay": alignment.delay_samples,
                    "polarity": alignment.polarity,
                    "confidence": round(alignment.confidence, 4),
                    "implementation": "gcc_phat+fractional/v1",
                }},
                Path(tmp_name), artifact,
            )
        except ImmutableWriteError:
            cached = True

    record = {
        "recipe_id": rid,
        "operation": "mixture_minus_source",
        "parents": [vocal_recipe_id],
        "artifact_pcm_sha256": artifact,
        "sample_rate_hz": sr,
        "channels": source_record["channel_layout"],
        "frames": len(mix),
        "sample_format": "float32",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cached": cached,
    }
    with Manifest(layout.manifest_sqlite) as man:
        man.upsert_candidate(record)
    return record


def render_candidate(layout, source_record: dict, *, model_filename: str, target: str,
                     construction: str, overlap: int, code_commit: str,
                     model_dir: str | Path = DEFAULT_MODEL_DIR,
                     executed_bundle_id: str | None = None,
                     expected_sha256: str | None = None) -> dict:
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
        if expected_sha256 and pre_hash.split(":")[-1] != expected_sha256.split(":")[-1]:
            raise RuntimeError(f"model lock hash mismatch for {model_filename!r}: "
                               f"expected {expected_sha256[:16]}, on-disk {pre_hash[:16]}")
        pre_recipe = build_separate_recipe(
            source_record, model_filename=model_filename, model_sha256=pre_hash,
            overlap=overlap, construction=construction, target=target, code_commit=code_commit,
            executed_bundle_id=executed_bundle_id,
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
    if expected_sha256 and out.model_sha256.split(":")[-1] != expected_sha256.split(":")[-1]:
        raise RuntimeError(f"model lock hash mismatch for {model_filename!r} after load: "
                           f"expected {expected_sha256[:16]}, got {out.model_sha256[:16]}")
    sr = out.sr
    mix, _ = sf.read(str(canonical), dtype="float64", always_2d=True)
    expected_grid = (
        int(source_record["frames"]), int(source_record["sample_rate_hz"]),
        len(source_record["channel_layout"]),
    )
    if (len(mix), int(sr), mix.shape[1]) != expected_grid:
        raise ValueError(
            f"separator/source grid mismatch: {(len(mix), int(sr), mix.shape[1])} "
            f"!= {expected_grid}"
        )

    def _stem(name: str):
        return out.stems.get(name)

    if construction == "native_primary":
        arr = _stem(target)
    elif construction == "native_secondary":
        arr = _stem("instrumental" if target == "vocals" else "vocals")
    elif construction in ("mixture_minus_primary", "mixture_minus_source"):
        prim = _stem(target)
        if prim is not None:
            if prim.shape != mix.shape:
                raise ValueError(f"separator stem grid mismatch: {prim.shape} != {mix.shape}")
            arr = mix - prim
        else:
            arr = None
    else:
        raise ValueError(f"unknown construction: {construction!r}")

    if arr is None:  # native stem absent -> residual fallback from whichever stem exists
        prim = _residual_primary(out.stems)
        if prim is None:
            raise RuntimeError("separator produced no usable stem")
        if prim.shape != mix.shape:
            raise ValueError(f"separator fallback stem grid mismatch: {prim.shape} != {mix.shape}")
        arr = mix - prim

    if arr.shape != mix.shape:
        raise ValueError(f"separator output grid mismatch: {arr.shape} != {mix.shape}")

    recipe = build_separate_recipe(
        source_record, model_filename=model_filename, model_sha256=out.model_sha256,
        overlap=overlap, construction=construction, target=target, code_commit=code_commit,
        executed_bundle_id=executed_bundle_id,
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
