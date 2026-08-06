"""Immutable content-addressed storage layout (docs/v2 §1.6).

```
lib/<track_id>/
  source/       source.json, original.<ext>, canonical.f32.wav
  passages/     passages.v1.json
  candidates/<recipe_id>/  recipe.json, execution.json, output.f32.wav, output.pcm.sha256, metrics.json
  renders/<recipe_id>/     delivery.wav, delivery.m4a
  manifest.sqlite
```

Candidates are **append-only**: a completed ``output.f32.wav`` is never rewritten.
A changed result is a new ``recipe_id`` node. The write helpers enforce this so a
downstream stage cannot silently mutate a parent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _safe_recipe_dirname(recipe_id: str) -> str:
    # recipe_id is "sha256:<hex>"; keep it filesystem-safe and unambiguous.
    return recipe_id.replace(":", "_")


class ImmutableWriteError(RuntimeError):
    """Attempted to overwrite a completed immutable artifact."""


class TrackLayout:
    def __init__(self, lib_root: str | Path, track_id: str):
        self.root = Path(lib_root) / track_id
        self.track_id = track_id

    # --- directories -----------------------------------------------------
    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def passages_dir(self) -> Path:
        return self.root / "passages"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def manifest_sqlite(self) -> Path:
        return self.root / "manifest.sqlite"

    def candidate_dir(self, recipe_id: str) -> Path:
        return self.candidates_dir / _safe_recipe_dirname(recipe_id)

    def render_dir(self, recipe_id: str) -> Path:
        return self.renders_dir / _safe_recipe_dirname(recipe_id)

    def ensure(self) -> "TrackLayout":
        for d in (self.source_dir, self.passages_dir, self.candidates_dir, self.renders_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # --- immutable candidate writes (atomic policy, v2.1 §7.5) ------------
    def write_candidate(
        self,
        recipe_id: str,
        recipe_obj: dict,
        execution: dict,
        output_f32_path: Path | None,
        artifact_pcm_sha256: str,
    ) -> Path:
        """Materialize a candidate directory atomically:

        1. build everything in a temp dir under the same parent;
        2. verify the written audio by re-hashing it against the expected PCM hash;
        3. write the ``COMPLETE`` marker last;
        4. atomically rename into place.

        A crash mid-write leaves only a temp dir, never a half-complete candidate;
        a completed candidate is never overwritten.
        """
        cdir = self.candidate_dir(recipe_id)
        if (cdir / "output.f32.wav").exists():
            raise ImmutableWriteError(
                f"candidate {recipe_id} already has output.f32.wav; candidates are append-only"
            )
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.candidates_dir / f".tmp-{cdir.name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            (tmp / "recipe.json").write_text(json.dumps(recipe_obj, indent=2, sort_keys=True))
            (tmp / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True))
            (tmp / "output.pcm.sha256").write_text(artifact_pcm_sha256 + "\n")
            if output_f32_path is not None:
                Path(output_f32_path).replace(tmp / "output.f32.wav")
                self._verify_pcm(tmp / "output.f32.wav", artifact_pcm_sha256)
            (tmp / "COMPLETE").write_text("")
            if cdir.exists():  # raced by a concurrent writer of the same recipe
                shutil.rmtree(tmp)
                raise ImmutableWriteError(f"candidate {recipe_id} completed concurrently")
            tmp.rename(cdir)
        except Exception:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            raise
        return cdir

    @staticmethod
    def _verify_pcm(wav_path: Path, expected_pcm_sha256: str) -> None:
        """Read back the written audio and verify it hashes to the expected value."""
        import soundfile as sf

        from . import identity

        arr, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        layout = {1: ["FC"], 2: ["FL", "FR"]}.get(arr.shape[1],
                                                  [f"CH{i}" for i in range(arr.shape[1])])
        actual = identity.artifact_pcm_sha256(arr, int(sr), layout, int(arr.shape[0]))
        if actual != expected_pcm_sha256:
            raise ImmutableWriteError(
                f"read-back hash mismatch for {wav_path.name}: "
                f"expected {expected_pcm_sha256[:20]}, got {actual[:20]}")
