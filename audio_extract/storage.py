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

    # --- immutable candidate writes -------------------------------------
    def write_candidate(
        self,
        recipe_id: str,
        recipe_obj: dict,
        execution: dict,
        output_f32_path: Path | None,
        artifact_pcm_sha256: str,
    ) -> Path:
        """Materialize a candidate directory. Refuses to overwrite a completed one.

        ``output_f32_path`` is an already-written float32 WAV to be moved/linked in
        by the caller; here we only manage the sidecar records and the guard.
        """
        cdir = self.candidate_dir(recipe_id)
        out_wav = cdir / "output.f32.wav"
        if out_wav.exists():
            raise ImmutableWriteError(
                f"candidate {recipe_id} already has output.f32.wav; candidates are append-only"
            )
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "recipe.json").write_text(json.dumps(recipe_obj, indent=2, sort_keys=True))
        (cdir / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True))
        (cdir / "output.pcm.sha256").write_text(artifact_pcm_sha256 + "\n")
        if output_f32_path is not None:
            Path(output_f32_path).replace(out_wav)
        return cdir
