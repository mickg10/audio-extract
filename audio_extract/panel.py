"""Load and validate ``panel.yaml`` — the seed model panel (docs/v2 §3).

Model identity is the *bundle* (weight hash + config hash + adapter revision +
target), never the checkpoint filename. The loader refuses ambiguous aliases and
surfaces checkpoints whose config hash is still a ``REQUIRED_AT_IMPORT``
placeholder so the importer (not a production run) resolves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PANEL_SCHEMA = "audio-extract/model-panel/v1"
_IMPORT_PLACEHOLDER = "REQUIRED_AT_IMPORT"


class PanelError(ValueError):
    """The panel file is malformed or a model bundle is under-specified."""


@dataclass
class ModelBundle:
    id: str
    family: str
    adapter: str
    target: str
    checkpoint_filename: str
    checkpoint_sha256: str
    constructions: list[str]
    source_revision: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_import(self) -> bool:
        """True while any identity-bearing hash is still a placeholder."""
        if self.checkpoint_sha256 in ("", _IMPORT_PLACEHOLDER):
            return True
        cfg_hash = self.config.get("sha256")
        return cfg_hash == _IMPORT_PLACEHOLDER


@dataclass
class Panel:
    schema: str
    defaults: dict[str, Any]
    models: list[ModelBundle]

    def by_id(self, model_id: str) -> ModelBundle:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)


def load_panel(path: str | Path) -> Panel:
    import yaml  # lazy: keeps yaml optional for pure-identity use

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise PanelError("panel file must be a mapping")
    if raw.get("schema") != PANEL_SCHEMA:
        raise PanelError(f"panel schema must be {PANEL_SCHEMA!r}, got {raw.get('schema')!r}")

    defaults = raw.get("defaults", {}) or {}
    models_raw = raw.get("models") or []
    if not models_raw:
        raise PanelError("panel has no models")

    seen: set[str] = set()
    models: list[ModelBundle] = []
    for entry in models_raw:
        mid = entry.get("id")
        if not mid:
            raise PanelError("a model entry is missing 'id'")
        if mid in seen:
            raise PanelError(f"duplicate model id: {mid!r}")
        seen.add(mid)
        ckpt = entry.get("checkpoint") or {}
        if "sha256" not in ckpt:
            raise PanelError(f"model {mid!r} checkpoint has no sha256 (identity is the hash)")
        models.append(
            ModelBundle(
                id=mid,
                family=entry.get("family", ""),
                adapter=entry.get("adapter", ""),
                target=entry.get("target", ""),
                checkpoint_filename=ckpt.get("filename", ""),
                checkpoint_sha256=ckpt.get("sha256", ""),
                source_revision=ckpt.get("source_revision"),
                constructions=list(entry.get("constructions", [])),
                config=entry.get("config", {}) or {},
                sweep=entry.get("sweep", {}) or {},
            )
        )
    return Panel(schema=raw["schema"], defaults=defaults, models=models)
