"""Exact executed model lock (v2.1 WP1 / §6).

Every candidate recipe must identify the exact bytes and effective configuration
actually executed — a registry alias or related HF checkpoint is not enough. The
lock file (``configs/model-lock.json``) maps a **logical id** to an
``executed-model-bundle`` whose identity is the SHA-256 over its canonical JSON
(file byte-hashes included). ONNX and CKPT variants are separate bundles; no
inferred equivalence. Lock entries are immutable: re-importing a logical id with
different bytes is an error, not an update.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import canon

BUNDLE_SCHEMA = "audio-extract/executed-model-bundle/v1"
_BUNDLE_DOMAIN = b"audio-extract-model-bundle-v1\x00"


class ModelLockError(ValueError):
    """Unresolved identity, hash mismatch, or an attempt to mutate a lock entry."""


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_sha256(bundle: dict[str, Any]) -> str:
    body = {k: v for k, v in bundle.items() if k != "bundle_sha256"}
    return hashlib.sha256(_BUNDLE_DOMAIN + canon.canonicalize(body)).hexdigest()


def build_bundle(*, logical_id: str, family: str, target_stem: str,
                 registry_alias: str, files: list[tuple[str, Path]],
                 adapter_version: str, adapter_revision: str,
                 effective_defaults: dict[str, Any]) -> dict[str, Any]:
    """Hash every identity-bearing file and assemble the bundle. Refuses missing
    files and placeholder hashes by construction (hashes are computed, not typed)."""
    file_entries = []
    for role, path in files:
        p = Path(path)
        if not p.is_file():
            raise ModelLockError(f"bundle {logical_id!r}: {role} file missing: {p}")
        file_entries.append({"role": role, "filename": p.name, "sha256": _sha256_file(p)})
    bundle: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "logical_id": logical_id,
        "family": family,
        "target_stem": target_stem,
        "adapter": {"name": "audio-separator", "version": adapter_version,
                    "adapter_revision": adapter_revision},
        "registry": {"alias": registry_alias},
        "files": file_entries,
        "effective_defaults": effective_defaults,
    }
    bundle["bundle_sha256"] = bundle_sha256(bundle)
    return bundle


class ModelLock:
    """The authoritative model registry for execution. When a lock file exists,
    ``panel render`` / ``candidate render`` resolve logical ids (or aliases)
    through it and abort on any mismatch (§6.7 exit gate)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            doc = json.loads(self.path.read_text())
            self._entries = {b["logical_id"]: b for b in doc.get("bundles", [])}

    @property
    def exists(self) -> bool:
        return bool(self._entries)

    def logical_ids(self) -> list[str]:
        return sorted(self._entries)

    def add(self, bundle: dict[str, Any]) -> None:
        lid = bundle["logical_id"]
        existing = self._entries.get(lid)
        if existing is not None:
            if existing["bundle_sha256"] != bundle["bundle_sha256"]:
                raise ModelLockError(
                    f"lock entry {lid!r} is immutable; re-import with different bytes "
                    f"({existing['bundle_sha256'][:12]} != {bundle['bundle_sha256'][:12]}) "
                    "requires a NEW logical_id")
            return  # identical re-import is a no-op
        self._entries[lid] = bundle

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"schema": "audio-extract/model-lock/v1",
               "bundles": [self._entries[k] for k in sorted(self._entries)]}
        self.path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def resolve(self, name: str) -> dict[str, Any]:
        """Resolve a logical id (preferred) or a registry alias to its bundle."""
        if name in self._entries:
            return self._entries[name]
        matches = [b for b in self._entries.values() if b["registry"]["alias"] == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ModelLockError(f"alias {name!r} is ambiguous across bundles: "
                                 f"{[b['logical_id'] for b in matches]}")
        raise ModelLockError(f"model {name!r} is not in the lock ({self.path}); "
                             "run `audio-extract models import` first")

    def verify(self, name: str, model_dir: str | Path) -> dict[str, Any]:
        """Resolve and re-hash the on-disk files against the lock (§6.7). Returns
        the bundle on success; raises on missing files or hash mismatch."""
        bundle = self.resolve(name)
        mdir = Path(model_dir)
        for f in bundle["files"]:
            p = mdir / f["filename"]
            if not p.is_file():
                raise ModelLockError(f"{bundle['logical_id']}: locked file missing on disk: {p}")
            actual = _sha256_file(p)
            if actual != f["sha256"]:
                raise ModelLockError(
                    f"{bundle['logical_id']}: hash mismatch for {f['filename']}: "
                    f"locked {f['sha256'][:12]} != on-disk {actual[:12]}")
        return bundle


def default_logical_id(registry_alias: str) -> str:
    """Derive a stable logical id from a registry filename, keeping the artifact
    kind visible (onnx vs ckpt stay distinct bundles)."""
    stem = registry_alias.rsplit(".", 1)[0].lower()
    kind = registry_alias.rsplit(".", 1)[-1].lower() if "." in registry_alias else "bin"
    slug = "".join(c if c.isalnum() else "_" for c in stem).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{slug}_{kind}"


def guess_family(registry_alias: str) -> str:
    s = registry_alias.lower()
    if "mel_band" in s or "melband" in s or "mel-band" in s:
        return "mel_band_roformer"
    if "bs_roformer" in s or "bs-roformer" in s or "roformer" in s:
        return "bs_roformer"
    if "mdx23c" in s:
        return "mdx23c"
    if "htdemucs" in s or "demucs" in s:
        return "htdemucs"
    if s.endswith(".onnx"):
        return "mdx"
    return "unknown"
