"""Global cross-scene candidate identity for certified oracle routing.

Per-work decoded-PCM deduplication is insufficient for a route plan that is
reused on controls: two logical estimators can be byte-identical on one scene
and different on another. This module assigns vertices from each alias's full
PCM signature across every required voiced and control scene.

Aliases collapse only when their entire cross-scene signatures are identical.
The resulting vertex order and alias partition are therefore stable everywhere,
even when two global vertices happen to share one scene-local artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json

from .oracle_routing_basis_v2 import DeduplicatedCandidate


GLOBAL_BASIS_SCHEMA = "audio-extract/oracle-routing-global-basis/v2"
DEFAULT_ALIAS_ORDER = (
    "median_mdx_mel_bs",
    "geomedian_mdx_mel_bs",
    "convex_fusion_uniform",
    "residual_mdx23c",
    "residual_melband",
    "residual_bs_roformer",
    "htdemucs_04573f0d",
    "htdemucs_955717e8",
)


class GlobalBasisError(ValueError):
    pass


@dataclass(frozen=True)
class GlobalVertex:
    index: int
    canonical_name: str
    aliases: tuple[str, ...]
    scene_pcm_sha256: tuple[tuple[str, str], ...]
    signature_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["scene_pcm_sha256"] = {
            scene: digest for scene, digest in self.scene_pcm_sha256
        }
        value["schema"] = GLOBAL_BASIS_SCHEMA
        return value


@dataclass(frozen=True)
class GlobalBasis:
    scenes: tuple[str, ...]
    required_aliases: tuple[str, ...]
    vertices: tuple[GlobalVertex, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_BASIS_SCHEMA,
            "scenes": list(self.scenes),
            "required_aliases": list(self.required_aliases),
            "vertices": [vertex.to_dict() for vertex in self.vertices],
        }


def _sha_identity(value: str, label: str) -> str:
    result = str(value).lower()
    if not result.startswith("sha256:") or len(result) != 71:
        raise GlobalBasisError(f"{label} is not sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise GlobalBasisError(f"{label} is not sha256:<64 hex>") from exc
    return result


def _alias_map(
    scene: str,
    rows: Sequence[DeduplicatedCandidate],
) -> dict[str, DeduplicatedCandidate]:
    if not rows:
        raise GlobalBasisError(f"scene {scene!r} has no candidate rows")
    result: dict[str, DeduplicatedCandidate] = {}
    for row in rows:
        _sha_identity(
            row.artifact_pcm_sha256,
            f"{scene}/{row.canonical_name}",
        )
        for alias in row.aliases:
            if not alias:
                raise GlobalBasisError(
                    f"scene {scene!r} contains an empty alias"
                )
            if alias in result:
                raise GlobalBasisError(
                    f"scene {scene!r} resolves alias {alias!r} more than once"
                )
            result[alias] = row
    return result


def _signature_sha(
    aliases: Sequence[str],
    signature: Sequence[tuple[str, str]],
) -> str:
    payload = json.dumps({
        "schema": GLOBAL_BASIS_SCHEMA,
        "aliases": sorted(aliases),
        "scene_pcm_sha256": list(signature),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_global_basis(
    scenes: Mapping[str, Sequence[DeduplicatedCandidate]],
    *,
    required_aliases: Sequence[str] = DEFAULT_ALIAS_ORDER,
    alias_order: Sequence[str] = DEFAULT_ALIAS_ORDER,
) -> GlobalBasis:
    """Build one stable vertex panel from full alias signatures.

    ``scenes`` must contain every voiced work and every control scene that will
    receive a reused plan. Scene keys are sorted into the identity, so mapping
    insertion order cannot change the panel.
    """

    scene_names = tuple(sorted(str(scene) for scene in scenes))
    if not scene_names or len(scene_names) != len(scenes):
        raise GlobalBasisError("scene names must be unique and non-empty")
    required = tuple(str(alias) for alias in required_aliases)
    if not required or any(not alias for alias in required):
        raise GlobalBasisError("required aliases must be non-empty")
    if len(set(required)) != len(required):
        raise GlobalBasisError("required aliases must be unique")
    priority = {
        str(alias): index for index, alias in enumerate(alias_order)
    }
    if set(required) - set(priority):
        raise GlobalBasisError(
            "alias_order omits one or more required aliases"
        )

    per_scene = {
        scene: _alias_map(scene, scenes[scene])
        for scene in scene_names
    }
    for scene, aliases in per_scene.items():
        missing = set(required) - set(aliases)
        if missing:
            raise GlobalBasisError(
                f"scene {scene!r} lacks required aliases: {sorted(missing)}"
            )

    alias_signatures: dict[str, tuple[tuple[str, str], ...]] = {}
    for alias in required:
        alias_signatures[alias] = tuple(
            (
                scene,
                _sha_identity(
                    per_scene[scene][alias].artifact_pcm_sha256,
                    f"{scene}/{alias}",
                ),
            )
            for scene in scene_names
        )

    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for alias, signature in alias_signatures.items():
        groups.setdefault(signature, []).append(alias)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: min(priority[alias] for alias in item[1]),
    )
    vertices = []
    for index, (signature, aliases) in enumerate(ordered_groups):
        ordered_aliases = tuple(
            sorted(aliases, key=priority.__getitem__)
        )
        canonical = ordered_aliases[0]
        vertices.append(GlobalVertex(
            index=index,
            canonical_name=canonical,
            aliases=ordered_aliases,
            scene_pcm_sha256=signature,
            signature_sha256=_signature_sha(
                ordered_aliases, signature
            ),
        ))
    return GlobalBasis(scene_names, required, tuple(vertices))


def materialize_scene_basis(
    basis: GlobalBasis,
    scene: str,
    rows: Sequence[DeduplicatedCandidate],
) -> tuple[DeduplicatedCandidate, ...]:
    """Return scene-local artifacts in the stable global vertex order."""

    if scene not in basis.scenes:
        raise GlobalBasisError(
            f"scene {scene!r} is outside the global basis"
        )
    aliases = _alias_map(scene, rows)
    result = tuple(
        aliases[vertex.canonical_name] for vertex in basis.vertices
    )
    for vertex, row in zip(basis.vertices, result):
        expected = dict(vertex.scene_pcm_sha256)[scene]
        if row.artifact_pcm_sha256 != expected:
            raise GlobalBasisError(
                f"scene {scene!r} vertex {vertex.index} PCM changed: "
                f"{row.artifact_pcm_sha256} != {expected}"
            )
    return result
