"""The v2 recipe object: schema, normalization, and validation.

A *recipe* is the canonical description of a requested computation. Its canonical
hash (:func:`audio_extract.identity.recipe_id`) is the cache key and DAG parent
pointer. Normalization here implements the docs/v2 §1.2 rules: materialize every
effective default (so an omitted default and an explicit default collapse to the
same identity), reject non-finite numbers and unknown fields, and keep exact
quantities as integers.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "audio-extract/recipe/v2"
CANON = "rfc8785+jcs-schema-v1"

# Operation types are first-class DAG nodes (docs/v2 §1.4). Normalization and
# delivery are their own nodes and must not be folded into a separator's identity.
OPERATION_TYPES: frozenset[str] = frozenset(
    {
        "decode",
        "resample",
        "channel_map",
        "separate",
        "mixture_minus_source",
        "sum_stems",
        "waveform_ensemble",
        "spectral_ensemble",
        "cleanup",
        "global_gain",
        "phrase_route",
        "dither_quantize",
        "encode_delivery",
        "measure",
        "judge",
    }
)

# Output constructions understood by the separator/ensemble stages.
CONSTRUCTIONS: frozenset[str] = frozenset(
    {
        "native_primary",
        "native_secondary",
        "mixture_minus_primary",
        "mixture_minus_source",
        "waveform_ensemble",
        "spectral_ensemble",
    }
)

_REQUIRED_TOP_LEVEL = ("schema", "canon", "input_pcm", "operation", "model", "effective_config", "software")
_ALLOWED_TOP_LEVEL = frozenset(_REQUIRED_TOP_LEVEL)

# Effective-config defaults that must be materialized before hashing. These mirror
# the adapter's resolved behavior; an omitted key and an explicit default value
# must produce the same recipe_id.
_EFFECTIVE_CONFIG_DEFAULTS: dict[str, Any] = {
    "batch_size": 1,
    "pitch_shift_semitones": 0,
    "input_peak_normalization": "disabled",
    "output_peak_normalization": "disabled",
    "output_sample_format": "float32",
}


class RecipeError(ValueError):
    """The recipe object is malformed or violates the schema policy."""


def _reject_nonfinite(node: Any, path: str = "") -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, float) and not math.isfinite(node):
        raise RecipeError(f"non-finite number at {path or '<root>'}: {node!r}")
    if isinstance(node, dict):
        for k, v in node.items():
            _reject_nonfinite(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _reject_nonfinite(v, f"{path}[{i}]")


def normalize_recipe(recipe: Any) -> dict[str, Any]:
    """Return a normalized copy with defaults materialized and schema pinned.

    Raises :class:`RecipeError` for unknown top-level fields, missing required
    sections, or non-finite numbers. This is the exact object that gets hashed.
    """
    if not isinstance(recipe, dict):
        raise RecipeError(f"recipe must be an object, got {type(recipe).__name__}")

    unknown = set(recipe) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise RecipeError(f"unknown top-level field(s): {sorted(unknown)}")

    out: dict[str, Any] = {k: recipe[k] for k in recipe}
    out["schema"] = SCHEMA
    out["canon"] = CANON

    for section in ("input_pcm", "operation", "model", "effective_config", "software"):
        if section not in out:
            raise RecipeError(f"missing required section: {section!r}")
        if not isinstance(out[section], dict):
            raise RecipeError(f"section {section!r} must be an object")

    op = out["operation"]
    if op.get("type") not in OPERATION_TYPES:
        raise RecipeError(f"operation.type {op.get('type')!r} not in {sorted(OPERATION_TYPES)}")
    construction = op.get("construction")
    if construction is not None and construction not in CONSTRUCTIONS:
        raise RecipeError(f"operation.construction {construction!r} not in {sorted(CONSTRUCTIONS)}")

    eff = dict(out["effective_config"])
    for key, default in _EFFECTIVE_CONFIG_DEFAULTS.items():
        eff.setdefault(key, default)
    out["effective_config"] = eff

    _reject_nonfinite(out)
    return out


def validate_recipe(recipe: Any) -> list[str]:
    """Return a list of human-readable problems (empty ⇒ valid). Non-raising."""
    problems: list[str] = []
    try:
        normalize_recipe(recipe)
    except RecipeError as exc:
        problems.append(str(exc))
        return problems

    ipcm = recipe.get("input_pcm", {})
    for key in ("sha256", "sample_rate_hz", "frames", "sample_format"):
        if key not in ipcm:
            problems.append(f"input_pcm missing {key!r}")
    if not isinstance(ipcm.get("sample_rate_hz", 0), int):
        problems.append("input_pcm.sample_rate_hz must be an integer")
    if not isinstance(ipcm.get("frames", 0), int):
        problems.append("input_pcm.frames must be an integer")

    model = recipe.get("model", {})
    if not model.get("weights_sha256"):
        problems.append("model.weights_sha256 is required (identity is the weight hash, not the filename)")
    return problems
