"""RFC 8785 JSON Canonicalization Scheme (constrained subset).

The v2 recipe identity is ``SHA256("audio-extract-recipe-v2\\0" + JCS(recipe))``,
so canonicalization must be byte-exact and stable. We implement the subset of
RFC 8785 that the recipe schema actually uses and *reject* anything outside it,
rather than silently diverge from the spec:

* objects with string keys, sorted by UTF-16 code unit;
* arrays (sequence order preserved — order can change the computation);
* strings (RFC 8785 / ECMAScript ``JSON.stringify`` escaping);
* integers, and floats that are exactly integer-valued;
* booleans and null.

Non-integer floats are refused on purpose: the schema mandates exact quantities
as integers or decimal *strings* (see docs/v2 §1.2), which side-steps the one
genuinely hard part of JCS (ECMAScript shortest-round-trip float formatting) and
keeps identity reproducible across languages.
"""

from __future__ import annotations

import math
from typing import Any


class CanonError(ValueError):
    """A value cannot be deterministically canonicalized under the schema policy."""


# Short escapes mandated by RFC 8785 §3.2.2.2 (ECMAScript JSON.stringify).
_SHORT_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}


def _escape_string(s: str) -> str:
    out = ['"']
    for ch in s:
        cp = ord(ch)
        esc = _SHORT_ESCAPES.get(cp)
        if esc is not None:
            out.append(esc)
        elif cp < 0x20:
            out.append(f"\\u{cp:04x}")
        else:
            # All other code points (including non-ASCII) are emitted verbatim;
            # the final UTF-8 encoding carries them.
            out.append(ch)
    out.append('"')
    return "".join(out)


def _number(value: Any) -> str:
    # bool is a subclass of int — must be handled by the caller before this point.
    if isinstance(value, bool):  # pragma: no cover - guarded by caller
        raise CanonError("bool reached number serializer")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonError(f"non-finite number is not canonicalizable: {value!r}")
        if value.is_integer():
            # -0.0 collapses to 0, matching ECMAScript number formatting.
            return str(int(value))
        raise CanonError(
            f"non-integer float {value!r} is not allowed in a canonical recipe; "
            "represent exact quantities as integers or decimal strings (see docs/v2 §1.2)"
        )
    raise CanonError(f"unsupported number type: {type(value).__name__}")


def _canon(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, dict):
        items = []
        for key in _sorted_keys(value):
            items.append(f"{_escape_string(key)}:{_canon(value[key])}")
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canon(v) for v in value) + "]"
    raise CanonError(f"unsupported type in canonical object: {type(value).__name__}")


def _sorted_keys(obj: dict) -> list[str]:
    keys = list(obj.keys())
    for k in keys:
        if not isinstance(k, str):
            raise CanonError(f"object keys must be strings, got {type(k).__name__}")
    # RFC 8785 sorts by UTF-16 code units; comparing UTF-16-BE byte strings is
    # equivalent to code-unit lexicographic order (correct for non-BMP too).
    return sorted(keys, key=lambda k: k.encode("utf-16-be"))


def canonicalize(value: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 encoding of ``value``."""
    return _canon(value).encode("utf-8")


def canonicalize_str(value: Any) -> str:
    """Return the canonical form as a ``str`` (useful for debugging / display)."""
    return _canon(value)
