"""audio-extract v2 deterministic core.

This package is the *deterministic tool layer* the v2 design calls for: canonical
candidate identity, an immutable content-addressed DAG, a passage miner, an
objective QA metric bank, and a JSON CLI. A bounded Pi + DeepSeek conductor sits
*above* these tools (see docs/v2/); the tools themselves never call an LLM.

v1 (`convert.py`, `web/`) is left untouched; v2 lives entirely under this package.
"""

try:  # single source of truth is pyproject.toml; fall back when not installed
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("audio-extract")
except Exception:  # pragma: no cover
    __version__ = "0.2.0"
