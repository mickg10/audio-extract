from pathlib import Path

import pytest

from audio_extract import panel as panel_mod

PANEL_PATH = Path(__file__).resolve().parent.parent / "configs" / "panel.yaml"


def test_seed_panel_loads():
    p = panel_mod.load_panel(PANEL_PATH)
    assert p.schema == panel_mod.PANEL_SCHEMA
    assert len(p.models) >= 6
    ids = {m.id for m in p.models}
    assert {"viperx_1297", "kim_vocal_2", "mdx23c_hq"} <= ids


def test_needs_import_flag():
    p = panel_mod.load_panel(PANEL_PATH)
    # viperx config sha256 is a REQUIRED_AT_IMPORT placeholder -> needs import.
    assert p.by_id("viperx_1297").needs_import is True
    # mdx23c has a pinned checkpoint hash and no unresolved config hash.
    assert p.by_id("mdx23c_hq").needs_import is False


def test_missing_checkpoint_hash_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema: audio-extract/model-panel/v1\n"
        "models:\n"
        "  - id: nohash\n"
        "    checkpoint:\n"
        "      filename: x.ckpt\n"
    )
    with pytest.raises(panel_mod.PanelError):
        panel_mod.load_panel(bad)


def test_duplicate_id_rejected(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "schema: audio-extract/model-panel/v1\n"
        "models:\n"
        "  - id: dup\n"
        "    checkpoint: {filename: a.ckpt, sha256: aa}\n"
        "  - id: dup\n"
        "    checkpoint: {filename: b.ckpt, sha256: bb}\n"
    )
    with pytest.raises(panel_mod.PanelError):
        panel_mod.load_panel(bad)


def test_wrong_schema_rejected(tmp_path):
    bad = tmp_path / "wrong.yaml"
    bad.write_text("schema: something/else\nmodels: []\n")
    with pytest.raises(panel_mod.PanelError):
        panel_mod.load_panel(bad)
