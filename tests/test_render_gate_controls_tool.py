import importlib.util
from pathlib import Path

import pytest


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "render_gate_controls.py"
_SPEC = importlib.util.spec_from_file_location("render_gate_controls_tool", _TOOL)
assert _SPEC and _SPEC.loader
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def test_parser_requires_explicit_control_members_and_paths(tmp_path):
    args = module.build_parser().parse_args([
        "--truth-root", str(tmp_path), "--work", "opera",
        "--control", "vocal_only", "--members", "htdemucs_04573f0d", "bs_roformer",
        "--output-store", str(tmp_path / "store"),
        "--model-dir", str(tmp_path / "models"),
        "--baseline-config", str(tmp_path / "baseline.yaml"),
        "--demucs-045-config", str(tmp_path / "045.yaml"),
        "--demucs-955-config", str(tmp_path / "955.yaml"),
        "--include-composites",
        "--output", str(tmp_path / "report.json"),
    ])
    assert args.control == "vocal_only"
    assert args.members == ["htdemucs_04573f0d", "bs_roformer"]
    assert args.include_composites


def test_parser_refuses_unknown_member(tmp_path):
    with pytest.raises(SystemExit):
        module.build_parser().parse_args([
            "--truth-root", str(tmp_path), "--work", "opera",
            "--control", "no_vocal", "--members", "not-a-model",
            "--output-store", str(tmp_path), "--model-dir", str(tmp_path),
            "--baseline-config", str(tmp_path / "b"),
            "--demucs-045-config", str(tmp_path / "a"),
            "--demucs-955-config", str(tmp_path / "c"),
            "--output", str(tmp_path / "o"),
        ])
