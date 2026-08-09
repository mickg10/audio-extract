import importlib.util
from pathlib import Path


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_classical_gate_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_classical_gate_manifest_tool", _TOOL)
assert _SPEC and _SPEC.loader
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def test_parser_binds_oracle_controls_members_and_output(tmp_path):
    args = module.build_parser().parse_args([
        "--oracle-report", str(tmp_path / "oracle.json"),
        "--truth-root", str(tmp_path / "truth"),
        "--control-root", str(tmp_path / "controls"),
        "--materialized-root", str(tmp_path / "materialized"),
        "--dataset-manifest", str(tmp_path / "dataset.jsonl"),
        "--conservative", "htdemucs_04573f0d",
        "--aggressive", "bs_roformer", "--works", "a", "b",
        "--output", str(tmp_path / "gate.json"),
    ])
    assert args.conservative == "htdemucs_04573f0d"
    assert args.aggressive == "bs_roformer"
    assert args.works == ["a", "b"]
