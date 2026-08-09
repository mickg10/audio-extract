import importlib.util
from pathlib import Path


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "export_o2_teacher.py"
_SPEC = importlib.util.spec_from_file_location("export_o2_teacher_tool", _TOOL)
assert _SPEC and _SPEC.loader
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def test_parser_requires_frozen_report_metadata_truth_and_new_output(tmp_path):
    args = module.build_parser().parse_args([
        "--routing-report", str(tmp_path / "report.json"),
        "--execution-root", str(tmp_path / "metadata"),
        "--truth-root", str(tmp_path / "truth"),
        "--output-root", str(tmp_path / "teacher"),
    ])
    assert args.output_root == tmp_path / "teacher"
