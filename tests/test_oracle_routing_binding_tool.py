import importlib.util
from pathlib import Path

import numpy as np


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "oracle_routing_binding.py"
_SPEC = importlib.util.spec_from_file_location("oracle_routing_binding_tool", _TOOL)
assert _SPEC and _SPEC.loader
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def _member(name: str, pcm: str):
    return {"name": name, "artifact_pcm_sha256": pcm}


def test_binding_basis_deduplicates_decoded_pcm_and_retains_alias_indices():
    members = [
        _member("median_mdx_mel_bs", "median"),
        _member("mdx23c", "mdx"),
        _member("melband", "mel"),
        _member("bs_roformer", "bs"),
        _member("geomedian_mdx_mel_bs", "median"),
    ]
    unique, aliases, indices = module._deduplicate_members(members)
    assert [row["name"] for row in unique] == [
        "median_mdx_mel_bs", "mdx23c", "melband", "bs_roformer"
    ]
    assert aliases["geomedian_mdx_mel_bs"] == "median_mdx_mel_bs"
    assert indices["geomedian_mdx_mel_bs"] == indices["median_mdx_mel_bs"]


def test_resolution_scaling_is_frozen_and_physical():
    routing_2, certified_2 = module._scaled_configs(2.0)
    routing_half, certified_half = module._scaled_configs(0.5)
    assert routing_2.temporal_switch_penalty == 0.05
    assert routing_half.temporal_switch_penalty == 0.2
    assert certified_2.temporal_weight_smoothness == 0.05
    assert certified_half.temporal_weight_smoothness == 0.8
    assert routing_half.frequency_switch_penalty == routing_2.frequency_switch_penalty


def test_frequency_boundary_diagnostic_is_finite():
    spectrum = np.ones((2, 5, 4), dtype=np.complex64)
    spectrum[:, 2:] *= 2
    result = module._frequency_boundary_check(
        spectrum, ((0, 2), (2, 5))
    )
    assert result["boundary_count"] == 1
    assert np.isfinite(result["max_boundary_over_p99_adjacent"])


def test_parser_accepts_single_work_fragment(tmp_path):
    args = module.build_parser().parse_args([
        "--candidate-manifest", str(tmp_path / "c.jsonl"),
        "--truth-root", str(tmp_path / "truth"),
        "--lib-root", str(tmp_path / "store"),
        "--htdemucs-045", str(tmp_path / "045"),
        "--htdemucs-955", str(tmp_path / "955"),
        "--control-root", str(tmp_path / "controls"),
        "--work", "bologna_verdi",
        "--output", str(tmp_path / "report.json"),
    ])
    assert args.work == ["bologna_verdi"]
