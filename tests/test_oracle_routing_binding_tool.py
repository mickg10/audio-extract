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


def test_paired_dedup_requires_identity_in_full_and_no_vocal_scopes():
    full = [
        _member(name, "same" if index < 2 else f"full-{index}")
        for index, name in enumerate(module.BASE_ORDER)
    ]
    control = [
        _member(name, "same" if index == 0 else f"control-{index}")
        for index, name in enumerate(module.BASE_ORDER)
    ]
    unique_full, unique_control, aliases, indices = (
        module._deduplicate_paired_members(full, control)
    )
    assert len(unique_full) == len(unique_control) == len(module.BASE_ORDER)
    assert aliases[module.BASE_ORDER[1]] == module.BASE_ORDER[1]
    assert indices[module.BASE_ORDER[1]] == 1

    control[1]["artifact_pcm_sha256"] = "same"
    unique_full, unique_control, aliases, indices = (
        module._deduplicate_paired_members(full, control)
    )
    assert len(unique_full) == len(unique_control) == len(module.BASE_ORDER) - 1
    assert aliases[module.BASE_ORDER[1]] == module.BASE_ORDER[0]
    assert indices[module.BASE_ORDER[1]] == 0


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


def test_solver_recipe_identity_materializes_decimal_strings():
    value = module._identity_values({
        "objective": np.float64(0.125),
        "nested": [1, 0.25, None],
    })
    assert value == {
        "objective": "0.125", "nested": [1, "0.25", None]
    }


def test_route_metric_cache_reuses_labels_and_fidelity(monkeypatch):
    calls = {"labels": 0, "metrics": 0}

    def source_metrics(*_args):
        calls["labels"] += 1
        return {}, ("labels",)

    def exact_metrics(*_args, labels=None):
        calls["metrics"] += 1
        assert labels == ("labels",)
        return {
            "metric": 1.0,
            "transient_loss/v2": 0.1,
            "transient_excess/v2": 0.2,
        }

    monkeypatch.setattr(module, "_pcm", lambda _value: "sha256:cached")
    monkeypatch.setattr(module, "_source_metrics", source_metrics)
    monkeypatch.setattr(module, "exact_metrics", exact_metrics)
    monkeypatch.setattr(
        module,
        "_worst_identifiable",
        lambda _labels: {"available": True, "composite_risk": 0.0},
    )
    monkeypatch.setattr(
        module,
        "hall_tail_preservation_metrics",
        lambda *_args, **_kwargs: {"hall_tail_error_db/v1": 0.0},
    )
    audio = np.zeros((8, 2), dtype="float32")
    cache = {}
    first = module._route_metrics(audio, audio, audio, cache)
    second = module._route_metrics(audio, audio, audio, cache)
    assert first == second
    assert calls == {"labels": 1, "metrics": 1}
