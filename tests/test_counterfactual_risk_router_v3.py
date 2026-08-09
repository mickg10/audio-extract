from types import SimpleNamespace

import numpy as np
import pytest

import audio_extract.counterfactual_risk_router_v3 as router
from audio_extract.counterfactual_risk_router_v3 import (
    CalibrationCertificateV3,
    CandidateIdentityV3,
    MetricCoverageV3,
    RawParentBypassCertificateV3,
    RiskPanelIdentityV3,
    RiskPanelV3,
    RiskRouterConfigV3,
    VerifiedModelBundleV3,
    candidate_costs_v3,
    candidate_panel_sha256,
    exhaustive_optimal_routes_v3,
    solve_risk_route_v3,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def git(digit: str) -> str:
    return digit * 40


def candidate(recipe: str, pcm: str) -> CandidateIdentityV3:
    return CandidateIdentityV3(sha(recipe), sha(pcm))


def bundle(digit: str) -> VerifiedModelBundleV3:
    return VerifiedModelBundleV3(
        weights_sha256=sha(digit),
        config_sha256=sha(chr(ord(digit) + 1) if digit < "9" else "a"),
        adapter_sha256=sha(chr(ord(digit) + 2) if digit < "8" else "b"),
        code_commit=git(digit),
    )


def identity() -> RiskPanelIdentityV3:
    candidates = (
        candidate("1", "4"),
        candidate("2", "5"),
        candidate("3", "6"),
    )
    model = bundle("7")
    query = VerifiedModelBundleV3(
        weights_sha256=sha("a"),
        config_sha256=sha("b"),
        adapter_sha256=sha("c"),
        code_commit=git("d"),
    )
    feature = sha("e")
    policy = sha("f")
    calibration = CalibrationCertificateV3(
        risk_model_bundle_sha256=model.sha256,
        feature_contract_sha256=feature,
        candidate_panel_sha256=candidate_panel_sha256(candidates),
        calibration_policy_sha256=policy,
        calibration_data_manifest_sha256=sha("0"),
        split_manifest_sha256=sha("9"),
        metric_coverages=tuple(
            MetricCoverageV3(name, 0.90, 0.91, 12)
            for name in ("voice", "hole", "artifact", "secondary")
        ),
    )
    parent_certificate = RawParentBypassCertificateV3(
        source_pcm_sha256=sha("8"),
        parent=candidates[0],
        parent_recipe_input_pcm_sha256=sha("8"),
        parent_recipe_id=candidates[0].recipe_id,
        parent_artifact_pcm_sha256=candidates[0].artifact_pcm_sha256,
        raw_candidate_copy_verified=True,
        no_transform_after_candidate=True,
        verifier_commit=git("e"),
    )
    return RiskPanelIdentityV3(
        source_pcm_sha256=sha("8"),
        candidates=candidates,
        metric_names=("voice", "hole", "artifact", "secondary"),
        feature_contract_sha256=feature,
        risk_model_bundle=model,
        calibration_policy_sha256=policy,
        calibration_certificate=calibration,
        conservative_parent=candidates[0],
        conservative_parent_certificate=parent_certificate,
        query_encoder_bundle=query,
        query_condition_sha256=sha("d"),
    )


def config(**kwargs) -> RiskRouterConfigV3:
    values = dict(
        critical_thresholds=(
            ("voice", 1.0),
            ("hole", 1.0),
            ("artifact", 1.0),
        ),
        secondary_weights=(("secondary", 0.1),),
        temporal_switch_penalty=0.03,
        frequency_switch_penalty=0.04,
        milp_time_limit_seconds=30.0,
        milp_relative_gap=0.0,
    )
    values.update(kwargs)
    return RiskRouterConfigV3(**values)


def panel(values, available=None) -> RiskPanelV3:
    values = np.asarray(values, dtype=np.float64)
    if available is None:
        available = np.ones_like(values, dtype=bool)
    return RiskPanelV3(identity(), values, np.asarray(available, dtype=bool))


def test_identity_requires_passed_bound_calibration_and_query():
    row = identity()
    row.validate()
    assert row.calibration_certificate.sha256.startswith("sha256:")
    assert row.conservative_parent_certificate.sha256.startswith("sha256:")

    broken_cal = CalibrationCertificateV3(
        **{
            **row.calibration_certificate.__dict__,
            "metric_coverages": (
                MetricCoverageV3("voice", 0.90, 0.89, 12),
                *row.calibration_certificate.metric_coverages[1:],
            ),
        }
    )
    with pytest.raises(ValueError, match="fails target"):
        RiskPanelIdentityV3(
            **{**row.__dict__, "calibration_certificate": broken_cal}
        ).validate()

    with pytest.raises(ValueError, match="mandatory"):
        RiskPanelIdentityV3(
            **{**row.__dict__, "query_encoder_bundle": None}
        ).validate()


def test_parent_certificate_binds_source_recipe_artifact_and_raw_copy():
    row = identity()
    wrong_source = RawParentBypassCertificateV3(
        **{
            **row.conservative_parent_certificate.__dict__,
            "parent_recipe_input_pcm_sha256": sha("7"),
        }
    )
    with pytest.raises(ValueError, match="exact source PCM"):
        RiskPanelIdentityV3(
            **{
                **row.__dict__,
                "conservative_parent_certificate": wrong_source,
            }
        ).validate()

    transformed = RawParentBypassCertificateV3(
        **{
            **row.conservative_parent_certificate.__dict__,
            "no_transform_after_candidate": False,
        }
    )
    with pytest.raises(ValueError, match="contains a transform"):
        RiskPanelIdentityV3(
            **{
                **row.__dict__,
                "conservative_parent_certificate": transformed,
            }
        ).validate()


def test_certified_config_requires_zero_requested_mip_gap():
    with pytest.raises(ValueError, match="exactly zero"):
        config(milp_relative_gap=0.1).validate(identity())


def test_unique_complementary_route_matches_exhaustive_mirror():
    values = np.asarray([
        [[
            [0.10, 0.10, 0.10, 0.0],
            [0.70, 0.70, 0.70, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
        [[
            [0.70, 0.70, 0.70, 0.0],
            [0.10, 0.10, 0.10, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
    ])
    row = panel(values)
    cfg = config(temporal_switch_penalty=0.01)
    cost, feasible = candidate_costs_v3(row, cfg)
    optimum, routes = exhaustive_optimal_routes_v3(cost, feasible, cfg)
    assert len(routes) == 1
    result = solve_risk_route_v3(row, cfg)
    assert result.status == "ROUTE"
    assert result.labels.tolist() == routes[0].tolist()
    assert result.objective == pytest.approx(optimum)


def test_equal_optima_abstain_instead_of_using_solver_tie_break():
    values = np.full((1, 1, 3, 4), 0.2)
    result = solve_risk_route_v3(
        panel(values),
        config(
            temporal_switch_penalty=0.0,
            frequency_switch_penalty=0.0,
        ),
    )
    assert result.status == "ABSTAIN_NONUNIQUE_OPTIMUM"
    assert result.raw_parent_bypass


def test_no_feasible_candidate_returns_verified_parent():
    values = np.asarray([
        [[
            [1.2, 0.1, 0.1, 0.0],
            [0.1, 1.2, 0.1, 0.0],
            [0.1, 0.1, 1.2, 0.0],
        ]]
    ])
    result = solve_risk_route_v3(panel(values), config())
    assert result.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.raw_parent_bypass
    assert result.conservative_parent == identity().conservative_parent
    assert result.conservative_parent_certificate_sha256 == (
        identity().conservative_parent_certificate.sha256
    )


def test_unavailable_nan_is_masked_and_infeasible():
    values = np.asarray([
        [[
            [np.nan, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            [0.3, 0.3, 0.3, 0.0],
        ]]
    ])
    available = np.ones_like(values, dtype=bool)
    available[0, 0, 0, 0] = False
    cost, feasible = candidate_costs_v3(panel(values, available), config())
    assert np.isfinite(cost).all()
    assert not feasible[0, 0, 0]
    assert feasible[0, 0, 1]


def test_zero_gap_check_rejects_approximate_primary(monkeypatch):
    values = np.asarray([
        [[
            [0.1, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            [0.3, 0.3, 0.3, 0.0],
        ]]
    ])
    fake = SimpleNamespace(
        success=True,
        status=0,
        x=np.asarray([1.0, 0.0, 0.0]),
        fun=0.1,
        mip_gap=1e-4,
        message="approximate",
    )
    monkeypatch.setattr(router, "_solve", lambda problem, cfg: fake)
    result = solve_risk_route_v3(panel(values), config())
    assert result.status == "ABSTAIN_SOLVER_UNCERTIFIED"


def test_alternate_is_discretely_recomputed_before_uniqueness(monkeypatch):
    values = np.asarray([
        [[
            [0.1, 0.1, 0.1, 0.0],
            [0.1, 0.1, 0.1, 0.0],
            [0.9, 0.9, 0.9, 0.0],
        ]]
    ])
    calls = []
    primary = SimpleNamespace(
        success=True,
        status=0,
        x=np.asarray([1.0, 0.0, 0.0]),
        fun=0.1,
        mip_gap=0.0,
        message="optimal",
    )
    alternate = SimpleNamespace(
        success=True,
        status=0,
        x=np.asarray([1e-10, 1.0 - 1e-10, 0.0]),
        fun=0.1 + 1e-11,
        mip_gap=0.0,
        message="optimal",
    )

    def fake(problem, cfg):
        calls.append(problem)
        return primary if len(calls) == 1 else alternate

    monkeypatch.setattr(router, "_solve", fake)
    result = solve_risk_route_v3(
        panel(values),
        config(
            temporal_switch_penalty=0.0,
            frequency_switch_penalty=0.0,
        ),
    )
    assert result.status == "ABSTAIN_NONUNIQUE_OPTIMUM"


def test_exhaustive_mirror_finds_multiple_tied_routes():
    cost = np.zeros((1, 2, 2))
    feasible = np.ones_like(cost, dtype=bool)
    optimum, routes = exhaustive_optimal_routes_v3(
        cost,
        feasible,
        config(
            temporal_switch_penalty=0.0,
            frequency_switch_penalty=0.0,
        ),
    )
    assert optimum == 0.0
    assert len(routes) == 4
    assert routes[0].tolist() == [[0, 0]]
    assert routes[-1].tolist() == [[1, 1]]


def test_plan_hash_changes_with_calibration_and_query_identity():
    values = np.asarray([
        [[
            [0.1, 0.1, 0.1, 0.0],
            [0.3, 0.3, 0.3, 0.0],
            [0.7, 0.7, 0.7, 0.0],
        ]]
    ])
    first = solve_risk_route_v3(panel(values), config())
    row = identity()
    changed_query = RiskPanelIdentityV3(
        **{**row.__dict__, "query_condition_sha256": sha("c")}
    )
    second = solve_risk_route_v3(
        RiskPanelV3(
            changed_query,
            values,
            np.ones_like(values, dtype=bool),
        ),
        config(),
    )
    assert first.routing_plan_sha256 != second.routing_plan_sha256
