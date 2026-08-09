import numpy as np
import pytest

from audio_extract.counterfactual_risk_router_v2 import (
    CandidateIdentity,
    RiskPanelIdentityV2,
    RiskPanelV2,
    RiskRouterConfigV2,
    candidate_costs_v2,
    solve_risk_route_v2,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def candidate(recipe: str, pcm: str) -> CandidateIdentity:
    return CandidateIdentity(sha(recipe), sha(pcm))


def identity() -> RiskPanelIdentityV2:
    candidates = (
        candidate("1", "4"),
        candidate("2", "5"),
        candidate("3", "6"),
    )
    return RiskPanelIdentityV2(
        source_pcm_sha256=sha("0"),
        candidates=candidates,
        metric_names=("voice", "hole", "artifact", "secondary"),
        feature_contract_sha256=sha("7"),
        risk_model_bundle_sha256=sha("8"),
        calibration_policy_sha256=sha("9"),
        calibration_certificate_sha256=sha("a"),
        conservative_parent=candidates[0],
        conservative_parent_certificate_sha256=sha("b"),
        query_encoder_bundle_sha256=sha("c"),
        query_condition_sha256=sha("d"),
    )


def config(**kwargs) -> RiskRouterConfigV2:
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
    return RiskRouterConfigV2(**values)


def panel(values, available=None) -> RiskPanelV2:
    values = np.asarray(values, dtype=np.float64)
    if available is None:
        available = np.ones_like(values, dtype=bool)
    return RiskPanelV2(identity(), values, np.asarray(available, dtype=bool))


def test_identity_binds_recipes_pcm_bundles_calibration_and_parent():
    row = identity()
    row.validate()
    assert row.conservative_index == 0
    assert row.sha256.startswith("sha256:")
    broken = RiskPanelIdentityV2(
        **{
            **row.__dict__,
            "conservative_parent": candidate("e", "f"),
        }
    )
    with pytest.raises(ValueError, match="not in the frozen candidate bank"):
        broken.validate()


def test_noncanonical_hash_case_is_refused_before_uniqueness():
    row = identity()
    bad = CandidateIdentity("SHA256:" + "1" * 64, sha("4"))
    altered = RiskPanelIdentityV2(
        **{**row.__dict__, "candidates": (row.candidates[0], bad)}
    )
    with pytest.raises(ValueError, match="canonical"):
        altered.validate()


def test_zero_weight_secondary_metric_is_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        config(secondary_weights=(("secondary", 0.0),)).validate(identity())


def test_unique_complementary_route_is_certified():
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
    result = solve_risk_route_v2(
        panel(values),
        config(temporal_switch_penalty=0.01),
    )
    assert result.status == "ROUTE"
    assert not result.raw_parent_bypass
    assert result.labels[:, 0].tolist() == [0, 1]
    assert result.conservative_parent == identity().candidates[0]


def test_equal_optima_abstain_instead_of_solver_dependent_tie():
    values = np.full((1, 1, 3, 4), 0.2)
    result = solve_risk_route_v2(
        panel(values),
        config(temporal_switch_penalty=0.0, frequency_switch_penalty=0.0),
    )
    assert result.status == "ABSTAIN_NONUNIQUE_OPTIMUM"
    assert result.raw_parent_bypass
    assert result.labels.tolist() == [[0]]


def test_no_feasible_candidate_returns_verified_raw_parent_bypass():
    values = np.asarray([
        [[
            [1.2, 0.1, 0.1, 0.0],
            [0.1, 1.2, 0.1, 0.0],
            [0.1, 0.1, 1.2, 0.0],
        ]]
    ])
    result = solve_risk_route_v2(panel(values), config())
    assert result.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.raw_parent_bypass
    assert result.conservative_parent == identity().conservative_parent
    assert result.conservative_parent_certificate_sha256 == sha("b")


def test_unavailable_nan_is_masked_and_candidate_becomes_infeasible():
    values = np.asarray([
        [[
            [np.nan, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            [0.3, 0.3, 0.3, 0.0],
        ]]
    ])
    available = np.ones_like(values, dtype=bool)
    available[0, 0, 0, 0] = False
    cost, feasible = candidate_costs_v2(panel(values, available), config())
    assert np.isfinite(cost).all()
    assert not feasible[0, 0, 0]
    assert feasible[0, 0, 1]


def test_plan_hash_changes_with_risk_tensor_and_calibration():
    values = np.asarray([
        [[
            [0.1, 0.1, 0.1, 0.0],
            [0.3, 0.3, 0.3, 0.0],
            [0.7, 0.7, 0.7, 0.0],
        ]]
    ])
    first = solve_risk_route_v2(panel(values), config())
    changed = values.copy()
    changed[0, 0, 2, 0] = 0.71
    second = solve_risk_route_v2(panel(changed), config())
    assert first.labels.tolist() == second.labels.tolist()
    assert first.routing_plan_sha256 != second.routing_plan_sha256

    row = identity()
    other_identity = RiskPanelIdentityV2(
        **{**row.__dict__, "calibration_certificate_sha256": sha("e")}
    )
    third = solve_risk_route_v2(
        RiskPanelV2(other_identity, values, np.ones_like(values, bool)),
        config(),
    )
    assert first.routing_plan_sha256 != third.routing_plan_sha256


def test_candidate_recipe_and_pcm_are_both_identity_bearing():
    row = identity()
    changed_candidates = list(row.candidates)
    changed_candidates[1] = CandidateIdentity(
        changed_candidates[1].recipe_id,
        sha("f"),
    )
    altered = RiskPanelIdentityV2(
        **{**row.__dict__, "candidates": tuple(changed_candidates)}
    )
    assert altered.sha256 != row.sha256
