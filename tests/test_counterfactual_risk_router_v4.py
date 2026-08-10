import dataclasses

import numpy as np
import pytest

from audio_extract.counterfactual_risk_router_v3 import (
    CandidateIdentityV3,
    MetricCoverageV3,
    RawParentBypassCertificateV3,
    RiskRouterConfigV3,
    VerifiedModelBundleV3,
    candidate_panel_sha256,
)
from audio_extract.counterfactual_risk_router_v4 import (
    CalibrationCertificateV4,
    QueryCalibrationScopeV4,
    RiskPanelIdentityV4,
    RiskPanelV4,
    RouteDecisionV4,
    exhaustive_optimal_routes_v4,
    solve_risk_route_v4,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def git(digit: str) -> str:
    return digit * 40


def candidate(recipe: str, pcm: str):
    return CandidateIdentityV3(sha(recipe), sha(pcm))


def bundle(seed: int):
    return VerifiedModelBundleV3(
        weights_sha256="sha256:" + f"{seed:064x}",
        config_sha256="sha256:" + f"{seed + 1:064x}",
        adapter_sha256="sha256:" + f"{seed + 2:064x}",
        code_commit=git("a"),
    )


def identity(**changes):
    candidates = (candidate("1", "2"), candidate("3", "4"))
    risk_bundle = bundle(5)
    query_bundle = bundle(8)
    query_scope = QueryCalibrationScopeV4(
        query_encoder_bundle_sha256=query_bundle.sha256,
        query_quality_stratum_sha256=sha("b"),
        selected_quantile_level=0.95,
        calibration_algorithm_sha256=sha("c"),
    )
    certificate = CalibrationCertificateV4(
        risk_model_bundle_sha256=risk_bundle.sha256,
        feature_contract_sha256=sha("d"),
        candidate_panel_sha256=candidate_panel_sha256(candidates),
        calibration_policy_sha256=sha("e"),
        calibration_data_manifest_sha256=sha("f"),
        split_manifest_sha256=sha("0"),
        query_scope=query_scope,
        metric_coverages=(
            MetricCoverageV3("voice", 0.9, 0.92, 20),
            MetricCoverageV3("hole", 0.9, 0.91, 20),
        ),
    )
    parent_certificate = RawParentBypassCertificateV3(
        source_pcm_sha256=sha("a"),
        parent=candidates[0],
        parent_recipe_input_pcm_sha256=sha("a"),
        parent_recipe_id=candidates[0].recipe_id,
        parent_artifact_pcm_sha256=candidates[0].artifact_pcm_sha256,
        raw_candidate_copy_verified=True,
        no_transform_after_candidate=True,
        verifier_commit=git("b"),
    )
    values = dict(
        source_pcm_sha256=sha("a"),
        candidates=candidates,
        metric_names=("voice", "hole"),
        feature_contract_sha256=sha("d"),
        risk_model_bundle=risk_bundle,
        calibration_policy_sha256=sha("e"),
        calibration_certificate=certificate,
        conservative_parent=candidates[0],
        conservative_parent_certificate=parent_certificate,
        query_encoder_bundle=query_bundle,
        query_condition_sha256=sha("1"),
        query_quality_stratum_sha256=sha("b"),
        selected_quantile_level=0.95,
    )
    values.update(changes)
    return RiskPanelIdentityV4(**values)


def panel(**changes):
    values = dict(
        identity=identity(),
        upper=np.asarray([[[[0.1, 0.1], [0.7, 0.7]]]], dtype=np.float64),
        available=np.ones((1, 1, 2, 2), dtype=bool),
    )
    values.update(changes)
    return RiskPanelV4(**values)


def config(**changes):
    values = dict(
        critical_thresholds=(("voice", 1.0), ("hole", 1.0)),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
        uniqueness_tolerance=1e-9,
    )
    values.update(changes)
    return RiskRouterConfigV3(**values)


def test_calibration_is_bound_to_query_encoder_bundle():
    value = identity()
    other = bundle(12)
    with pytest.raises(ValueError, match="query encoder bundle mismatch"):
        dataclasses.replace(value, query_encoder_bundle=other).validate()


def test_calibration_is_bound_to_query_quality_stratum():
    value = identity()
    with pytest.raises(ValueError, match="query quality stratum mismatch"):
        dataclasses.replace(value, query_quality_stratum_sha256=sha("2")).validate()


def test_calibration_is_bound_to_selected_quantile():
    value = identity()
    with pytest.raises(ValueError, match="selected quantile mismatch"):
        dataclasses.replace(value, selected_quantile_level=0.9).validate()


def test_query_scope_changes_panel_identity():
    first = identity()
    changed_scope = dataclasses.replace(
        first.calibration_certificate.query_scope,
        query_quality_stratum_sha256=sha("2"),
    )
    changed_certificate = dataclasses.replace(
        first.calibration_certificate,
        query_scope=changed_scope,
    )
    second = dataclasses.replace(
        first,
        query_quality_stratum_sha256=sha("2"),
        calibration_certificate=changed_certificate,
    )
    first.validate(); second.validate()
    assert first.sha256 != second.sha256


def test_route_json_contains_exact_labels_and_content_hash():
    value = solve_risk_route_v4(panel(), config())
    payload = value.to_dict()
    assert payload["labels"] == [[0]]
    assert payload["labels_shape"] == [1, 1]
    assert payload["labels_dtype"] == "<i4"
    assert payload["labels_sha256"].startswith("sha256:")


def test_exhaustive_mirror_uses_true_minimum_before_tolerance():
    tolerance = 1e-9
    costs = np.asarray([[[0.0, 0.75 * tolerance, 1.5 * tolerance]]])
    feasible = np.ones_like(costs, dtype=bool)
    best, routes = exhaustive_optimal_routes_v4(
        costs,
        feasible,
        config(critical_thresholds=(("voice", 1.0),), uniqueness_tolerance=tolerance),
    )
    assert best == 0.0
    assert [int(row[0, 0]) for row in routes] == [0, 1]


def test_exhaustive_mirror_sorts_equal_routes_lexicographically():
    costs = np.zeros((1, 2, 2), dtype=np.float64)
    feasible = np.ones_like(costs, dtype=bool)
    _, routes = exhaustive_optimal_routes_v4(
        costs,
        feasible,
        config(critical_thresholds=(("voice", 1.0),), uniqueness_tolerance=0.0),
    )
    assert [tuple(row.ravel()) for row in routes] == [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]


def test_v4_solver_plan_identity_changes_with_query_stratum():
    first_panel = panel()
    first = solve_risk_route_v4(first_panel, config())
    original = first_panel.identity
    scope = dataclasses.replace(
        original.calibration_certificate.query_scope,
        query_quality_stratum_sha256=sha("2"),
    )
    cert = dataclasses.replace(original.calibration_certificate, query_scope=scope)
    second_identity = dataclasses.replace(
        original,
        query_quality_stratum_sha256=sha("2"),
        calibration_certificate=cert,
    )
    second = solve_risk_route_v4(
        dataclasses.replace(first_panel, identity=second_identity),
        config(),
    )
    assert first.labels.tolist() == second.labels.tolist()
    assert first.routing_plan_sha256 != second.routing_plan_sha256


def test_decision_refuses_non_grid_labels_at_json_boundary():
    source = solve_risk_route_v4(panel(), config())
    broken = dataclasses.replace(source, labels=np.asarray([0, 1]))
    with pytest.raises(Exception, match="two-dimensional"):
        broken.to_dict()


def test_calibration_metric_order_is_identity_bearing():
    value = identity()
    reversed_certificate = dataclasses.replace(
        value.calibration_certificate,
        metric_coverages=tuple(
            reversed(value.calibration_certificate.metric_coverages)
        ),
    )
    with pytest.raises(ValueError, match="in order"):
        dataclasses.replace(
            value,
            calibration_certificate=reversed_certificate,
        ).validate()
