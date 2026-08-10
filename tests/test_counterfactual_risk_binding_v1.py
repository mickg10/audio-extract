import dataclasses

import numpy as np
import pytest

from audio_extract.counterfactual_risk_binding_v1 import (
    RiskInferenceBindingError,
    bind_inference_payload_v1,
    candidate_axis_ids,
    solve_bound_risk_route_v1,
)
from audio_extract.counterfactual_risk_model_v2 import RiskPanelPayloadV2
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
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def git(digit: str) -> str:
    return digit * 40


def candidate(recipe: str, pcm: str) -> CandidateIdentityV3:
    return CandidateIdentityV3(sha(recipe), sha(pcm))


def bundle(weights: str, config: str, adapter: str) -> VerifiedModelBundleV3:
    return VerifiedModelBundleV3(
        weights_sha256=sha(weights),
        config_sha256=sha(config),
        adapter_sha256=sha(adapter),
        code_commit=git("a"),
    )


def fixture():
    candidates = (candidate("1", "2"), candidate("3", "4"))
    risk_bundle = bundle("5", "6", "7")
    query_bundle = bundle("8", "9", "a")
    scope = QueryCalibrationScopeV4(
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
        query_scope=scope,
        metric_coverages=(
            MetricCoverageV3("voice", 0.9, 0.92, 20),
            MetricCoverageV3("hole", 0.9, 0.91, 20),
        ),
    )
    parent_certificate = RawParentBypassCertificateV3(
        source_pcm_sha256=sha("f"),
        parent=candidates[0],
        parent_recipe_input_pcm_sha256=sha("f"),
        parent_recipe_id=candidates[0].recipe_id,
        parent_artifact_pcm_sha256=candidates[0].artifact_pcm_sha256,
        raw_candidate_copy_verified=True,
        no_transform_after_candidate=True,
        verifier_commit=git("b"),
    )
    payload = RiskPanelPayloadV2(
        source_pcm_sha256=sha("f"),
        upper=np.asarray([[[[0.1, 0.1], [0.7, 0.7]]]], dtype=np.float64),
        available=np.ones((1, 1, 2, 2), dtype=bool),
        candidate_ids=candidate_axis_ids(candidates),
        metric_names=("voice", "hole"),
        feature_contract_sha256=sha("d"),
        feature_values_sha256=sha("1"),
        hard_feature_available_sha256=sha("2"),
        model_config_sha256=risk_bundle.config_sha256,
        risk_model_state_sha256=risk_bundle.weights_sha256,
        query_encoder_sha256=query_bundle.sha256,
        query_condition_sha256=sha("3"),
        quantile_levels=(0.5, 0.9, 0.95),
        selected_quantile_level=0.95,
        selected_quantile_index=2,
        calibration_policy_sha256=sha("e"),
        availability_probability_threshold=0.9,
    )
    return {
        "payload": payload,
        "candidates": candidates,
        "risk_model_bundle": risk_bundle,
        "query_encoder_bundle": query_bundle,
        "query_quality_stratum_sha256": sha("b"),
        "calibration_certificate": certificate,
        "conservative_parent": candidates[0],
        "conservative_parent_certificate": parent_certificate,
    }


def bind(**changes):
    values = fixture()
    values.update(changes)
    return bind_inference_payload_v1(**values)


def config():
    return RiskRouterConfigV3(
        critical_thresholds=(("voice", 1.0), ("hole", 1.0)),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
    )


def test_exact_payload_binds_and_routes():
    bound = bind()
    bound.validate()
    result = solve_bound_risk_route_v1(bound, config())
    assert result.decision.labels.tolist() == [[0]]
    assert result.bound_risk_panel_sha256 == bound.sha256
    assert result.routing_plan_sha256.startswith("sha256:")
    assert result.to_dict()["decision"]["labels"] == [[0]]


def test_candidate_order_substitution_is_refused():
    values = fixture()
    payload = dataclasses.replace(
        values["payload"],
        candidate_ids=tuple(reversed(values["payload"].candidate_ids)),
    )
    with pytest.raises(RiskInferenceBindingError, match="candidate IDs/order"):
        bind(payload=payload)


def test_candidate_pcm_substitution_changes_slot_identity_and_is_refused():
    values = fixture()
    candidates = list(values["candidates"])
    candidates[1] = CandidateIdentityV3(candidates[1].recipe_id, sha("0"))
    with pytest.raises(RiskInferenceBindingError, match="candidate IDs/order"):
        bind(candidates=tuple(candidates))


def test_post_inference_model_state_or_config_substitution_is_refused():
    values = fixture()
    with pytest.raises(RiskInferenceBindingError, match="weights/state"):
        bind(
            risk_model_bundle=dataclasses.replace(
                values["risk_model_bundle"], weights_sha256=sha("0")
            )
        )
    with pytest.raises(RiskInferenceBindingError, match="bundle config"):
        bind(
            risk_model_bundle=dataclasses.replace(
                values["risk_model_bundle"], config_sha256=sha("0")
            )
        )


def test_query_encoder_substitution_is_refused():
    values = fixture()
    other = bundle("0", "1", "2")
    with pytest.raises(RiskInferenceBindingError, match="query bundle"):
        bind(query_encoder_bundle=other)


def test_calibration_policy_and_quantile_must_match_certificate():
    values = fixture()
    payload = dataclasses.replace(
        values["payload"], calibration_policy_sha256=sha("0")
    )
    with pytest.raises(RiskInferenceBindingError, match="policies differ"):
        bind(payload=payload)
    payload = dataclasses.replace(
        values["payload"],
        selected_quantile_level=0.9,
        selected_quantile_index=1,
    )
    with pytest.raises(ValueError, match="selected quantile mismatch"):
        bind(payload=payload)


def test_metric_order_substitution_is_refused_by_calibration():
    values = fixture()
    payload = dataclasses.replace(
        values["payload"], metric_names=("hole", "voice")
    )
    with pytest.raises(ValueError, match="in order"):
        bind(payload=payload)


def test_bound_panel_snapshots_mutable_payload_arrays():
    values = fixture()
    payload = values["payload"]
    bound = bind(payload=payload)
    before = bound.panel.sha256
    payload.upper[0, 0, 0, 0] = 999.0
    payload.available[0, 0, 0, 0] = False
    assert bound.panel.sha256 == before
    assert bound.panel.upper.flags.writeable is False
    assert bound.panel.available.flags.writeable is False


def test_binding_tamper_is_detected_before_route():
    bound = bind()
    broken = dataclasses.replace(
        bound,
        binding=dataclasses.replace(
            bound.binding, query_condition_sha256=sha("0")
        ),
    )
    with pytest.raises(RiskInferenceBindingError, match="query condition"):
        solve_bound_risk_route_v1(broken, config())


def test_query_condition_is_mandatory_for_v4_bridge():
    values = fixture()
    payload = dataclasses.replace(
        values["payload"],
        query_encoder_sha256=None,
        query_condition_sha256=None,
    )
    with pytest.raises(RiskInferenceBindingError, match="query bundle"):
        bind(payload=payload)
