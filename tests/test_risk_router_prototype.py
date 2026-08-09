import numpy as np

from audio_extract.risk_router_prototype import (
    confident_cell_mask,
    decode_potts,
    fit_linear_risk_student,
    gather_shared_stereo,
)


def test_counterfactual_heads_calibrate_upper_risks():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(80, 3))
    coefficients = rng.uniform(0.05, 0.3, size=(3, 2, 4))
    risks = np.maximum(np.einsum("nf,fkd->nkd", features, coefficients) + 0.5, 0.0)
    student = fit_linear_risk_student(
        features[:60], risks[:60], features[60:], risks[60:], quantile=0.9
    )
    upper = student.predict_upper(features[60:])
    assert upper.shape == (20, 2, 4)
    assert np.mean(upper >= risks[60:]) >= 0.85


def test_deterministic_potts_route_is_shared_by_stereo_channels():
    # Three bounded defects followed by one secondary distortion head.
    upper = np.full((2, 2, 2, 4), 0.1)
    upper[:, 0, 0, 3] = 0.0
    upper[:, 0, 1, 3] = 1.0
    upper[:, 1, 0, 3] = 1.0
    upper[:, 1, 1, 3] = 0.0
    result = decode_potts(
        upper, [0.5, 0.5, 0.5], [1.0],
        time_switch_cost=0.0, frequency_switch_cost=0.1,
    )
    assert result.decision == "route"
    assert result.route.tolist() == [[0, 1], [0, 1]]
    assert result.plan_id.startswith("sha256:")

    cells = np.zeros((2, 2, 2, 2))
    cells[:, :, 0] = [10.0, 11.0]
    cells[:, :, 1] = [20.0, 21.0]
    selected = gather_shared_stereo(cells, result.route)
    assert selected.tolist() == [
        [[10.0, 11.0], [20.0, 21.0]],
        [[10.0, 11.0], [20.0, 21.0]],
    ]


def test_near_tie_is_masked_and_forces_conservative_fallback():
    exact = np.array([[[[0.1, 0.1, 0.1, 0.20], [0.1, 0.1, 0.1, 0.205]]]])
    confidence = confident_cell_mask(
        exact, [0.5, 0.5, 0.5], [1.0], minimum_margin=0.01
    )
    assert confidence.tolist() == [[False]]
    result = decode_potts(
        exact, [0.5, 0.5, 0.5], [1.0],
        time_switch_cost=0.1, frequency_switch_cost=0.1,
        fallback_candidate=1, confidence_mask=confidence,
    )
    assert result.decision == "abstain"
    assert result.reason == "near_tie_or_low_confidence"
    assert result.fallback_route.tolist() == [[1]]
    unmasked = decode_potts(
        exact, [0.5, 0.5, 0.5], [1.0],
        time_switch_cost=0.1, frequency_switch_cost=0.1, fallback_candidate=1,
    )
    assert unmasked.decision == "route"
    assert unmasked.plan_id != result.plan_id


def test_no_feasible_candidate_abstains_instead_of_hiding_failure():
    upper = np.array([[[[0.8, 0.1, 0.1, 0.0], [0.1, 0.9, 0.1, 0.0]]]])
    result = decode_potts(
        upper, [0.5, 0.5, 0.5], [1.0],
        time_switch_cost=0.1, frequency_switch_cost=0.1,
    )
    assert result.decision == "abstain"
    assert result.reason == "no_feasible_candidate"
    assert result.route is None
