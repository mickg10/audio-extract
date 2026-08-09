import inspect

import numpy as np
import pytest

from audio_extract.group_conformal_risk_student import (
    GroupConformalRiskError,
    fit_grouped_linear_risk_student,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def git(digit: str) -> str:
    return digit * 40


def base_data(calibration_groups: int = 9):
    train_x = np.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.5],
        [0.5, 2.0],
    ])
    # Two candidates x three defects, with a simple nonnegative linear relation.
    train_y = np.empty((len(train_x), 2, 3), dtype=np.float64)
    for row, (first, second) in enumerate(train_x):
        train_y[row, 0] = (0.2 + 0.3 * first, 0.1 + 0.2 * second, 0.05)
        train_y[row, 1] = (0.4 + 0.1 * second, 0.3 + 0.1 * first, 0.08)

    calibration_x = np.column_stack((
        np.linspace(0.1, 1.8, calibration_groups),
        np.linspace(1.7, 0.2, calibration_groups),
    ))
    calibration_y = np.empty(
        (calibration_groups, 2, 3), dtype=np.float64
    )
    for row, (first, second) in enumerate(calibration_x):
        calibration_y[row, 0] = (
            0.2 + 0.3 * first + 0.01,
            0.1 + 0.2 * second + 0.02,
            0.06,
        )
        calibration_y[row, 1] = (
            0.4 + 0.1 * second + 0.015,
            0.3 + 0.1 * first + 0.01,
            0.09,
        )
    return (
        train_x,
        train_y,
        np.asarray([f"train-{index // 2}" for index in range(len(train_x))]),
        calibration_x,
        calibration_y,
        np.asarray([f"cal-{index}" for index in range(calibration_groups)]),
    )


def fit(data=None, **kwargs):
    values = data or base_data()
    options = dict(
        metric_scales=(1.0, 1.0, 0.1),
        target_coverage=0.90,
        ridge=1e-6,
        candidate_panel_sha256=sha("1"),
        feature_contract_sha256=sha("2"),
        metric_contract_sha256=sha("3"),
        training_manifest_sha256=sha("4"),
        calibration_manifest_sha256=sha("5"),
        split_manifest_sha256=sha("6"),
        code_commit=git("7"),
    )
    options.update(kwargs)
    return fit_grouped_linear_risk_student(*values, **options)


def test_nine_groups_support_finite_ninety_percent_certificate():
    student = fit()
    student.validate()
    certificate = student.calibration
    assert certificate.calibration_group_count == 9
    assert certificate.conformal_rank == 9
    assert certificate.guaranteed_group_coverage == pytest.approx(0.90)
    assert certificate.guaranteed_group_coverage >= certificate.target_coverage
    assert certificate.simultaneous_offset >= 0
    assert student.sha256.startswith("sha256:")


def test_eight_groups_cannot_claim_finite_ninety_percent_bound():
    with pytest.raises(
        GroupConformalRiskError,
        match="insufficient independent calibration groups",
    ):
        fit(base_data(calibration_groups=8))


def test_train_and_calibration_group_overlap_is_refused():
    values = list(base_data())
    values[5] = values[5].copy()
    values[5][0] = values[2][0]
    with pytest.raises(GroupConformalRiskError, match="groups overlap"):
        fit(tuple(values))


def test_duplicating_cells_inside_one_calibration_group_does_not_change_offset():
    values = base_data()
    first = fit(values)
    duplicated = list(values)
    duplicated[3] = np.repeat(values[3], 2, axis=0)
    duplicated[4] = np.repeat(values[4], 2, axis=0)
    duplicated[5] = np.repeat(values[5], 2, axis=0)
    second = fit(
        tuple(duplicated),
        calibration_manifest_sha256=sha("8"),
    )
    assert second.calibration.calibration_group_count == 9
    assert second.calibration.simultaneous_offset == pytest.approx(
        first.calibration.simultaneous_offset
    )
    assert np.array_equal(second.coefficients, first.coefficients)


def test_one_bad_cell_controls_the_group_max_at_ninety_percent():
    values = list(base_data())
    baseline = fit(tuple(values))
    values[4] = values[4].copy()
    values[4][-1, 1, 2] += 10.0
    damaged = fit(
        tuple(values),
        calibration_manifest_sha256=sha("8"),
    )
    assert damaged.calibration.simultaneous_offset > (
        baseline.calibration.simultaneous_offset
    )


def test_calibration_example_order_does_not_change_the_bound():
    values = list(base_data())
    first = fit(tuple(values))
    permutation = np.asarray([8, 0, 7, 1, 6, 2, 5, 3, 4])
    values[3] = values[3][permutation]
    values[4] = values[4][permutation]
    values[5] = values[5][permutation]
    second = fit(tuple(values))
    assert second.calibration.simultaneous_offset == pytest.approx(
        first.calibration.simultaneous_offset
    )
    assert second.calibration.conformal_rank == first.calibration.conformal_rank


def test_predict_upper_uses_features_only_and_dominates_point():
    student = fit()
    features = np.asarray([[0.25, 0.75], [1.5, 0.2]])
    point = student.predict_point(features)
    upper = student.predict_upper(features)
    assert point.shape == (2, 2, 3)
    assert upper.shape == point.shape
    assert np.all(point >= 0)
    assert np.all(upper >= point)
    assert "risks" not in inspect.signature(student.predict_upper).parameters
    assert "groups" not in inspect.signature(student.predict_upper).parameters


def test_metric_scale_axis_and_feature_count_are_identity_checked():
    with pytest.raises(GroupConformalRiskError, match="metric_scales"):
        fit(metric_scales=(1.0, 1.0))
    student = fit()
    with pytest.raises(GroupConformalRiskError, match="feature count differs"):
        student.predict_upper(np.zeros((1, 3)))


def test_coefficient_or_contract_substitution_breaks_validation():
    student = fit()
    changed = student.__class__(
        **{
            **student.__dict__,
            "coefficients": student.coefficients + 0.01,
        }
    )
    with pytest.raises(
        GroupConformalRiskError,
        match="different coefficients",
    ):
        changed.validate()

    changed_contract = student.__class__(
        **{
            **student.__dict__,
            "metric_contract_sha256": sha("9"),
        }
    )
    with pytest.raises(
        GroupConformalRiskError,
        match="different metric contract",
    ):
        changed_contract.validate()
