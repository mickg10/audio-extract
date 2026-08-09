import dataclasses
import inspect

import numpy as np
import pytest

from audio_extract.group_conformal_risk_student_v2 import (
    ALGORITHM_REVISION,
    ExchangeabilityScopeV2,
    FrozenSelectionProtocolV2,
    GroupConformalRiskV2Error,
    fit_grouped_linear_risk_student_v2,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def git(digit: str) -> str:
    return digit * 40


def family(index: int) -> str:
    return "sha256:" + f"{index:064x}"


def scope() -> ExchangeabilityScopeV2:
    return ExchangeabilityScopeV2(
        exchangeability_scope_sha256=sha("a"),
        query_quality_stratum_sha256=sha("b"),
    )


def selection() -> FrozenSelectionProtocolV2:
    return FrozenSelectionProtocolV2(
        selected_model_spec_sha256=sha("1"),
        selected_hyperparameters_sha256=sha("2"),
        selection_protocol_sha256=sha("3"),
        training_manifest_sha256=sha("4"),
        tuning_manifest_sha256=sha("5"),
    )


def base_data(calibration_family_count: int = 9):
    train_x = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.5],
            [0.5, 2.0],
        ],
        dtype=np.float64,
    )
    train_y = np.empty((len(train_x), 2, 3), dtype=np.float64)
    for row, (first, second) in enumerate(train_x):
        train_y[row, 0] = (
            0.2 + 0.3 * first,
            0.1 + 0.2 * second,
            0.05,
        )
        train_y[row, 1] = (
            0.4 + 0.1 * second,
            0.3 + 0.1 * first,
            0.08,
        )

    calibration_x = np.column_stack(
        (
            np.linspace(0.1, 1.8, calibration_family_count),
            np.linspace(1.7, 0.2, calibration_family_count),
        )
    )
    calibration_y = np.empty(
        (calibration_family_count, 2, 3), dtype=np.float64
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
    train_families = np.asarray(
        [family(10 + index // 2) for index in range(len(train_x))],
        dtype=object,
    )
    calibration_families = np.asarray(
        [family(100 + index) for index in range(calibration_family_count)],
        dtype=object,
    )
    return (
        train_x,
        train_y,
        train_families,
        calibration_x,
        calibration_y,
        calibration_families,
    )


def fit(data=None, **kwargs):
    values = data or base_data()
    options = dict(
        tuning_source_families=(family(50), family(51)),
        metric_scales=(1.0, 1.0, 0.1),
        target_coverage=0.90,
        ridge=1e-6,
        svd_rcond=1e-12,
        max_augmented_condition=1e8,
        scope=scope(),
        selection_protocol=selection(),
        candidate_panel_sha256=sha("6"),
        feature_contract_sha256=sha("7"),
        metric_contract_sha256=sha("8"),
        calibration_manifest_sha256=sha("9"),
        split_manifest_sha256=sha("c"),
        code_commit=git("d"),
    )
    options.update(kwargs)
    return fit_grouped_linear_risk_student_v2(*values, **options)


def test_nine_independent_families_support_exact_ninety_percent_rank():
    student = fit()
    certificate = student.calibration
    assert certificate.calibration_group_count == 9
    assert certificate.conformal_rank == 9
    assert certificate.guaranteed_group_coverage == pytest.approx(0.9)
    assert certificate.scope.family_axis == "source_family_sha256"
    assert certificate.scope.calibration_algorithm_revision == ALGORITHM_REVISION
    assert certificate.augmented_design_condition >= 1.0
    student.validate()


def test_eight_families_cannot_claim_ninety_percent_coverage():
    with pytest.raises(
        GroupConformalRiskV2Error,
        match="insufficient independent calibration source families",
    ):
        fit(base_data(calibration_family_count=8))


def test_rank_and_reported_coverage_are_recomputed_during_validation():
    certificate = fit().calibration
    forged_rank = dataclasses.replace(certificate, conformal_rank=1)
    with pytest.raises(GroupConformalRiskV2Error, match="conformal rank"):
        forged_rank.validate(metric_count=3)
    forged_coverage = dataclasses.replace(
        certificate, guaranteed_group_coverage=0.99
    )
    with pytest.raises(
        GroupConformalRiskV2Error,
        match="guaranteed_group_coverage",
    ):
        forged_coverage.validate(metric_count=3)


def test_source_family_overlap_is_refused_even_when_rows_have_other_ids():
    values = list(base_data())
    values[5] = values[5].copy()
    values[5][0] = values[2][0]
    assert values[5][0] == values[2][0]
    with pytest.raises(GroupConformalRiskV2Error, match="source families overlap"):
        fit(tuple(values))


def test_tuning_families_are_reserved_from_training_and_calibration():
    values = base_data()
    with pytest.raises(GroupConformalRiskV2Error, match="source families overlap"):
        fit(tuning_source_families=(values[2][0], family(51)))
    with pytest.raises(GroupConformalRiskV2Error, match="source families overlap"):
        fit(tuning_source_families=(values[5][0], family(51)))


def test_calibration_cannot_be_reused_for_model_selection():
    protocol = dataclasses.replace(
        selection(), calibration_data_consulted=True
    )
    with pytest.raises(
        GroupConformalRiskV2Error,
        match="may not influence model/hyperparameter selection",
    ):
        fit(selection_protocol=protocol)
    protocol = dataclasses.replace(
        selection(), tuning_manifest_sha256=sha("9")
    )
    with pytest.raises(
        GroupConformalRiskV2Error,
        match="reuses selection data",
    ):
        fit(selection_protocol=protocol)


def test_scope_and_query_stratum_are_identity_bearing():
    first = fit()
    changed = fit(
        scope=dataclasses.replace(
            scope(), query_quality_stratum_sha256=sha("e")
        )
    )
    assert first.sha256 != changed.sha256
    bad = dataclasses.replace(
        first.calibration.scope,
        calibration_algorithm_revision="other",
    )
    with pytest.raises(
        GroupConformalRiskV2Error,
        match="algorithm revision",
    ):
        bad.validate()


def test_augmented_lstsq_handles_nearly_collinear_features_stably():
    values = list(base_data())
    eps = 1e-10
    values[0] = np.column_stack((values[0][:, 0], values[0][:, 0] + eps))
    values[3] = np.column_stack((values[3][:, 0], values[3][:, 0] + eps))
    student = fit(
        tuple(values),
        ridge=1e-4,
        max_augmented_condition=1e10,
    )
    assert np.isfinite(student.coefficients).all()
    assert student.calibration.augmented_design_condition < 1e10


def test_condition_limit_fails_closed():
    with pytest.raises(
        GroupConformalRiskV2Error,
        match="max_augmented_condition",
    ):
        fit(max_augmented_condition=1.01)


def test_negative_order_statistic_is_conservatively_clamped_to_zero():
    values = list(base_data())
    values[4] = np.maximum(values[4] - 100.0, 0.0)
    student = fit(tuple(values))
    assert student.calibration.raw_order_statistic <= 0.0
    assert student.calibration.simultaneous_offset == 0.0


def test_inference_api_uses_features_only_and_upper_dominates_point():
    student = fit()
    features = np.asarray([[0.25, 0.75], [1.5, 0.2]])
    point = student.predict_point(features)
    upper = student.predict_upper(features)
    assert point.shape == (2, 2, 3)
    assert upper.shape == point.shape
    assert np.all(point >= 0)
    assert np.all(upper >= point)
    assert "risks" not in inspect.signature(student.predict_upper).parameters
    assert "famil" not in " ".join(
        inspect.signature(student.predict_upper).parameters
    )


def test_noncanonical_family_ids_are_refused():
    values = list(base_data())
    values[5] = values[5].copy()
    values[5][0] = "work-derivative-a"
    with pytest.raises(
        GroupConformalRiskV2Error,
        match="canonical sha256",
    ):
        fit(tuple(values))
