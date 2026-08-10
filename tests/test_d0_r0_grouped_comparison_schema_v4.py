import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from tests.test_counterfactual_risk_grouped_report_artifact_v4 import (
    fixture as report_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "schemas" / "d0-r0-grouped-comparison-report-v3.schema.json"
V4_PATH = ROOT / "schemas" / "d0-r0-grouped-comparison-report-v4.schema.json"


def validator():
    v3 = json.loads(V3_PATH.read_text())
    v4 = json.loads(V4_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(v3)
    jsonschema.Draft202012Validator.check_schema(v4)
    resolver = jsonschema.RefResolver.from_schema(
        v4,
        store={
            v3["$id"]: v3,
            V3_PATH.name: v3,
        },
    )
    return jsonschema.Draft202012Validator(v4, resolver=resolver)


def payload():
    _prereg, _report, artifact = report_fixture()
    return artifact.identity_dict()


def assert_invalid(value):
    errors = list(validator().iter_errors(value))
    assert errors, "mutated report unexpectedly passed v4 schema"


def test_real_v4_artifact_validates_against_paired_schema():
    value = payload()
    assert "source_v3_verifier_commit" in value
    assert "paired_regret_tolerance" in value
    validator().validate(value)


@pytest.mark.parametrize(
    "field",
    ["source_v3_verifier_commit", "paired_regret_tolerance"],
)
def test_identity_bearing_v4_root_fields_are_required(field):
    value = payload()
    del value[field]
    assert_invalid(value)


def test_nested_arm_substitution_is_rejected():
    value = payload()
    pair = value["paired_units"][0]
    pair["d0"]["route_provenance"]["submission"]["arm_id"] = "R0"
    assert_invalid(value)


def test_route_evaluation_cannot_be_backed_by_abstention_lineage():
    value = payload()
    work = value["paired_units"][0]["d0"]
    output = work["route_provenance"]["route_artifact"]["route_output"]
    submission = work["route_provenance"]["submission"]
    output["status"] = "ABSTAIN"
    output["labels_shape"] = None
    output["labels_sha256"] = None
    submission["status"] = "ABSTAIN"
    submission["labels_shape"] = None
    submission["labels_sha256"] = None
    assert_invalid(value)


def test_oracle_available_abstention_requires_numeric_oracle_objective():
    value = payload()
    work = value["paired_units"][0]["d0"]
    output = work["route_provenance"]["route_artifact"]["route_output"]
    submission = work["route_provenance"]["submission"]
    evaluation = work["complete_route_evaluation"]
    output["status"] = "ABSTAIN"
    output["labels_shape"] = None
    output["labels_sha256"] = None
    submission["status"] = "ABSTAIN"
    submission["labels_shape"] = None
    submission["labels_sha256"] = None
    evaluation["status"] = "ABSTAIN"
    evaluation["exact_oracle_objective"] = None
    evaluation["submitted_exact_objective"] = None
    evaluation["selection_regret"] = None
    evaluation["temporal_switches"] = None
    evaluation["frequency_switches"] = None
    assert_invalid(value)


def test_one_sided_oracle_unavailable_pair_is_rejected():
    value = payload()
    evaluation = value["paired_units"][0]["d0"][
        "complete_route_evaluation"
    ]
    evaluation["status"] = "ORACLE_UNAVAILABLE"
    evaluation["exact_oracle_objective"] = None
    evaluation["submitted_exact_objective"] = None
    evaluation["selection_regret"] = None
    evaluation["temporal_switches"] = None
    evaluation["frequency_switches"] = None
    assert_invalid(value)


def test_cross_tab_cannot_contain_one_sided_oracle_unavailable():
    value = payload()
    value["pairwise_summary"]["status_cross_tab"] = [
        ["ORACLE_UNAVAILABLE", "ROUTE_SAFE", 1]
    ]
    assert_invalid(value)


def test_critical_and_secondary_violation_kinds_are_not_interchangeable():
    value = payload()
    evaluation = value["paired_units"][0]["d0"][
        "complete_route_evaluation"
    ]
    evaluation["status"] = "ROUTE_CATASTROPHIC_FALSE_SAFE"
    evaluation["critical_false_safe_violations"] = [
        {
            "schema": "audio-extract/counterfactual-route-violation/v1",
            "time_index": 0,
            "band_index": 0,
            "candidate_index": 0,
            "metric_name": "artifact",
            "kind": "required_secondary_missing",
            "threshold": None,
            "value": None,
        }
    ]
    evaluation["submitted_exact_objective"] = None
    evaluation["selection_regret"] = None
    assert_invalid(value)


def test_above_threshold_violation_requires_positive_numeric_facts():
    value = payload()
    evaluation = value["paired_units"][0]["d0"][
        "complete_route_evaluation"
    ]
    evaluation["status"] = "ROUTE_CATASTROPHIC_FALSE_SAFE"
    evaluation["critical_false_safe_violations"] = [
        {
            "schema": "audio-extract/counterfactual-route-violation/v1",
            "time_index": 0,
            "band_index": 0,
            "candidate_index": 0,
            "metric_name": "voice",
            "kind": "critical_above_threshold",
            "threshold": 1.0,
            "value": 0.0,
        }
    ]
    evaluation["submitted_exact_objective"] = None
    evaluation["selection_regret"] = None
    assert_invalid(value)


def test_secondary_missing_violation_must_not_carry_a_value():
    value = payload()
    evaluation = value["paired_units"][0]["d0"][
        "complete_route_evaluation"
    ]
    evaluation["status"] = "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE"
    evaluation["incomplete_secondary_violations"] = [
        {
            "schema": "audio-extract/counterfactual-route-violation/v1",
            "time_index": 0,
            "band_index": 0,
            "candidate_index": 0,
            "metric_name": "artifact",
            "kind": "required_secondary_missing",
            "threshold": None,
            "value": 0.1,
        }
    ]
    evaluation["submitted_exact_objective"] = None
    evaluation["selection_regret"] = None
    assert_invalid(value)


def test_d0_and_r0_summaries_cannot_be_swapped():
    value = payload()
    value["d0_summary"], value["r0_summary"] = (
        value["r0_summary"],
        value["d0_summary"],
    )
    assert_invalid(value)
