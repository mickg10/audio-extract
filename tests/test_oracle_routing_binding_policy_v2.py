from audio_extract.oracle_routing_binding_policy_v2 import (
    BindingPolicyConfig,
    evaluate_report_strict,
)
from audio_extract.oracle_routing_decision_v2 import REPORT_SCHEMA
from audio_extract.oracle_routing_work_contract_v3 import (
    REQUIRED_WORKS,
    WORK_CONTRACT_SHA256,
)
from audio_extract.oracle_routing_work_contract_v3 import (
    identity_dict as work_contract_identity,
)

METHODS = ("O2_global_medoid", "O3_certified_convex")
RESOLUTIONS = ("1.0", "2.0", "0.5")


def artifact(seed="1"):
    return {
        "subtype": "FLOAT",
        "reopen_verified": True,
        "sample_rate_hz": 44_100,
        "channels": ["FL", "FR"],
        "frames": 4_410_000,
        "artifact_pcm_sha256": "sha256:" + seed * 64,
        "container_sha256": "sha256:" + seed * 64,
    }


def metrics(*, voice=-5.0, hole=1.0, artifact_ratio=0.10):
    return {
        "retained_voice_db_p90": voice,
        "event_hole_db_p90": hole,
        "artifact_ratio_p90": artifact_ratio,
        "stereo_width_dev_db/v2": 0.20,
        "interchannel_coherence_dev/v2": 0.02,
        "hall_tail_dev_db/v2": 0.30,
        "transient_loss_db/v2": 0.20,
        "transient_excess_db/v2": 0.20,
    }


def event(risk=1.0):
    return {"available": True, "composite_risk": risk}


def base_method(values=None, *, seed="1"):
    return {
        "metrics": values or metrics(),
        "artifact": artifact(seed),
        "worst_identifiable_event": event(),
    }


def certificate(name):
    if name == "O2_global_medoid":
        return {
            "kind": "global_potts_milp",
            "success": True,
            "mip_gap": 0.0,
            "objective_recomputed": True,
            "complete_grid": True,
        }
    return {
        "kind": "convex_projected_gradient",
        "converged": True,
        "monotone_objective": True,
        "starts_agree": True,
        "projected_gradient_norm": 1e-9,
        "max_simplex_error": 1e-10,
        "objective_recomputed": True,
        "complete_grid": True,
    }


def routed(name, values=None, *, seed="2"):
    result = base_method(values, seed=seed)
    result.update(
        {
            "optimizer_certificate": certificate(name),
            "seam_check": {
                "max_time_boundary_jump_over_p99_derivative": 1.0,
                "max_frequency_ringing_ratio": 1.0,
            },
            "worst_identifiable_event": event(1.05),
        }
    )
    return result


def identity_method(seed="3"):
    return {
        "artifact": artifact(seed),
        "identity_roundtrip": {"max_abs": 1e-6, "rms": 1e-7},
    }


def work_evidence():
    methods = {
        "median_raw": base_method(metrics(), seed="1"),
        "residual_mdx23c": base_method(
            metrics(voice=-4.5, hole=1.2, artifact_ratio=0.12), seed="4"
        ),
        "median_stft_identity": identity_method("5"),
        "O1_stft_identity": identity_method("6"),
        "O2_global_medoid": routed("O2_global_medoid", metrics(), seed="2"),
        "O3_certified_convex": routed("O3_certified_convex", metrics(), seed="3"),
    }
    return {
        "methods": methods,
        "no_vocal": {
            "median_raw": {"false_positive_energy_ratio": 0.01},
            "O2_global_medoid": {"false_positive_energy_ratio": 0.0102},
            "O3_certified_convex": {"false_positive_energy_ratio": 0.0101},
        },
    }


def complete_report():
    return {
        "schema": REPORT_SCHEMA,
        "work_contract": work_contract_identity(),
        "work_contract_sha256": WORK_CONTRACT_SHA256,
        "resolutions": {
            resolution: {"works": {work: work_evidence() for work in REQUIRED_WORKS}}
            for resolution in RESOLUTIONS
        },
    }


def make_o2_actionable(report, resolution):
    report["resolutions"][resolution]["works"]["bologna_verdi"]["methods"][
        "O2_global_medoid"
    ] = routed(
        "O2_global_medoid",
        metrics(voice=-7.0, hole=1.2),
        seed="7",
    )


def make_o3_actionable(report, resolution):
    report["resolutions"][resolution]["works"]["bologna_verdi"]["methods"][
        "O3_certified_convex"
    ] = routed(
        "O3_certified_convex",
        metrics(voice=-7.0, hole=1.2),
        seed="8",
    )


def test_only_preregistered_method_at_primary_can_be_actionable():
    report = complete_report()
    make_o3_actionable(report, "1.0")
    result = evaluate_report_strict(report)
    assert result["decision"] == "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"

    make_o2_actionable(report, "1.0")
    result = evaluate_report_strict(report)
    assert result["decision"] == "ACTIONABLE_ROUTING_GAP"
    assert result["selected"]["method"] == "O2_global_medoid"
    assert result["selected"]["resolution_seconds"] == "1.0"


def test_sensitivity_only_pass_is_non_promoting():
    report = complete_report()
    make_o2_actionable(report, "0.5")
    result = evaluate_report_strict(report)
    assert result["decision"] == "RESOLUTION_SENSITIVE_GAP"
    assert result["selected"] is None
    assert len(result["sensitivity_hits"]) == 1


def test_any_invalid_required_row_blocks_even_primary_pass():
    report = complete_report()
    make_o2_actionable(report, "1.0")
    report["resolutions"]["0.5"]["works"]["bologna_verdi"]["methods"][
        "O3_certified_convex"
    ]["optimizer_certificate"]["converged"] = False
    result = evaluate_report_strict(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert result["selected"] is None


def test_missing_required_resolution_is_invalid():
    report = complete_report()
    del report["resolutions"]["2.0"]
    result = evaluate_report_strict(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert any(
        "missing required resolution 2.0" in failure
        for row in result["method_decisions"]
        for failure in row["failures"]
    )


def test_other_method_sensitivity_cannot_promote_selected_method():
    report = complete_report()
    make_o3_actionable(report, "2.0")
    result = evaluate_report_strict(report)
    assert result["decision"] == "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"


def test_policy_is_closed_and_identity_bearing():
    policy = BindingPolicyConfig.from_mapping(
        {
            "schema": "audio-extract/oracle-routing-binding-policy/v2",
            "task_id": "soloist_vs_rest",
            "selected_method": "O2_global_medoid",
            "primary_resolution": "1.0",
            "sensitivity_resolutions": ["2.0", "0.5"],
            "required_methods": list(METHODS),
        }
    )
    assert policy.required_resolutions == RESOLUTIONS
    assert policy.identity_dict()["selected_method"] == "O2_global_medoid"


def test_wrong_schema_is_fail_closed():
    result = evaluate_report_strict({"schema": "wrong"})
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
