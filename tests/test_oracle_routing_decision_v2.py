from audio_extract.oracle_routing_decision_v2 import (
    REPORT_SCHEMA,
    evaluate_report,
)
from audio_extract.oracle_routing_work_contract_v3 import (
    REQUIRED_WORKS,
    WORK_CONTRACT_SHA256,
)
from audio_extract.oracle_routing_work_contract_v3 import (
    identity_dict as work_contract_identity,
)

WORKS = REQUIRED_WORKS
RESOLUTIONS = ("2.0", "1.0", "0.5")


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


def routed(name, values=None, *, seed="2", seam=1.0, ringing=1.0):
    result = base_method(values, seed=seed)
    result.update(
        {
            "optimizer_certificate": certificate(name),
            "seam_check": {
                "max_time_boundary_jump_over_p99_derivative": seam,
                "max_frequency_ringing_ratio": ringing,
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


def work_evidence(*, actionable_o2=False):
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


def complete_report(*, actionable_at=None):
    report = {
        "schema": REPORT_SCHEMA,
        "work_contract": work_contract_identity(),
        "work_contract_sha256": WORK_CONTRACT_SHA256,
        "resolutions": {},
    }
    for resolution in RESOLUTIONS:
        works = {work: work_evidence() for work in WORKS}
        if resolution == actionable_at:
            works["bologna_verdi"]["methods"]["O2_global_medoid"] = routed(
                "O2_global_medoid",
                metrics(voice=-7.0, hole=1.2),
                seed="7",
            )
        report["resolutions"][resolution] = {"works": works}
    return report


def decisions_for(result, method):
    return [item for item in result["method_decisions"] if item["method"] == method]


def test_actionable_certified_o2_selects_target_singer_gate():
    result = evaluate_report(complete_report(actionable_at="1.0"))
    assert result["decision"] == "ACTIONABLE_ROUTING_GAP"
    assert result["recommendation"] == ("TRAIN_TARGET_SINGER_FROZEN_MEMBER_GATE")
    assert result["selected"]["method"] == "O2_global_medoid"
    assert result["selected"]["resolution_seconds"] == "1.0"


def test_sensitivity_only_o2_pass_cannot_promote():
    result = evaluate_report(complete_report(actionable_at="0.5"))
    assert result["decision"] == "RESOLUTION_SENSITIVE_GAP"
    assert result["recommendation"] == (
        "DO_NOT_PROMOTE__REVIEW_FROZEN_SCALE_SENSITIVITY"
    )
    assert result["selected"] is None


def test_invalid_required_cell_blocks_primary_o2_pass():
    report = complete_report(actionable_at="1.0")
    report["resolutions"]["0.5"]["works"]["bologna_verdi"]["methods"][
        "O3_certified_convex"
    ]["optimizer_certificate"]["converged"] = False
    result = evaluate_report(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert result["selected"] is None


def test_o3_cannot_promote_even_at_primary_resolution():
    report = complete_report()
    report["resolutions"]["1.0"]["works"]["bologna_verdi"]["methods"][
        "O3_certified_convex"
    ] = routed(
        "O3_certified_convex",
        metrics(voice=-7.0, hole=1.2),
        seed="8",
    )
    result = evaluate_report(report)
    assert result["decision"] == "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"
    assert result["selected"] is None


def test_all_complete_nonwinning_resolutions_select_basis_or_correction():
    result = evaluate_report(complete_report())
    assert result["decision"] == "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"
    assert result["recommendation"] == (
        "CHANGE_BASIS_OR_BUILD_TARGET_SINGER_CORRECTION"
    )
    assert all(item["evidence_valid"] for item in result["method_decisions"])


def test_missing_resolution_is_incomplete_evidence():
    report = complete_report()
    del report["resolutions"]["1.0"]
    result = evaluate_report(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert any(
        "missing required resolution" in failure
        for item in result["method_decisions"]
        for failure in item["failures"]
    )


def test_missing_hall_work_or_contract_identity_is_incomplete():
    report = complete_report()
    del report["resolutions"]["1.0"]["works"]["aalto_mozart_hall"]
    result = evaluate_report(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert any(
        "hall-bearing contract" in failure
        for row in result["method_decisions"]
        for failure in row["failures"]
    )

    report = complete_report()
    report["work_contract_sha256"] = "sha256:" + "0" * 64
    result = evaluate_report(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert result["method_decisions"] == []


def test_uncertified_o3_blocks_a_no_gap_conclusion():
    report = complete_report()
    report["resolutions"]["0.5"]["works"]["bologna_verdi"]["methods"][
        "O3_certified_convex"
    ]["optimizer_certificate"]["converged"] = False
    result = evaluate_report(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert any(
        not item["evidence_valid"]
        for item in decisions_for(result, "O3_certified_convex")
    )


def test_no_vocal_regression_blocks_an_otherwise_large_gap():
    report = complete_report(actionable_at="0.5")
    report["resolutions"]["0.5"]["works"]["aalto_mozart_dry"]["no_vocal"][
        "O2_global_medoid"
    ]["false_positive_energy_ratio"] = 0.02
    result = evaluate_report(report)
    assert result["decision"] == "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"
    selected_resolution = next(
        item
        for item in decisions_for(result, "O2_global_medoid")
        if item["resolution_seconds"] == "0.5"
    )
    assert selected_resolution["evidence_valid"]
    assert any("no-vocal" in item for item in selected_resolution["failures"])


def test_bad_route_seam_is_a_valid_gate_failure_not_missing_evidence():
    report = complete_report(actionable_at="0.5")
    candidate = report["resolutions"]["0.5"]["works"]["bologna_verdi"]["methods"][
        "O2_global_medoid"
    ]
    candidate["seam_check"]["max_time_boundary_jump_over_p99_derivative"] = 10.0
    result = evaluate_report(report)
    assert result["decision"] == "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"
    row = next(
        item
        for item in decisions_for(result, "O2_global_medoid")
        if item["resolution_seconds"] == "0.5"
    )
    assert row["evidence_valid"]
    assert any("time-boundary" in item for item in row["failures"])


def test_missing_artifact_identity_fails_closed():
    report = complete_report()
    del report["resolutions"]["2.0"]["works"]["bologna_verdi"]["methods"][
        "O2_global_medoid"
    ]["artifact"]["artifact_pcm_sha256"]
    result = evaluate_report(report)
    assert result["decision"] == "INCOMPLETE_EVIDENCE"


def test_route_must_beat_median_not_merely_a_hypothetical_single():
    report = complete_report()
    # A +1 dB Verdi gain is plausible improvement over a weaker single member,
    # but it does not beat the frozen +1.5 dB median-champion threshold.
    report["resolutions"]["0.5"]["works"]["bologna_verdi"]["methods"][
        "O2_global_medoid"
    ] = routed("O2_global_medoid", metrics(voice=-6.0, hole=1.1), seed="8")
    result = evaluate_report(report)
    assert result["decision"] == "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"


def test_wrong_schema_fails_closed_without_exception():
    result = evaluate_report({"schema": "wrong"})
    assert result["decision"] == "INCOMPLETE_EVIDENCE"
    assert result["method_decisions"] == []
