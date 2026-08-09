import math

import pytest

from audio_extract.oracle_routing_binding_gate import (
    REPORT_SCHEMA,
    default_opera_binding_config,
    evaluate_binding_report,
)


BASELINES = (
    "median_mdx_mel_bs",
    "mdx23c",
    "best_whole_track_single",
)


def _sha(seed):
    return f"sha256:{seed:064x}"


def _artifact(seed):
    return {
        "artifact_pcm_sha256": _sha(seed),
        "container_sha256": _sha(seed + 1000),
        "frames": 441000,
        "sample_rate_hz": 44100,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
    }


def _metrics(
    *,
    voice=-10.0,
    hole=2.0,
    artifact=0.10,
    width=0.20,
    coherence=0.02,
    hall=0.10,
    transient_loss=0.10,
    transient_excess=0.10,
    worst=0.20,
    seam=1.0,
    ringing=1.0,
):
    return {
        "retained_voice_db_p90": voice,
        "event_hole_db_p90": hole,
        "artifact_ratio_p90": artifact,
        "stereo_width_dev_db/v2": width,
        "interchannel_coherence_dev/v2": coherence,
        "hall_tail_error_db/v1": hall,
        "transient_loss_ratio/v2": transient_loss,
        "transient_excess_ratio/v2": transient_excess,
        "worst_event_composite_risk": worst,
        "seam_max_boundary_jump_over_p99_derivative": seam,
        "frequency_boundary_ringing_ratio": ringing,
    }


def _output(seed, method, **metrics):
    result = {"artifact": _artifact(seed), "metrics": _metrics(**metrics)}
    if method in ("O2", "O3"):
        result.update(
            {
                "recipe_id": f"route-{method}-{seed}",
                "routing_plan_sha256": _sha(seed + 2000),
            }
        )
    return result


def _work(seed, *, route_voice=-12.0, route_hole=1.8):
    outputs = {}
    specifications = {
        "median_mdx_mel_bs": dict(
            voice=-10.0, hole=2.0, artifact=0.10
        ),
        "mdx23c": dict(voice=-9.0, hole=1.8, artifact=0.12),
        "best_whole_track_single": dict(
            voice=-9.5, hole=2.1, artifact=0.11
        ),
        "O2": dict(
            voice=route_voice, hole=route_hole, artifact=0.10
        ),
        "O3": dict(
            voice=route_voice + 0.2, hole=route_hole, artifact=0.10
        ),
    }
    for index, (method, metrics) in enumerate(specifications.items()):
        outputs[method] = _output(seed + index, method, **metrics)
    return {"outputs": outputs}


def _no_vocal_entry(seed, ratio, *, routed=False):
    result = {
        "artifact": _artifact(seed),
        "false_positive_energy_ratio": ratio,
    }
    if routed:
        result["routing_plan_sha256"] = _sha(seed + 3000)
    return result


def _resolution(seed):
    works = {
        "bologna_verdi": _work(
            seed, route_voice=-12.0, route_hole=1.8
        ),
        "bologna_donizetti": _work(
            seed + 20, route_voice=-10.0, route_hole=0.0
        ),
        "bologna_puccini": _work(seed + 40),
        "aalto_mozart_dry": _work(
            seed + 60, route_voice=-10.0, route_hole=1.8
        ),
    }
    works["aalto_mozart_dry"]["no_vocal"] = {
        "median_mdx_mel_bs": _no_vocal_entry(seed + 100, 0.0100),
        "mdx23c": _no_vocal_entry(seed + 101, 0.0120),
        "best_whole_track_single": _no_vocal_entry(
            seed + 102, 0.0110
        ),
        "O2": _no_vocal_entry(seed + 103, 0.0104, routed=True),
        "O3": _no_vocal_entry(seed + 104, 0.0103, routed=True),
    }
    return {
        "works": works,
        "transform_controls": {
            "O1": {
                "raw_artifact_pcm_sha256": _sha(seed + 200),
                "stft_artifact_pcm_sha256": _sha(seed + 201),
                "max_abs": 1e-6,
                "rms": 1e-7,
                "spectral_error": 1e-7,
                "stereo_error": 1e-7,
            },
            "median_mdx_mel_bs": {
                "raw_artifact_pcm_sha256": _sha(seed + 202),
                "stft_artifact_pcm_sha256": _sha(seed + 203),
                "max_abs": 1e-6,
                "rms": 1e-7,
                "spectral_error": 1e-7,
                "stereo_error": 1e-7,
            },
        },
    }


def _report():
    roles = (
        "median_mdx_mel_bs",
        "geomedian_mdx_mel_bs",
        "convex_fusion_uniform",
        "mdx23c",
        "melband",
        "bs_roformer",
        "htdemucs_04573f0d",
        "htdemucs_955717e8",
    )
    return {
        "schema": REPORT_SCHEMA,
        "code_commit": _sha(4001),
        "candidate_manifest_sha256": _sha(4002),
        "truth_manifest_sha256": _sha(4003),
        "routing_config_sha256": _sha(4004),
        "basis": [
            {
                "role": role,
                "candidate_id": f"candidate-{index}",
                "artifact_pcm_sha256": _sha(index + 5000),
            }
            for index, role in enumerate(roles)
        ],
        "resolutions": {
            "1": _resolution(1),
            "0.5": _resolution(1001),
            "2": _resolution(2001),
        },
    }


def _config():
    return default_opera_binding_config(
        baseline_methods=BASELINES,
        primary_resolution_seconds=1.0,
    )


def _remove_primary_gains(report):
    for method in ("O2", "O3"):
        report["resolutions"]["1"]["works"]["bologna_verdi"][
            "outputs"
        ][method]["metrics"]["retained_voice_db_p90"] = -9.7
        report["resolutions"]["1"]["works"]["bologna_donizetti"][
            "outputs"
        ][method]["metrics"]["event_hole_db_p90"] = 1.7


def test_actionable_route_must_beat_metric_wise_frontier():
    decision = evaluate_binding_report(_report(), _config())
    assert decision["status"] == "ACTIONABLE"
    assert set(decision["actionable_methods"]) == {"O2", "O3"}


def test_beating_weak_single_but_not_median_is_not_actionable():
    report = _report()
    _remove_primary_gains(report)
    decision = evaluate_binding_report(report, _config())
    assert decision["status"] == "RESOLUTION_SENSITIVE"
    assert not decision["actionable_methods"]


def test_missing_or_nonfinite_metric_is_invalid_evidence():
    report = _report()
    del report["resolutions"]["1"]["works"]["bologna_verdi"][
        "outputs"
    ]["O2"]["metrics"]["artifact_ratio_p90"]
    assert (
        evaluate_binding_report(report, _config())["status"]
        == "INVALID_EVIDENCE"
    )
    report = _report()
    report["resolutions"]["1"]["works"]["bologna_verdi"][
        "outputs"
    ]["O2"]["metrics"]["artifact_ratio_p90"] = math.nan
    assert (
        evaluate_binding_report(report, _config())["status"]
        == "INVALID_EVIDENCE"
    )


def test_transform_roundtrip_failure_invalidates_entire_gate():
    report = _report()
    report["resolutions"]["1"]["transform_controls"]["O1"][
        "max_abs"
    ] = 0.01
    decision = evaluate_binding_report(report, _config())
    assert decision["status"] == "INVALID_EVIDENCE"
    assert "transform control" in decision["failures"][0]


def test_no_vocal_regression_blocks_route():
    report = _report()
    for method in ("O2", "O3"):
        report["resolutions"]["1"]["works"]["aalto_mozart_dry"][
            "no_vocal"
        ][method]["false_positive_energy_ratio"] = 0.02
    decision = evaluate_binding_report(report, _config())
    assert decision["status"] == "RESOLUTION_SENSITIVE"
    assert not decision["actionable_methods"]
    assert all(
        any("no-vocal" in failure for failure in result["failures"])
        for result in decision["primary"]["methods"].values()
    )


def test_relative_artifact_regression_blocks_route():
    report = _report()
    for method in ("O2", "O3"):
        report["resolutions"]["1"]["works"]["bologna_verdi"][
            "outputs"
        ][method]["metrics"]["artifact_ratio_p90"] = 0.111
    decision = evaluate_binding_report(report, _config())
    assert not decision["actionable_methods"]
    rows = decision["primary"]["methods"]["O2"]["stability_axes"]
    artifact = next(
        row
        for row in rows
        if row["work_id"] == "bologna_verdi"
        and row["metric"] == "artifact_ratio_p90"
    )
    assert artifact["regression_kind"] == "relative"
    assert artifact["regression"] == pytest.approx(0.11)


def test_seam_or_ringing_failure_blocks_route():
    report = _report()
    for method in ("O2", "O3"):
        report["resolutions"]["1"]["works"]["bologna_donizetti"][
            "outputs"
        ][method]["metrics"][
            "seam_max_boundary_jump_over_p99_derivative"
        ] = 5.0
    decision = evaluate_binding_report(report, _config())
    assert not decision["actionable_methods"]


def test_sensitivity_only_gap_is_not_silently_promoted():
    report = _report()
    _remove_primary_gains(report)
    decision = evaluate_binding_report(report, _config())
    assert decision["status"] == "RESOLUTION_SENSITIVE"
    assert not decision["actionable_methods"]
    assert decision["sensitivity_actionable_methods"]


def test_missing_required_basis_role_is_invalid():
    report = _report()
    report["basis"] = [
        row
        for row in report["basis"]
        if row["role"] != "htdemucs_955717e8"
    ]
    assert (
        evaluate_binding_report(report, _config())["status"]
        == "INVALID_EVIDENCE"
    )


def test_pcm_duplicate_roles_are_deduplicated_without_losing_coverage():
    report = _report()
    report["basis"][1]["artifact_pcm_sha256"] = report["basis"][0][
        "artifact_pcm_sha256"
    ]
    decision = evaluate_binding_report(report, _config())
    assert decision["status"] == "ACTIONABLE"
    assert decision["basis"]["deduplicated_roles"] == 1


def test_grid_mismatch_non_float_or_missing_route_identity_is_invalid():
    report = _report()
    report["resolutions"]["1"]["works"]["bologna_verdi"][
        "outputs"
    ]["O2"]["artifact"]["frames"] -= 1
    assert (
        evaluate_binding_report(report, _config())["status"]
        == "INVALID_EVIDENCE"
    )
    report = _report()
    report["resolutions"]["1"]["works"]["bologna_verdi"][
        "outputs"
    ]["O2"]["artifact"]["subtype"] = "PCM_24"
    assert (
        evaluate_binding_report(report, _config())["status"]
        == "INVALID_EVIDENCE"
    )
    report = _report()
    del report["resolutions"]["1"]["works"]["bologna_verdi"][
        "outputs"
    ]["O2"]["routing_plan_sha256"]
    assert (
        evaluate_binding_report(report, _config())["status"]
        == "INVALID_EVIDENCE"
    )


def test_missing_provenance_hash_is_invalid():
    report = _report()
    del report["truth_manifest_sha256"]
    decision = evaluate_binding_report(report, _config())
    assert decision["status"] == "INVALID_EVIDENCE"
