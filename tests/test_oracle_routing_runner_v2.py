import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.oracle_binding_preregistration import build_document, write_once
from audio_extract.oracle_routing_basis_v2 import DeduplicatedCandidate
from audio_extract.oracle_routing_runner_v2 import (
    CertifiedRoutingRunConfig,
    CertifiedRoutingRunError,
    ExactTruth,
    run_experiment,
    run_work_resolution,
)
from audio_extract.oracle_routing_spectral_v2 import RoutingSpectralConfig

SR = 44_100


def file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def scene(seconds=2.0):
    frames = round(seconds * SR)
    time = np.arange(frames) / SR
    envelope = np.zeros(frames)
    quarter = frames // 4
    envelope[:quarter] = 1.0
    envelope[quarter:2 * quarter] = np.linspace(1.0, 0.03, quarter)
    envelope[2 * quarter:3 * quarter] = 0.15
    envelope[3 * quarter:] = np.linspace(0.8, 0.03, frames - 3 * quarter)
    accompaniment = np.column_stack((
        envelope * np.sin(2 * np.pi * 230 * time),
        0.7 * envelope * np.sin(2 * np.pi * 310 * time + 0.2),
    )).astype("float32")
    vocal_envelope = np.zeros(frames)
    vocal_envelope[quarter:3 * quarter] = 0.25
    vocal = np.column_stack((
        vocal_envelope * np.sin(2 * np.pi * 440 * time),
        vocal_envelope * np.sin(2 * np.pi * 440 * time + 0.05),
    )).astype("float32")
    return accompaniment, vocal


def write_candidate(tmp_path, name, aliases, audio, seed):
    path = tmp_path / f"{name}.wav"
    sf.write(path, audio.astype("float32"), SR, subtype="FLOAT")
    reopened, _ = sf.read(path, dtype="float32", always_2d=True)
    return DeduplicatedCandidate(
        work_id="work",
        canonical_name=name,
        aliases=tuple(aliases),
        canonical_recipe_id="sha256:" + seed * 64,
        alias_recipe_ids=("sha256:" + seed * 64,),
        resolved_path=str(path),
        artifact_pcm_sha256=identity.artifact_pcm_sha256(
            reopened, SR, ["FL", "FR"], len(reopened)
        ),
        container_sha256=file_sha(path),
        frames=len(reopened),
        sample_rate_hz=SR,
        channels=("FL", "FR"),
        subtype="FLOAT",
        parent_recipe_ids=(),
        executed_model_bundle_hashes=(),
    )


def fixture(tmp_path):
    accompaniment, vocal = scene()
    mixture = accompaniment + vocal
    truth = ExactTruth(
        work_id="work",
        mixture=mixture.astype("float32"),
        accompaniment=accompaniment,
        vocal=vocal,
        sample_rate_hz=SR,
        frames=len(mixture),
        pcm_identities={
            "mixture": identity.artifact_pcm_sha256(
                mixture.astype("float32"), SR, ["FL", "FR"], len(mixture)
            ),
            "accompaniment": identity.artifact_pcm_sha256(
                accompaniment, SR, ["FL", "FR"], len(mixture)
            ),
            "vocal": identity.artifact_pcm_sha256(
                vocal, SR, ["FL", "FR"], len(mixture)
            ),
        },
    )
    rows = (
        write_candidate(
            tmp_path,
            "median_mdx_mel_bs",
            ("median_mdx_mel_bs", "geomedian_mdx_mel_bs",
             "convex_fusion_uniform"),
            accompaniment,
            "1",
        ),
        write_candidate(
            tmp_path,
            "residual_mdx23c",
            ("residual_mdx23c",),
            0.95 * accompaniment,
            "2",
        ),
        write_candidate(
            tmp_path,
            "other_basis",
            ("residual_melband", "residual_bs_roformer",
             "htdemucs_04573f0d", "htdemucs_955717e8"),
            accompaniment + 0.1 * vocal,
            "3",
        ),
    )
    config = CertifiedRoutingRunConfig(
        spectral=RoutingSpectralConfig(
            sample_rate_hz=SR,
            n_fft=64,
            hop_length=16,
            time_cell_seconds=0.1,
            band_edges_hz=(0, 5_000, 12_000, 22_050),
            min_source_power_relative=1e-8,
            reference_floor_relative=1e-10,
        ),
        resolutions_seconds=(0.1,),
        base_resolution_seconds=0.1,
        temporal_switch_penalty_at_base=0.0,
        frequency_switch_penalty_at_base=0.0,
        temporal_smoothness_at_base=0.0,
        frequency_smoothness_at_base=0.0,
        o2_time_limit_seconds=30.0,
        o3_tolerance=1e-8,
        o3_max_iterations=10_000,
    )
    return truth, rows, config


def test_run_work_resolution_writes_complete_replayable_artifacts(tmp_path):
    truth, rows, config = fixture(tmp_path)
    output = tmp_path / "output"
    report, plans, runtime = run_work_resolution(
        truth=truth,
        candidate_rows=rows,
        output_root=output,
        resolution_seconds=0.1,
        config=config,
        code_commit="test-commit",
    )
    assert set(report["methods"]) >= {
        "median_raw", "residual_mdx23c", "O1_raw",
        "median_stft_identity", "O1_stft_identity",
        "O2_global_medoid", "O3_certified_convex",
    }
    assert plans["O2"].shape == plans["O3"].shape
    assert report["methods"]["O2_global_medoid"][
        "optimizer_certificate"
    ]["objective_recomputed"]
    assert report["methods"]["O3_certified_convex"][
        "optimizer_certificate"
    ]["converged"]
    for method in (
        "median_stft_identity", "O1_stft_identity",
        "O2_global_medoid", "O3_certified_convex",
    ):
        artifact = report["methods"][method]["artifact"]
        assert Path(artifact["path"]).is_file()
        assert artifact["reopen_verified"]
    assert runtime["cell_report"]["total_cells"] > 0

    # A second run must reuse identical immutable artifacts and reports.
    replay, replay_plans, _ = run_work_resolution(
        truth=truth,
        candidate_rows=rows,
        output_root=output,
        resolution_seconds=0.1,
        config=config,
        code_commit="test-commit",
    )
    assert replay["methods"]["O2_global_medoid"]["artifact"] == (
        report["methods"]["O2_global_medoid"]["artifact"]
    )
    assert np.array_equal(replay_plans["O3"], plans["O3"])


def test_binding_experiment_refuses_missing_preregistration_before_audio(tmp_path):
    with pytest.raises(CertifiedRoutingRunError, match="preregistration"):
        run_experiment(
            basis_rows=(),
            no_vocal_basis_rows=(),
            truth_root=tmp_path / "truth",
            output_root=tmp_path / "must-not-exist",
            code_commit="a" * 40,
            preregistration=None,
            works=(),
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_binding_experiment_refuses_code_commit_mismatch_before_audio(tmp_path):
    voiced = tmp_path / "voiced.jsonl"
    no_vocal = tmp_path / "no-vocal.jsonl"
    voiced.write_text("voiced\n")
    no_vocal.write_text("no-vocal\n")
    preregistration = write_once(
        tmp_path / "preregistration.json",
        build_document(
            experiment_id="runner-mismatch",
            source_groups={"voiced": [voiced], "no_vocal": [no_vocal]},
            source_commit="a" * 40,
            truth_manifest_sha256="sha256:" + "1" * 64,
            basis_audit_sha256="sha256:" + "2" * 64,
            routing_config_sha256="sha256:" + "3" * 64,
        ),
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(CertifiedRoutingRunError, match="code commit"):
        run_experiment(
            basis_rows=(),
            no_vocal_basis_rows=(),
            truth_root=tmp_path / "truth",
            output_root=output,
            code_commit="b" * 40,
            preregistration=preregistration,
            works=(),
        )
    assert not output.exists()
