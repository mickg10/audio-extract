from audio_extract.cli import build_parser
from audio_extract.demucs_affine import DemucsAffine
from audio_extract.train_classical import (
    ClassicalDataset,
    V1_LOSS,
    V2_RESIDUAL_LOSS,
    _build_optimizer,
    _exact_fold_works,
    _manifest_train_works,
    _loss_config,
    _require_production_normalization,
    _require_task_contract,
    _separate_controls,
    _state_dict_sha256,
)
from audio_extract.classical_loss import ClassicalLossConfig
from audio_extract.classical_loss_v2 import ClassicalResidualLossConfig
from audio_extract.optimizer_contract import optimizer_provenance


def test_train_classical_cli_contract_parses():
    args = build_parser().parse_args([
        "train", "classical",
        "--manifest", "dataset.jsonl",
        "--split-manifest", "splits.json",
        "--config", "train.yaml",
        "--run-dir", "run",
    ])
    assert args.group == "train"
    assert args.cmd == "classical"
    assert args.steps is None
    assert args.resume is None


def test_train_classical_cli_accepts_explicit_zero_step_parity_mode():
    args = build_parser().parse_args([
        "train", "classical",
        "--manifest", "dataset.jsonl",
        "--split-manifest", "splits.json",
        "--config", "train.yaml",
        "--run-dir", "run",
        "--steps", "0",
    ])
    assert args.steps == 0


def test_dataset_refuses_an_incomplete_frozen_work_list(tmp_path):
    try:
        ClassicalDataset(tmp_path, ["present-only-in-manifest"], crop_frames=4)
    except ValueError as exc:
        assert str(exc) == (
            "missing immutable training materializations: present-only-in-manifest"
        )
    else:  # pragma: no cover
        raise AssertionError("missing frozen training work was silently skipped")


def test_dataset_verifies_recipe_and_decoded_pcm_identity(tmp_path):
    import numpy as np
    import soundfile as sf

    from audio_extract.materialize_classical_train import materialize_cantoria

    source = tmp_path / "source"
    output = tmp_path / "materialized"
    source.mkdir()
    accompaniment = np.linspace(-0.1, 0.1, 256, dtype="float32")[:, None]
    vocal = np.linspace(0.02, -0.02, 256, dtype="float32")[:, None]
    sf.write(source / "Cantoria_X_MixOrgan.wav", accompaniment + vocal, 44100,
             subtype="FLOAT")
    sf.write(source / "Cantoria_X_Mix.wav", vocal, 44100, subtype="FLOAT")
    materialize_cantoria(source, output, "X")

    dataset = ClassicalDataset(output, ["cantoria_X"], crop_frames=64)
    assert [item[0] for item in dataset.items] == ["cantoria_X"]

    mixture_path = output / "cantoria_X" / "M.f32.wav"
    mixture, rate = sf.read(mixture_path, dtype="float32", always_2d=True)
    mixture[0, 0] += 0.125
    sf.write(mixture_path, mixture, rate, subtype="FLOAT")
    try:
        ClassicalDataset(output, ["cantoria_X"], crop_frames=64)
    except ValueError as exc:
        assert "materialized PCM hash mismatch for cantoria_X/M" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("corrupted materialized PCM was accepted")


def test_trainer_dispatches_only_versioned_loss_contracts():
    v1_name, v1 = _loss_config({"loss": {
        "implementation": V1_LOSS,
        "waveform_l1": 1.0,
        "complex_stft": 0.5,
        "mixture_consistency": 0.0,
        "no_vocal_false_positive": 0.25,
        "vocal_only_false_negative": 0.25,
        "source_coordinate_alpha_beta_R": 0.1,
        "stereo_coherence": 0.05,
        "exact_event_weighting": 0.25,
    }})
    assert v1_name == V1_LOSS
    assert isinstance(v1, ClassicalLossConfig)

    v2_name, v2 = _loss_config({"loss": {
        "implementation": V2_RESIDUAL_LOSS,
        "residual_waveform": 1.0,
        "residual_complex_stft": 0.5,
        "no_vocal_false_positive": 1.0,
        "vocal_only_false_negative": 0.5,
        "source_coordinate": 0.1,
        "stereo_accompaniment": 0.05,
        "event_weighted_residual": 0.25,
        "stft_ffts": [512, 1024],
        "waveform_reference_floor": 0.001,
        "stft_reference_floor": 0.001,
        "source_coord_ridge": 0.000001,
        "source_coord_max_condition": 1000000.0,
        "target_consistency_tolerance": 0.00001,
        "residual_consistency_tolerance": 0.000001,
        "identity_roundoff_ulps": 8.0,
        "eps": 0.00000001,
    }})
    assert v2_name == V2_RESIDUAL_LOSS
    assert isinstance(v2, ClassicalResidualLossConfig)
    assert v2.stft_ffts == (512, 1024)

    try:
        _loss_config({"loss": {"implementation": "unknown"}})
    except ValueError as exc:
        assert "unrecognized loss implementation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown training loss was accepted")


def test_manifest_train_works_uses_group_manifest_without_resplitting(tmp_path):
    manifest = tmp_path / "dataset.jsonl"
    splits = tmp_path / "splits.json"
    manifest.write_text(
        '{"work_id":"one","group_id":"family-one","split":"train"}\n'
        '{"work_id":"two","group_id":"family-two","split":"test-v1"}\n'
    )
    splits.write_text(
        '{"group_split":{"family-one":"train","family-two":"test-v1"}}\n'
    )
    assert _manifest_train_works(manifest, splits, {"train"}) == ["one"]


def test_manifest_split_disagreement_is_refused(tmp_path):
    manifest = tmp_path / "dataset.jsonl"
    splits = tmp_path / "splits.json"
    manifest.write_text('{"work_id":"one","group_id":"family","split":"train"}\n')
    splits.write_text('{"group_split":{"family":"test"}}\n')
    try:
        _manifest_train_works(manifest, splits, {"train"})
    except ValueError as exc:
        assert "manifest/split disagreement" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("split disagreement was accepted")


def test_manifest_training_contract_refuses_wrong_task_or_integrity(tmp_path):
    import json

    manifest = tmp_path / "dataset.jsonl"
    splits = tmp_path / "splits.json"
    row = {
        "work_id": "one",
        "group_id": "family",
        "split": "train",
        "integrity_class": "same_take_paired_target",
        "task": "soloist_vs_rest",
        "eligible_training_targets": ["accompaniment_A"],
    }
    manifest.write_text(json.dumps(row) + "\n")
    splits.write_text('{"group_split":{"family":"train"}}\n')
    try:
        _manifest_train_works(
            manifest,
            splits,
            {"train"},
            required_integrity="same_take_paired_target",
            required_task="all_voices_vs_nonvocal",
        )
    except ValueError as exc:
        assert "manifest task mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong task identity was accepted")

    row["task"] = "all_voices_vs_nonvocal"
    row["integrity_class"] = "matched_program"
    manifest.write_text(json.dumps(row) + "\n")
    try:
        _manifest_train_works(
            manifest,
            splits,
            {"train"},
            required_integrity="same_take_paired_target",
            required_task="all_voices_vs_nonvocal",
        )
    except ValueError as exc:
        assert "manifest integrity mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong integrity class was accepted")


def test_exact_fold_rejects_nonexact_and_group_overlap(tmp_path):
    manifest = tmp_path / "dataset.jsonl"
    splits = tmp_path / "splits.json"
    base = {
        "integrity_class": "linear_exact",
        "files": {"M": {}, "A": {}, "V": {}},
        "eligible_training_targets": ["accompaniment_A"],
        "split": "test-v1",
    }
    one = {**base, "work_id": "one", "group_id": "shared"}
    two = {**base, "work_id": "two", "group_id": "shared"}
    manifest.write_text("\n".join((__import__("json").dumps(one), __import__("json").dumps(two))) + "\n")
    splits.write_text('{"group_split":{"shared":"test-v1"}}\n')
    try:
        _exact_fold_works(manifest, splits, ["one"], ["two"])
    except ValueError as exc:
        assert "source derivatives cross fold roles" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("group overlap was accepted")


def test_random_control_model_identity_is_state_based():
    import torch

    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2)
    initial = _state_dict_sha256(model)
    assert initial == _state_dict_sha256(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert _state_dict_sha256(model) != initial


def test_two_source_control_uses_direct_accompaniment_not_residual():
    import torch

    class ToyModel:
        def __call__(self, audio):
            return torch.stack((audio * 2, audio * 3), dim=1)

    mixture = torch.ones(1, 2, 8)
    accompaniment = mixture * 4
    vocals = mixture * 5
    affine = DemucsAffine(mean=torch.zeros(1, 1), scale=torch.ones(1, 1))
    affines = {role: affine for role in ("M", "A", "V")}
    direct = _separate_controls(
        ToyModel(), mixture, accompaniment, vocals, 1, "direct_source", 0, affines
    )
    residual = _separate_controls(
        ToyModel(), mixture, accompaniment, vocals, 1, "mixture_residual", None, affines
    )
    assert torch.equal(direct["A_hat"], mixture * 2)
    assert torch.equal(direct["V_hat"], mixture * 3)
    assert torch.equal(residual["A_hat"], mixture - mixture * 3)


def test_random_control_optimizer_has_provenance_group_name():
    import torch

    optimizer = _build_optimizer(torch.nn.Linear(2, 1), {
        "base_checkpoint": {"mode": "random_two_source_control"},
        "optim": {
            "name": "adam", "lr": 0.0003, "betas": [0.9, 0.999],
            "eps": 1e-8, "weight_decay": 0.0,
        },
    })
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]["group_name"] == "all_parameters"
    assert optimizer_provenance(optimizer)["groups"][0]["weight_decay"] == 0.0


def test_pretrained_optimizer_honors_declared_class_and_explicit_defaults():
    import torch

    config = {
        "base_checkpoint": {"signature": "test"},
        "optim": {
            "name": "adam", "betas": [0.8, 0.98], "eps": 1e-7,
            "weight_decay": 0.0,
            "schedule": [{
                "upper_decoder_lr": 1e-4,
                "transformer_decoder_lr": 3e-5,
                "lower_encoder_lr": 0,
            }],
        },
    }
    optimizer = _build_optimizer(torch.nn.Linear(2, 1), config)
    provenance = optimizer_provenance(optimizer)

    assert type(optimizer) is torch.optim.Adam
    assert provenance["class"].endswith(".Adam")
    assert all(group["betas"] == [0.8, 0.98] for group in provenance["groups"])
    assert all(group["eps"] == 1e-7 for group in provenance["groups"])
    assert all(group["weight_decay"] == 0.0 for group in provenance["groups"])


def test_training_controls_use_persisted_full_track_affines_independently():
    import torch

    class CaptureModel:
        def __call__(self, audio):
            self.seen = audio.detach().clone()
            return torch.stack((audio, audio), dim=1)

    model = CaptureModel()
    mixture = torch.full((1, 2, 8), 3.0)
    accompaniment = torch.full((1, 2, 8), 10.0)
    vocals = torch.full((1, 2, 8), -2.0)
    affines = {
        "M": DemucsAffine(torch.tensor([[1.0]]), torch.tensor([[2.0]])),
        "A": DemucsAffine(torch.tensor([[4.0]]), torch.tensor([[3.0]])),
        "V": DemucsAffine(torch.tensor([[-4.0]]), torch.tensor([[0.5]])),
    }
    _separate_controls(
        model, mixture, accompaniment, vocals, 1, "mixture_residual", None, affines
    )
    torch.testing.assert_close(model.seen[0], (mixture[0] - 1.0) / 2.0)
    torch.testing.assert_close(model.seen[1], (accompaniment[0] - 4.0) / 3.0)
    torch.testing.assert_close(model.seen[2], (vocals[0] + 4.0) / 0.5)


def test_trainer_refuses_raw_input_normalization_contract():
    try:
        _require_production_normalization({})
    except ValueError as exc:
        assert "non-production Demucs normalization" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("raw-input training was accepted")

    _require_production_normalization({"normalization": {
        "implementation": "demucs-full-track-affine/v1",
        "training_statistics_scope": "full_track_per_work_per_control",
        "evaluation_statistics_scope": "complete_input",
    }})


def test_trainer_requires_canonical_task_and_explicit_role_scope():
    roles = {
        "removed": ["featured_soloists"],
        "retained": ["orchestra", "chorus", "non_target_soloists"],
    }
    _require_task_contract({"data": {"task": "soloist_vs_rest", "task_roles": roles}})

    for data in (
        {"task": "featured_soloist_vs_rest", "task_roles": roles},
        {"task": "soloist_vs_rest"},
    ):
        try:
            _require_task_contract({"data": data})
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid task contract was accepted: {data}")
