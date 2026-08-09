from audio_extract.cli import build_parser
from audio_extract.train_classical import (
    _build_optimizer,
    _exact_fold_works,
    _manifest_train_works,
    _separate_controls,
    _state_dict_sha256,
)


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
    direct = _separate_controls(
        ToyModel(), mixture, accompaniment, vocals, 1, "direct_source", 0
    )
    residual = _separate_controls(
        ToyModel(), mixture, accompaniment, vocals, 1, "mixture_residual", None
    )
    assert torch.equal(direct["A_hat"], mixture * 2)
    assert torch.equal(direct["V_hat"], mixture * 3)
    assert torch.equal(residual["A_hat"], mixture - mixture * 3)


def test_random_control_optimizer_has_provenance_group_name():
    import torch

    optimizer = _build_optimizer(torch.nn.Linear(2, 1), {
        "base_checkpoint": {"mode": "random_two_source_control"},
        "optim": {"name": "adam", "lr": 0.0003},
    })
    assert optimizer.param_groups[0]["group_name"] == "all_parameters"
