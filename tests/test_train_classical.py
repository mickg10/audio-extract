from audio_extract.cli import build_parser
from audio_extract.train_classical import _manifest_train_works


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
