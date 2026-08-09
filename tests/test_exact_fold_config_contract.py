from __future__ import annotations

from pathlib import Path

import yaml

from audio_extract.challenges import EVALUATION_TASKS
from audio_extract.optimizer_contract import validate_optimizer_config


ROOT = Path(__file__).resolve().parents[1]
FOLDS = (
    ROOT / "configs/train/opera-exact-cv-fold-v.yaml",
    ROOT / "configs/train/opera-exact-cv-fold-d.yaml",
)


def test_exact_fold_configs_use_canonical_task_and_explicit_optimizer():
    for path in FOLDS:
        config = yaml.safe_load(path.read_text())
        assert config["data"]["task"] in EVALUATION_TASKS
        assert config["data"]["task"] == "soloist_vs_rest"
        assert config["data"]["integrity_required"] == "linear_exact"
        resolved = validate_optimizer_config(config["optim"])
        assert resolved["name"] == "adam"
        assert resolved["weight_decay"] == 0.0


def test_exact_fold_roles_are_disjoint_and_expected():
    fold_v = yaml.safe_load(FOLDS[0].read_text())
    fold_d = yaml.safe_load(FOLDS[1].read_text())

    for config in (fold_v, fold_d):
        train = set(config["data"]["train_work_ids"])
        evaluate = set(config["data"]["eval_work_ids"])
        assert train.isdisjoint(evaluate)
        assert "aalto_mozart_dry" in evaluate

    assert "bologna_verdi" in fold_v["data"]["eval_work_ids"]
    assert "bologna_donizetti" in fold_d["data"]["eval_work_ids"]
