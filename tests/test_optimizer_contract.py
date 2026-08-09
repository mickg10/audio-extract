from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from audio_extract.optimizer_contract import (
    OptimizerConfigError,
    build_optimizer,
    optimizer_provenance,
    validate_optimizer_config,
)


def _parameter_groups():
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    return [
        {"params": [first], "lr": 1e-4, "group_name": "upper_decoder"},
        {"params": [second], "lr": 3e-5, "group_name": "transformer_decoder"},
    ]


def _config(name: str) -> dict:
    return {
        "name": name,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }


def test_adam_config_constructs_adam_not_adamw():
    optimizer = build_optimizer(torch, _parameter_groups(), _config("adam"))
    assert type(optimizer) is torch.optim.Adam


def test_adamw_config_constructs_adamw():
    config = _config("adamw")
    config["weight_decay"] = 0.01
    optimizer = build_optimizer(torch, _parameter_groups(), config)
    assert type(optimizer) is torch.optim.AdamW


def test_missing_identity_bearing_default_is_refused():
    config = _config("adam")
    del config["weight_decay"]
    with pytest.raises(OptimizerConfigError, match="weight_decay"):
        validate_optimizer_config(config)


def test_unknown_optimizer_is_refused():
    with pytest.raises(OptimizerConfigError, match="unsupported optimizer"):
        validate_optimizer_config({
            "name": "mystery",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        })


def test_effective_provenance_contains_group_hyperparameters():
    optimizer = build_optimizer(torch, _parameter_groups(), _config("adam"))
    report = optimizer_provenance(optimizer)

    assert report["class"].endswith(".Adam")
    assert [group["group_name"] for group in report["groups"]] == [
        "upper_decoder", "transformer_decoder"
    ]
    assert [group["lr"] for group in report["groups"]] == [1e-4, 3e-5]
    assert all(group["betas"] == [0.9, 0.999] for group in report["groups"])
    assert all(group["weight_decay"] == 0.0 for group in report["groups"])
