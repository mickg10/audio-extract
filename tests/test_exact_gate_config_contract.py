from pathlib import Path

import yaml

from audio_extract.train_classical_gate import CONFIG_SCHEMA, _config_objects


ROOT = Path(__file__).resolve().parents[1]


def test_exact_gate_folds_share_one_bounded_contract_and_hold_out_groups():
    configs = [
        yaml.safe_load((ROOT / "configs/train/oracle-route-gate-fold-v.yaml").read_text()),
        yaml.safe_load((ROOT / "configs/train/oracle-route-gate-fold-d.yaml").read_text()),
    ]
    assert all(config["schema"] == CONFIG_SCHEMA for config in configs)
    assert all(config["members"] == {
        "conservative": "htdemucs_04573f0d", "aggressive": "bs_roformer"
    } for config in configs)
    assert all(config["optim"]["steps"] == 100 for config in configs)
    assert all(config["evaluation_steps"] == [0, 100] for config in configs)
    assert all(config["optim"]["weight_decay"] == 0.0 for config in configs)
    assert all(config["loss_implementation"].endswith(
        "classical_loss_v2:classical_residual_loss_v2"
    ) for config in configs)
    assert all(not (
        set(config["data"]["train_work_ids"])
        & set(config["data"]["eval_work_ids"])
    ) for config in configs)
    gate_configs, loss_configs = zip(*(_config_objects(config) for config in configs))
    assert gate_configs[0] == gate_configs[1]
    assert loss_configs[0] == loss_configs[1]
