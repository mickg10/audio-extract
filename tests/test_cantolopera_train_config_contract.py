from pathlib import Path

import yaml

from audio_extract.challenges import EVALUATION_TASKS


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/train/cantolopera-tier-a-htdemucs-045-100.yaml"


def test_cantolopera_pilot_is_pinned_to_strict_all_voice_continuation():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["experiment"] == "CANTOLOPERA-TIER-A-HTDEMUCS-045-RESIDUAL-V2-100"
    assert cfg["base_checkpoint"] == {
        "signature": "04573f0d",
        "role": "released vocals-specialized single model",
        "sha256": "f3cf25b222c4eed7cd49dd8b2c9597d50c18bd154090f7b919cfa5f93cf22c49",
    }
    assert cfg["sources"] == ["drums", "bass", "other", "vocals"]
    assert cfg["vocal_source_index"] == 3
    assert cfg["data"]["task"] == "all_voices_vs_nonvocal"
    assert cfg["data"]["task"] in EVALUATION_TASKS
    assert cfg["data"]["integrity_required"] == "same_take_paired_target"
    assert cfg["data"]["train_splits"] == ["train"]
    assert cfg["optim"]["steps_first_run"] == 100
    assert cfg["optim"]["evaluation_steps"] == [0, 25, 50, 100]
    assert cfg["loss"]["implementation"] == (
        "audio_extract.classical_loss_v2:classical_residual_loss_v2"
    )
    assert cfg["loss"]["no_vocal_false_positive"] == 1.0
    assert cfg["loss"]["vocal_only_false_negative"] == 0.5
    assert cfg["normalization"] == {
        "implementation": "demucs-full-track-affine/v1",
        "training_statistics_scope": "full_track_per_work_per_control",
        "evaluation_statistics_scope": "complete_input",
        "epsilon": 1e-8,
    }
