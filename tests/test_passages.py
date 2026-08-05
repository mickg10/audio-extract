import json

from audio_extract import fixtures as fx
from audio_extract import passages as pm

SR = 44100
DUR = 6.0


def _small_cfg():
    # Synthetic clips are seconds long; shrink the design's 8–20 s windows to fit.
    return pm.MinerConfig(
        default_window_s=2.0, min_window_s=1.0, max_window_s=3.0,
        min_center_distance_s=1.0, quotas={
            "high_soprano": 2, "hall_tail": 2, "no_vocal_control": 2,
            "random_control": 2, "quiet_backing": 1, "dense_accompaniment": 1,
            "vocal_overlap": 2,
        },
    )


def _mine():
    vocal, _ = fx.synth_vocal(SR, DUR)  # default f0 = C5 (523.25 Hz)
    orch = fx.synth_orchestra(SR, DUR)
    n = min(len(vocal), len(orch))
    return pm.mine_passages(vocal[:n], orch[:n], SR, _small_cfg()), n


def test_miner_returns_passages_within_bounds():
    passages, n = _mine()
    assert len(passages) >= 1
    for p in passages:
        assert 0 <= p.start_sample < p.end_sample <= n


def test_miner_tags_high_soprano_and_controls():
    passages, _ = _mine()
    all_tags = {t for p in passages for t in p.tags}
    assert "high_soprano" in all_tags       # f0 = C5, above the C5 threshold
    assert "no_vocal_control" in all_tags    # inactive spans between phrases


def test_miner_nms_limits_overlap():
    passages, _ = _mine()
    spans = [(p.start_sample, p.end_sample) for p in passages]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            assert pm._iou(spans[i], spans[j]) <= 0.30


def test_write_passages(tmp_path):
    passages, _ = _mine()
    out = tmp_path / "passages.v1.json"
    pm.write_passages(passages, out)
    doc = json.loads(out.read_text())
    assert doc["schema"] == "audio-extract/passages/v1"
    assert len(doc["passages"]) == len(passages)
    assert "detectors" in doc["passages"][0]
