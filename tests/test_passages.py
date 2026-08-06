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
    vocal, _ = fx.synth_vocal(SR, DUR)  # default f0 = C5
    orch = fx.synth_orchestra(SR, DUR)
    n = min(len(vocal), len(orch))
    return pm.mine_passages(vocal[:n], orch[:n], SR, _small_cfg()), n


def test_miner_returns_passages_within_bounds():
    passages, n = _mine()
    assert len(passages) >= 1
    for p in passages:
        assert 0 <= p.start_sample < p.end_sample <= n


def test_miner_tags_controls():
    passages, _ = _mine()
    all_tags = {t for p in passages for t in p.tags}
    assert "no_vocal_control" in all_tags    # inactive spans between phrases
    # (soprano tagging is covered deterministically by test_soprano_tags)


def test_soprano_tags():
    from audio_extract.passages import soprano_tags
    # absolutely high on a low-pitched track -> tagged (max reduces to the C5 threshold)
    assert "high_soprano" in soprano_tags(660.0, 700.0, p80=250.0, p95=260.0, has_voiced=True)
    # low note on a low track -> NOT tagged (the old `or`/`min` over-fired here)
    assert "high_soprano" not in soprano_tags(400.0, 420.0, p80=250.0, p95=260.0, has_voiced=True)
    # high but below the track's top-20% on a soprano track -> NOT high
    assert "high_soprano" not in soprano_tags(600.0, 650.0, p80=700.0, p95=750.0, has_voiced=True)
    assert "extreme_soprano" in soprano_tags(720.0, 760.0, p80=250.0, p95=260.0, has_voiced=True)


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
