"""Passage miner (docs/v2 §2).

Selects hard evaluation windows from a *provisional* vocal stem + accompaniment:
vocal-energy + pitch fusion (not speech VAD) drives activity; events are tagged
(high soprano, dense accompaniment, hall tail, quiet backing, controls), proposed
as 8–20 s windows, de-clustered by interval NMS, then chosen by stratified
farthest-point selection against category quotas.

Pitch uses ``librosa.pyin`` — the deterministic tracker the design names as the
pyin fallback; PESTO can be layered in later as the primary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import dsp

C5_HZ = 523.25
F5_HZ = 698.46


@dataclass
class MinerConfig:
    hop_ms: float = 10.0
    win_ms: float = 40.0
    # window sizing (seconds) — design defaults are for full opera tracks
    default_window_s: float = 12.0
    min_window_s: float = 8.0
    max_window_s: float = 20.0
    # activity hysteresis
    on_threshold: float = 0.60
    off_threshold: float = 0.35
    min_event_ms: float = 150.0
    close_gap_ms: float = 120.0
    pitch_conf: float = 0.55
    # NMS
    max_iou: float = 0.25
    min_center_distance_s: float = 8.0
    quotas: dict = field(default_factory=lambda: {
        "high_soprano": 3, "dense_accompaniment": 3, "quiet_backing": 2,
        "hall_tail": 4, "no_vocal_control": 4, "random_control": 3,
    })
    seed: int = 0


@dataclass
class Passage:
    passage_id: str
    start_sample: int
    end_sample: int
    tags: list[str]
    features: dict
    detectors: dict
    reason: str = ""


def _pyin(vocal_mono: np.ndarray, sr: int, hop: int):
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        vocal_mono, sr=sr, fmin=65.0, fmax=1200.0, frame_length=2048, hop_length=hop
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0, voiced_flag.astype(bool), voiced_prob


def _events_from_activity(active: np.ndarray, min_len: int, close_gap: int) -> list[tuple[int, int]]:
    """Contiguous active runs (in frames), closing short gaps and dropping shorts."""
    a = active.copy()
    # close gaps
    i = 0
    n = len(a)
    while i < n:
        if not a[i]:
            j = i
            while j < n and not a[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= close_gap:
                a[i:j] = True
            i = j
        else:
            i += 1
    events = []
    i = 0
    while i < n:
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if (j - i) >= min_len:
                events.append((i, j))
            i = j
        else:
            i += 1
    return events


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _interval_nms(cands: list[tuple[tuple[int, int], float]], max_iou: float,
                  min_center: int) -> list[tuple[int, int]]:
    """cands = [(span, score)]; keep high-score, suppress overlapping/close ones."""
    kept: list[tuple[int, int]] = []
    for span, _ in sorted(cands, key=lambda c: -c[1]):
        c = 0.5 * (span[0] + span[1])
        ok = True
        for k in kept:
            kc = 0.5 * (k[0] + k[1])
            if _iou(span, k) > max_iou or abs(c - kc) < min_center:
                ok = False
                break
        if ok:
            kept.append(span)
    return kept


def _farthest_point(feature_rows: np.ndarray, k: int) -> list[int]:
    """Standardized farthest-point selection; returns row indices (max k)."""
    n = len(feature_rows)
    if n <= k:
        return list(range(n))
    X = feature_rows.astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    chosen = [int(np.argmax(np.linalg.norm(Xs - Xs.mean(0), axis=1)))]
    while len(chosen) < k:
        d = np.min([np.linalg.norm(Xs - Xs[c], axis=1) for c in chosen], axis=0)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return chosen


def soprano_tags(med_f0: float, peak_f0: float, p80: float, p95: float,
                 has_voiced: bool) -> list[str]:
    """High/extreme-soprano tags: a note must be absolutely high (C5/F5) AND, on a
    high-voiced track, also above the track percentile — ``max(absolute, relative)``
    (oracle review §3). ``max`` avoids the over-fire of the earlier ``or`` (which
    reduced to ``min`` and tagged low notes on low-pitched tracks)."""
    hi_thr = max(C5_HZ, p80) if has_voiced else C5_HZ
    ex_thr = max(F5_HZ, p95) if has_voiced else F5_HZ
    tags: list[str] = []
    if med_f0 >= hi_thr:
        tags.append("high_soprano")
    if peak_f0 >= ex_thr:
        tags.append("extreme_soprano")
    return tags


def mine_passages(vocal: np.ndarray, accompaniment: np.ndarray, sr: int,
                  config: MinerConfig | None = None) -> list[Passage]:
    cfg = config or MinerConfig()
    hop = max(1, int(sr * cfg.hop_ms / 1000.0))
    v_mono = dsp.mono(vocal)

    v_rms = dsp.frame_rms_db(vocal, sr, cfg.win_ms, cfg.hop_ms)
    a_rms = dsp.frame_rms_db(accompaniment, sr, cfg.win_ms, cfg.hop_ms)
    f0, voiced, vprob = _pyin(v_mono, sr, hop)
    T = min(len(v_rms), len(a_rms), len(f0), len(vprob))
    v_rms, a_rms, f0, voiced, vprob = v_rms[:T], a_rms[:T], f0[:T], voiced[:T], vprob[:T]

    # activity score: normalized vocal loudness fused with pitch confidence
    noise_floor = np.percentile(v_rms, 20)
    p35, p85 = np.percentile(v_rms, 35), np.percentile(v_rms, 85)
    # design rule: (loud AND pitch-confident) OR exposed — pitch_conf is applied here
    voiced_loud = (v_rms >= max(noise_floor + 10.0, p35)) & (vprob >= cfg.pitch_conf)
    exposed = v_rms >= p85
    eligible = voiced_loud | exposed
    score = 0.5 * np.clip((v_rms - noise_floor) / 40.0, 0, 1) + 0.5 * vprob
    active = np.zeros(T, dtype=bool)
    on = False
    for i in range(T):
        if not on and score[i] >= cfg.on_threshold and eligible[i]:
            on = True
        elif on and score[i] <= cfg.off_threshold:
            on = False
        active[i] = on

    events = _events_from_activity(
        active, int(cfg.min_event_ms / cfg.hop_ms), int(cfg.close_gap_ms / cfg.hop_ms)
    )

    voiced_f0 = f0[voiced & (f0 > 0)]
    p80 = np.percentile(voiced_f0, 80) if voiced_f0.size else C5_HZ
    p95 = np.percentile(voiced_f0, 95) if voiced_f0.size else F5_HZ

    default_w = int(cfg.default_window_s * sr)
    min_w, max_w = int(cfg.min_window_s * sr), int(cfg.max_window_s * sr)
    n_samples = min(len(dsp.mono(vocal)), len(dsp.mono(accompaniment)))

    def frame_to_sample(fr: int) -> int:
        return int(fr * hop)

    proposals: list[tuple[tuple[int, int], float, dict, list[str]]] = []
    for (e0, e1) in events:
        ef0 = f0[e0:e1][voiced[e0:e1] & (f0[e0:e1] > 0)]
        med_f0 = float(np.median(ef0)) if ef0.size else 0.0
        peak_f0 = float(np.percentile(ef0, 95)) if ef0.size else 0.0  # smoothed, not raw max (octave-robust)
        var_db = float(np.mean(v_rms[e0:e1]) - np.mean(a_rms[e0:e1]))
        acc_loud_pct = float((a_rms < np.mean(a_rms[e0:e1])).mean())
        tags = soprano_tags(med_f0, peak_f0, p80, p95, bool(voiced_f0.size))
        if var_db > 3.0:
            tags.append("quiet_backing")
        if acc_loud_pct >= 0.80:
            tags.append("dense_accompaniment")

        center = frame_to_sample((e0 + e1) // 2)
        half = max(min_w, min(max_w, default_w)) // 2
        span = (max(0, center - half), min(n_samples, center + half))
        feats = {
            "median_f0_hz": round(med_f0, 2),
            "peak_f0_hz": round(peak_f0, 2),
            "vocal_accompaniment_ratio_db": round(var_db, 2),
            "event_duration_ms": round((e1 - e0) * cfg.hop_ms, 1),
        }
        score_e = (e1 - e0) + (1000 if "high_soprano" in tags else 0)
        proposals.append((span, float(score_e), feats, tags or ["vocal_overlap"]))

        # hall-tail window just after the offset
        tail0 = frame_to_sample(e1)
        tail1 = min(n_samples, tail0 + int(3.0 * sr))
        if tail1 - tail0 > int(0.5 * sr):
            proposals.append(((tail0, tail1), float(e1 - e0), {"tail_after_offset": True}, ["hall_tail"]))

    # no-vocal controls: sustained inactive spans
    inactive_events = _events_from_activity(~active, int(1000 / cfg.hop_ms), 0)
    for (i0, i1) in inactive_events:
        c = frame_to_sample((i0 + i1) // 2)
        half = default_w // 2
        proposals.append(((max(0, c - half), min(n_samples, c + half)),
                          float(i1 - i0), {"control": "no_vocal"}, ["no_vocal_control"]))

    # random controls
    rng = np.random.default_rng(cfg.seed)
    for _ in range(4):
        if n_samples <= default_w:
            s = 0
        else:
            s = int(rng.integers(0, n_samples - default_w))
        proposals.append(((s, min(n_samples, s + default_w)), 0.1, {"control": "random"}, ["random_control"]))

    # interval NMS across all proposals
    spans_scored = [(p[0], p[1]) for p in proposals]
    kept = set(_interval_nms(spans_scored, cfg.max_iou, int(cfg.min_center_distance_s * sr)))
    proposals = [p for p in proposals if p[0] in kept]

    # stratified selection: per category, farthest-point up to quota
    by_cat: dict[str, list[int]] = {}
    for idx, (_span, _sc, _f, tags) in enumerate(proposals):
        cat = tags[0]
        by_cat.setdefault(cat, []).append(idx)

    selected_idx: list[int] = []
    for cat, idxs in by_cat.items():
        quota = cfg.quotas.get(cat, 2)
        rows = np.array([
            [proposals[i][0][0], proposals[i][0][1] - proposals[i][0][0],
             proposals[i][2].get("median_f0_hz", 0.0),
             proposals[i][2].get("vocal_accompaniment_ratio_db", 0.0)]
            for i in idxs
        ], dtype=np.float64)
        pick = _farthest_point(rows, quota)
        selected_idx.extend(idxs[p] for p in pick)

    selected_idx.sort(key=lambda i: proposals[i][0][0])
    out: list[Passage] = []
    for n_, i in enumerate(selected_idx):
        span, _sc, feats, tags = proposals[i]
        out.append(Passage(
            passage_id=f"p_{n_:04d}",
            start_sample=int(span[0]),
            end_sample=int(span[1]),
            tags=tags,
            features=feats,
            detectors={"pitch": "librosa-pyin", "activity": "audio-extract-vocal-activity/v1"},
            reason=f"selected for category {tags[0]}",
        ))
    return out


def write_passages(passages: list[Passage], path: str | Path) -> None:
    doc = {"schema": "audio-extract/passages/v1", "passages": [asdict(p) for p in passages]}
    Path(path).write_text(json.dumps(doc, indent=2))
