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
    # absolute vocal-absence ceiling for no_vocal_control (GPU-validation finding:
    # on an all-sung clip, track-relative "inactive" windows still carried 76% vocal
    # energy and structurally contaminated the theft assays)
    no_vocal_abs_ratio_max: float = 0.15
    quotas: dict = field(default_factory=lambda: {
        "high_soprano": 3, "extreme_soprano": 1, "voice_dominant": 2,
        "difficult_overlap": 3, "dense_tutti": 2, "quiet_backing": 2,
        "abrupt_forte": 1, "hall_tail": 3, "no_vocal_control": 3,
        "random_control": 2, "vocal_overlap": 2,
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

    # --- accompaniment context (v2.1 §8.2): loudness, band occupancy, flux ----
    band_env = dsp.band_envelope_db(accompaniment, sr, dsp.PUMP_BANDS, cfg.win_ms, cfg.hop_ms)
    Tb = min(T, band_env.shape[1])
    band_env = band_env[:, :Tb]
    occ = (band_env > np.percentile(band_env, 30, axis=1, keepdims=True)).sum(axis=0).astype(np.float64)
    flux = np.concatenate([[0.0], np.clip(np.diff(band_env.sum(axis=0)), 0, None)])
    occ_p70 = float(np.percentile(occ, 70))
    flux_p70 = float(np.percentile(flux, 70))
    a_p80 = float(np.percentile(a_rms[:Tb], 80))
    silent = a_rms[:Tb] < (np.percentile(a_rms[:Tb], 10) + 3.0)

    def _dense_fraction(e0: int, e1: int) -> float:
        """Fraction of event frames meeting >=3 of the 4 dense-tutti conditions."""
        e0c, e1c = min(e0, Tb - 1), min(e1, Tb)
        if e1c <= e0c:
            return 0.0
        conds = ((a_rms[e0c:e1c] >= a_p80).astype(int)
                 + (occ[e0c:e1c] >= occ_p70).astype(int)
                 + (flux[e0c:e1c] >= flux_p70).astype(int)
                 + (~silent[e0c:e1c]).astype(int))
        return float((conds >= 3).mean())

    acc2 = dsp.as2d(accompaniment)
    hi_share_bands = band_env[3:, :]  # >=2 kHz bands as the brightness proxy

    def _span_features(span: tuple[int, int]) -> dict:
        f0_, f1_ = int(span[0] / hop), max(int(span[0] / hop) + 1, int(span[1] / hop))
        f0_, f1_ = min(f0_, Tb - 1), min(f1_, Tb)
        density = float((occ[f0_:f1_] / max(1, len(dsp.PUMP_BANDS))).mean()) if f1_ > f0_ else 0.0
        total = band_env[:, f0_:f1_].sum() if f1_ > f0_ else 0.0
        bright = float(hi_share_bands[:, f0_:f1_].sum() / total) if total else 0.0
        width = float(dsp.stereo_side_ratio(acc2[span[0]:span[1]]))
        return {"density": round(density, 3), "brightness_share": round(bright, 3),
                "stereo_width": round(width, 4)}

    proposals: list[dict] = []

    def _add(span, score, feats, tags):
        proposals.append({"span": (int(span[0]), int(span[1])), "score": float(score),
                          "feats": feats, "tags": tags})

    for (e0, e1) in events:
        ef0 = f0[e0:e1][voiced[e0:e1] & (f0[e0:e1] > 0)]
        med_f0 = float(np.median(ef0)) if ef0.size else 0.0
        peak_f0 = float(np.percentile(ef0, 95)) if ef0.size else 0.0  # octave-robust
        var_db = float(np.mean(v_rms[e0:e1]) - np.mean(a_rms[e0:e1]))
        dense_frac = _dense_fraction(e0, e1)
        event_ms = (e1 - e0) * cfg.hop_ms

        tags = soprano_tags(med_f0, peak_f0, p80, p95, bool(voiced_f0.size))
        if var_db > 3.0:
            tags.append("quiet_backing")
        if var_db >= -6.0:
            tags.append("voice_dominant")
        if dense_frac >= 0.5 and event_ms >= 500.0:
            tags.append("dense_tutti")
        if -12.0 <= var_db <= 3.0 and dense_frac >= 0.3:
            tags.append("difficult_overlap")
        onset_win = a_rms[e0:min(e0 + int(300 / cfg.hop_ms), Tb)]
        if onset_win.size > 1 and float(np.max(np.diff(onset_win))) >= 10.0:
            tags.append("abrupt_forte")

        center = frame_to_sample((e0 + e1) // 2)
        half = max(min_w, min(max_w, default_w)) // 2
        span = (max(0, center - half), min(n_samples, center + half))
        feats = {
            "median_f0_hz": round(med_f0, 2),
            "peak_f0_hz": round(peak_f0, 2),
            "vocal_accompaniment_ratio_db": round(var_db, 2),
            "dense_fraction": round(dense_frac, 3),
            "event_duration_ms": round(event_ms, 1),
            **_span_features(span),
        }
        score_e = (e1 - e0) + (1000 if "high_soprano" in tags else 0)
        _add(span, score_e, feats, tags or ["vocal_overlap"])

        # hall-tail child just after the offset (kept adjacent to its parent event)
        tail0 = frame_to_sample(e1)
        tail1 = min(n_samples, tail0 + int(3.0 * sr))
        if tail1 - tail0 > int(0.5 * sr):
            _add((tail0, tail1), e1 - e0,
                 {"tail_after_offset": True, "parent_span": [span[0], span[1]],
                  **_span_features((tail0, tail1))}, ["hall_tail"])

    # controls — with an ABSOLUTE vocal-absence gate: a "no-vocal" control must
    # genuinely lack voice, not merely be track-relatively quiet. A contaminated
    # control poisons every downstream theft assay and exact-reference challenge.
    v2_arr = dsp.as2d(vocal)
    a2_arr = dsp.as2d(accompaniment)

    def _vocal_ratio(span: tuple[int, int]) -> float:
        vs = v2_arr[span[0]:span[1]]
        as_ = a2_arr[span[0]:span[1]]
        ev = float(np.sqrt(np.mean(vs ** 2))) if vs.size else 0.0
        em = float(np.sqrt(np.mean((vs + as_[: len(vs)]) ** 2))) if vs.size else 1.0
        return ev / (em + 1e-12)

    inactive_events = _events_from_activity(~active, int(1000 / cfg.hop_ms), 0)
    for (i0, i1) in inactive_events:
        # the control window stays INSIDE the inactive event — a window centered on
        # the gap but spilling into sung phrases is not a control
        e_s = frame_to_sample(i0)
        e_e = min(n_samples, frame_to_sample(i1))
        if e_e - e_s < int(1.0 * sr):
            continue  # too short to be a usable control
        span = (e_s, min(e_e, e_s + max_w))
        ratio = _vocal_ratio(span)
        if ratio > cfg.no_vocal_abs_ratio_max:
            continue  # declined: this "quiet" span still carries voice
        _add(span, i1 - i0,
             {"control": "no_vocal", "vocal_energy_ratio": round(ratio, 4),
              **_span_features(span)}, ["no_vocal_control"])

    rng = np.random.default_rng(cfg.seed)
    for _ in range(4):
        s = 0 if n_samples <= default_w else int(rng.integers(0, n_samples - default_w))
        span = (s, min(n_samples, s + default_w))
        _add(span, 0.1, {"control": "random", **_span_features(span)}, ["random_control"])

    # --- selection v2 (§8.5): NMS WITHIN category first --------------------
    by_cat_all: dict[str, list[int]] = {}
    for i, p in enumerate(proposals):
        for t in p["tags"]:
            by_cat_all.setdefault(t, []).append(i)
    kept_idx: set[int] = set()
    for cat, idxs in by_cat_all.items():
        spans_scored = [(proposals[i]["span"], proposals[i]["score"]) for i in idxs]
        kept_spans = set(_interval_nms(spans_scored, cfg.max_iou, int(cfg.min_center_distance_s * sr)))
        kept_idx.update(i for i in idxs if proposals[i]["span"] in kept_spans)

    # light cross-category dedup; hall-tail children stay near their parent event
    pool: list[int] = []
    for i in sorted(kept_idx, key=lambda i: -proposals[i]["score"]):
        dup = False
        for j in pool:
            if _iou(proposals[i]["span"], proposals[j]["span"]) > 0.6:
                if ("hall_tail" in proposals[i]["tags"]) != ("hall_tail" in proposals[j]["tags"]):
                    continue  # parent/child pair is exempt
                dup = True
                break
        if not dup:
            pool.append(i)

    # multi-label quota accounting: a passage counts toward EVERY tag it carries
    def _row(i: int) -> list[float]:
        p = proposals[i]
        f = p["feats"]
        return [p["span"][0] / max(1, n_samples), p["span"][1] - p["span"][0],
                f.get("median_f0_hz", 0.0), f.get("vocal_accompaniment_ratio_db", 0.0),
                f.get("density", 0.0), f.get("brightness_share", 0.0),
                f.get("stereo_width", 0.0), 1.0 if "hall_tail" in p["tags"] else 0.0]

    counts: dict[str, int] = {c: 0 for c in cfg.quotas}
    sel_set: set[int] = set()
    for cat, quota in cfg.quotas.items():
        need = quota - counts.get(cat, 0)
        if need <= 0:
            continue
        cands = [i for i in pool if cat in proposals[i]["tags"] and i not in sel_set]
        if not cands:
            continue
        rows = np.array([_row(i) for i in cands], dtype=np.float64)
        for k in _farthest_point(rows, min(need, len(cands))):
            i = cands[k]
            sel_set.add(i)
            for t in proposals[i]["tags"]:
                if t in counts:
                    counts[t] += 1

    selected_idx = sorted(sel_set, key=lambda i: proposals[i]["span"][0])
    out: list[Passage] = []
    for n_, i in enumerate(selected_idx):
        p = proposals[i]
        out.append(Passage(
            passage_id=f"p_{n_:04d}",
            start_sample=p["span"][0],
            end_sample=p["span"][1],
            tags=p["tags"],
            features=p["feats"],
            detectors={"pitch": "librosa-pyin", "activity": "audio-extract-vocal-activity/v2"},
            reason=f"selected for categories {p['tags']}",
        ))
    return out


def activity_timebase(sr: int, cfg: MinerConfig) -> dict:
    """Explicit activity timebase record (v2.1 §8.3) — declares the frame grid so
    consumers never guess sample-vs-frame indexing."""
    return {
        "schema": "audio-extract/activity/v2",
        "hop_samples": max(1, int(sr * cfg.hop_ms / 1000.0)),
        "frame_length_samples": max(8, int(sr * cfg.win_ms / 1000.0)),
        "frame_origin_sample": 0,
        "thresholds": {"on": cfg.on_threshold, "off": cfg.off_threshold,
                       "pitch_conf": cfg.pitch_conf},
    }


def write_passages(passages: list[Passage], path: str | Path, timebase: dict | None = None) -> None:
    doc = {"schema": "audio-extract/passages/v1", "passages": [asdict(p) for p in passages]}
    if timebase is not None:
        doc["activity_timebase"] = timebase
    Path(path).write_text(json.dumps(doc, indent=2))
