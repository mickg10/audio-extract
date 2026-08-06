"""Benchmark program (v3 Phase C / §7): grouped splits + three tiers.

* **Grouped split policy**: the unit is a musical work/session — neighboring clips
  from one work are NEVER split across calibration and test. Assignment is a
  deterministic hash of the group key.
* **Tier 1** — exact linear mixtures (stems available): exact targets, selector labels.
* **Tier 2** — mastered transfer: a held-out production chain (hall → EQ tilt →
  compression → limiter → saturation) over Tier-1 material. The pre-master
  reference is kept but explicitly marked NOT exact (claims discipline).
* **Tier 3** — target-track diagnostics (no ground truth): built per run by the
  challenge engine (genuine controls, remixes, sensitivity probes).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .challenges import estimate_hall_ir

SPLITS = ("dev", "calibration", "regression")
_SPLIT_WEIGHTS = (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class GroupKey:
    work: str
    performer: str = ""
    session: str = ""
    venue: str = ""
    mastering_chain: str = ""

    def id(self) -> str:
        s = "|".join([self.work, self.performer, self.session, self.venue, self.mastering_chain])
        return hashlib.sha256(s.encode()).hexdigest()


def assign_split(group: GroupKey, *, seed: str = "audio-extract-bench-v1") -> str:
    """Deterministic, group-atomic split assignment — every clip of a group lands
    in the same split, always."""
    h = hashlib.sha256(f"{seed}\x00{group.id()}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    acc = 0.0
    for split, w in zip(SPLITS, _SPLIT_WEIGHTS):
        acc += w
        if u < acc:
            return split
    return SPLITS[-1]


def split_cases(cases: list[dict]) -> dict[str, list[dict]]:
    """``cases``: dicts carrying a ``group`` (GroupKey). Returns split -> cases,
    group-atomic by construction."""
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for c in cases:
        out[assign_split(c["group"])].append(c)
    return out


# ---------------------------------------------------------------------------
# Tier 2 — held-out production chains (deterministic by chain_id)
# ---------------------------------------------------------------------------
@dataclass
class ProductionChain:
    chain_id: str
    rt60_s: float
    tilt_db_per_oct: float      # spectral tilt (master EQ)
    comp_threshold_db: float    # soft-knee compression threshold
    comp_ratio: float
    limiter_ceiling: float
    saturation: float           # 0..1 tanh drive

    @classmethod
    def from_id(cls, chain_id: str) -> "ProductionChain":
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(chain_id.encode()).digest()[:8], "big"))
        return cls(
            chain_id=chain_id,
            rt60_s=float(rng.uniform(0.8, 2.2)),
            tilt_db_per_oct=float(rng.uniform(-1.5, 1.5)),
            comp_threshold_db=float(rng.uniform(-24.0, -12.0)),
            comp_ratio=float(rng.uniform(2.0, 5.0)),
            limiter_ceiling=float(rng.uniform(0.85, 0.98)),
            saturation=float(rng.uniform(0.0, 0.35)),
        )


def _apply_tilt(x: np.ndarray, sr: int, db_per_oct: float) -> np.ndarray:
    if abs(db_per_oct) < 1e-6:
        return x
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    gain = np.ones_like(freqs)
    nz = freqs > 20.0
    gain[nz] = 10 ** ((db_per_oct * np.log2(freqs[nz] / 1000.0)) / 20.0)
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = np.fft.irfft(np.fft.rfft(x[:, ch]) * gain, n)
    return out


def _compress(x: np.ndarray, sr: int, threshold_db: float, ratio: float) -> np.ndarray:
    env = dsp.frame_rms_db(x, sr, 40.0, 10.0)
    hop = max(1, int(sr * 0.010))
    gain_db = np.zeros(len(env))
    over = env > threshold_db
    gain_db[over] = (threshold_db - env[over]) * (1.0 - 1.0 / ratio)
    # smooth (attack/release ~ 100 ms) and upsample to samples
    k = 11
    kernel = np.ones(k) / k
    gain_db = np.convolve(gain_db, kernel, mode="same")
    per_sample = np.repeat(10 ** (gain_db / 20.0), hop)[: x.shape[0]]
    if len(per_sample) < x.shape[0]:
        per_sample = np.pad(per_sample, (0, x.shape[0] - len(per_sample)), mode="edge")
    return x * per_sample[:, None]


def apply_production_chain(x: np.ndarray, sr: int, chain: ProductionChain) -> np.ndarray:
    """Deterministic mastered version of a linear mixture. NOT invertible; the
    linear reference stays only a *pre-master* reference for Tier-2 cases."""
    a = dsp.as2d(x).astype(np.float64)
    ir = estimate_hall_ir(sr, rt60_s=chain.rt60_s,
                          seed=int.from_bytes(hashlib.sha256(
                              chain.chain_id.encode()).digest()[8:12], "big"))
    n = a.shape[0]
    nfft = 1 << (n + len(ir) - 2).bit_length()
    wet = np.empty_like(a)
    IR = np.fft.rfft(ir, nfft)
    for ch in range(a.shape[1]):
        wet[:, ch] = np.fft.irfft(np.fft.rfft(a[:, ch], nfft) * IR, nfft)[:n]
    peak_ref = np.max(np.abs(a)) + 1e-12
    wet = wet / (np.max(np.abs(wet)) + 1e-12) * peak_ref
    m = 0.75 * a + 0.25 * wet
    m = _apply_tilt(m, sr, chain.tilt_db_per_oct)
    m = _compress(m, sr, chain.comp_threshold_db, chain.comp_ratio)
    if chain.saturation > 0:
        drive = 1.0 + 4.0 * chain.saturation
        m = np.tanh(m * drive) / np.tanh(drive)
    peak = np.max(np.abs(m)) + 1e-12
    if peak > chain.limiter_ceiling:
        m = m * (chain.limiter_ceiling / peak)
    return m


@dataclass
class BenchCase:
    case_id: str
    tier: int
    group: GroupKey
    mixture: np.ndarray
    reference: np.ndarray            # exact for tier 1; PRE-MASTER (not exact) for tier 2
    reference_exact: bool
    meta: dict = field(default_factory=dict)


def build_tier2_case(tier1_mixture: np.ndarray, tier1_target: np.ndarray, sr: int,
                     group: GroupKey, chain_id: str) -> BenchCase:
    chain = ProductionChain.from_id(chain_id)
    mastered = apply_production_chain(tier1_mixture, sr, chain)
    cid = hashlib.sha256(f"t2|{group.id()}|{chain_id}".encode()).hexdigest()[:24]
    return BenchCase(case_id=f"t2_{cid}", tier=2, group=group, mixture=mastered,
                     reference=dsp.as2d(tier1_target), reference_exact=False,
                     meta={"chain_id": chain_id,
                           "claim": "pre-master reference only; NOT exact ground truth"})
