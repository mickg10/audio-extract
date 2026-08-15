# Verdict — Anti-forgetting fine-tune of the Mel-Band-Roformer ensemble member

**Result: the FIRST trained model to (partially) beat the delivered ensemble — gate-600 =
MelBand′-ensemble BEATS the champion on 4/5 held-out works, but by margins mostly at the
noise floor (0.01–0.08 dB), and it LOSES on the one hall/reverberant work (worse orchestral
gouging). Step-0 parity exact. Whether this "beats" the ensemble in any audible sense is an
owner's-ears call → A/B renders delivered.**

This is the 4th and strongest data point in the "can a trained model beat the ensemble" line
(after judge-tune=SHADOW, HTDemucs-FT=LOST, output-correction=TIES). It's the first that moves
the constrained objective in the right direction on most works — but marginally, and with a
hall regression — so the ensemble remains at/near optimal.

## Setup

- **Lever:** fine-tune an ensemble INPUT MEMBER (not the output). Target: `vocals_mel_band_roformer.ckpt`
  (Mel-Band-Roformer, **228,202,852 params**), loaded via audio-separator's in-tree `RoformerLoader` —
  the exact object used for inference (bit-exact round-trip by identity).
- **Anti-forgetting (vs the HTDemucs-FT catastrophic-drift lesson):** LR **1e-5**, **L2-SP** anchor
  λ=1.0 to a frozen pretrained copy θ₀, **600 steps**, grad-clip 1.0. It worked — no catastrophic
  drift; all changes tiny and controlled.
- **Data:** frozen classical-v1 train split (cantoría 14 organ + FreiDi 3 + donor 18 real-orchestra),
  exact M/V, RAM-preloaded, peak-normalized to match inference. Bologna/Aalto **HELD OUT**.
- **Gate:** at each checkpoint, export MelBand′ → run it through the **real `separate()`** path →
  `median(MDX23C, MelBand′, BS)` → A′ = M − median → exact-reference metrics vs the champion
  (`median(MDX23C, MelBand_pretrained, BS)`). MDX23C + BS vocals precomputed once.
- **Two acknowledged gaps:** no broad-replay corpus staged (relied on L2-SP + small LR + short
  schedule); Cantolopera §12-bridge pairs not materialized (classical-v1 exact only).

## Gate-0 parity: EXACT (all works d = +0.00) — MelBand′=pretrained ⇒ ensemble == champion.

## Gate-600 — per held-out work, MelBand′-ensemble (R) vs champion (C)

| work | hole_p90 R/C (Δ) | hole_max R/C | SI-SDR R/C (Δ) | retained_voice R/C (Δ) | call |
|---|---|---|---|---|---|
| bologna_verdi     | 1.64 / 1.65 (−0.01) | **13.8 / 16.4** | 17.45 / 17.37 (+0.08) | −16.83 / −16.97 (+0.14) | **BEAT** |
| bologna_puccini   | 0.45 / 0.45 (+0.00) | 7.1 / 7.1 | 25.15 / 25.15 (+0.00) | −24.86 / −24.89 (+0.03) | **BEAT** (≈tie) |
| bologna_donizetti | 3.70 / 3.75 (−0.05) | 13.5 / 13.5 | 26.66 / 26.64 (+0.02) | −14.52 / −14.52 (+0.00) | **BEAT** |
| aalto_mozart_dry  | 0.40 / 0.40 (+0.00) | 5.4 / 5.5 | 20.55 / 20.51 (+0.04) | −29.28 / −29.28 (+0.00) | **BEAT** (≈tie) |
| aalto_mozart_hall | **3.15 / 2.70 (+0.45)** | 7.2 / 6.5 | 9.67 / 8.27 (+1.40) | −8.59 / −7.98 (−0.61) | **LOSE** |
| **aggregate** | 4/5 hole no-worse | | 5/5 SI-SDR ≥ champ | | **BEAT 4/5** |

(Gates 100/300 also 4/5; the hall regression is present from step 100 and grows: hole_p90 Δ +0.31→+0.46→+0.45.)

## Honest interpretation

- **Only one clear, above-noise improvement:** bologna_verdi worst-case hole `hole_max 16.4 → 13.8`
  (−2.6 dB less gouging at the worst moment) plus +0.08 SI-SDR. The other 3 "wins" (puccini, donizetti,
  dry) are **at the measurement noise floor** (Δ ≤ 0.05 dB) — effectively ties.
- **Hall regression is real and directional:** on aalto_mozart_hall the fine-tune removes MORE
  (SI-SDR +1.4, voice-removal better by 0.6 dB) but **gouges more orchestral holes** (hole_p90 +0.45,
  hole_max +0.7). It trades aggressiveness for theft — bad for reverberant/hall material, which is a
  meaningful slice of the real library.
- **Anti-forgetting succeeded:** contrast HTDemucs-FT (catastrophic drift, lost 0/5). Here the 228M
  member stayed within 0.01–0.5 dB of pretrained everywhere — L2-SP + tiny LR + short schedule held.
- **Net:** the ensemble is at/near optimal. The member fine-tune yields a marginal, mixed result —
  a small worst-case-hole win on dense-voice works, a hall-gouging regression — with per-work margins
  below obvious audibility on paper. **The deciding test is the owner's ears (A/B renders).**

## Artifacts

- gates.json, run_report.json, train.log, config in this commit.
- Checkpoints (5 × 913 MB, NOT in git): NAS `mickg@nas642tail:/tanksmall/MICKG2/mickg/models/roformer_ft1/`
  (melband_ft_step{0,100,300,600,final}.ckpt; raw state_dict, drop-in for audio-separator).
- A/B full-length renders (6 tracks, owner's-ears test): `<track>__A_champion.m4a` / `__B_melband_ft.m4a`.
- Repro: `./run.sh python -m audio_extract.roformer_ft --run-dir runs/roformer_ft1 --max-step 600 --lr 1e-5 --l2sp 1.0 --crop-s 7 --batch 1`
