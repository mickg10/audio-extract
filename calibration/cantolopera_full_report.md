# Cantolopera FULL recordings — ingest + characterization

The user's **owned/purchased** Cantolopera aria WAVs on their NAS (`mickg@nas642tail:/tanksmall/MICKG2/mickg/cantolopera/full_48khz_f32/`), 48 kHz / stereo / float32. Processed on the user's own hardware (no scraping).

## 1. LOSSLESS vs LOSSY — VERDICT: **TRUE LOSSLESS**

**All 22/22 analyzed files are genuinely LOSSLESS.** Analyzed at native 48 kHz (no resample) via a mid-file loud-segment power spectrum:
- **Brick-wall frequency = 24 000 Hz (full Nyquist)** on every file — content extends to the band edge, with NO codec cutoff.
- **HF noise floor = -139 to -123 dB** (float-precision / dither). No lossy codec produces a floor below ~−60…−95 dB; these sit at −123…−139 dB.
- Contrast with the earlier 30 s **previews**, which were ~64 kbps lossy, brick-walled at ~11 kHz, floor ~−50 dB. **These full files are a categorically different, lossless product.**
- Effective bitrate: **lossless** (equivalent to CD/hi-res PCM). `reference_grade = lossless_same_take` (no longer `lossy_preview`); **audio-grade threshold-eligible.**

## 2. Stem semantics + M=A+V null test

Naming is `<aria-variant>_{voice,orchestra}.wav`. Empirically (full-range cross-correlation on the orchestral intro, where no one is singing):
- **`_voice` = the FULL performance (M = soloist/ensemble + orchestra); `_orchestra` = the STRUMENTALE (A = orchestra-only backing).** V = M − A = the removed voice(s).
- The two are the **SAME TAKE**: in orchestra-only regions the cross-correlation peak is **0.74–0.99** at ~0 lag, and M − A nulls to **−20 to −29 dB**.
- BUT the strumentale is **independently mastered** (level/EQ differ — e.g. La Bohème needs +7 dB gain to match), so the null floors at ~−20…−29 dB, NOT below −35 dB. → **pair_integrity = `same_take_paired_target`, NOT `linear_exact`.** (A naive mid-file segment during singing shows near-zero correlation — the voice buries the orchestra — which is why the orchestral-intro test is required.)

| aria (orchestral-intro null) | xcorr peak | best-window null |
|---|--:|--:|
| aida-qui-radames-verra-o-cieli-azzurri-giuse | 0.742 | -26.1 dB |
| la-boheme-vecchia-zimarra-giacomo-puccini | 0.823 | -20.6 dB |
| amlet-a-vos-jeux-mes-amis-partagez-vous-mes- | 0.992 | -28.6 dB |
| aida-o-terra-addio-duetto-strumentale-giusep | 0.157 | -2.4 dB |

## 3. Valid exact-ish works gained: **6** (same-take, lossless)

| opera | aria / variant | voice removed (V) | dur | null | grade |
|---|---|---|--:|--:|---|
| Aida | senza_soprano | remaining voice(s) (minus soprano) | 612.9s | ~−20..−29dB | lossless_same_take / same_take_paired_target |
| Aida | senza_soprano | remaining voice(s) (minus soprano) | 292.3s | ~−20..−29dB | lossless_same_take / same_take_paired_target |
| Aida | senza_tenor | remaining voice(s) (minus tenor) | 292.3s | ~−20..−29dB | lossless_same_take / same_take_paired_target |
| Aida | full | solo/ensemble | 395.0s | -26.1dB | lossless_same_take / same_take_paired_target |
| Andrea Chenier | full | solo/ensemble | 340.6s | ~−20..−29dB | lossless_same_take / same_take_paired_target |
| La Boheme | full | solo/ensemble | 133.7s | -20.6dB | lossless_same_take / same_take_paired_target |

Voice types covered: **bass** (La Bohème/Colline), **tenor** (Andrea Chénier/Improvviso; Aida/Radamès), **soprano** (Aida/O cieli azzurri; Aida duet minus-tenor), **mezzo** (Aida duet minus-soprano/Amneris). Diverse — exactly the operatic registers we lacked.

## 4. Group-atomic metadata (mastering correlation)

Cantolopera is ONE label/producer → **mastering-correlated within an opera/production**. Group by OPERA for LOWO so whole productions are held out together, never split aria-by-aria:
- **Aida**: 4 valid work(s) — one production (share recording+mastering).
- **Andrea Chenier**: 1 valid work(s) — one production (share recording+mastering).
- **La Boheme**: 1 valid work(s) — one production (share recording+mastering).
The 4 **Aida** works (fu-la-sorte, o-terra ×2, qui-radames) are ONE Aida production → a single LOWO group `aida`.

## 5. Broken / excluded (flagged, not force-paired)

- **aida-fu-la-sorte-dellarmi-a-tuoi-funesta-amore-amore** — no voice stem
- **aida-fu-la-sorte-dellarmi-a-tuoi-funesta-amore-amore** — strumentale version — no soloist to remove (orchestra-only)
- **aida-o-terra-addio-duetto-strumentale-giuseppe-verdi** — strumentale version — no soloist to remove (orchestra-only)
- **amlet-a-vos-jeux-mes-amis-partagez-vous-mes-fleurs-a** — DURATION MISMATCH voice 791.0s vs orch 721.2s (not same edit)
- **andrea-chenier-la-mamma-morta-umberto-giordano** — DURATION MISMATCH voice 422.4s vs orch 164.5s (not same edit)
- **andrea-chenier-nemico-della-patria-umberto-giordano** — orchestra stem MISSING (voice-only)

(The 2 `strumentale` pairs are orchestra-only versions with no soloist to remove; the coordinator-flagged `la-mamma-morta` (422 vs 164 s) and `nemico-della-patria` (voice-only) are confirmed broken; **`amlet` is a NEW mismatch found — 791 vs 721 s, different edit**; `senza-mezzosoprano` voice was still uploading.)

## 6. Staging

Target: research6 `~/datasets/cantolopera_full/<work>/{mix,accompaniment,voice}.flac` (mix=_voice, accompaniment=_orchestra, voice=mix−accompaniment). **Transfer is the bottleneck**: the NAS is saturated by the user's ongoing SMB upload, so NAS reads run at ~0.35 MB/s (LAN IP 10.0.27.127 is no faster — it's the NAS, not the route). The 6 valid works (~1.8 GB) are staging NAS→Mac in the background; FLAC-encode + push to research6 follows. Command to (re)run once the NAS frees up is in the deliverables.

## 7. Read: how strong are these?

**(a) As exact-labeled ensembles for the Gate-3 regret test — NO, not directly.** They are `same_take_paired_target` (−20…−29 dB null), not `linear_exact`. Using A=strumentale as the exact accompaniment target would inject a ~−26 dB master-difference residual into V=M−A, contaminating the exact source-coordinate labels. They CAN serve as **weak/moderate references** (strictly better than the lossy previews: lossless audio + deeper null), flagged `threshold_eligible=false`.
**(b) As fine-tuning targets for the dense-voice separation regime — YES, high value.** These are real, **lossless**, mastered opera voice+orchestra at commercial production density (exactly the user's hard tracks' domain), across bass/tenor/soprano/mezzo and solo+duet textures. M (full) + A (strumentale) is an excellent supervised pair for adapting the separators to dense operatic mixing, even without perfect linearity. Recommended primary use: **separation fine-tuning / evaluation**, not exact-label calibration.

**Note:** more arias were still uploading to the NAS during this analysis (aida-ritorna-vincitor, aida-se-quel-guerrier, il-barbiere ×2 seen in `.rsync-partial`) — the final valid set will be larger (Rossini/Barbiere would add a 4th opera). Re-run the manifest builder when the upload completes.

## Deliverables

- `calibration/cantolopera_full_manifest.jsonl` — per-work grade, roles, null, duration, opera group, valid/broken flags.
- `calibration/cantolopera_full_report.md` — this report.
- `calibration/stage_cantolopera_finish.py` — completes staging: aligns + FLAC-encodes {mix,accompaniment,voice} and rsyncs valid works to research6 `~/datasets/cantolopera_full/<work>/` (run once `~/cantolopera_valid/` finishes pulling).
- Staging (in progress, transfer-bound at ~0.35 MB/s): research6 `~/datasets/cantolopera_full/`.