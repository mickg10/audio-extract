> **⚠️ SUPERSEDED (2026-08-08).** The objective below calibrated its caps on the judge's OWN
> predictions over the same pool — a circular rule (oracle P0 #3). The frozen exact-label
> **selection-regret test** (`judge_selection_regret_report.md`) supersedes this: under honest
> evaluation the judge does **NOT** beat always-median at selection. The '11/25 beats champion'
> claim here is not reliable. Use the median-champion deliverables (`deliverables/`), not
> `deliverables_judge/`, until a promoted judge exists.

# Judge-in-the-loop instrumental selection — v3 judge over the library

The graduated **v3 judge** (`~/judge_models_v3/`, production-trustworthy for the classical/operatic domain) picks the best instrumental per library track, reference-free (no ground truth). Host research6, 5 locked separators, `configs/tuning_panel.yaml` panel.

## Objective

For each track, among the candidate instrumentals **Y** on mixture **M** (D = M−Y), predict the **5 trustworthy heads** (retained_voice_db_p90, retained_voice_coef_p90, event_hole_db_p90, event_hole_db_max, alpha_error_p90) — **artifact_ratio is NOT used** (v3 §5, it does not generalize). Then:

> **Choose the candidate that minimizes predicted `retained_voice_db_p90` (remove the voice as thoroughly as possible), subject to `event_hole_db_p90 ≤ 11.80 dB` AND `alpha_error_p90 ≤ 0.744` (do NOT gouge the accompaniment); tie-break by lower `event_hole_db_p90`.**

The two caps are the **75th percentile** of each metric across the 80 in-domain candidates (library-wide calibration) — they reject the worst-25% gouging/accompaniment-distorting candidates. Candidates whose reference-free feature vector falls outside the v3 training manifold (**Mahalanobis > 11.05**, i.e. > 1.5× the 99th-pct train self-distance) are flagged **OOD**; a track whose champion candidate is OOD **falls back to the v4-lite global champion** (median-MDX23C+MelBand+BS) rather than trusting an extrapolated score.

## Candidate panel (per track)

From `configs/tuning_panel.yaml`: 2 anchors + challengers. Runnable here: **residual:MDX23C**, **median(MDX23C,MelBand,BS)**, **STFT geometric-median(3)**, **convex-fusion (uniform simplex)**. The `htdemucs_ft` challenger is **skipped** — that checkpoint is not in the locked model dir (consistent with v1–v3).

## Results — 25 tracks

- **Judge selection differs from the v4-lite champion on 11/25 tracks.**
- **5/25 tracks flagged OOD** (fell back to champion): 11_turkish_dance_guten_mor, 18_drink_trick_longer, 20_requiem_for_humanity, drink_trick_short, turksh_dnc_22s.
- On the 11 differing tracks, the judged pick improves predicted voice removal by **-4.22 dB** retained_voice on average and changes event_hole by -0.83 dB (i.e. it removes more voice while staying under the gouging cap).
- Chosen-recipe counts: median 14, convex-fusion 5, residual:MDX23C 5, geo-median 1.

| track | chosen | OOD | vs champ | ret_voice_db | ret_voice_coef | hole_p90 | hole_max | alpha_err |
|---|---|:--:|:--:|--:|--:|--:|--:|--:|
| 11_turkish_dance_guten_morgen_ | median | OOD | same | -9.9 | 0.90 | 6.8 | 15.8 | 0.288 |
| 12_13_scarydog_stabat_mater_20 | convex-fusion |  | **DIFF** | +7.3 | 0.69 | -2.9 | 2.0 | 0.045 |
| 15_irish_dance_2026_07_06 | median |  | same | +4.7 | 1.06 | 3.9 | 6.2 | 0.317 |
| 16_doll_track_novoice_gb_05_18 | geo-median |  | **DIFF** | +6.2 | 0.98 | -0.3 | -0.4 | 0.255 |
| 17_marseillaise_applause_05_17 | residual:MDX23C |  | **DIFF** | -0.1 | 0.94 | 0.6 | 1.0 | 0.264 |
| 18_drink_trick_longer_2026_07_ | median | OOD | same | -28.5 | 0.12 | -3.5 | 7.4 | 0.279 |
| 19_cabaret_drumroll_new_2026_0 | median |  | same | -6.1 | 0.99 | 9.7 | 10.4 | 0.284 |
| 1_habanera_last_2026_06_18 | residual:MDX23C |  | **DIFF** | -3.2 | 1.07 | 10.6 | 6.4 | 0.450 |
| 20_requiem_for_humanity_2026_0 | median | OOD | same | -6.4 | 0.95 | 3.1 | 4.9 | 0.189 |
| 21_barber_agnus_dei_2026_06_22 | convex-fusion |  | **DIFF** | +7.7 | 0.88 | 11.7 | 26.4 | 0.503 |
| 22_valkyries_new_7242026 | convex-fusion |  | **DIFF** | +9.9 | 1.09 | 10.4 | 8.3 | 0.654 |
| 2_mosquito_2026_02_07_1 | convex-fusion |  | **DIFF** | -4.1 | 1.07 | 8.4 | -0.4 | 0.250 |
| 3_bedbug_tchaik_long_7242026 | convex-fusion |  | **DIFF** | +6.8 | 0.94 | 1.9 | 2.3 | 0.023 |
| 5_fly_swan_traviata_2026_05_02 | median |  | same | +6.6 | 0.99 | -0.2 | -2.4 | 0.023 |
| 6_la_wally_children_2026_07_06 | median |  | same | -1.5 | 1.05 | 10.8 | 16.5 | 0.504 |
| 7_verdi_slow_voices_05_14_26_1 | median |  | same | +12.0 | 1.02 | 14.4 | 30.7 | 0.899 |
| bedbug_tchaik_old_2026_06_22 | residual:MDX23C |  | **DIFF** | +6.7 | 0.95 | 1.5 | -3.0 | 0.023 |
| cabaret_willkommen_05_18_2026 | residual:MDX23C |  | **DIFF** | -9.9 | 0.96 | 0.4 | -3.3 | -0.024 |
| doll_draft_05_14_2026 | median |  | same | +13.1 | 0.99 | 10.4 | 22.3 | 0.787 |
| drink_trick_short_2026_06_17_1 | median | OOD | same | -17.4 | 0.17 | 1.9 | 1.3 | 0.383 |
| marseillaise_appl_police_7_23_ | residual:MDX23C |  | **DIFF** | +0.6 | 1.00 | -0.6 | -0.3 | 0.217 |
| turksh_dnc_22s | median | OOD | same | -22.7 | 0.09 | 5.1 | 38.9 | 0.429 |
| verdi_fast_voices_2026_05_07_1 | median |  | same | +12.2 | 1.00 | 13.9 | 29.3 | 0.908 |
| yt_donna_a8_vzjny10k | median |  | same | +3.8 | 0.96 | 12.1 | 42.0 | 0.900 |
| yt_nessun_suj_2sbsfks | median |  | same | +7.5 | 0.90 | 19.5 | 50.1 | 1.074 |

(retained_voice_db / coef: lower = less voice left = better. hole_p90/max, alpha_err: lower = less accompaniment damage. All are v3-judge PREDICTIONS on a 45 s central window — reference-free, no ground truth.)

## Honest notes

- **Reference-free estimates.** These are the judge's *predicted* qualities, not measured against a true instrumental (which does not exist for these masters). They are trustworthy in-domain because the v3 judge beats the median baseline on these 5 heads under held-out-ensemble LOWO (report v3 §4).
- **In-domain only.** The 5 OOD tracks are the non-operatic novelties (turkish/irish-adjacent dance, drink-trick skits, the 'requiem for humanity' piece) whose features leave the classical/operatic training manifold; for those the judge defers to the v4-lite champion rather than extrapolate.
- **artifact_ratio deliberately excluded** — it is the one v3 head that does not generalize, so musical-noise/artifacts are NOT part of this objective; a track that trades a little musical noise for much less retained voice will be preferred. Add a trustworthy artifact head to gate that.
- **Panel gap:** htdemucs_ft (a genuinely different waveform family) is not installed, so the panel is 4 spectral-mask/ensemble candidates; adding HTDemucs could change picks on transient-heavy tracks.

## Deliverables

- `deliverables_judge/<track>_instrumental.m4a` — 25 full-length judge-selected instrumentals (AAC 192k).
- `calibration/judge_selection_report.md` — this report. Machine-readable: research6 `~/judge_select/selection.json` (+ `scores.json`, all 100 candidate scores).