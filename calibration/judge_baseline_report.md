# Voice-Removal Quality Judge — FIRST Baseline (PILOT)

**Honest claim label: PILOT / SYNTHETICALLY-VALIDATED baseline — NOT a trustworthy production judge.**

- Host: research6 (`mickg@100.73.131.92`), GPU **NVIDIA RTX PRO 4500 Blackwell** (32 GB), CUDA 13 / torch 2.13.
- Repo commit: `2837caa`  ·  pipeline: `audio_extract.judge_train` + `judge_features_v2` + `judge_labels`.
- Model lock: 3 anchor separators (`configs/model-lock.json`) — MDX23C-8KFFT-InstVoc_HQ, vocals_mel_band_roformer, model_bs_roformer_ep_317. HTDemucs NOT in the model dir -> HTDemucs challenger SKIPPED.
- Trained 6 per-head HistGradientBoosting regressors; evaluation = **LeaveOneGroupOut over works** (never a random clip split).

## 1. Dataset composition

- **Independent REAL linear works: 4** — aalto, donizetti, puccini, verdi (Bologna anechoic Verdi/Puccini/Donizetti + Aalto Mozart; Aalto dry+hall grouped as one work). All `linear_exact`: M, A, V known exactly (V from the isolated voice_ref).
- **Independent SYNTHETIC-orchestra self-remix groups: 6** — 11_turkish_dance, 15_irish_dance_t, 16_doll_track_no, 19_cabaret_drumr, 2_mosquito_tobac, 3_bedbug_tchaik_ — built from vocal-free library-track control passages (approach of `calibration/works_inventory.md` §FREE self-remix works), each screened to voiced-fraction ~0. Weak-independent (real orchestra, synthetic injected voice).
- **Synthetic augmentation on the real orchestras** (grouped WITH their parent real work, so LOWO holds them out together — no leakage): 48 examples across aalto, donizetti, puccini, verdi. Adds level (g)/RT60/register ladders, NOT new independent works.
- **Cantolopera weak pairs: 0** — the STRUMENTALE same-take pairs are a *purchase shortlist* (`works_inventory.md`), not owned data; absent on this host.
- **Voices for all synthetic cases are deterministic synthetic singers** (`fixtures.synth_vocal`, f0 = 523/392/196 Hz). VocalSet donors live on tt-quietbox2 and were not reachable in this run -> realism limitation, fully leak-free.

**Totals:** 53 mixtures -> **106 examples** (2 anchor recipes each) across **10 LOWO groups**.
Examples by corpus: real_linear=10, synthetic-on-real-orchestra=48, synthetic-on-independent-library-orchestra=48.

| LOWO group | kind | examples | with usable label tiles |
|---|---|---:|---:|
| aalto | independent REAL work | 16 | 16/16 |
| donizetti | independent REAL work | 14 | 14/14 |
| puccini | independent REAL work | 14 | 14/14 |
| verdi | independent REAL work | 14 | 14/14 |
| 11_turkish_dance | independent synth-orchestra | 8 | 8/8 |
| 15_irish_dance_t | independent synth-orchestra | 8 | 8/8 |
| 16_doll_track_no | independent synth-orchestra | 8 | 8/8 |
| 19_cabaret_drumr | independent synth-orchestra | 8 | 8/8 |
| 2_mosquito_tobac | independent synth-orchestra | 8 | 8/8 |
| 3_bedbug_tchaik_ | independent synth-orchestra | 8 | 8/8 |

## 2. Recipes (candidate accompaniment Y from mixture M)

Both anchors from `configs/tuning_panel.yaml`:
- `residual_mdx23c`: **Y = M − vocals(MDX23C)**.
- `median_mdx_mel_bs`: **Y = M − median_t(align(vocals_t → M))** over {MDX23C, MelBand-Roformer, BS-Roformer} (bake-off global champion).

## 3. Per-head LOWO results (the honest generalization number)

`beats_baseline` = LOWO MAE < median-baseline MAE (predicting the train-median). It is the real signal that the reference-free features carry information about the exact label.

| head | unit | LOWO MAE | median-baseline MAE | beats baseline? | target mean±std |
|---|---|---:|---:|:--:|---|
| event_hole_db_p90 | dB | 2.436 | 1.756 | no | 1.574 ± 3.581 |
| event_hole_db_max | dB | 2.942 | 2.432 | no | 2.371 ± 4.432 |
| retained_voice_db_p90 | dB | 8.687 | 12.062 | YES | 0.600 ± 15.359 |
| retained_voice_coef_p90 | coef | 0.102 | 0.092 | no | 0.908 ± 0.173 |
| artifact_ratio_p90 | ratio | 0.063 | 0.111 | YES | 0.150 ± 0.156 |
| alpha_error_p90 | abs | 0.134 | 0.119 | no | 0.115 ± 0.203 |

**Heads that LEARN now (beat baseline under LOWO): retained_voice_db_p90, artifact_ratio_p90.**
Heads that do NOT beat baseline yet: event_hole_db_p90, event_hole_db_max, retained_voice_coef_p90, alpha_error_p90.

## 4. GPU + compute timing

- Separation (load-once per model, RTX PRO 4500), total wall over all 53 mixtures ×3 models: mdx23c 321.1s, melband 139.8s, bs 242.7s.
- Feature+label (CPU, librosa pYIN dominated): 1082.5s for 106 examples.
- Reference 92 s work through all 3 models ≈ 29 s; synthetic ~11 s clips ≈ 3–4 s each.

## 5. Honest interpretation & the concrete gap

This is a **PILOT** baseline. Only **4 truly-independent real opera works** anchor it (Bologna×3 + Aalto); the remaining 6 LOWO groups are synthetic-orchestra self-remixes and the bulk of rows are synthetic augmentation. With so few independent real works, the LOWO wins below demonstrate the **feature+label workbench is sound and the labels track separator quality** — on Verdi the exact labels score the median-ensemble recipe better than residual-MDX23C on every head (accompaniment hole 0.51 vs 1.19 dB, retained voice −21.9 vs −18.4 dB, artifact ratio 0.14 vs 0.20, α-error 0.057 vs 0.128), matching the bake-off's median-ensemble win — but they do NOT certify a judge that generalizes to unseen REAL programs.

**Learnable now vs not (from the actual LOWO table):** two heads already carry work-generalizing signal and beat the median baseline — **retained_voice_db_p90** (voice-leakage energy; LOWO MAE 8.69 dB vs 12.06 baseline, **−28%**) and **artifact_ratio_p90** (orthogonal non-source artifacts; 0.063 vs 0.111, **−43%**). Both are "how much non-orchestra content leaked into Y" axes, which the reference-free features (the D=M−Y band-deficit, residual-voice-alignment and spectral-flux distributions) expose directly. The four heads that do NOT yet beat baseline are the accompaniment-transfer axes — **event_hole_db_p90/max** and **alpha_error_p90** — plus **retained_voice_coef_p90**; note event_hole's cross-work target std (3.58 dB) is >2× its mean (1.57 dB), so with only 4 independent real works the constant median is already hard to beat. These heads are **starved of independent works, not proven unlearnable** — the honest signal is that 2/6 heads already generalize on this tiny corpus.

**Concrete gap to move PILOT -> trustworthy:** raise independent REAL works from **4** toward the **20–30** design target. Per `calibration/works_inventory.md`, ~36 independent real-voice works and the donor pools (VocalSet 20 singers × PHENICX 4 orchestra donors × measured RIRs = 100+ synthesizable voice+orchestra works, Cantoría 14 organ-exact, Spheres Mozart/Tchaikovsky, choir sets) are **already acquired on tt-quietbox2** but were not staged on this GPU host. Immediate steps: (1) stage Spheres (+2 -> restores the intended ~6 real works); (2) ingest VocalSet×PHENICX synthesizable works (real singers, real orchestras, exact linear targets) to add 15–25 independent groups; (3) acquire the Cantolopera STRUMENTALE same-take pairs for weak real-world rows. Re-run this exact pipeline on that set; only then should the label change from PILOT to a work-certified judge.

## 6. Saved artifacts (on research6)

- `~/judge_models/judge_<head>.joblib` — the 6 fitted per-head HistGradientBoosting regressors (full-fit).
- `~/judge_build/judge_dataset.npz` — X (features), per-head y_*, groups, feature_names.
- `~/judge_build/rows_meta.json` — per (mixture, recipe) example labels + provenance.
- `~/judge_build/report.json` — machine-readable LOWO report.
- `~/judge_baseline_report.md` — this report (copied to Mac `calibration/judge_baseline_report.md`).
