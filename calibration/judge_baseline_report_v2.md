# Voice-Removal Quality Judge — Baseline v2 (REAL-VOICE, work-validated)

**Honest claim label: WORK-VALIDATED baseline (approaching-trustworthy) — NOT yet production-trustworthy.**
21 independent **real-voice** works; the reference-free features beat the median baseline on **5/6 heads under held-out-WORK LOWO** and **5/6 (4 robustly) under the stricter held-out-ENSEMBLE LOWO**. The remaining limit is ensemble/accompaniment *diversity* (4 corpora, 2 accompaniment families), not the workbench.

- Host: research6 (`mickg@100.73.131.92`), GPU **RTX PRO 4500 Blackwell**, CUDA 13 / torch 2.13. Repo `2837caa`.
- Pipeline unchanged from v1 (`judge_train`/`judge_features_v2`/`judge_labels`); the ONLY change is the **data**: real recordings staged from tt-quietbox2 (`ttuser@100.91.242.69`) replace v1's synthetic voices.
- Fixes carried from v1: MDXC short-clip tile-pad; `OMP_NUM_THREADS=1` (training 4.5 s vs a 10-min OpenMP spin-wait); feature cache.

## 1. What changed vs v1
v1 was a PILOT: **4** real works (Bologna×3+Aalto) + synthetic-voice augmentation → only **2/6** heads beat baseline. v2 stages the real library and drops all synthetic voices:

| | v1 (pilot) | v2 (this) |
|---|---|---|
| independent real-voice works | 4 | **21** |
| distinct ensembles/corpora | 2 | **4** |
| voices | synthetic (`synth_vocal`) | **all real recordings** |
| heads beating baseline (per-work LOWO) | 2/6 | **5/6** |

## 2. Dataset composition — 21 independent REAL-VOICE works (44 examples)

All exact references (M = A + V verified per work). 2 anchor recipes each (residual:MDX23C, median:MDX23C+MelBand+BS). 60 s central analysis window.

| corpus / ensemble | works (LOWO groups) | voice | accompaniment | integrity | M=A+V residual |
|---|---|---|---|---|---|
| **Bologna** (reused v1) | verdi, puccini, donizetti | solo opera | orchestra (anechoic) | linear_exact | ≤ −78 dB |
| **Aalto** (reused v1) | aalto (Mozart, dry+hall) | solo soprano | orchestra (anechoic) | linear_exact | ≤ −78 dB |
| **FreiDi** (Weber, *Der Freischütz*) | freidi_no06, no08, no09 | opera ensemble (1–3 singers) | orchestra (spot-mic) | same_performance_bleed; M=A+V built | −111…−134 dB |
| **Cantoría** (Iberian Golden-Age) | CEA, EJB1, EJB2, HCB, LBM1, LBM2, LJT1, LJT2, LNG, RRC, SSS, THM, VBP, YSM | SATB quartet | **organ** (A = MixOrgan−Mix) | linear_exact | **−240 dB (exact)** |

- Examples by corpus: Cantoria 28 (14×2 recipes), FreiDi 6 (3×2), Bologna/Aalto reused 10.
- Accompaniment families: **orchestra** (Bologna/Aalto/FreiDi) and **organ** (Cantoría).
- **Cantolopera weak pairs: still 0** (lossy STRUMENTALE previews on the Mac; not yet ingested — see gap). VocalSet×PHENICX/Spheres synthesizable and choir sets: available on tt-quietbox2, not yet added (see gap).

## 3. Per-head results — held-out-WORK LOWO (21 groups)

`beats_baseline` = LOWO MAE < median-baseline MAE (predicting the train-median). The honest generalization signal.

| head | unit | LOWO MAE | median-baseline MAE | beats | Δ vs baseline |
|---|---|--:|--:|:--:|--:|
| event_hole_db_p90 | dB | 7.320 | 14.753 | **YES** | −50.4% |
| event_hole_db_max | dB | 18.383 | 26.323 | **YES** | −30.2% |
| retained_voice_db_p90 | dB | 6.594 | 8.937 | **YES** | −26.2% |
| retained_voice_coef_p90 | coef | 0.196 | 0.242 | **YES** | −18.8% |
| artifact_ratio_p90 | ratio | 0.125 | 0.111 | no | +11.9% |
| alpha_error_p90 | abs | 0.174 | 0.299 | **YES** | −42.0% |

**5/6 heads generalize across held-out works.** The event_hole heads — which v1 (4 works, synthetic voices) could NOT learn — now beat baseline by 30–50%, exactly the "more works + real voices" prediction: real organ/ensemble accompaniments produce large, varied separator holes (event_hole target mean 18 dB vs 1.6 dB in v1) that the D=M−Y band-deficit features track.

## 4. Per-head results — held-out-ENSEMBLE LOWO (4 corpora) — the stricter honesty check

Cantoría's 14 pieces share ONE SATB quartet + organ, so per-work LOWO there tests *unseen piece, same singers*. This stricter split holds out a whole ensemble/accompaniment family (train on 3 corpora, predict the 4th — a real domain shift, e.g. train orchestra→predict organ):

| head | LOWO MAE | baseline MAE | beats | Δ |
|---|--:|--:|:--:|--:|
| event_hole_db_p90 | 13.801 | 12.351 | no | +11.7% |
| event_hole_db_max | 28.670 | 37.403 | **YES** | −23.3% |
| retained_voice_db_p90 | 9.677 | 11.007 | **YES** | −12.1% |
| retained_voice_coef_p90 | 0.472 | 0.576 | **YES** | −18.1% |
| artifact_ratio_p90 | 0.1667 | 0.1674 | **YES** (tie) | −0.4% |
| alpha_error_p90 | 0.379 | 0.497 | **YES** | −23.8% |

**Even across entirely unseen ensembles, 5/6 heads still beat baseline.** The **4 heads that generalize under BOTH tests** — `event_hole_db_max`, `retained_voice_db_p90`, `retained_voice_coef_p90`, `alpha_error_p90` — are the robust, deployable set today. `event_hole_db_p90`'s p90 magnitude is very ensemble-specific (organ vs orchestra holes differ), so it beats per-work but not per-ensemble; `artifact_ratio` is the mirror (per-ensemble tie, per-work slightly worse).

## 5. GPU + compute timing
- Separation (load-once/model, 17 works ×3): mdx23c 224 s, melband 90 s, bs 140 s.
- Feature+label (CPU pYIN, `OMP_NUM_THREADS=1`): 1151 s for 34 new examples (+10 reused instantly from the v1 cache).
- Training (LOWO baseline, per-work): **4.5 s**.

## 6. Honest interpretation & the remaining gap

v2 is no longer a pilot: with **21 independent real-voice works** the interpretable baseline **generalizes on 5/6 heads to unseen works and 4/6 robustly to unseen ensembles** — the reference-free judge carries real, transferable signal about separator quality (holes, voice-leakage, artifact, α-error). The labels also keep ranking the median ensemble above residual-MDX23C.

**Why it is not yet production-trustworthy:** the 21 works span only **4 ensembles and 2 accompaniment families** (orchestra ×3 corpora + organ). 14 of 21 groups are one Cantoría quartet — the per-work "21 groups" over-counts singer/venue diversity, which is exactly why the per-ensemble table above matters. Real deployment (pop/jazz/varied orchestras, many singers, many languages/venues) is still out of sample.

**Concrete gap → trustworthy:** add ensemble/accompaniment DIVERSITY, all already on tt-quietbox2:
1. **VocalSet (20 real singers) × PHENICX/Spheres orchestra donors × measured RIRs (Pori/Spheres)** — group-atomic by (singer, orchestra-donor, RIR); adds many distinct real-voice×real-orchestra ensembles with exact linear targets. *(This v2 delivered the real-work core first; the synthesizable expansion is the next increment.)*
2. **Cantolopera** STRUMENTALE lossy pairs (Mac `calibration/input_data`, 63 wav) → weak `cantolopera_corpus` group, `lossy_preview`, threshold-ineligible.
3. Choir sets (Choral Singing / Dagstuhl / ESMUC — a-cappella negative/stress) and Spheres orchestra works.
Target: ≥8–10 distinct ensembles across ≥4 accompaniment families, then re-run this exact pipeline. Only then does the label graduate from "work-validated / approaching-trustworthy" to "production-trustworthy".

## 7. Saved artifacts
- research6 `~/judge_models_v2/judge_<head>.joblib` (6 regressors) + `MODELS_INDEX.json`
- research6 `~/judge_build_v2/`: `judge_dataset.npz` (X 44×20, per-head y, per-work groups), `report.json` (per-work LOWO), `report_corpus.json` (per-ensemble LOWO), `rows_meta.json`, `examples_cache.npz`
- Mac `calibration/judge_baseline_report_v2.md` (this) + `judge_models_v2_summary.md`
- v1 baseline retained at `~/judge_models/` and `calibration/judge_baseline_report.md`.
