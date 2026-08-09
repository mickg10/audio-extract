# Judge baseline — trained model summary

- Location (research6): `~/judge_models/judge_<head>.joblib` (6 files) + `MODELS_INDEX.json`
- Repo commit `2837caa` · 2026-08-08 · GPU RTX PRO 4500 Blackwell
- Dataset: **106 examples**, **10 LOWO groups** = 4 independent real opera works (verdi, puccini, donizetti, aalto) + 6 independent synthetic-orchestra groups (ae-opera vocal-free controls + synthetic voice). 2 anchor recipes each (residual:MDX23C, median:MDX23C+MelBand+BS).
- Estimator: per-head `HistGradientBoostingRegressor(max_depth=3, max_iter=150, lr=0.08, l2=1.0)`, full-fit; evaluation = LeaveOneGroupOut over works.

| head (model file `judge_<head>.joblib`) | LOWO MAE | median-baseline MAE | beats baseline |
|---|---:|---:|:--:|
| event_hole_db_p90 | 2.436 | 1.756 | no |
| event_hole_db_max | 2.942 | 2.432 | no |
| retained_voice_db_p90 | 8.687 | 12.062 | **YES (−28%)** |
| retained_voice_coef_p90 | 0.102 | 0.092 | no |
| artifact_ratio_p90 | 0.063 | 0.111 | **YES (−43%)** |
| alpha_error_p90 | 0.134 | 0.119 | no |

- **Learn now (generalize across held-out works):** `retained_voice_db_p90`, `artifact_ratio_p90` (the voice-leakage / orthogonal-artifact axes).
- **Honest claim label:** PILOT / synthetically-validated baseline — NOT a trustworthy production judge.
- **Concrete gap:** raise independent REAL works 4 → 20–30 (Spheres +2, then VocalSet×PHENICX synthesizable + Cantoría, all already on tt-quietbox2), re-run this pipeline. Full write-up: `judge_baseline_report.md`.
