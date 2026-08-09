# Judge baseline v2 — trained model summary (REAL voices)

- Location (research6): `~/judge_models_v2/judge_<head>.joblib` (6) + `MODELS_INDEX.json`
- Commit `2837caa` · 2026-08-08 · GPU RTX PRO 4500. Estimator: `HistGradientBoostingRegressor(max_depth=3, max_iter=150, lr=0.08, l2=1.0)`, full-fit; eval = LeaveOneGroupOut over works.
- **Dataset: 44 examples, 21 independent REAL-VOICE works** = Bologna verdi/puccini/donizetti + Aalto (reused v1) + FreiDi 06/08/09 + Cantoría ×14. 4 ensembles / 2 accompaniment families (orchestra, organ). 2 anchor recipes each.

| head | per-work LOWO MAE | baseline | beats | per-ensemble LOWO | beats |
|---|--:|--:|:--:|--:|:--:|
| event_hole_db_p90 | 7.320 | 14.753 | **YES −50%** | 13.801 / 12.351 | no |
| event_hole_db_max | 18.383 | 26.323 | **YES −30%** | 28.670 / 37.403 | **YES** |
| retained_voice_db_p90 | 6.594 | 8.937 | **YES −26%** | 9.677 / 11.007 | **YES** |
| retained_voice_coef_p90 | 0.196 | 0.242 | **YES −19%** | 0.472 / 0.576 | **YES** |
| artifact_ratio_p90 | 0.125 | 0.111 | no | 0.1667 / 0.1674 | tie |
| alpha_error_p90 | 0.174 | 0.299 | **YES −42%** | 0.379 / 0.497 | **YES** |

- **Per-work LOWO: 5/6 beat baseline. Per-ensemble LOWO: 5/6 beat.** Robust under BOTH: event_hole_db_max, retained_voice_db_p90, retained_voice_coef_p90, alpha_error_p90.
- **Claim: WORK-VALIDATED baseline (approaching-trustworthy)** — NOT yet production-trustworthy (only 4 ensembles / 2 accompaniment families; Cantoría is 14 pieces of one quartet).
- **Gap:** add VocalSet×PHENICX/Spheres synthesizable (real singer×orchestra×RIR), Cantolopera weak rows, choir sets → ≥8–10 distinct ensembles, then re-run. Full write-up: `judge_baseline_report_v2.md`.
