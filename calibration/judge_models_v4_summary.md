# Judge v4 (fixed contract 88884a2) — model + regret-test summary

- research6 `~/judge_models_v4/judge_<head>.joblib` (6) + `MODELS_INDEX.json`. **32-dim** features (task one-hot + availability bits); inference MUST use `judge_train.predict_coherent` (nonnegative heads, hole_max ≥ hole_p90). artifact_ratio head trained but **DISABLED** for policy.
- Trained on the frozen candidate PANEL: 40 exact-labeled works × 4 candidates = 160 (feature, EXACT-label) rows. Per-head LOWO over 8 ensembles (coherent):

| head | lowo_mae | calibrated_p90_abs_err | beats baseline |
|---|--:|--:|:--:|
| event_hole_db_p90 | 10.617 | 24.085 | yes |
| event_hole_db_max | 22.807 | 53.608 | yes |
| retained_voice_db_p90 | 7.880 | 16.226 | yes |
| retained_voice_coef_p90 | 0.200 | 0.478 | yes |
| artifact_ratio_p90 | 0.179 | 0.293 | no |
| alpha_error_p90 | 0.164 | 0.430 | yes |

- **Frozen selection-regret test (Gate 3/4): VERDICT = SHADOW/RESEARCH-BETA.** Per-ensemble held-out selection; thresholds from TRAIN exact labels (hole>18 dB, |β|>0.30, alpha from exact dist); safety margin = calibrated_p90. Terminal states: successful 0 / no_feasible 26 / OOD_fallback 14 (coverage 0.35).
- **Judge regret 0.352 dB (conservative) / 0.722 dB (raw ranking) vs always-median 0.234, always-MDX23C 2.707.** Beats MDX23C, does NOT beat median (ties it by deferring; raw ranking is worse). The per-head calibrated errors (≥16–53 dB) exceed the caps and the inter-candidate gaps, so the judge can neither confidently abstain-calibrate nor out-rank the median heuristic. Selective-risk improves mildly (0.352→0.257).
- **Not a production tuner yet.** Fix path: frozen-encoder student (cut per-head error below inter-candidate gap), wider panel (add HTDemucs), more exact-labeled ensembles. Full write-up: `calibration/judge_selection_regret_report.md`.
