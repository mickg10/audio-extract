# Judge baseline v3 — trained model summary (diversity-closed)

- research6 `~/judge_models_v3/judge_<head>.joblib` (6) + `MODELS_INDEX.json`. Commit `2837caa`, RTX PRO 4500.
- **108 examples · 40 per-work groups · 9 ensembles · 5 accompaniment families.**
  Families: orchestra (bologna/aalto/freidi), organ (cantoria), synthesizable_orchestra (VocalSet×PHENICX×Pori), choir_vocal_accompaniment (Choral Singing), cantolopera_orchestra_lossy (weak). All added data from tt-quietbox2 (no purchase).

| head | per-work LOWO | base | beats | per-ENSEMBLE LOWO | base | beats |
|---|--:|--:|:--:|--:|--:|:--:|
| event_hole_db_p90 | 5.016 | 15.738 | **YES −68%** | 9.280 | 15.363 | **YES −40%** |
| event_hole_db_max | 10.773 | 27.762 | **YES −61%** | 17.498 | 28.624 | **YES −39%** |
| retained_voice_db_p90 | 6.202 | 12.455 | **YES −50%** | 9.820 | 15.032 | **YES −35%** |
| retained_voice_coef_p90 | 0.121 | 0.254 | **YES −52%** | 0.208 | 0.385 | **YES −46%** |
| artifact_ratio_p90 | 0.144 | 0.126 | no +15% | 0.249 | 0.188 | no +33% |
| alpha_error_p90 | 0.093 | 0.416 | **YES −78%** | 0.151 | 0.490 | **YES −69%** |

- **5/6 heads beat baseline under BOTH per-work and the strict per-ensemble LOWO.** event_hole_db_p90 now passes per-ensemble (failed in v2).
- **artifact_ratio_p90 is the sole non-generalizing head — do NOT gate on it.** The other 5 are trustworthy.
- **Claim: PRODUCTION-TRUSTWORTHY (operatic/classical domain)** per the pre-registered rule (≥8 ensembles / ≥4 families / ≥5/6 per-ensemble — all met). Exceptions: 5/6 not 6/6 (artifact_ratio); synthesizable ensembles share VocalSet singers (real ensembles carry the test); scope classical/operatic, not pop/rock.
- Progression: v1 2/6 (PILOT) → v2 5/6 per-work (work-validated) → v3 5/6 per-ensemble across 9 ensembles / 5 families. Full write-up: `judge_baseline_report_v3.md`.
