# Voice-Removal Judge — Frozen Selection-Regret Test (oracle Gate 3/4)

**VERDICT: SHADOW/RESEARCH-BETA.**  Promotion rule: leave SHADOW only if the judge policy beats BOTH fixed baselines on regret AND selective performance improves as uncertain works are abstained. Result: beats always-median = **False**, beats always-MDX23C = **True**, selective-risk improves = **True**.

- Host research6, repo `88884a2` (P0 contract fixes). 32-dim features (task one-hot + availability bits), **coherent** predictions (`predict_coherent`: nonnegative heads, hole_max ≥ hole_p90). **Full-track** scoring. **artifact_ratio head DISABLED** (v3 showed it does not generalize) — excluded from the selection policy.
- This REPLACES the old circular-threshold selection: feasibility thresholds are calibrated on the **training groups' EXACT labels**, never on the judge's own predicted scores over the target pool (that circularity was the P0 bug).

## 1. What is tested

Leave-one-**ensemble/session**-out. For each held-out ensemble G: train the judge on the other ensembles; calibrate feasibility caps from TRAIN exact labels; for each held-out WORK, the judge predicts every candidate (coherent) and **selects the predicted-feasible candidate minimizing predicted `retained_voice_db_p90`**, or **ABSTAINS**, or **OOD-falls-back** to median. The CHOSEN candidate is then scored on EXACT labels and compared to the exact-oracle choice (regret).

**Feasibility caps** (domain truth, not the pool): accompaniment-hole depth `event_hole_db_p90 ≤ 18.0 dB`, retained-voice `|β| ≤ 0.3`, and an `alpha_error` cap taken per-fold from the TRAIN exact-label 90th percentile. A per-head **safety margin = the head's calibrated p90 absolute error** is subtracted from each cap, so the judge only accepts a candidate it is confidently under-cap on (else it abstains).

## 2. Corpus + candidate panel

- **40 exact-labeled works** across **8 ensembles**: aalto(2), bologna(3), cantoria(14), choir(3), freidi(3), synth_beethoven(5), synth_bruckner(5), synth_mahler(5). (Cantolopera excluded — lossy, no exact A,V.)
- Candidate panel per work: **residual:MDX23C, median(MDX23C,MelBand,BS), STFT geometric-median, convex-fusion(uniform)** — 160 (work,candidate) rows. `htdemucs_ft` **absent** from the locked model dir → logged and skipped (panel = 4).

## 3. Terminal states (Gate 4 — distinct + counted)

| state | count | meaning |
|---|--:|---|
| successful_selection | 0 | judge found ≥1 confident-feasible candidate and chose |
| no_feasible_candidate | 26 | in-domain but no candidate confidently under caps → ABSTAIN |
| OOD_fallback | 14 | work outside training manifold → defer to median champion |

- Coverage (judge made a choice) = **0.350**; abstention rate = **0.650**.
- Correct abstentions (no exact-feasible option existed) = 15; missed opportunities (abstained though an exact-feasible option existed) = 11.

## 4. Regret vs baselines (exact-label evaluation)

Regret = chosen candidate's exact `retained_voice_db_p90` − exact-oracle's (lower = closer to the best possible; 0 = matched the oracle). retained_voice_db: lower = less voice left.

| policy | mean top-1 regret (dB) | worst-work regret | notes |
|---|--:|--:|---|
| **judge (conservative)** | **0.352** | 0.736 | margin+OOD; here = OOD-fallback→median on all covered works |
| judge (point / raw ranking) | 0.722 | — | no margin, always-on — tests the judge's own ranking |
| always-median | 0.234 | — | v4-lite champion |
| always-MDX23C | 2.707 | — | cheap fallback |
| exact-oracle | 0.000 | 0.000 | upper bound (knows A,V) |

- **Point-estimate policy** (judge's raw ranking, no safety margin): regret **0.722 dB** vs always-median 0.234 → improvement over median **-0.488 dB** (negative = WORSE than median), with **10** catastrophic hole violations (vs median's 5). So even without the conservative margin, the judge's ranking of the 4 candidates does not beat the fixed median.
- Judge **top-2** regret (oracle within judge's 2 best predicted-feasible): n/a dB.
- Improvement over always-median = **0.000 dB** (paired, +=judge better); over always-MDX23C = **1.590 dB**.
- Catastrophic violations among ACCEPTED choices: retained-voice |β|>0.3: **0**; event-hole>18.0dB: **5**; any-cap: **5**.

## 5. Selective-risk curve (reject-option)

Abstain the most-uncertain works (highest Mahalanobis to the train manifold) and recompute on the kept set. If mean regret / violations fall as the reject fraction rises, the judge's uncertainty is meaningful.

| reject frac | n kept | mean regret | worst regret | violation rate |
|--:|--:|--:|--:|--:|
| 0.0 | 9 | 0.352 | 0.736 | 0.000 |
| 0.1 | 8 | 0.344 | 0.736 | 0.000 |
| 0.2 | 7 | 0.288 | 0.736 | 0.000 |
| 0.3 | 6 | 0.336 | 0.736 | 0.000 |
| 0.4 | 5 | 0.257 | 0.506 | 0.000 |
| 0.5 | 4 | 0.321 | 0.506 | 0.000 |

## 6. Per-ensemble breakdown

| held-out ensemble | works | successful | no_feasible | OOD | judge mean regret |
|---|--:|--:|--:|--:|--:|
| aalto | 2 | 0 | 2 | 0 | n/a |
| bologna | 3 | 0 | 2 | 1 | 0.000 |
| cantoria | 14 | 0 | 5 | 9 | 0.393 |
| choir | 3 | 0 | 0 | 3 | n/a |
| freidi | 3 | 0 | 3 | 0 | n/a |
| synth_beethoven | 5 | 0 | 4 | 1 | 0.413 |
| synth_bruckner | 5 | 0 | 5 | 0 | n/a |
| synth_mahler | 5 | 0 | 5 | 0 | n/a |

## 7. Verdict + honest failure attribution

**Stays SHADOW/RESEARCH-BETA.** It does NOT beat always-median on regret.

**Two independent failures, both honest:**
1. **Calibration too wide for confident abstention.** `successful_selection = 0`: the per-head calibrated p90 absolute errors (event_hole ≈ 24–53 dB, |β| ≈ 0.39–0.73, retained_voice ≈ 14–17 dB across folds) are LARGER than the caps themselves (hole 18 dB, |β| 0.30), so `cap − margin` is negative and NO candidate is ever confidently under-cap. The judge therefore only ever abstains (26/40) or OOD-defers to median (14/40) — under this policy it simply IS always-median on every work it covers (improvement over median = 0.000).
2. **Ranking no better than median even without the margin.** The point-estimate policy (raw predictions, no abstention) has regret 0.722 dB — WORSE than always-median (0.234 dB, improvement -0.488) — and picks 10 cap-violating instrumentals vs median's 5. So the problem is NOT only calibration: the judge's per-head error exceeds the small exact-label gaps BETWEEN the 4 near-tied candidates, so it cannot rank them better than the fixed median heuristic. This is the whole point of Gate 3 — a judge that beats the median baseline on per-head MAE (v3) does NOT beat always-median at the actual selection task.

**Path forward (oracle's menu):** (a) a **frozen-encoder student** to cut per-head error below the inter-candidate gap and produce usable uncertainty; (b) a **wider, more-separated panel** — install HTDemucs (a different waveform family) so the candidates are not 4 near-tied spectral-mask residuals; (c) **more exact-labeled ensembles** so both the OOD/abstain machinery and the thresholds have denser support. Note the median champion itself violated the hole cap on 5 works, so 'always-median' is a floor, not a safe ceiling — a real tuner is still worth building, just not shippable on this contract yet.

## 8. Deliverables

- research6 `~/judge_models_v4/` (6 coherent per-head regressors, full-fit on all 40 works) + MODELS_INDEX.
- research6 `~/judge_build_v4/`: `panel.npz` + `panel_rows.json` (frozen candidate panel: 32-dim features + EXACT labels), `regret_summary.json`, `regret_per_work.json`, per-fold `regret_fold_<G>.json`.
- Mac `calibration/judge_selection_regret_report.md` (this).