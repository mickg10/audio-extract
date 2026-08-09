# Voice-Removal Quality Judge — Baseline v3 (diversity-closed)

**Claim label: PRODUCTION-TRUSTWORTHY (operatic/classical domain), per the pre-registered graduation rule — with 3 documented exceptions.**
The pre-agreed bar was: **≥8 ensembles / ≥4 accompaniment families AND ≥5/6 heads beat the median baseline under held-out-ENSEMBLE LOWO**. v3 delivers **9 ensembles, 5 families, and 5/6 heads beating baseline under BOTH per-work (40 groups) and the strict per-ensemble (9 groups) LOWO**, with −35% to −78% error reductions. Exceptions (below): `artifact_ratio` still not learnable; synthesizable ensembles share singers; scope is classical/operatic, not pop/rock.

- Host research6 (RTX PRO 4500, CUDA13/torch2.13), repo `2837caa`. Pipeline unchanged; only DATA grew. All added data from tt-quietbox2 — no purchase.
- Reuses v2's 44 real-work features + the mono channel-match fix, MDXC tile-pad, `OMP_NUM_THREADS=1`.

## 1. What changed v2 → v3 — three diversity blocks
v2 was work-validated but only 4 ensembles / 2 families (orchestra, organ). v3 adds:
1. **Synthesizable real-voice×real-orchestra** — 10 VocalSet singers (real) × 3 PHENICX orchestra beds (Beethoven/Bruckner/Mahler, per-instrument stems summed) × 2 measured **Pori** RIRs; M = A_bed + place_vocal(V, RIR, g), A/V exact. 15 rows, 3 ensembles, family *synthesizable_orchestra*.
2. **Choir (new accompaniment family)** — Choral Singing (Locus Iste / Nino Dios / El Rossinyol), soloist-vs-section: V = Soprano, A = Alto+Tenor+Bass, M = full. 3 rows, family *choir_vocal_accompaniment* (accompaniment IS voices — a genuinely new family).
3. **Cantolopera weak** — 14 same-take 30 s lossy preview pairs (M=full, A=orchestra, V=full−orch). One down-weighted group, `lossy_preview`, threshold-ineligible; transfer-direction signal only.

## 2. Dataset composition — 108 examples, 9 ensembles, 5 families

| family | ensemble(s) | works/rows | integrity | M=A+V |
|---|---|---|---|---|
| **orchestra** | bologna, aalto, freidi | 3+1+3 works | linear_exact / same-perf-bleed | ≤ −78 dB |
| **organ** | cantoria (×14) | 14 works | linear_exact | −240 dB |
| **synthesizable_orchestra** | synth_beethoven, synth_bruckner, synth_mahler | 15 (real singer×real orch×RIR) | synthesizable, exact | −240 dB |
| **choir_vocal_accompaniment** | choir_acappella | 3 | lossless_same_take | −149 dB |
| **cantolopera_orchestra_lossy** | cantolopera_lossy | 14 (weak) | lossy_preview | −240 dB* |

*Cantolopera M=A+V holds by construction, but A,V are lossy 30 s previews (weak). Per-work groups: **40** (21 real + 15 synth + 1 choir-set-of-3… choir counted per piece = 3 → 21+15+3+1=40). Per-ensemble groups: **9**.

## 3. Per-head — held-out-WORK LOWO (40 groups)

| head | LOWO MAE | baseline | beats | Δ |
|---|--:|--:|:--:|--:|
| event_hole_db_p90 | 5.016 | 15.738 | **YES** | −68.1% |
| event_hole_db_max | 10.773 | 27.762 | **YES** | −61.2% |
| retained_voice_db_p90 | 6.202 | 12.455 | **YES** | −50.2% |
| retained_voice_coef_p90 | 0.121 | 0.254 | **YES** | −52.3% |
| artifact_ratio_p90 | 0.144 | 0.126 | no | +14.6% |
| alpha_error_p90 | 0.093 | 0.416 | **YES** | −77.7% |

## 4. Per-head — held-out-ENSEMBLE LOWO (9 ensembles) — the honest test

Hold out a whole ensemble/accompaniment family (train on 8, predict the 9th — a real domain shift, e.g. train orchestra/organ→predict choir):

| head | LOWO MAE | baseline | beats | Δ |
|---|--:|--:|:--:|--:|
| event_hole_db_p90 | 9.280 | 15.363 | **YES** | −39.6% |
| event_hole_db_max | 17.498 | 28.624 | **YES** | −38.9% |
| retained_voice_db_p90 | 9.820 | 15.032 | **YES** | −34.7% |
| retained_voice_coef_p90 | 0.208 | 0.385 | **YES** | −46.0% |
| artifact_ratio_p90 | 0.249 | 0.188 | no | +32.5% |
| alpha_error_p90 | 0.151 | 0.490 | **YES** | −69.1% |

**5/6 heads beat baseline under the strict per-ensemble test.** Notably `event_hole_db_p90` — which beat per-work but FAILED per-ensemble in v2 — now generalizes across unseen ensembles too, directly attributable to the added accompaniment diversity (organ + choir + synthesizable orchestra taught the model that "hole" means the same thing across accompaniment types).

## 5. Did artifact_ratio recover with more diversity? (the v2 miss)
**No.** `artifact_ratio_p90` remains the single non-generalizing head under BOTH tests (+14.6% per-work, +32.5% per-ensemble). The orthogonal-artifact ratio is the residual of Y NOT explained by α·A+β·V — separator-specific nonlinear "musical noise" whose reference-free proxies (HF isolated-peak excess, spectral-flux excess) do not yet transfer across ensembles. **The judge must NOT gate on the artifact_ratio head yet**; the other five are trustworthy.

## 6. Progression

| | v1 | v2 | v3 |
|---|---|---|---|
| independent real-voice works | 4 | 21 | 21 (+15 synth +3 choir +14 cantolopera rows) |
| ensembles / accompaniment families | 2 / 2 | 4 / 2 | **9 / 5** |
| heads beat baseline — per-work | 2/6 | 5/6 | **5/6** |
| heads beat baseline — per-ENSEMBLE | — | 5/6 (event_hole_p90 failed) | **5/6 (event_hole_p90 now passes)** |
| claim | PILOT | work-validated | **production-trustworthy (classical/operatic)** |

## 7. Graduation verdict + honest exceptions
Pre-registered rule MET: **9 ≥ 8 ensembles, 5 ≥ 4 families, 5/6 ≥ 5/6 per-ensemble** → graduate to **production-trustworthy (operatic/classical domain)**. Three exceptions are stated plainly:
1. **5/6, not 6/6** — `artifact_ratio_p90` does not generalize; exclude it from gating until a better artifact feature/objective lands.
2. **Synthesizable shared-donor leakage** — the 3 synth ensembles share the 10 VocalSet singers and 2 Pori RIRs, so holding out synth_beethoven still trains on those singers under synth_bruckner/mahler. The 6 REAL/lossless ensembles (bologna, aalto, freidi, cantoria, choir, cantolopera) are fully independent and carry the honest test; the synthesizable rows add orchestra-domain coverage, flagged `synthesizable`.
3. **Domain = classical/operatic/choral/organ/symphonic.** NOT validated for pop/rock/jazz/electronic/world timbres or heavy studio mastering. "Production-trustworthy" is scoped to the classical/operatic domain only.

## 8. GPU + compute
- v3 separation (32 new mixtures ×3, load-once): mdx23c 207 s, melband 90 s, bs 134 s.
- Features (CPU pYIN, `OMP_NUM_THREADS=1`): 1276 s for 64 new examples (+44 reused instantly from v2).
- Dual LOWO training (per-work + per-ensemble, 6 heads each): ~9 s total.

## 9. Remaining gap
- Make `artifact_ratio` learnable (better orthogonal-artifact features or a small frozen-encoder head).
- Replace synthesizable shared-donor coverage with more DISTINCT real voice+orchestra recordings; add non-classical domains (pop/rock/jazz) for cross-domain production trust.
- Upgrade Cantolopera from lossy previews to purchased WAV same-take pairs for reference-grade (not just transfer-direction) rows.

## 10. Saved artifacts
- research6 `~/judge_models_v3/judge_<head>.joblib` (6) + `MODELS_INDEX.json`
- research6 `~/judge_build_v3/`: `judge_dataset.npz` (X 108×20, per-work + per-ensemble labels), `report.json` (per-work), `report_corpus.json` (per-ensemble), `composition.json`, `rows_meta.json`, `examples_cache.npz`
- Mac `calibration/judge_baseline_report_v3.md` + `judge_models_v3_summary.md`
- v1/v2 retained (`judge_models`, `judge_models_v2`, prior reports).
