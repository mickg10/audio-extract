# Step 5 — WORK-LEVEL calibration_v2 + LOWO frozen-selector (first work-level certification numbers)

_Generated 2026-08-07T03:11:03.751271+00:00 on research6._

GPU: `NVIDIA RTX PRO 4500 Blackwell, 32623 MiB, 28317 MiB, 595.84`  |  total scoring wall: **1749.8 s**

> **CLAIM DISCIPLINE.** This is reported as **PIPELINE VALIDATION** (does the frozen calibration -> select_v3 path run end-to-end and terminate correctly per held-out work). The statistical **risk claim is DEFERRED** until the ~29-independent-work bar; with 5 fit works (Verdi held external) the work-atomic certified-risk bound cannot reach the targets, and abstention is the CORRECT outcome, not a failure.

## 1. Expanded work / condition inventory

Grouping unit = **WORK** (a work's conditions never split across train/test). Verdi is the frozen external test-v1 and is in **no** calibration fit.

| work | conditions | n_cond | linear-exact (dB) | voc_frac | stereo side/mid | source |
|---|---|---|---|---|---|---|
| donizetti | dry | 1 | -91.6 | 0.11 | 0.0 | Bologna anechoic Italian opera - bologna_d |
| puccini | dry | 1 | -94.0 | 0.0406 | 0.0 | Bologna anechoic Italian opera - bologna_p |
| aalto_mozart | dry, hall | 2 | -78.4/-92.1 | 0.5959/0.5654 | 0.0/0.0 | Aalto anechoic aalto_mozart (Mozart, Don G |
| spheres_mozart | dry, hall, stereo | 3 | -94.7/-82.7/-82.7 | 0.1199/0.0477/0.048 | 0.0/0.0/0.772837 | The Spheres Dataset (Mozart) orchestra + b |
| spheres_tchaikovsky | dry, hall, stereo | 3 | -106.3/-93.3/-93.3 | 0.12/0.0612/0.0595 | 0.0/0.0/0.901329 | The Spheres Dataset (Tchaikovsky) orchestr |
| verdi | dry | 1 | -97.9 | 0.0901 | 0.0 | Bologna anechoic Italian opera - bologna_v |

**Measured-hall / stereo rendering recipe (Spheres-Mozart, Spheres-Tchaikovsky):**
- Orchestra = mono sum of every Spheres instrument stem over a fixed 90 s window (14 stems, 48k->44.1k). Voice = borrowed anechoic Aalto soprano (['mozart_sopr_6.mp3']), scaled to ~12% mix energy.
- **RIR:** The Spheres measured RIR `source_Vln_1` (shape [23, 32768] = receivers x taps @48k, resampled to 44.1k). The SAME RIR is applied IDENTICALLY to voice and orchestra -> target A stays exact by linear construction (mix = H(orch)+H(voice) = H(orch+voice), target = H(orch)).
- Conditions: **dry** (H=identity, dual-mono); **hall** (H = RIR receiver 0, dual-mono, MEASURED reverb); **stereo** (H = RIR receivers {0,12}, a spaced stereo receiver pair -> genuine measured-hall stereo). Aalto-Mozart keeps its dry + synthetic-RT60 hall (one work).
- Note: convolution lowers hall/stereo voc_frac (~0.05) vs dry (0.12) — a real acoustic effect (the continuous orchestra gains more steady-state reverb energy than the intermittent voice).

## 2. Per (work, recipe) worst proxies + truth flags

truth thresholds: orchestral_theft=20.0, vocal_leakage=0.3, event_hole=18.0, fullness=6.0, stereo=0.5

| work | recipe | theft_mean | min_rc_si | eh_depth | v_intf | band | stereo_err | bad(theft/leak/hole/full/st) |
|---|---|---|---|---|---|---|---|---|
| aalto_mozart | ens_median:MDX23C/MelBand/BS | 0.005325 | 40.91 | 8.15341 | 0.03496 | 1.42 | 2e-05 | ..... |
| aalto_mozart | residual:BS | 0.013175 | 35.117 | 18.69312 | 0.10457 | 3.015 | 2e-05 | ..X.. |
| aalto_mozart | residual:MDX23C | 0.06317 | 20.326 | 9.38095 | 0.04314 | 1.834 | 0.0 | ..... |
| aalto_mozart | residual:MelBand | 0.062384999999999996 | 18.09 | 4.92077 | 0.02755 | 1.592 | 4e-05 | X.... |
| donizetti | ens_median:MDX23C/MelBand/BS | 0.0042 | 47.661 | 16.02523 | 0.01553 | 1.419 | 0.0 | ..... |
| donizetti | residual:BS | 0.01751 | 35.135 | 27.19518 | 0.00786 | 2.255 | 0.0 | ..X.. |
| donizetti | residual:MDX23C | 0.01326 | 37.669 | 15.13426 | 0.02172 | 2.16 | 0.0 | ..... |
| donizetti | residual:MelBand | 0.122 | 18.222 | 15.23927 | 0.01964 | 1.435 | 1e-05 | X.... |
| puccini | ens_median:MDX23C/MelBand/BS | 0.00882 | 41.11 | 9.09496 | 0.08252 | 0.252 | 0.0 | ..... |
| puccini | residual:BS | 0.02292 | 32.797 | 10.53059 | 0.03986 | 0.242 | 0.0 | ..... |
| puccini | residual:MDX23C | 0.01678 | 35.544 | 8.7273 | 0.07024 | 0.535 | 0.0 | ..... |
| puccini | residual:MelBand | 0.03723 | 28.594 | 12.43218 | 0.44931 | 0.291 | 1e-05 | .X... |
| spheres_mozart | ens_median:MDX23C/MelBand/BS | 0.0006833333333333332 | 62.297 | 5.79269 | 0.03966 | 0.525 | 0.00174 | ..... |
| spheres_mozart | residual:BS | 0.0041199999999999995 | 42.214 | 5.6311 | 0.03689 | 0.482 | 0.00078 | ..... |
| spheres_mozart | residual:MDX23C | 0.034010000000000006 | 23.212 | 8.93545 | 0.04619 | 1.245 | 0.00417 | ..... |
| spheres_mozart | residual:MelBand | 0.0007833333333333334 | 61.638 | 5.5582 | 0.0409 | 0.522 | 0.00111 | ..... |
| spheres_tchaikovsky | ens_median:MDX23C/MelBand/BS | 0.004113333333333333 | 44.403 | 8.15307 | 0.1743 | 0.395 | 0.00286 | ..... |
| spheres_tchaikovsky | residual:BS | 0.0020566666666666663 | 49.246 | 10.52722 | 0.1621 | 0.354 | 0.0022 | ..... |
| spheres_tchaikovsky | residual:MDX23C | 0.05033666666666667 | 17.566 | 5.8846 | 0.21082 | 0.657 | 0.00581 | X.... |
| spheres_tchaikovsky | residual:MelBand | 0.04938333333333333 | 22.582 | 10.61666 | 0.16411 | 0.433 | 0.0024 | ..... |
| verdi | ens_median:MDX23C/MelBand/BS | 0.02057 | 33.793 | 6.32108 | 0.09511 | 0.442 | 1e-05 | ..... |
| verdi | residual:BS | 0.01829 | 34.755 | 20.89275 | 0.18218 | 0.652 | 0.0 | ..X.. |
| verdi | residual:MDX23C | 0.03518 | 29.169 | 11.98782 | 0.10761 | 1.161 | 0.0 | ..... |
| verdi | residual:MelBand | 0.0526 | 25.575 | 8.1738 | 0.21478 | 0.355 | 1e-05 | ..... |

## 3. calibration_v2 per-defect tau table (fit on 5 non-Verdi works)

cert = Learn-Then-Test certifiable (Clopper-Pearson one-sided UCB <= target, delta=0.05). tau = certification threshold on calibrated severity. **work_model** unit groups by (work,recipe); **work** unit groups by work (atomic — the honest independent-work count).

| defect | critical? | fit units (wm) | truly_bad | strongest cert (wm) | work_model cert @.10/.15/.20 | tau @.10/.15/.20 | **work-atomic** cert @.10/.15/.20 |
|---|---|---|---|---|---|---|---|
| orchestral_theft | True | 20 | 3 | 0.2 | n/n/Y | -/-/0.435 | n/n/n |
| vocal_leakage | True | 20 | 1 | 0.15 | n/Y/Y | -/0.351/0.351 | n/n/n |
| event_hole | True | 20 | 2 | 0.2 | n/n/Y | -/-/0.445 | n/n/n |
| fullness | False | 20 | 0 | 0.15 | n/Y/Y | -/0.251/0.251 | n/n/n |
| stereo | False | 20 | 0 | 0.15 | n/Y/Y | -/0.006/0.006 | n/n/n |

**Map knots (isotonic raw->severity, GLOBAL not per-recording):**
- `orchestral_theft`: 20 knots; x[min..max]=[0.000683..0.122], y[min..max]=[0.0..0.54445]
- `vocal_leakage`: 40 knots; x[min..max]=[0.00036..0.44931], y[min..max]=[0.0006..0.74885]
- `event_hole`: 40 knots; x[min..max]=[2.539945..27.195185], y[min..max]=[0.070554..0.755422]

## 4. Leave-one-work-out frozen-selector (the deliverable)

> **BUG FOUND (blocks the as-shipped certified path).** `cli_autonomous.severity_cells_from_store`: per-case cells are keyed by challenge_id (positional: the tuple (case['challenge_id'], r['candidate_recipe_id'], result) is unpacked as (recipe_id, model, result) and key=recipe_id). So every recipe loses its event_hole/vocal_leakage/fullness/stereo cells (theft is keyed correctly by candidate_recipe_id), and a phantom challenge-keyed candidate collects them. select_autonomous therefore returns no_acceptable_candidate for ALL works at ALL targets for a spurious reason (missing cells -> ucb=1.0).  
> **Impact:** Blocks the certified recipe-spec select path. LOWO outcomes below use a correctly-keyed select_v3 (build_candidates_correct) to show the TRUE risk-gate behavior; the as-shipped status is recorded per fold as 'shipped_status'.  
> **Fix:** `key by candidate_recipe_id: `key = model if str(model).startswith('sha256:') else model` (i.e. use r['candidate_recipe_id'], not case['challenge_id']).`

For each held-out work: freeze calibration on the OTHER works, run the COMPLETE select_v3 on the held-out work's cases. Verdi always excluded from every fit. Because of the bug above, the as-shipped `cli_autonomous.select_autonomous(calibration_path=frozen)` returns `no_acceptable_candidate` for EVERY work/target (spurious — missing cells). The table below is the **correctly-keyed select_v3** (same frozen maps + taus, cells pooled per recipe) = the intended certified behavior; the as-shipped status is shown alongside. Primary target = **0.20** (loosest).

| held-out work | fit works | select_v3 @0.20 | winner recipe | bounding (non-cert critical taus) | as-shipped | reason |
|---|---|---|---|---|---|---|
| donizetti | 4 | **no_acceptable_candidate** | ens_median:MDX23C/MelBand/BS | ['orchestral_theft'] | no_acceptable_candidate | no candidate passes the calibrated risk gates |
| puccini | 4 | **no_acceptable_candidate** | residual:BS | ['orchestral_theft'] | no_acceptable_candidate | no candidate passes the calibrated risk gates |
| aalto_mozart | 4 | **no_acceptable_candidate** | ens_median:MDX23C/MelBand/BS | [] | no_acceptable_candidate | no candidate passes the calibrated risk gates |
| spheres_mozart | 4 | **no_acceptable_candidate** | residual:BS | ['orchestral_theft'] | no_acceptable_candidate | no candidate passes the calibrated risk gates |
| spheres_tchaikovsky | 4 | **no_acceptable_candidate** | residual:MDX23C | ['orchestral_theft'] | no_acceptable_candidate | no candidate passes the calibrated risk gates |
| verdi (TEST-v1) | 5 | **final** | ens_median:MDX23C/MelBand/BS | [] | no_acceptable_candidate | passes hard + calibrated gates; leave-one-evidence |

**All targets per held-out work:**
- **donizetti**: .1:no_acceptable_candidate(ens_median:MDX23C/MelBand/BS) | .15:no_acceptable_candidate(ens_median:MDX23C/MelBand/BS) | .2:no_acceptable_candidate(ens_median:MDX23C/MelBand/BS)
- **puccini**: .1:no_acceptable_candidate(residual:BS) | .15:no_acceptable_candidate(residual:BS) | .2:no_acceptable_candidate(residual:BS)
- **aalto_mozart**: .1:no_acceptable_candidate(ens_median:MDX23C/MelBand/BS) | .15:no_acceptable_candidate(ens_median:MDX23C/MelBand/BS) | .2:no_acceptable_candidate(ens_median:MDX23C/MelBand/BS)
- **spheres_mozart**: .1:no_acceptable_candidate(residual:BS) | .15:no_acceptable_candidate(residual:BS) | .2:no_acceptable_candidate(residual:BS)
- **spheres_tchaikovsky**: .1:no_acceptable_candidate(residual:BS) | .15:no_acceptable_candidate(residual:BS) | .2:no_acceptable_candidate(residual:MDX23C)
- **verdi**: .1:no_acceptable_candidate(residual:MelBand) | .15:no_acceptable_candidate(ens_median:MDX23C/MelBand/BS) | .2:final(ens_median:MDX23C/MelBand/BS)

**Aggregate coverage / abstention (non-Verdi works):**

| target | n_final | n_abstain | n_works | verdi (test-v1) |
|---|---|---|---|---|
| 0.1 | 0 | 5 | 5 | no_acceptable_candidate |
| 0.15 | 0 | 5 | 5 | no_acceptable_candidate |
| 0.2 | 0 | 5 | 5 | final |

**Accepted-outcome failures** (a `final` whose winner is truly_bad on a critical defect on its OWN held-out work — the risk-relevant error):
- NONE at target 0.20 (every `final` winner is truly-good on all critical defects on its held-out work).

## 5. Key questions

**Does the median ensemble certify anywhere?** YES, once — `ens_median:MDX23C/MelBand/BS` reaches a certified `final` for ['verdi'] (at target risk **0.20** only, with the fuller **5-work** calibration, via the correctly-keyed select_v3). It passes hard + calibrated gates, is leave-one-evidence-out stable, and is separated from the runner-up. This is the FIRST work-level certified selection. It does NOT certify on the 4-work LOWO folds: leaving a work out drops the orchestral_theft tau below certifiable at 0.20 (the n-boundary — 5 works barely certifies theft@0.20, 4 does not), and it never certifies at 0.10/0.15 (n far too small). The risk claim stays DEFERRED to the ~29-work bar.

**Does measured-hall change certifiability vs dry?** Critical-defect truly_bad counts by condition:

| condition | n(rec-cases) | theft-bad | leak-bad | hole-bad |
|---|---|---|---|---|
| dry | 24 | 3 | 1 | 3 |
| hall | 12 | 2 | 0 | 0 |
| stereo | 8 | 1 | 0 | 0 |

_Honest, nuanced answer:_ within this small n, measured-hall does NOT change the certification OUTCOME — every fold abstains regardless of condition, and Verdi (dry) certifies; the binding constraint is the orchestral_theft tau's n-sensitivity (5 vs 4 fit works), a WORK-count effect, not a dry-vs-hall effect (theft is a work-level control-side quantity). At the CONDITION level the effects are mixed and instructive: reverb SMEARS event-holes (all event_hole truly_bad cases are on DRY — BS's deep holes; hall/stereo depths fall below the 18 dB threshold), and it crushes broadband SI-SDR (Aalto median dry->hall 23.7->12.5 dB) and raises masking uncertainty — but SI-SDR is not a critical gate. The MEASURED-hall Spheres cases behave like the synthetic-hall Aalto case (reverb lowers holes, raises interference). The STEREO condition is well-preserved by every recipe (stereo_width_err ~0.001-0.006 << 0.5 -> stereo is never a binding defect). Net: hall is a genuine additional CONDITION axis but, at n=6 works, certifiability is gated by work-count, not reverb.

### Verdi frozen test-v1 (never in any fit) — reported separately
- target 0.1: **no_acceptable_candidate** (best_available `residual:MelBand`, non-cert taus ['orchestral_theft', 'vocal_leakage', 'event_hole']); as-shipped select_autonomous: no_acceptable_candidate.
- target 0.15: **no_acceptable_candidate** (best_available `ens_median:MDX23C/MelBand/BS`, non-cert taus ['orchestral_theft', 'event_hole']); as-shipped select_autonomous: no_acceptable_candidate.
- target 0.2: **final** -> winner `ens_median:MDX23C/MelBand/BS`; as-shipped select_autonomous: no_acceptable_candidate.
Verdi's median-ensemble `final` at 0.20 is a legitimate accept: on Verdi the median residual is truly-good on ALL critical defects (theft min-construction SI-SDR 33.8 dB > 20; event-hole 6.3 dB < 18; vocal-interference 0.095 < 0.30), so it is NOT an accepted-outcome failure.

## 6. GPU timing / disk / bugs
- separations: 66, total sep time 577.8 s, mean 8.8 s. Stereo stems observed: {(4851000, 2), (4062448, 2), (4674600, 2), (3969000, 2)}
- scoring wall: 1749.8 s (~29.2 min), 6 works / 11 (work,condition) cases / 4 recipes on RTX PRO 4500. Disk held flat (concatenated controls deleted per work, stems symlinked, temp cleared per separation).
- **BUG (blocks certified path):** `cli_autonomous.severity_cells_from_store` keys per-case cells by `challenge_id` instead of `candidate_recipe_id` (see the boxed note in section 4). The whole LOWO table therefore uses a correctly-keyed select_v3 re-run; as-shipped select_autonomous returns no_acceptable everywhere. No other tracebacks; the GPU scoring run completed cleanly.
- Stereo separation verified: stereo mixes -> stereo stems (2-channel residuals); dual-mono cases -> width_err ~= 0 as expected.

---
_Artifacts: `calibration_v2.json` (fit on 5 non-Verdi works), `calib_lowo_<work>.json` (per-fold frozen artifacts), `step5_results.json` (raw scored matrix)._
