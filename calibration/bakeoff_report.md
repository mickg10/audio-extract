# audio-extract Step 2 RECIPE BAKE-OFF — report

Generated from `/home/mickg/bakeoff_work/bakeoff_results.json` on research6 (RTX PRO 4500 Blackwell). Repo HEAD `6fc73b9` (recipe-spec challenge evaluation).

All scoring by the shipped modules only: `audio_extract.cli_autonomous.run_challenges(recipe_specs=...)` -> `challenges.exact_reference_error` + `metrics_v2.event_holes` (exact event-conditioned labels; injected vocal V = mixture - exact target).

## Executive summary

**Two best FEASIBLE recipes** (asymmetric constraint, pooled DEV/CALIB): **#1 `ens_median:MDX23C/MelBand/BS`** and **#2 `residual:MDX23C`**.

- **Winner: `ens_median:MDX23C/MelBand/BS`** — the per-sample MEDIAN of the three vocal estimates, then residual. It is the ONLY recipe that is simultaneously best-in-class on residual voice (pooled 0.033), spectral fullness (band 0.902 dB) and control theft (0.009..), with a competitive event-hole (10.211 dB) and the highest broadband SI-SDR. The median rejects each model's idiosyncratic failure (BS's gouging, MelBand's voice-passthrough) without inheriting it.
- **Runner-up: `residual:MDX23C` (== `native:MDX23C`)** — the robust single model. MDX23C's native instrumental head and its M-minus-vocals residual are numerically identical here (within 0.01 on every metric), so they are one recipe in two guises. Safe voice removal on every work, no catastrophic case.
- **Disqualified by the primary voice gate:** `residual:MelBand` and the MelBand-heavy mean ensembles. Mel-Band leaves almost the ENTIRE soprano on **Puccini** (interference 0.449, SI-SIR -0.3 dB) — a product-fatal residual-voice failure that dry/hall test tracks alone would NOT reveal.
- **Disqualified by the event-hole gate:** `residual:BS`. BS-Roformer removes voice aggressively (often the best interference) and has the best broadband SI-SDR, but it GOUGES the orchestra during vocal events (pooled hole 16.043 dB, worst; 27.2 dB on Donizetti) and has the worst spectral fullness. This is exactly the failure that event-conditioned hole depth exposes and SI-SDR hides.
- **Ensemble vs single:** the MEDIAN ensemble beats every single viable model on the metrics that matter; the weighted-MEAN ensembles do NOT — they trade voice for holes and inherit Mel-Band's voice/theft liabilities.
- **Regime:** dry is easy (all recipes remove voice cleanly); hall and especially real dense orchestra + chorus are the discriminating regimes. The winner (`ens_median`) is stable across all three; the mean-ensembles only look good on dry/hall holes and collapse on real-orchestra voice.

## Pair cases (prior one-mic truth recipe, reused)

Rendering recipe (unchanged from the truth run, `~/truth_pairs/build_pairs.py`): every source stem mono-downmixed (mean of mic channels) into ONE virtual capture; `orchestra_only` = sum of all NON-soloist family desk takes (take-001 of every desk; **chorus RETAINED on the orchestra/rest side** for the `soloist_vs_rest` ontology); `voice_ref` = single soloist take; `mix = orchestra_only + voice_ref` (linear-exact); one shared -3 dBFS peak gain applied identically to all three; dual-mono 44.1k float32. Aalto hall = same mix convolved with a fixed-seed RT60=1.5 s IR identically on both sides.

| case | role | variant | dur s | voice frac of mix |
|---|---|---|---|---|
| bologna_donizetti | dev | dry | 110.0 | 0.1100 |
| bologna_puccini | calib | dry | 106.0 | 0.0406 |
| aalto_mozart_dry | dev | dry | 90.0 | 0.5959 |
| aalto_mozart_hall | dev | hall | 90.0 | 0.5654 |
| bologna_verdi | test-v1 | dry | 92.1 | 0.0901 |

`bologna_donizetti` = DEV (soloist CANTO_SOLO + chorus retained as accompaniment); `bologna_puccini` = CALIB; `aalto_mozart_dry/hall` = DEV (Mozart Donna Elvira). **`bologna_verdi` = FROZEN test-v1** — scored below for context, EXCLUDED from selection & ceilings.

## Recipe grid

Native-instrumental probe (does the checkpoint emit an instrumental-tagged stem?): **MDX23C** -> ['instrumental', 'vocals'], **MelBand** -> ['other', 'vocals'], **BS** -> ['instrumental', 'vocals']. 
`native:MelBand` is **SKIPPED** — Mel-Band emits `(vocals)`+`(other)`, no instrumental-tagged stem, so a 'native' recipe would silently equal its residual (not faked). `native:BS` not requested by the grid.

| recipe id (friendly) | kind | recipe_id (sha) |
|---|---|---|
| residual:MDX23C | residual | `sha256:1fbb0f0a487ae2e3c7405d405665c030d65e58c3295ffb1bf6e47b5dca47754f` |
| residual:MelBand | residual | `sha256:e21228241f2dd677fa18a1027bd4f03d4818384b13eee080f3491fb248540c5d` |
| residual:BS | residual | `sha256:ad65af4f239ab60f911bc258f6f861c58a03f3bda8b987d684061453739e738f` |
| native:MDX23C | native | `sha256:6f3513af1980bd8af72b936540d4de8e122c598d293603598dfe9623b01d41ba` |
| ens_mean:MDX23C0.8/MelBand0.2 | ensemble_residual | `sha256:97c5461beb20f9f63441941116f52d93974b35a57ec19e39eee441c943cedc97` |
| ens_mean:MDX23C0.6/MelBand0.4 | ensemble_residual | `sha256:dd66bb4654c50b12f352d99f97c77c4845722dcb679d87e11d7bb13dfbfb1f35` |
| ens_mean:MDX23C0.4/MelBand0.6 | ensemble_residual | `sha256:a81449a5d400ac51f9f7d03ac804882e8324ed633396408057613d3a7da85b20` |
| ens_median:MDX23C/MelBand/BS | ensemble_residual | `sha256:e847597ef33f2e2603c415b5a0546c4179f552842bd513f96e7f6fb17686ce0c` |

## Full recipe x case x metric matrix

### event_hole_depth_db  (PRIMARY; multiband deficit during vocal events; lower=better)

Pooled column = mean over DEV/CALIB cases only (Verdi shown but not pooled).

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi | pooled(dev/calib) |
|---|---|---|---|---|---|---|
| residual:MDX23C | 17.355 | 13.441 | 8.723 | 9.381 | 12.689 | 12.225 |
| residual:MelBand | 15.718 | 12.432 | 5.019 | 4.921 | 8.174 | 9.523 |
| residual:BS | 27.195 | 11.072 | 18.693 | 7.211 | 20.893 | 16.043 |
| native:MDX23C | 17.316 | 13.398 | 8.728 | 9.382 | 12.630 | 12.206 |
| ens_mean:MDX23C0.8/MelBand0.2 | 15.716 | 10.453 | 6.909 | 7.371 | 7.782 | 10.112 |
| ens_mean:MDX23C0.6/MelBand0.4 | 14.634 | 8.522 | 6.058 | 6.326 | 5.514 | 8.885 |
| ens_mean:MDX23C0.4/MelBand0.6 | 14.206 | 7.778 | 5.518 | 5.596 | 5.883 | 8.274 |
| ens_median:MDX23C/MelBand/BS | 17.240 | 9.633 | 8.153 | 5.819 | 8.381 | 10.211 |

### vocal_interference_ratio  (residual solo-voice leak into accompaniment; lower=better)

Pooled column = mean over DEV/CALIB cases only (Verdi shown but not pooled).

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi | pooled(dev/calib) |
|---|---|---|---|---|---|---|
| residual:MDX23C | 0.022 | 0.070 | 0.000 | 0.043 | 0.107 | 0.034 |
| residual:MelBand | 0.020 | 0.449 | 0.003 | 0.028 | 0.215 | 0.125 |
| residual:BS | 0.008 | 0.040 | 0.001 | 0.105 | 0.182 | 0.038 |
| native:MDX23C | 0.022 | 0.070 | 0.000 | 0.044 | 0.107 | 0.034 |
| ens_mean:MDX23C0.8/MelBand0.2 | 0.021 | 0.146 | 0.000 | 0.040 | 0.129 | 0.052 |
| ens_mean:MDX23C0.6/MelBand0.4 | 0.021 | 0.222 | 0.001 | 0.037 | 0.150 | 0.070 |
| ens_mean:MDX23C0.4/MelBand0.6 | 0.020 | 0.298 | 0.001 | 0.034 | 0.172 | 0.088 |
| ens_median:MDX23C/MelBand/BS | 0.016 | 0.082 | 0.001 | 0.035 | 0.095 | 0.033 |

### vocal_si_sir_db  (SI-SIR of the accompaniment error; dB)

Pooled column = mean over DEV/CALIB cases only (Verdi shown but not pooled).

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi | pooled(dev/calib) |
|---|---|---|---|---|---|---|
| residual:MDX23C | -16.580 | -11.920 | -46.550 | -13.700 | -10.270 | -22.188 |
| residual:MelBand | -19.460 | -0.290 | -27.330 | -17.760 | -5.080 | -16.210 |
| residual:BS | -22.900 | -15.770 | -41.920 | -9.640 | -6.560 | -22.558 |
| native:MDX23C | -16.510 | -11.910 | -48.720 | -13.680 | -10.260 | -22.705 |
| ens_mean:MDX23C0.8/MelBand0.2 | -16.320 | -5.030 | -48.430 | -14.190 | -7.700 | -20.992 |
| ens_mean:MDX23C0.6/MelBand0.4 | -16.550 | -1.970 | -37.230 | -14.830 | -5.900 | -17.645 |
| ens_mean:MDX23C0.4/MelBand0.6 | -17.250 | -0.750 | -32.220 | -15.630 | -4.990 | -16.462 |
| ens_median:MDX23C/MelBand/BS | -17.400 | -8.890 | -37.330 | -15.120 | -8.040 | -19.685 |

### si_sdr_db vs exact orchestra  (broadband, secondary; higher=better)

Pooled column = mean over DEV/CALIB cases only (Verdi shown but not pooled).

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi | pooled(dev/calib) |
|---|---|---|---|---|---|---|
| residual:MDX23C | 25.661 | 24.612 | 21.458 | 12.028 | 18.723 | 20.940 |
| residual:MelBand | 23.695 | 17.475 | 22.990 | 11.916 | 17.094 | 19.019 |
| residual:BS | 28.246 | 25.834 | 20.073 | 8.040 | 17.366 | 20.548 |
| native:MDX23C | 25.661 | 24.613 | 21.460 | 12.028 | 18.725 | 20.941 |
| ens_mean:MDX23C0.8/MelBand0.2 | 26.084 | 24.219 | 22.297 | 12.213 | 19.424 | 21.203 |
| ens_mean:MDX23C0.6/MelBand0.4 | 26.035 | 22.683 | 22.964 | 12.295 | 19.563 | 20.994 |
| ens_mean:MDX23C0.4/MelBand0.6 | 25.526 | 20.821 | 23.345 | 12.268 | 19.098 | 20.490 |
| ens_median:MDX23C/MelBand/BS | 27.773 | 25.975 | 23.674 | 12.482 | 21.783 | 22.476 |

### band_envelope_err_db  (spectral fullness distortion; lower=better)

Pooled column = mean over DEV/CALIB cases only (Verdi shown but not pooled).

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi | pooled(dev/calib) |
|---|---|---|---|---|---|---|
| residual:MDX23C | 2.160 | 0.535 | 0.697 | 1.834 | 1.161 | 1.306 |
| residual:MelBand | 1.435 | 0.291 | 0.398 | 1.592 | 0.355 | 0.929 |
| residual:BS | 2.255 | 0.242 | 1.377 | 3.015 | 0.652 | 1.722 |
| native:MDX23C | 2.190 | 0.535 | 0.697 | 1.832 | 1.163 | 1.313 |
| ens_mean:MDX23C0.8/MelBand0.2 | 1.906 | 0.451 | 0.624 | 1.757 | 0.944 | 1.184 |
| ens_mean:MDX23C0.6/MelBand0.4 | 1.672 | 0.378 | 0.552 | 1.689 | 0.749 | 1.073 |
| ens_mean:MDX23C0.4/MelBand0.6 | 1.489 | 0.318 | 0.487 | 1.636 | 0.578 | 0.983 |
| ens_median:MDX23C/MelBand/BS | 1.419 | 0.252 | 0.517 | 1.420 | 0.442 | 0.902 |

### stft_distance  (multires spectral; lower=better)

Pooled column = mean over DEV/CALIB cases only (Verdi shown but not pooled).

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi | pooled(dev/calib) |
|---|---|---|---|---|---|---|
| residual:MDX23C | 0.532 | 0.169 | 0.436 | 0.539 | 0.302 | 0.419 |
| residual:MelBand | 0.341 | 0.109 | 0.332 | 0.497 | 0.140 | 0.320 |
| residual:BS | 0.503 | 0.134 | 0.575 | 1.006 | 0.212 | 0.555 |
| native:MDX23C | 0.533 | 0.168 | 0.435 | 0.534 | 0.301 | 0.418 |
| ens_mean:MDX23C0.8/MelBand0.2 | 0.454 | 0.149 | 0.410 | 0.517 | 0.250 | 0.382 |
| ens_mean:MDX23C0.6/MelBand0.4 | 0.396 | 0.132 | 0.387 | 0.500 | 0.208 | 0.354 |
| ens_mean:MDX23C0.4/MelBand0.6 | 0.355 | 0.119 | 0.366 | 0.489 | 0.174 | 0.332 |
| ens_median:MDX23C/MelBand/BS | 0.414 | 0.198 | 0.608 | 0.743 | 0.278 | 0.491 |

### stereo_width_err  (lower=better)

Pooled column = mean over DEV/CALIB cases only (Verdi shown but not pooled).

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi | pooled(dev/calib) |
|---|---|---|---|---|---|---|
| residual:MDX23C | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| residual:MelBand | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| residual:BS | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native:MDX23C | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ens_mean:MDX23C0.8/MelBand0.2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ens_mean:MDX23C0.6/MelBand0.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ens_mean:MDX23C0.4/MelBand0.6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ens_median:MDX23C/MelBand/BS | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### theft_mean on genuine no-vocal control (this work's orchestra_only; lower=better)

| recipe | bologna_donizetti | bologna_puccini | aalto_mozart_dry | aalto_mozart_hall | bologna_verdi |
|---|---|---|---|---|---|
| residual:MDX23C | 0.013 | 0.017 | 0.098 | 0.028 | 0.035 |
| residual:MelBand | 0.122 | 0.037 | 0.124 | 0.001 | 0.053 |
| residual:BS | 0.018 | 0.023 | 0.018 | 0.009 | 0.018 |
| native:MDX23C | 0.013 | 0.017 | 0.098 | 0.028 | 0.035 |
| ens_mean:MDX23C0.8/MelBand0.2 | 0.027 | 0.016 | 0.083 | 0.023 | 0.033 |
| ens_mean:MDX23C0.6/MelBand0.4 | 0.050 | 0.019 | 0.078 | 0.017 | 0.034 |
| ens_mean:MDX23C0.4/MelBand0.6 | 0.074 | 0.024 | 0.085 | 0.011 | 0.038 |
| ens_median:MDX23C/MelBand/BS | 0.004 | 0.009 | 0.009 | 0.002 | 0.021 |

## Constrained-feasibility ranking (asymmetric)

Ceilings placed in the NATURAL GAPS of the pooled DEV/CALIB distributions (robust, not knife-edge): **C_voice = 0.045** on `vocal_interference_ratio` (gap between the good-remover cluster <=0.038 and the next tier 0.052); **C_hole = 14.0** on `event_hole_depth_db` (gap between the MDX23C twins ~12.2 and the BS gouger 16.0). Gate order is lexicographic: voice, THEN hole, THEN distortion tie-break.

| recipe | pooled voice | worst voice | pooled hole | pooled band | pooled stft | pooled stereo | pooled SI-SDR | voice<=Cv | hole<=Ch | FEASIBLE |
|---|---|---|---|---|---|---|---|---|---|---|
| residual:MDX23C | 0.034 | 0.070 | 12.225 | 1.306 | 0.41867 | 0.00000 | 20.94 | Y | Y | **Y** |
| residual:MelBand | 0.125 | 0.449 | 9.523 | 0.929 | 0.31956 | 0.00003 | 19.02 | . | Y | . |
| residual:BS | 0.038 | 0.105 | 16.043 | 1.722 | 0.55461 | 0.00001 | 20.55 | Y | . | . |
| native:MDX23C | 0.034 | 0.070 | 12.206 | 1.313 | 0.41751 | 0.00000 | 20.94 | Y | Y | **Y** |
| ens_mean:MDX23C0.8/MelBand0.2 | 0.052 | 0.146 | 10.112 | 1.184 | 0.38231 | 0.00000 | 21.20 | . | Y | . |
| ens_mean:MDX23C0.6/MelBand0.4 | 0.070 | 0.222 | 8.885 | 1.073 | 0.35384 | 0.00001 | 20.99 | . | Y | . |
| ens_mean:MDX23C0.4/MelBand0.6 | 0.088 | 0.298 | 8.274 | 0.983 | 0.33205 | 0.00001 | 20.49 | . | Y | . |
| ens_median:MDX23C/MelBand/BS | 0.033 | 0.082 | 10.211 | 0.902 | 0.49106 | 0.00001 | 22.48 | Y | Y | **Y** |

Feasible recipes ranked by distortion tie-break (band -> stft -> stereo):

1. **ens_median:MDX23C/MelBand/BS** (`sha256:e847597ef33f2e2603c415b5a0546c4179f552842bd513f96e7f6fb17686ce0c`) — band=0.902 stft=0.49106 stereo=0.00001 | voice=0.033 hole=10.211
2. **residual:MDX23C** (`sha256:1fbb0f0a487ae2e3c7405d405665c030d65e58c3295ffb1bf6e47b5dca47754f`) — band=1.306 stft=0.41867 stereo=0.00000 | voice=0.034 hole=12.225
3. **native:MDX23C** (`sha256:6f3513af1980bd8af72b936540d4de8e122c598d293603598dfe9623b01d41ba`) — band=1.313 stft=0.41751 stereo=0.00000 | voice=0.034 hole=12.206

### TWO BEST FEASIBLE RECIPES

**#1: ens_median:MDX23C/MelBand/BS**  
`sha256:e847597ef33f2e2603c415b5a0546c4179f552842bd513f96e7f6fb17686ce0c`  
pooled voice=0.033, hole=10.211, band=0.902, stft=0.49106, stereo=0.00001, SI-SDR=22.48 dB

**#2: residual:MDX23C**  
`sha256:1fbb0f0a487ae2e3c7405d405665c030d65e58c3295ffb1bf6e47b5dca47754f`  
pooled voice=0.034, hole=12.225, band=1.306, stft=0.41867, stereo=0.00000, SI-SDR=20.94 dB

### Ceiling sensitivity (top-2 as ceilings sweep percentiles)

| C_voice | C_hole | top-2 feasible |
|---|---|---|
| 0.0383 | 12.225 | ens_median:MDX23C/MelBand/BS, residual:MDX23C |
| 0.0383 | 12.225 | ens_median:MDX23C/MelBand/BS, residual:MDX23C |
| 0.0519 | 12.206 | ens_median:MDX23C/MelBand/BS, ens_mean:MDX23C0.8/MelBand0.2 |
| 0.0519 | 12.225 | ens_median:MDX23C/MelBand/BS, ens_mean:MDX23C0.8/MelBand0.2 |
| 0.0519 | 12.206 | ens_median:MDX23C/MelBand/BS, ens_mean:MDX23C0.8/MelBand0.2 |
| 0.0519 | 12.225 | ens_median:MDX23C/MelBand/BS, ens_mean:MDX23C0.8/MelBand0.2 |
| 0.0702 | 10.211 | ens_median:MDX23C/MelBand/BS, ens_mean:MDX23C0.6/MelBand0.4 |
| 0.0702 | 12.225 | ens_median:MDX23C/MelBand/BS, ens_mean:MDX23C0.6/MelBand0.4 |

## Regime sensitivity (dry vs hall vs real orchestra)

Per-regime pooled **voice / hole / band** for the two finalists and the two disqualified extremes (lower is better everywhere):

| recipe | DRY v/h/b | HALL v/h/b | REAL v/h/b |
|---|---|---|---|
| ens_median:MDX23C/MelBand/BS | 0.001 / 8.2 / 0.52 | 0.035 / 5.8 / 1.42 | 0.049 / 13.4 / 0.84 |
| residual:MDX23C | 0.000 / 8.7 / 0.70 | 0.043 / 9.4 / 1.83 | 0.046 / 15.4 / 1.35 |
| residual:BS | 0.001 / 18.7 / 1.38 | 0.105 / 7.2 / 3.02 | 0.024 / 19.1 / 1.25 |
| residual:MelBand | 0.003 / 5.0 / 0.40 | 0.028 / 4.9 / 1.59 | 0.234 / 14.1 / 0.86 |

- **Difficulty ordering DRY < HALL < REAL.** Voice removal is trivial DRY (interference ~1e-4, SI-SIR down to -49 dB for everyone), degrades under HALL reverb, and is HARDEST on REAL dense orchestra + chorus — every recipe's residual-voice rises there. So the real-orchestra works, not the anechoic ones, are what actually separate the recipes.
- **The winner is regime-stable.** `ens_median` has the best or near-best band in all three regimes and the best pooled voice overall; it never has a catastrophic case.
- **The mean-ensembles are a dry/hall mirage.** They own the lowest event-holes DRY/HALL (and `residual:MelBand` looks great there too), but on REAL orchestra their voice leak explodes (Mel-Band 0.449 on Puccini) — a recipe chosen on anechoic tracks alone would pick a MelBand-heavy ensemble and ship audible soprano on real opera. Applying the GLOBAL ceilings to the REAL regime alone leaves NO recipe feasible, because real orchestra is strictly harder than the dry-inclusive pool the ceilings were calibrated on — a signal that production thresholds must be set on real-orchestra references, not anechoic remixes.

## Ensemble vs single verdict

| ensemble | pooled voice (vs best single) | pooled hole (vs best single) | pooled band (vs best single) |
|---|---|---|---|
| ens_mean:MDX23C0.8/MelBand0.2 | 0.052 (best residual:MDX23C=0.034) | 10.112 (best residual:MelBand=9.523) | 1.184 (best residual:MelBand=0.929) |
| ens_mean:MDX23C0.6/MelBand0.4 | 0.070 (best residual:MDX23C=0.034) | 8.885 (best residual:MelBand=9.523) | 1.073 (best residual:MelBand=0.929) |
| ens_mean:MDX23C0.4/MelBand0.6 | 0.088 (best residual:MDX23C=0.034) | 8.274 (best residual:MelBand=9.523) | 0.983 (best residual:MelBand=0.929) |
| ens_median:MDX23C/MelBand/BS | 0.033 (best residual:MDX23C=0.034) | 10.211 (best residual:MelBand=9.523) | 0.902 (best residual:MelBand=0.929) |

**Verdict: yes, but only the MEDIAN ensemble.** `ens_median(MDX23C,MelBand,BS)` is the single best recipe overall — it beats every single model on pooled voice, band, theft and SI-SDR, and beats every *voice-viable* single (MDX23C, BS) on event-holes too. The median is a robust estimator: it discards each model's outlier per sample, so BS's gouging and Mel-Band's voice-passthrough are voted out instead of averaged in. The weighted-MEAN ensembles do the opposite — they *blend in* Mel-Band's failures, so their voice leak rises monotonically with Mel-Band weight (0.052 -> 0.070 -> 0.088) and they never clear the primary voice gate. Note the singles named 'best' in the table (Mel-Band on hole/band) are themselves voice-disqualified, so they are not deliverable baselines.

## GPU timing / disk

Per-model separation time (all separations, ~90-110 s tracks):

| model | n seps | median s | mean s |
|---|---|---|---|
| MDX23C | 10 | 12.0 | 12.9 |
| MelBand | 10 | 5.4 | 5.8 |
| BS | 10 | 8.2 | 8.7 |

Per-recipe GPU cost (one track) = sum of member-model median sep times (memoized: each (model,track) separated once and shared across recipes):

| recipe | member models | est GPU s/track |
|---|---|---|
| residual:MDX23C | MDX23C | 12.0 |
| residual:MelBand | MelBand | 5.4 |
| residual:BS | BS | 8.2 |
| native:MDX23C | MDX23C | 12.0 |
| ens_mean:MDX23C0.8/MelBand0.2 | MDX23C+MelBand | 17.4 |
| ens_mean:MDX23C0.6/MelBand0.4 | MDX23C+MelBand | 17.4 |
| ens_mean:MDX23C0.4/MelBand0.6 | MDX23C+MelBand | 17.4 |
| ens_median:MDX23C/MelBand/BS | MDX23C+MelBand+BS | 25.5 |

Total bake-off wall time: **1505.9 s** (~25 min) across 5 cases x 8 recipes (memoized: each (model,track) separated ONCE for the mix and once for the theft control, shared across all recipes).

**Disk:** research6 held steady at **27 GB free (82% used)** throughout — no change. `~/bakeoff_work` is **724 KB** total (per-case challenge dirs are symlinks to the existing `~/truth_pairs` WAVs; `persist_outputs=False`; every separation's stems deleted from `_stem_tmp` immediately after being read into memory; per-case stem cache cleared between cases). No other missions' directories were touched.

## Bugs / notes

- **No tracebacks.** All 8 recipes x 5 cases scored cleanly; run exit code 0.
- **`native:MelBand` correctly SKIPPED** (not faked): audio-separator tags Mel-Band's second stem `(other)`, which normalizes to `other`, not `instrumental`, so `compose_accompaniment`'s native path would have silently fallen back to the residual. Probe-gated it out. MDX23C and BS both emit a real `(Instrumental)` stem; only MDX23C's native recipe was in the requested grid.
- **`native:MDX23C` == `residual:MDX23C`** to <0.01 on every metric (the InstVoc instrumental head equals mixture-minus-vocals for this checkpoint). Kept both as distinct recipe ids for honesty; they are one recipe operationally.
- **Repo state:** `git pull` fast-forwarded a rewritten `origin/v2-design`; a prior local tip `8eda0c4` ('stem-name collision' fix) is now an orphan superseded by `6fc73b9` (which is the authoritative branch tip and already carries the `(Stem)`-tag parser). HEAD == origin/v2-design. Sanity `pytest tests/test_cli_autonomous.py` = 4 passed.
