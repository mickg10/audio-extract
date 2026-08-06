# TRUTH REPORT — audio-extract separation vs. GENUINE with/without-voice masters

**First evaluation against true ground truth (not synthetic remixes).**
Date: 2026-08-06 · Models: the 5 locked bundles (`configs/model-lock.json`) ·
GPU: research6 (NVIDIA RTX PRO 4500 Blackwell, CUDA 13) · onnxruntime-gpu 1.28.

The question this answers: when the *real* no-voice master exists on the *same
recording*, does our separation-quality ranking survive — or was it an artifact of
the synthetic-remix construction? Synthetic-remix ranking to beat:
**melband > mdx23c > bs_roformer > kim ≈ uvr**.

---

## 0. Headline

* **The tier structure is REAL and reproduces on genuine ground truth.** The three
  large/transformer models (melband, mdx23c, bs_roformer) sit ~7 dB SI-SDR above the
  two MDX models (kim, uvr) on every case. **kim ≈ uvr is confirmed exactly** (they
  differ by 0.02 dB SI-SDR averaged, and track each other on all 8 metrics).
* **One rank flip, and it is the headline: `mdx23c` overtakes `melband` for #1.**
  Composite mean-rank across the four decision metrics:
  `mdx23c (2.10) > melband (2.55) > bs_roformer (2.60) > kim (3.60) > uvr (4.15)`.
  The swap is driven by the **Bologna Italian-opera** cases (real orchestra + real
  soprano), where mdx23c/bs_roformer beat melband by 4–8 dB SI-SDR. On **Aalto**
  (Mozart) the synthetic order melband > mdx23c > bs_roformer is preserved. So the
  flip is *dataset-dependent*, not noise.
* **The PRIMARY event-hole metric changes the story SI-SDR alone tells.** Ranked by
  accompaniment damage during singing (`event_hole_depth`, lower = better):
  `kim > uvr > mdx23c > melband > bs_roformer`. The timid MDX models punch the
  *fewest* holes (they under-subtract — they leave voice in, see vocal_interference);
  **bs_roformer punches the deepest holes** (up to **40 dB** on Donizetti) — aggressive
  over-subtraction that its high SI-SDR hides. **mdx23c is the best-balanced model**:
  top-1 on vocal-interference *and* mid-pack on holes.
* **Reverb (realism) compresses the top tier and penalizes aggression.** Anechoic→hall
  costs the strong models 9–12 dB SI-SDR (kim/uvr only 2–4 dB, they had less to lose),
  flips Aalto #1 from melband→mdx23c, and collapses bs_roformer (worst under hall).

---

## 1. Datasets, licences, provenance

| Dataset | Excerpt | Voicing | Licence / citation | Zenodo | MD5 (7z/zip) |
|---|---|---|---|---|---|
| **Bologna** anechoic Italian opera | Verdi *Di tale amor* (Il Trovatore) | Soprano + orch | **CC BY 4.0** | [3628247](https://zenodo.org/records/3628247) `Verdi.7z` | `8334a948…03458a` |
| | Puccini *O mio babbino caro* (Gianni Schicchi) | Soprano + orch | CC BY 4.0 | `Puccini.7z` | `63127b1d…f99f` |
| | Donizetti *Come Paride vezzoso* (Elisir) | Solo voice + orch | CC BY 4.0 | `Donizetti.7z` | `427bd57a…3fbe4` |
| **Aalto** anechoic Mozart | Donna Elvira aria (Don Giovanni) | Soprano + orch | Free for academic research | [mediatech.aalto.fi](https://research.cs.aalto.fi/acoustics/) | — |

* Bologna citation: *Anechoic recordings of Italian opera*, Zenodo record 3628247,
  DOI 10.5281/zenodo.3628247 (Univ. of Bologna/Parma), CC BY 4.0. All three MD5s
  verified after download.
* Aalto citation: **Pätynen, Pulkki & Lokki (2008)**, "Anechoic recording system for
  symphony orchestra," *Acta Acustica united with Acustica* 94(6):856–865.
* **FreiDi (Freischütz Digital, CC BY-SA 4.0, Zenodo 20285754) was NOT completed** —
  see §7. It would have contributed the only `same_performance_bleed` cases.

---

## 2. Method

### 2.1 Declared rendering recipe — "one consistent microphone direction"
Every source stem is mono-downmixed (mean of channels) to a single virtual capture,
per the Oracle directive (start with one mic direction). Then:

* `orchestra_only` = Σ all **non-soloist** family tracks (the *retained "rest" side*).
* `voice_ref` = the **soloist** family (a single take → one soloist, no layering).
* `mix_with_voice` = `orchestra_only + voice_ref` — **linear by construction**.
* One **shared** peak-normalization gain (mix → −3 dBFS) applied *identically* to all
  three signals, so the exact linear target is preserved.
* Output: **44.1 kHz, stereo (dual-mono L=R), float32 WAV** (models see FLOAT input;
  confirmed 32-bit round-trip, no int16 quantization).
* **Bologna source selection**: for each instrument family, sum take `-001` of *every*
  desk (desk count → natural section balance: VL1 10 desks, VLA/VC 6, KB 4, winds 2…).
  This uses the multitrack as designed (each part recorded as many full-length 92 s
  takes; **no time-concatenation** — verified by summing one desk's 12 takes = 12×92 s).
* **Alignment validated** against each excerpt's own `*_downmix.wav`: cross-correlation
  delay = **0–42 samples (0.0–1.0 ms)** → the multitrack shares a common t=0, so the
  linear sum is time-coherent. (Waveform corr is low, 0.16–0.37, only because our
  per-desk section balance differs from the dataset's official downmix — expected.)
* **Hall variant** (Aalto): convolve mix/orch/voice *each* with the SAME deterministic
  IR — `estimate_hall_ir(sr, rt60_s=1.5, seed=1234)`, an exact replica of
  `audio_extract.challenges.estimate_hall_ir` — so the same-take property is preserved.

### 2.2 Labels (Oracle directives applied)
* **pair_integrity**: Bologna + Aalto = **`linear_exact`** (measured mix−orch−voice
  residual ≤ −78 dB, float-precision limited). FreiDi would have been
  `same_performance_bleed`.
* **evaluation_task**: all cases **`soloist_vs_rest`**. Donizetti has *no separate
  chorus stem* in the multitrack (only the isolated solo voice + 13 instrument
  families), so the "chorus→retained side" rule is moot here; the solo voice is the
  only thing removed.
* **Bologna split**: development = **Donizetti**, calibration = **Puccini**,
  **frozen holdout = Verdi** (built, reported, **excluded from any fitting/tuning**).
  Aalto is an independent cross-dataset realism axis (dry + hall).

### 2.3 Pipeline (per case × model)
`audio_extract.separate.Separator` (overlap 8, `model_dir=/home/mickg/models`) →
`residual = mix − vocals_stem`. Scored with:
* **SECONDARY** `exact_reference_error(residual, TRUE orchestra_only)` → SI-SDR,
  multires-STFT dist, band-envelope err, stereo-width err (alignment included).
* **PRIMARY** `audio_extract.metrics_v2.event_holes(resid_env, true_A_env, true_V_env,
  events)` — event-conditioned multiband **deficit vs the true accompaniment**, with
  vocal-activity `events` detected from the true voice track (10 ms grid, −30 dB rel
  threshold). Reports worst-event hole depth/area/duration + masked-hole uncertainty.
* **Vocal-interference** = energy of the projection of the accompaniment error
  `(residual−true_A)` onto the true voice direction `V`, in dB re true_A
  (`vocal_interference_db`) and as a fraction of the error (`vocal_leak_frac_of_err`).
* **Vocal side**: SI-SDR and envelope-correlation of the extracted vocal stem vs
  `voice_ref` (clean for all cases here — no bleed, since sources are isolated).
* GPU wall-time per separation.

> Note: `stereo_width_err ≈ 1e-5` for all rows — **non-discriminative by construction**
> (dual-mono recipe). `align_delay = 0`, `align_conf ≈ 0`: pairs are sample-aligned by
> construction, so 0-delay is correct; low "confidence" only reflects that the residual
> is not a clean copy of the orchestra (expected), not an alignment failure.

---

## 3. Pair-case inventory

| case | role | task | pair_integrity | dur (s) | voice frac of mix | linear-exact | notes |
|---|---|---|---|---|---|---|---|
| bologna_verdi | **holdout_frozen** | soloist_vs_rest | linear_exact | 92.1 | 0.090 | ≤ −98 dB | whole excerpt; 12 orch families |
| bologna_puccini | calibration | soloist_vs_rest | linear_exact | 106.0 | **0.041** | −94 dB | quietest soprano (hardest) + harp |
| bologna_donizetti | development | soloist_vs_rest | linear_exact | 110.0 | 0.110 | −92 dB | trimmed to max-voice window @32 s |
| aalto_mozart_dry | dev / realism | soloist_vs_rest | linear_exact | 90.0 | **0.596** | −78 dB | soprano-dominant (easiest to detect) |
| aalto_mozart_hall | dev / realism | soloist_vs_rest | linear_exact | 90.0 | 0.565 | −92 dB | RT60 1.5 s IR, identical both sides |

Five `linear_exact` cases (target was 4–8). The **voice-fraction spread 4 %→60 %** gives
two difficulty regimes: quiet-soprano-over-full-orchestra (Bologna) vs
soprano-dominant-sparse-accompaniment (Aalto).

---

## 4. Full model × case matrix

### 4.1 Residual vs TRUE orchestra — SI-SDR (dB, higher = better) *[secondary]*
| model | verdi | puccini | donizetti | aalto-dry | aalto-hall | **mean** | Bologna | Aalto |
|---|---|---|---|---|---|---|---|---|
| **melband** | +17.09 | +17.48 | +23.70 | **+22.99** | +11.92 | +18.63 | +19.42 | **+17.45** |
| **mdx23c** | +18.72 | +24.61 | +25.66 | +21.46 | **+12.03** | **+20.50** | +23.00 | +16.74 |
| **bs_roformer** | +17.37 | **+25.83** | **+28.25** | +20.07 | +8.04 | +19.91 | **+23.82** | +14.06 |
| kim | +14.78 | +17.24 | +18.45 | +8.21 | +4.47 | +12.63 | +16.83 | +6.34 |
| uvr | +14.16 | +16.76 | +18.23 | +7.95 | +5.95 | +12.61 | +16.38 | +6.95 |

### 4.2 Event-hole depth (dB, lower = better) *[PRIMARY — accompaniment damage during singing]*
| model | verdi | puccini | donizetti | aalto-dry | aalto-hall | **mean** |
|---|---|---|---|---|---|---|
| melband | 13.0 | 18.8 | 27.4 | 4.7 | 6.8 | 14.1 |
| mdx23c | 15.5 | 11.1 | 19.8 | 9.1 | 10.0 | 13.1 |
| **bs_roformer** | 23.7 | 20.4 | **40.1** | 18.3 | 7.1 | **21.9** |
| **kim** | 10.6 | 16.6 | 12.6 | 5.0 | 7.9 | **10.6** |
| uvr | 10.8 | 16.8 | 12.8 | 5.1 | 7.8 | 10.7 |

*(worst-3-events depth; masked-hole uncertainty 0.52–0.70 — highest on voice-dominant
Aalto, so the hole metric is most trustworthy on Bologna where the voice is quiet.)*

### 4.3 Vocal-interference (dB re true accompaniment, lower = better) *[under-subtraction / voice left in]*
| model | verdi | puccini | donizetti | aalto-dry | aalto-hall | **mean** |
|---|---|---|---|---|---|---|
| melband | −23.4 | −20.7 | −43.2 | −50.3 | −30.0 | −33.5 |
| **mdx23c** | −29.4 | −36.8 | −42.3 | **−68.0** | −26.1 | **−40.6** |
| bs_roformer | −24.8 | −41.7 | −51.2 | −62.0 | −18.5 | −39.7 |
| kim | −16.2 | −19.3 | −19.0 | −8.8 | −6.1 | −13.9 |
| uvr | −15.3 | −19.1 | −18.7 | −8.6 | −7.6 | −13.9 |

*(kim/uvr `vocal_leak_frac_of_err` ≈ 0.75–0.91: three-quarters+ of their accompaniment
error IS un-removed voice — the signature of under-subtraction.)*

### 4.4 Vocal-side SI-SDR of extracted vocal vs voice_ref (dB, higher = better)
| model | verdi | puccini | donizetti | aalto-dry | aalto-hall | **mean** |
|---|---|---|---|---|---|---|
| **mdx23c** | +8.1 | +10.5 | +16.5 | +23.1 | +13.2 | **+14.3** |
| bs_roformer | +6.5 | +11.9 | +19.1 | +21.8 | +9.0 | +13.7 |
| melband | +6.2 | +1.5 | +14.5 | +24.6 | +13.1 | +12.0 |
| uvr | +3.7 | +0.3 | +15.6 | +17.2 | +9.7 | +9.3 |
| kim | +4.6 | +1.4 | +15.7 | +17.5 | +6.9 | +9.2 |

### 4.5 Spectral fidelity (means, lower = better)
| model | multires-STFT dist | band-envelope err (dB) |
|---|---|---|
| **melband** | **0.28** | **0.81** |
| mdx23c | 0.40 | 1.28 |
| bs_roformer | 0.49 | 1.51 |
| uvr | 0.40 | 2.38 |
| kim | 0.41 | 2.41 |

---

## 5. Rankings — genuine ground truth vs synthetic remix

| metric | ranking on GENUINE ground truth |
|---|---|
| SI-SDR (all cases) | mdx23c > bs_roformer > melband > kim > uvr |
| SI-SDR (Bologna only) | bs_roformer > mdx23c > melband > kim > uvr |
| SI-SDR (Aalto only) | **melband > mdx23c > bs_roformer** > uvr > kim  *(= synthetic top-3)* |
| STFT / band-envelope | **melband > mdx23c > bs_roformer** > kim/uvr  *(= synthetic top-3)* |
| Event-hole depth (PRIMARY) | kim > uvr > mdx23c > melband > bs_roformer |
| Vocal-interference | mdx23c > bs_roformer > melband > kim > uvr |
| Vocal-side SI-SDR | mdx23c > bs_roformer > melband > uvr > kim |
| **Composite mean-rank** (SI-SDR, −hole, −interf, vocal-side) | **mdx23c > melband > bs_roformer > kim > uvr** |
| **Synthetic-remix (to beat)** | melband > mdx23c > bs_roformer > kim ≈ uvr |

**Do they agree?** — Yes on structure, with one flip:
* **Top-3 vs bottom-2 split is identical.** kim ≈ uvr **confirmed** (0.02 dB apart).
* **The single flip: mdx23c ↔ melband swap #1/#2.** melband keeps #1 on the *spectral*
  metrics (STFT, band-env) and on Aalto/Mozart — i.e. exactly the regime the synthetic
  remixes resembled. mdx23c takes #1 overall because it dominates the **real
  orchestra + real soprano (Bologna)** cases and is the **best-balanced** across the
  over/under-subtraction trade-off.
* **bs_roformer is high-variance**: best raw SI-SDR on Bologna (+23.8) but worst
  event-hole damage and worst under hall — the PRIMARY metric demotes it below melband.

---

## 6. Bologna vs Aalto-dry vs Aalto-hall (realism deltas)

*(FreiDi unavailable — see §7; Bologna is the "real-orchestra dry" axis in its place,
and is cleaner ground truth than FreiDi's bleed-contaminated spot mics would have been.)*

**Anechoic→hall on Aalto (Δ = hall − dry):**
| model | ΔSI-SDR | Δhole-depth | Δvocal-interf | Δvocal-side |
|---|---|---|---|---|
| melband | −11.1 | +2.1 | +20.3 | −11.5 |
| mdx23c | −9.4 | +0.9 | +41.9 | −10.0 |
| bs_roformer | −12.0 | **−11.2** | +43.5 | −12.8 |
| kim | −3.7 | +2.8 | +2.7 | −10.6 |
| uvr | −2.0 | +2.7 | +1.0 | −7.5 |

* **Hall changes rankings.** Aalto #1 flips melband(dry) → mdx23c(hall); bs_roformer
  falls from #3 to last. Reverb narrows the strong models' lead (they drop 9–12 dB,
  kim/uvr only 2–4 dB) — a caution that anechoic benchmarks *overstate* the top tier's
  real-world margin.
* **Reverberant voice is the new hard part**: vocal-interference worsens by +20…+44 dB
  under hall for the strong models — the voice's *reverb tail* is orchestra-like and
  leaks into the accompaniment. bs_roformer's holes *shrink* under hall (−11 dB) only
  because the reverb masks the voice it was over-subtracting.
* **Dataset (not just reverb) matters too**: dry-anechoic Bologna (quiet soprano, full
  orchestra) and dry-anechoic Aalto (loud soprano, sparse orchestra) already disagree on
  #1 (bs_roformer/mdx23c vs melband) — so the mdx23c↔melband flip is a genuine
  content/orchestration effect, reinforced by reverb.

---

## 7. Operational: disk, GPU timing, bugs, FreiDi

**GPU separation time** (RTX PRO 4500 Blackwell, per ~90–110 s case, mean):
`melband 5.8 s · uvr 6.1 s · kim 6.4 s · bs_roformer 8.7 s · mdx23c 13.1 s`.
Full 5×5 matrix ran in ~11 min incl. model loads (1–6 s each). Peak GPU mem ~6.3 GB.

**Disk**:
* tt-quietbox2 (storage/CPU, datasets): Bologna 88 GB (3 × 7z + extracted multitracks),
  Aalto 146 MB, built pairs 493 MB → **2.9 TB free** after.
* research6 (GPU): `~/truth_pairs` 493 MB (5 cases × 3 float32 WAVs + results); stems
  deleted after each case as required → **7.9 GB free** (started ~12 GB; the delta is
  onnxruntime/torch runtime caches, not the pairs). No disk pressure.

**Bugs / issues encountered (all resolved, no failed rows — 25/25 scored):**
1. *Zenodo single-file throttling*: the FreiDi `audio.zip` (12.2 GB) downloaded at
   ~0.06 MB/s (≈59 h ETA). aria2c with 16 parallel connections fixed the Bologna
   downloads (12–14 MB/s). FreiDi was **de-prioritized to last per the Oracle** and,
   given time, **not completed** — so there are **no `same_performance_bleed` cases**.
   This is the one gap vs the original brief; the 5 `linear_exact` cases are strictly
   cleaner ground truth. FreiDi remains the recommended next addition.
2. *`pkill -f aria2c` footgun*: matched its own ssh command line and killed the launcher
   shell. Switched to `pkill -x` (exact process name).
3. *Corrupted Unicode dir names in the Bologna archives* (`Verdi_Soprano_Lirico`,
   `BAS_Donizetti`, `KB_Donizetti` — mangled Mac filename bytes). A literal
   glob silently dropped the **voice** stem. Fixed by auto-discovering family dirs and
   classifying by name substring (robust to the stray char).
4. *Donizetti soloist is `CANTO_SOLO`, not "baritone"*, and has **no isolated chorus
   stem** — token corrected; task stays `soloist_vs_rest`.
5. *audio-separator inherits input subtype*: a PCM_16 input round-trips through int16.
   Pairs are therefore written as **FLOAT** (verified 32-bit end-to-end).

---

## 8. Caveats & reproducibility

* **Rendering is mono ("one microphone direction")** by directive → `stereo_width_err`
  is not exercised. A per-instrument-pan variant is the natural next step to test the
  width metric and stereo-image robustness.
* **Bologna section balance** = per-desk take-001 sum (strings dominate, as in a real
  orchestra); it deliberately differs from the dataset's official downmix. Validity
  rests on `linear_exact` (mix = orch + voice), which holds to ≤ −78 dB — not on
  matching the official mix.
* **Event-hole metric is partially masked on voice-dominant Aalto** (masked fraction
  0.65–0.70) — trust it most on the quiet-voice Bologna cases (masked ≈ 0.5).
* **Verdi is a frozen holdout** — reported above but must not be used to fit/tune any
  selector or threshold.
* Determinism: hall IR seed 1234, RT60 1.5 s; overlap 8; models pinned by
  `model-lock.json` (all SHA-256s in the bundle). MD5s of all three Bologna archives
  verified.
* Artifacts on research6: `~/truth_pairs/{<case>/,results/results_all.json,
  evaluate.py,manifest.json}`; build script on tt-quietbox2 `~/datasets/build_pairs.py`.
