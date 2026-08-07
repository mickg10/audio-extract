# Dataset Acquisition Report — Steps 3–4 (measured-room/stereo + free transfer pairs)

Date: 2026-08-07
Storage/compute: tt-quietbox2 (`ttuser@100.91.242.69`), CPU only, `~/datasets/`
Downloader: `aria2c -c -x16 -s16` (via `~/datasets/dl.sh`, md5-verified)
Scope honored: open/free/legal only; no shadow/copyrighted bulk audio; 30 s previews are triage-only.

---

## PART A — measured-room + stereo exact material

### A1. The Spheres Dataset — LANDED + md5-verified
- Source (canonical): Zenodo record **17347681**, DOI `10.5281/zenodo.17347681`
  - Tampere Univ. portal: https://researchportal.tuni.fi/en/datasets/the-spheres-dataset/ · paper: arXiv:2511.21247 · code: github.com/repertorium/TheSpheresDataset-Experiments
- License: **CC BY-SA 4.0**
- Content: real multitrack orchestra (Colibrì Ensemble) — **Mozart** (~1814 s) + **Tchaikovsky** Romeo & Juliet (~1361 s), 48 kHz, plus **measured room impulse responses per instrument position**.
- Downloaded (the STEREO package + RIRs, NOT the 25 GB multichannel or 18.5 GB claps/sweeps):
  - `spheres/TheSpheresDataset-StereoMix.zip` — **2,796,218,501 B (2.80 GB)** — md5 `43a5f317d33645ea1446bbefbc931482` ✅
  - `spheres/TheSpheresDataset-RIRs.zip` — **23,747,209 B (23.7 MB)** — md5 `5d61f86a50fea7dcacb5485efdf0de61` ✅
- Verified contents:
  - StereoMix = **17 per-instrument STEREO FLAC stems** per work (2 ch, 48 kHz, PCM_16) + `TheSpheres_metadata_{songs,Mozart,Tchaikovsky}.json`. (Confirmed e.g. Harp.flac = 2 ch / 48000 Hz / 1340 s.)
  - RIRs = **17 source positions**, each a **23-channel MEASURED impulse response** (`source_*.npy`, float64, shape (23, 32768) = 0.68 s @ 48 kHz) + per-source PDF plots. Instruments: Vln1/2, Vla, Vcl, DBass1/2, Flute, Oboe, Clarinet, Bassoon, Horns, Trumpets, Trombones, Tuba, Timpani, Bass Drum, Harp.
  - Integrity class: **same_take / linear_exact by construction** (per-instrument stems of one performance → exact sums).

### A2. FreiDi (Freischütz Digital multitrack) — LANDED + md5-verified (throttle FIXED)
- Source: Zenodo record **20285754**, "Freischütz Digital Multitrack Dataset"
- License: **CC BY-SA 4.0**
- `freidi/audio.zip` — **12,249,252,088 B (12.25 GB)** — md5 `a7ace091327ff9c5af065b093ed743f3` ✅
- The prior 0.06 MB/s single-connection throttle is **resolved**: `aria2c -x16 -s16` pulled the full 12.25 GB in ~2–3 min (~100 MB/s). Same simultaneous-performance opera set → integrity class **same_performance_bleed**.

### A3. Additional measured-hall options (noted, not pulled)
- **3D-MARCo** — Zenodo **3474286** — St Paul's concert hall, Huddersfield, 71 mics; 3D music recordings **+ room impulse responses**. License **CC BY 3.0** (Zenodo metadata). 10 files, **~23.6 GB**. → the strongest measured-concert-hall complement.
- **dEchorate** — github.com/Chutlhu/dEchorate — calibrated multichannel RIR database with annotated early-echo timings / 3D source-mic geometry (echo-aware). Measured, controlled wall configs.
- **Aalto / Pori concert-hall IRs** — legacy.spa.aalto.fi/projects/poririrs/ — concert-hall IRs, 3 stage sources × 7 receivers, free non-commercial.
- "**ECHO**" could not be resolved to a single canonical dataset (WebSearch budget for the session was exhausted; the above are the locatable measured-hall options). Recommend 3D-MARCo as the primary measured-hall add if/when needed.

### What Part A unlocks (the stereo/hall gap the truth run exposed)
- **Measured (not synthetic) hall convolution:** convolve the dry Bologna / Aalto-anechoic voice+orchestra stems with the Spheres **measured** per-instrument RIRs — applied **identically to voice and orchestra** — so the exact accompaniment target is preserved by linear construction while the interference carries a *real* room. This certifies reverberant cases the anechoic build cannot.
- **Stereo rendering:** pick a stereo receiver pair (or downmix) from the 23-channel RIRs, or use the Spheres StereoMix per-instrument stereo stems directly, to build **stereo** exact cases with per-instrument ground truth (measured stereo width, not dual-mono).
- **3D-MARCo** (if added) extends this to a full concert hall with 71 simultaneous captures.

---

## PART B — first free transfer-pair batch (Deezer 30 s previews)

Pipeline (`~/datasets/transfer_previews.py`, self-contained; audit logic **vendored verbatim** from
`audio_extract.{dsp,alignment,challenges}` — `audit_pair` thresholds byte-identical):
top pairs by `focus_score` (deduped by ISRC) → resolve **both** sides via **Deezer ISRC API**
(`api.deezer.com/track/isrc:<ISRC>`, no auth) → coarse broadband cross-correlation to find the
multi-second window lag → trim to overlap → `audit_pair` (delay + one fixed gain → null residual)
to LABEL → keep only `linear_exact` / `same_take_paired_target`.

**Key method note (why the label works):** run naively over the whole 30 s window, the
ever-present vocal dominates the null and *every* pair reads `matched_program` (whole-clip
residual −2…−6 dB). `audit_pair` is designed to measure the null on the **solo-inactive** regions
(its `solo_inactive_mask` arg). We estimate that mask as the quietest 25 % of frames by
vocal-estimate (`full − g·base`) energy — where a **same-take** orchestra cancels but a
**re-record cannot**. The label uses that solo-inactive residual; `audit_wholeclip` is kept for
reference. An independent **same-recording xcorr** flag (broadband xcorr ≥ 0.5 at |lag| ≤ 2 s)
corroborates each label — a re-record cannot phase-correlate.

### Counts (top 60 by focus_score, deduped)
| stage | n |
|---|---|
| selected (unique pairs) | 60 |
| both previews on Deezer | 52 (**~87 % coverage**; 8 misses) |
| usable overlap (≥5 s) | 52 |
| **KEPT `same_take_paired_target`** | **7** |
| `matched_program` | 45 |
| same-recording xcorr evidence | 34 |

Integrity-class breakdown of the 52 audited: **same_take_paired_target 7 / matched_program 45**
(no `linear_exact` — independent MP3 encodes never reach the −35 dB floor, as expected for lossy previews).
Of the 45 `matched_program`, **~12 are "borderline"**: strong same-recording xcorr but a null floor of
−8…−11.7 dB set by the previews' independent mastering (below the −12 dB same_take bar) — correctly NOT certified.

### The 7 usable same_take pairs (example rows)
| pair_id | artist | base → variant | ISRCs | si_resid | lag | xconf |
|---|---|---|---|---|---|---|
| 24502 | Andrea Bocelli | Ama Credi E Vai → Instrumental | ITZ040600044/…46 | **−19.0 dB** | 0.0 s | 0.68 |
| 5479 | Avantasia | Bring On The Night → Instrumental | ATN262436806/…24 | **−20.2 dB** | 0.0 s | 0.88 |
| 728903 | Yes | Long Distance Runaround (2024 S. Wilson Remix) → Instrumental | USAT22400399/…407 | **−17.3 dB** | 0.0 s | 0.83 |
| 460161 | Equilibrium | Wirtshaus Gaudi → Instrumental | DED831400415/…27 | −14.0 dB | 0.03 s | 0.91 |
| 399109 | Avantasia | Creepshow → Instrumental | ATN262436801/…19 | −12.8 dB | 0.0 s | 0.87 |
| 680304 | Epica | Crimson Bow and Arrow → Instrumental | DED831800500/…04 | −12.7 dB | −0.09 s | 0.84 |
| 393611 | We Are the Catalyst | The Enemy Inside → Instrumental | SE4JK2500401/…02 | −12.4 dB | 0.0 s | 0.89 |

Genre note: the highest-focus_score pairs skew to symphonic/crossover acts with **official "Instrumental"
companion tracks** (same master, vocals muted) — an ideal `same_take_paired_target` source. Andrea Bocelli is
the pure operatic one; Paul Potts landed in the borderline set.

### Landed artifacts
- `~/datasets/transfer_previews/manifest.json` — full record for all 60 (pair_id, artists, titles, ISRCs,
  Deezer ids/preview urls, coarse lag/overlap/xcorr, `audit_wholeclip` + `audit_solo_inactive`,
  `same_recording_xcorr`, `recommended_integrity`, `kept`).
- `~/datasets/transfer_previews/clips/<pair_id>_{full,accomp}.wav` — 14 files (7 kept pairs × {vocal-side, sample-aligned instrumental}), 44.1 kHz stereo PCM_16, trimmed to the overlap.
- `~/datasets/transfer_previews/raw/*.mp3` — the resolved 30 s previews (provenance).
- `~/datasets/pairs_classical.jsonl` — copy of the mined pair set (10,344).

**LOSSY CAVEAT (as instructed):** these are 128 kbps 30 s previews for transfer **direction / triage** only.
Labels are lossy; do **not** use them for final brightness/stereo thresholds — that is what Spheres (stereo,
lossless) and Bologna certify.

---

## Disk + blockers
- Disk used this mission: FreiDi 12 GB + Spheres 2.7 GB + transfer_previews 119 MB ≈ **~14.8 GB**. Free on `/`: **2.8 TB** (19 % used). Did not disturb other missions (bologna/aalto/annas-archive untouched).
- **Zenodo throttle: RESOLVED** — `-x16 -s16` restored ~100 MB/s; all three Zenodo files md5-verified.
- **Deezer coverage ~87 %** (8/60 ISRCs had no preview / "no data"). Good enough for a first batch; can widen by pulling more of the 10,344 pairs and/or adding Spotify/Apple preview fallbacks.
- **Spotify `preview_url` not used**: the mined `pairs_classical.jsonl` carries no preview_url and Spotify deprecated the field on most API responses (late 2024); Deezer ISRC lookups were the reliable free path and gave both sides cleanly.
