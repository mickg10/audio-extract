# TRAINING WORKS — independent soloist+orchestra corpus for the voice-removal judge

Date: 2026-08-08 · Box: **tt-quietbox2** (`ttuser@100.91.242.69`), CPU-only, `~/datasets/`
Downloader: `aria2c -x16` / `curl`. Scope: **open/free/legal only**, licenses cited; **count = WORKS not files**.
Machine-readable companion: `~/datasets/works_manifest.jsonl` (35 rows, one per work/corpus/donor).
This file re-counts **honestly against the binding constraint the reviewers named: INDEPENDENT REAL soloist+orchestra works.** The prior `WORKS_INVENTORY.md` "~36" figure conflated a-cappella choir + choir+organ + donors with the binding tier; those are separated out below.

---

## 1. HEADLINE — honest independent-work count

| tier | what it is | # independent works | lossless? | counts toward "soloist+orchestra"? |
|---|---|---|---|---|
| **A** | **REAL soloist/ensemble + ORCHESTRA** | **7** | 6 lossless + 1 lossy | **YES (the binding tier)** |
| **B** | REAL voice + **organ** (exact voice-removed target) | **14** (Cantoría) | lossless | no (organ, not orchestra) — but strongest real linear-exact refs |
| **C** | REAL voice + other instr. accompaniment (**OOD**) | **~3** (Saraga Hindustani) | lossy | no (OOD, harmonium+tabla) |
| **A+B+C** | **REAL, independent, with a real accompaniment to preserve** | **≈24** | mostly lossless | broadened in-domain |
| D | a-cappella choir (target≈silence → stress/negative) | 11 pieces | lossless | no (no accompaniment) |
| E | orchestra donors (no voice) | 6 (PHENICX 4 + Spheres 2) | lossless | donor |
| F | isolated voice donors | 20 (VocalSet) + jingju OOD | lossless | donor |
| G | measured concert-hall RIR banks | 3 (Spheres / St Paul's / **Pori NEW**) | lossless | venue |
| H | synthetic multitrack | 20 (ChoralSynth) | lossless | augmentation |
| I | our-library self-remix ladder | 23/27 tracks (one own-corpus) | lossless | augmentation, NOT independent |

### The count that matters
- **Strict real soloist+orchestra: 7 works** → **gap to 40 = 33.**
- **Broadened real (voice + any real instrumental accompaniment, separable, independent): ≈24 works** → **gap to 40 = 16.**
- **Open data is now exhausted for the strict tier.** An exhaustive Zenodo API sweep (22 queries) + targeted web verification found **no new open, separable, real soloist+orchestra multitrack** beyond what we already hold (details §6). The last 16 must come from **purchase (Cantolopera lossless) + one request (MedleyDB)** — see §7.

### Change vs prior inventory (why 7, not "~36")
- **FreiDi split 1 → 3** (Weber Nos. 6/8/9 are three distinct numbers; **built + verified this mission**): **+2**.
- **Spheres removed from the real tier**: Spheres is the Colibrì **orchestra** ensemble with **no singer** — its "voice" is *borrowed*. Honestly it is **2 orchestra donors + measured RIRs (tier E/G)**, not 2 real voice works. **−2** from the inflated count.
- Net real soloist+orchestra: Bologna 3 + Aalto 1 + FreiDi 3 = **7** (the "~7" the reviewers cited, now correctly composed).
- Choir (a-cappella) and choir+organ are **not** soloist+orchestra and are tiered separately (D, B).

---

## 2. COUNT MATRIX — by integrity class × voice type × venue (REAL works, tiers A–C)

**By integrity class**
- `linear_exact` (per-source sums / exact difference): Bologna 3 (anechoic), Aalto 1 (anechoic, lossy), Cantoría 14 (organ-by-difference) = **18**
- `same_performance_bleed` (one simultaneous take, spot-mic bleed): FreiDi 3 = **3**
- `same_take` (multitrack subset, OOD): Saraga Hindustani ~3 = **3**

**By voice type**
- Solo opera (soprano/tenor/…): Bologna 3, Aalto 1 = 4
- Opera ensemble (2–3 named voices): FreiDi 3
- SATB quartet, one-pro-per-part: Cantoría 14
- OOD (Hindustani khyal): Saraga ~3

**By venue / mastering-RIR family**
- Anechoic / dry studio: Bologna 3, Aalto 1 (both dry), Cantoría 14 (studio close) = 18
- Close-mic spot (real room bleed): FreiDi 3
- Concert/studio mixed (OOD): Saraga ~3
- (Measured-hall regime is available for ALL of the above via convolution with the tier-G RIR banks — applied identically to voice & orchestra, preserving the linear target.)

---

## 3. THE REAL soloist+orchestra WORKS (tier A) — verified

| work_id | piece | voices | integrity | lossless | separable-how | path (box) |
|---|---|---|---|---|---|---|
| bologna_verdi | Verdi aria | solo | linear_exact | ✅ | anechoic multitrack incl. `CANTO_SOLO` → downmix | `~/datasets/bologna/Verdi_ex/` |
| bologna_puccini | Puccini aria | solo | linear_exact | ✅ | " | `~/datasets/bologna/Puccini_ex/` |
| bologna_donizetti | Donizetti aria | solo | linear_exact | ✅ | " (CANTO_SOLO_Donizetti confirmed) | `~/datasets/bologna/Donizetti_ex/` |
| aalto_mozart | Mozart, Donna Elvira aria | soprano | linear_exact | ❌ mp3 | anechoic incl. `mozart_sopr` stem | `~/datasets/aalto/mozart/` |
| **freidi_freischuetz_no06** | Weber Freischütz No.6 | 2 (s1,s3) | same_perf_bleed | ✅ | built: A=Σ instr. stems, V=stems-singers, M=A+V | `~/datasets/freidi/built/06/` |
| **freidi_freischuetz_no08** | Weber Freischütz No.8 | 1 (s3) | same_perf_bleed | ✅ | " (510 s) | `~/datasets/freidi/built/08/` |
| **freidi_freischuetz_no09** | Weber Freischütz No.9 | 3 (s1,s2,s3) | same_perf_bleed | ✅ | " (428 s) | `~/datasets/freidi/built/09/` |

**Note on Aalto**: the Aalto anechoic Mozart aria is a *real* soprano+orchestra recording but we hold only the **lossy MP3**. Its lossless companion **PHENICX-Anechoic `mozart/`** ships the **orchestra stems WITHOUT the voice** (verified: clarinet/flute/bassoon/horns/strings only) — so PHENICX gives a lossless orchestra donor of the *same session*, but not a lossless voice. A lossless soprano would need the raw Aalto session (request).

---

## 4. NEW this mission — what was acquired / built

### 4.1 FreiDi with/without-voice mixes — BUILT (the +2 works)
Extracted the 195-file `audio.zip` selectively (`stems-*`, `auto-*`; ~4.5 GB) and constructed, per number, a `(mix, accomp, voice)` triple with **`mix = accomp + voice` exact by construction**:
- **Orchestra target `accomp` = Σ per-instrument spot-mic stems** {vn1,vn2,va,vc,db,fl,ob,cl,bn,horns}. **Voice `voice` = `stems-singers`.**
- **`ab` (the aa/bb ambient main-pair) was EXCLUDED** from the target — probed & confirmed it sits at **≈orchestra level** (`ab_to_orch` = −0.8/+0.6/+1.4 dB) and carries **more voice** than the instrument stems (`r(ab,voice)` = 0.11–0.19 vs `r(orch,voice)` ≈ 0) → it is a whole-room capture that would contaminate the "voice-removed" reference.
- Coarse `strings`/`woodwind` stems were NOT used (they are alt mic-blends, **not** linear sums of the fine stems: residual only −8 / −1.5 dB) → mixing granularities avoided.
- Integrity = **same_performance_bleed** (spot mics of one simultaneous take; the target retains bounded singer bleed — realistic, not a perfect null).
- Metrics (voice-to-orchestra RMS): No.6 +0.0 dB, No.8 +0.6 dB, No.9 +4.8 dB (voice clearly present/prominent).
- Output: `~/datasets/freidi/built/{06,08,09}/{mix,accomp,voice}.flac` (24-bit, 48 kHz; jointly scaled where the instrument sum clipped — scale in `freidi_build_report.json`, preserving M=A+V and the separation).

### 4.2 Pori concert-hall RIRs — DOWNLOADED (new measured venue)
`~/datasets/works/rir_pori/` — **Pori Promenadikeskus** (real Finnish concert hall), 48 kHz, source×receiver grid, `omni/cardioid/binaural/soundfield` (≈28 MB). Free for research/non-commercial (Aalto). **3rd real measured hall** for the synthesizable pool (adds to Spheres + St Paul's/3D-MARCo).

---

## 5. SYNTHESIZABLE POOL — effective-independence estimate

The synthesizable pool = **isolated voice donor × orchestra donor × measured-RIR venue**, rendered so the same RIR hits voice & orchestra identically (linear target preserved).

| axis | count | members |
|---|---|---|
| **in-domain operatic voice donors (singers)** | **20** | VocalSet: 9F/11M pros, all ranges, operatic techniques |
| + OOD voice donors | +N | Jingju a-cappella (OOD timbre) ; + per-voice SATB stems from Cantoría(4)/CSD(16)/Dagstuhl(~13)/ESMUC(~12) ≈ **+45 additional isolated singers** |
| **orchestra donors** | **6** | PHENICX: Beethoven Sym7 / Bruckner Sym8 / Mahler Sym1 / Mozart(orch) ; Spheres: Mozart / Tchaikovsky R&J |
| **measured concert-hall venues** | **3** (+dry) | Spheres hall · St Paul's Hall (3D-MARCo) · **Pori (new)** · + anechoic/dry regime |

**Effective independence.** Render combinatorics = 20 × 6 × 4 = **480 renders**. But renders share donors, so they are **augmentation, not 480 independent works**. The axis that actually matters for a *voice-removal* judge is the **accompaniment** (that IS the target): **6 orchestras × 4 venue-regimes = 24 distinct, accompaniment-independent orchestral beds**, each presentable with 20 voice timbres. So the pool contributes **≈24 accompaniment-independent scenarios** (≈3× the 7 real orchestral beds) with wide voice variety — strong generalization coverage, **but donor-shared** (do not double-count against the independent-work target; keep as a labelled augmentation group).

---

## 6. Why the strict tier is capped from OPEN data (search evidence)

- **Zenodo API sweep**: 22 queries (opera/aria/orchestra/anechoic/cantata/oratorio/lieder/… multitrack). Every in-domain hit was already held or is a **donor/choir/instrumental** set. The only *new* voice+accompaniment record — **"30 Recordings of 19th Century Viennese Songs" (Zenodo 15363113, CC BY 4.0, 3 GB)** — is **single stereo mixes (voice+fortepiano), NO stems** → not separable, no ground truth → cannot be a training reference (usable only as unlabeled eval input).
- **Confirmed gated / OOD / non-open** (do not count): MedleyDB full (request-gated — but see §7), URMP (no voice), Erkomaishvili (CC BY-NC, a-cappella, files restricted), jaCappella (contact-gate, a-cappella), GTSinger (CC BY-NC), Voice+piano Zenodo 12928912 (license=None), Saraga Carnatic (CC BY-NC-SA), Haydn string-quartet anechoic (Zenodo 4955282, CC BY-NC, no voice), ChoraleBricks (Zenodo 20849469, CC BY, wind, no voice), PCD-K piano concerto (no voice).
- **Verdict**: no open, separable, real soloist+orchestra multitrack exists that we don't have. The strict tier is **saturated at 7** from open sources.

---

## 7. RANKED "ACQUIRE NEXT" — most independent works/singers per dollar

| # | item | type | gets us | cost | license/URL |
|---|---|---|---|---|---|
| **1** | **Cantolopera lossless** (Casa Ricordi / Hal Leonard) | **purchase** | **the ONLY cheap MASTERED same-take voice+orchestra with official minus-voice companion.** ~4–6 vols → **15–40 same-take aria pairs** (soprano+tenor, Verdi/Puccini/Mozart/Bellini/Donizetti). Closes most of the gap to 40. | ~$20–30/vol; **$150–200** total | ricordi/halleonard product pages |
| **2** | **MedleyDB full** (v1/v2) | request (free) | **6 classical voice+PIANO multitracks** (Mozart *Dies Bildnis*/*Bester Jüngling*, Handel *Tornami a vagheggiar*, Schubert *Erstarrung*, Schumann *Mignon*, Debussy) = **+6 real voice+piano works** | free | medleydb.weebly.com / Zenodo permission-request (NYU) |
| 3 | **CoVox** (OSF `osf.io/cgexn/`) | open* | **22 female singers incl. OPERA-ARIA style** → +22 operatic voice donors (≈2× the VocalSet axis) | free | *LICENSE UNCONFIRMED (no OSF node license set) — VERIFY CC-BY before use* |
| 4 | Haydn string-quartet anechoic (Zenodo 4955282) | open (NC) | +1 anechoic chamber-string **donor** ensemble | free | CC BY-NC 4.0 |
| 5 | ChoraleBricks (Zenodo 20849469) | open | +wind/brass-timbre orchestral **donors** (10 chorales) | free | CC BY 4.0 |
| 6 | OpenAIR concert halls (Jack Lyons, cathedrals) | open | +more real **venues** (site intermittent Aug-2026) | free | mostly CC BY-SA |
| 7 | Saraga Carnatic (14 GB) | open (NC) | +OOD voice+violin+mridangam works | free | CC BY-NC-SA 4.0 |
| 8 | Music Minus One – Opera/Vocal | purchase | orchestra-only side (pair w/ commercial full — NOT same-take → weaker null) | ~$20/vol | Hal Leonard MMONE |

**Recommendation**: buy **4–6 Cantolopera volumes** (#1) and file the **MedleyDB request** (#2) now — together they add ~20–45 real same-take/near-exact voice+accompaniment works and take the honest independent count **past 40**. Everything else is donor/venue augmentation.

---

## 8. SELF-REMIX LADDER on our library (augmentation, tier I — NOT independent)

From `calibration/novocal_result.json`: **23 of 27** library-track separations have ≥3 s vocal-free (orchestral-only) spans (`_vocal_ratio ≤ 0.15`), **≈1,180 s** vocal-free total. Each span → an **exact self-remix reference** (target vocal = silence; target accompaniment = the mix). Longest single spans: VERDI_1m44s 73.4 s, VERDI_1m39s 70.5 s, VERDI_1m33s 70.6 s, MARSEILLAISE 56.9 s, HABANERA 33.4 s. Through-sung (no qualifying span): 11_TURKISH_DANCE, 18_DRINK_TRICK_LONGER, DRINK_TRICK_SHORT, TURKSH_DNC-22s. **Grouped as ONE own-corpus** for training augmentation; must not be counted as independent test works.

---

## 9. EXACT PATHS + integrity classes (quick index)

- Real solo+orch: `~/datasets/bologna/{Verdi,Puccini,Donizetti}_ex/` · `~/datasets/aalto/mozart/` · `~/datasets/freidi/built/{06,08,09}/`
- Voice+organ (14): `~/datasets/works/cantoria/CantoriaDataset_v1.0.0/Audio/`
- OOD: `~/datasets/works/saraga_hindustani/`
- Choir (D): `~/datasets/works/{choral_singing,dagstuhl_choir,esmuc_choir,marco3d}/`
- Orchestra donors (E): `~/datasets/works/phenicx_anechoic/` · `~/datasets/spheres/`
- Voice donors (F): `~/datasets/works/vocalset/` · `~/datasets/works/jingju_acappella/`
- RIR venues (G): `~/datasets/spheres/RIRs_ext/` · `~/datasets/works/marco3d/` · `~/datasets/works/rir_pori/` (NEW)
- Synthetic (H): `~/datasets/works/choralsynth/`
- FreiDi build report: `~/datasets/freidi/built/freidi_build_report.json`
- Manifest: `~/datasets/works_manifest.jsonl` · this file: `~/datasets/TRAINING_WORKS.md`

## 10. Disk / discipline
- New this mission: FreiDi stems extract ~4.5 GB + built FLACs ~1 GB + Pori RIRs 28 MB. Free on `/`: ~2.8 TB. Other missions' dirs (`bologna/aalto/spheres/annas-archive`) untouched; `annas-archive` (copyrighted bulk) NOT counted or used.
- Licenses cited per row in `works_manifest.jsonl`. Open/CC only in the counted tiers; NC items flagged; `annas-archive`/Cantolopera-lossy excluded from the open count.
