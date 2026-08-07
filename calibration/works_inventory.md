# WORKS INVENTORY — ground-truth for the vocal-separation quality judge

Date: 2026-08-07 · Storage/compute: **tt-quietbox2** (`ttuser@100.91.242.69`), CPU-only, `~/datasets/works/`
Downloader: `aria2c -c -x16 -s16` (Zenodo throttle avoided; all sizes match the Zenodo API byte counts).
Scope honored: open/free/legal only, licenses cited; **count = WORKS, not files**; huge multichannel sets pulled selectively.

---

## Headline count

| tier | before | acquired this mission | **now** |
|---|---|---|---|
| **Independent REAL-VOICE separable works** | ~6 | **+30** | **≈36** |
| Synthetic multitrack choral works (linear-exact by construction) | 0 | +20 (ChoralSynth) | 20 |
| Isolated-vocal **donor** pool (singers) → synthesizable voice+orch works | 0 | +VocalSet (20 pros) + Jingju a-cap | ~20 donors |
| Orchestra **donor** pieces (no voice) → synthesizable voice+orch works | 2 (Spheres) | +4 (PHENICX Beethoven/Bruckner/Mahler/Mozart) | 6 |
| Measured concert-hall RIR banks (for hall regime) | 1 (Spheres) | +1 (3D-MARCo, St Paul's, 71-mic) | 2 |

**Real-voice pool went from ~6 → ~36 distinct works** (target was 20–30). Counting each choir-set piece as one work; ESMUC counted conservatively at 4 pieces (7 take-sets). The donor pools additionally enable **100+ synthesizable** voice+orchestra works (each VocalSet singer × each orchestra donor × each RIR regime).

---

## WORKS TABLE

Legend — integrity class: **linear_exact** = per-source stems / difference sum to the mix by construction; **same_take** = real per-singer/section stems of ONE performance, near-linear (measured −30…−33 dB reconstruction residual); **synthesizable** = isolated voice × orchestra donor, or dry stems × measured RIR; **same_perf_bleed** = one simultaneous take with cross-mic bleed; **hall_measured** = real concert-hall capture.

| source | work id / pieces | voice type | accompaniment | separable-how | license | size | integrity class | path (on box) |
|---|---|---|---|---|---|---|---|---|
| **Cantoría** (Zenodo 5878677/5851070) | **14 pieces** (CEA, EJB1, EJB2, HCB, LBM1, LBM2, LJT1, LJT2, LNG, RRC, SSS, THM, VBP, YSM) | SATB quartet (pro, one-per-part) | **organ** (+ a-cappella) | per-piece **Mix** (voices) vs **MixOrgan** (voices+organ) → organ target exact by difference (organ ≈ −3…−5 dB); S/A/T/B stems sum to Mix (−31…−33 dB) | **CC BY 4.0** | 861 MB | **linear_exact** (organ) + **same_take** (SATB) | `works/cantoria/` (extracted) |
| **Choral Singing Dataset** (Zenodo 1286570) | **3 pieces** (Locus Iste / Nino Dios / El Rossinyol) | SATB, **16 singers** (4×S/A/T/B) | a cappella | per-singer stems → linear sum to section/choir | **CC BY 4.0** | 937 MB | same_take | `works/choral_singing/` (extracted) |
| **Dagstuhl ChoirSet** (Zenodo 4608395) | **3 pieces** (DCS_LI Locus Iste, DCS_SE, DCS_TP) | SATB choir + soloists; per-singer + **larynx** mics | a cappella | per-singer/section stems + room mics; multiple takes | **CC BY 4.0** | 5.1 GB | same_take | `works/dagstuhl_choir/` (zip) |
| **ESMUC Choir Dataset** (Zenodo 5848990) | **7 take-sets ≈ 4 pieces** (DG, DH1/DH2, SC1/SC2/SC3, WU) | SATB choir, per-singer (S1-4,A1-3,B1-2) | a cappella | per-singer stems + **ORTF/AB** room-mic mixtures | **CC BY 4.0** | 2.3 GB | same_take | `works/esmuc_choir/` (zip) |
| **3D-MARCo — Acappella** (Zenodo 3474286) | 1 vocal-quartet performance | SATB a-cappella group | a cappella | **real St Paul's concert hall**, 71-mic 3D arrays (OCT3D, 2LCube, Decca, PCMA3D…) | **CC BY 3.0** | 1.7 GB | hall_measured | `works/marco3d/09*Acappella.zip` |
| **3D-MARCo — Impulse Responses** | RIR bank | — (donor) | — | measured concert-hall RIRs → hall-regime synthesis (complements Spheres RIRs) | CC BY 3.0 | 741 MB | (RIR donor) | `works/marco3d/03*Impulse*.zip` |
| **PHENICX-Anechoic** (Zenodo 840025) | **4 orchestra donors**: Beethoven Sym7, Bruckner Sym8, Mahler Sym1, Mozart | — (no voice) | orchestra (per-instrument anechoic) | 10–39 per-instrument anechoic stems/piece → sum to orchestra; convolve with RIRs for hall | **CC BY 4.0** | 764 MB | linear_exact (donor) | `works/phenicx_anechoic/` (extracted) |
| **VocalSet** (Zenodo 10200775) | **20 professional singers** (opera + techniques, M/F, all ranges) | solo, isolated | none (donor) | isolated vocals × orchestra donor × RIR = synthesizable voice+orch work | **CC BY 4.0** | 2.5 GB | synthesizable (donor) | `works/vocalset/` (zip) |
| **ChoralSynth** (Zenodo 10161065) | **20 multitrack choral songs** | synthetic SATB+ | a cappella (synth) | per-part synthesized stems → exact sums | **CC BY-SA 4.0** | 96 MB | linear_exact (synthetic) | `works/choralsynth/` (extracted) |
| **Jingju a-Cappella (JaCRC)** (Zenodo 6536490) | Beijing/jingju operatic singing (many singers/segments) | solo jingju (dan/laosheng roles) — **OOD** | a cappella | isolated OOD-timbre voice → donor; OOD stress set | **CC BY 4.0** | 7.1 GB | synthesizable (donor) / OOD | `works/jingju_acappella/` (zip) |
| **Saraga 1.5 — Hindustani** (Zenodo 4301737) | Indian classical (khyal) — **≥3 ragas** | solo voice — **OOD** | **harmonium + tabla** (multitrack) | per-source multitrack stems (voice/harmonium/tabla) | **CC BY-NC-SA 4.0** (non-commercial) | 4.1 GB | same_take / OOD | `works/saraga_hindustani/` (zip) |
| **MedleyVox** (Zenodo 7984549) | duet / unison / rest — **voice-on-voice** eval | multiple singing voices | (voices only) | multi-singer mixtures + isolated → voice-vs-voice stress set | **CC BY 4.0** | 904 MB | same_take (stress) | `works/medleyvox/` (zip) |
| **MedleyDB Sample** (Zenodo 1438309) | LizNelson_Rainfall (1 vocal multitrack) | solo (pop/folk) | band | per-stem multitrack (marginal — not classical) | **CC BY-SA 4.0** | 415 MB | same_take | `works/medleydb_sample/` (zip) |

### Already-held works (context — not re-counted as new)
| source | pieces | voice | accompaniment | integrity | license |
|---|---|---|---|---|---|
| Bologna anechoic | Verdi, Puccini, Donizetti (3) | solo opera | orchestra | linear_exact (dry) | (held) |
| Aalto anechoic | Mozart (dry+hall) | soprano | orchestra | linear_exact | (held) |
| The Spheres | Mozart, Tchaikovsky (dry/hall/stereo) | (borrowed voice) | orchestra + measured RIR | linear_exact | CC BY-SA 4.0 |
| FreiDi | Freischütz (opera) | ensemble | orchestra | same_perf_bleed | CC BY-SA 4.0 |

---

## COUNT by regime / voice / accompaniment (real-voice works)

**By accompaniment**
- Voice ensemble + **organ**: Cantoría 14
- **A-cappella choir** (SATB): Choral Singing 3 + Dagstuhl 3 + ESMUC ~4 + 3D-MARCo 1 = **11**
- Solo voice + **orchestra** (real, held): Bologna 3 + Aalto 1 + Spheres 2 + FreiDi 1 = **7**
- **OOD** voice + ensemble: Saraga-Hindustani ≥3 (voice+harmonium+tabla) + Jingju 1 (a-cap) = **≥4**
- Voice-on-voice: MedleyVox (set) · Pop multitrack: MedleyDB-Sample 1
- Synthetic choral: ChoralSynth 20

**By regime**
- **dry / studio**: Cantoría, Choral Singing, Dagstuhl, ESMUC, VocalSet, ChoralSynth, Saraga, Jingju, Bologna, Aalto-dry (majority)
- **hall / measured-room**: 3D-MARCo Acappella (real hall) + 3D-MARCo & Spheres RIRs (synth hall) + Aalto-hall + Spheres-hall
- **stereo**: Spheres-stereo + 3D-MARCo multi-array stereo

**By voice type**
- SATB choir: Cantoría, Choral Singing, Dagstuhl, ESMUC, ChoralSynth, 3D-MARCo
- Solo soprano/tenor + orchestra: Bologna, Aalto, Spheres
- Isolated solo (all ranges, technique-labelled): VocalSet
- OOD operatic: Jingju (jingju), Saraga (Hindustani khyal)

---

## FREE self-remix works — library tracks with ≥3 s vocal-free (orchestral-only) spans

Detector reuses the passage miner's defining `no_vocal_control` gate (`audio_extract.passages`): a span is vocal-free when `_vocal_ratio = rms(vocal)/rms(vocal+accomp) ≤ 0.15` (`no_vocal_abs_ratio_max`), applied continuously with the repo's `dsp.frame_rms_db` framing, runs ≥3 s. **23 of 27** library-track separations qualify → each yields exact self-remix reference cases (target vocal = silence, target accompaniment = the mix, over the span).

| track | n spans (≥3 s) | total vocal-free s | longest s |
|---|---|---|---|
| 1_HABANERA_LAST | 8 | 123.4 | 33.4 |
| VERDI_1m44s_NO_VOICES | 3 | 102.9 | 73.4 |
| VERDI_1m39s_NO_VOICES | 3 | 98.8 | 70.5 |
| VERDI_1m33s_original_tempo_NO_VOICES | 2 | 93.2 | 70.6 |
| MARSEILLAISE_APPL_Police_7_23_26 | 3 | 71.7 | 56.9 |
| 17_MARSEILLAISE_APPLAUSE | 2 | 63.6 | 56.9 |
| 5_FLY_SWAN_TRAVIATA | 5 | 51.1 | 16.0 |
| 6_LA_WALLY_CHILDREN | 4 | 50.6 | 17.5 |
| 16_DOLL_TRACK_NoVoice_Gb | 5 | 48.0 | 26.7 |
| 12_13_SCARYDOG_STABAT_MATER | 4 | 44.9 | 24.9 |
| 22_VALKYRIES_NEW | 4 | 41.0 | 22.7 |
| 19_CABARET_DRUMROLL_NEW | 2 | 35.8 | 32.2 |
| 15_IRISH_DANCE | 1 | 30.5 | 30.5 |
| 2_MOSQUITO | 1 | 28.3 | 28.3 |
| 3_BEDBUG_TCHAIK_LONG | 1 | 25.2 | 25.2 |
| CABARET_Willkommen | 1 | 23.0 | 23.0 |
| 7_VERDI_SLOW_Voices | 2 | 15.9 | 9.5 |
| DOLL_DRAFT | 2 | 14.8 | 9.8 |
| VERDI_FAST_Voices (×2 variants) | 2 | 14.4 | 8.6 |
| BEDBUG_TCHAIK_OLD | 1 | 10.9 | 10.9 |
| 20_REQUIEM_FOR_HUMANITY | 1 | 8.0 | 8.0 |
| 21_BARBER_AGNUS_DEI | 1 | 6.8 | 6.8 |

**No qualifying span (through-sung):** 11_TURKISH_DANCE_GUTEN_MORGEN, 18_DRINK_TRICK_LONGER, DRINK_TRICK_SHORT, TURKSH_DNC-22s.
Per-span start/end timestamps: `calibration/novocal_result.json`.

---

## PURCHASE SHORTLIST — real same-take MASTERED voice+orchestra pairs (for the user to buy)

The open sets above are anechoic/dry or amateur-choir; the one tier we can only *buy* is **professionally mastered opera with an official orchestra-only (minus-voice) companion of the SAME take**. Two product lines ship exactly that:

**① Cantolopera (Casa Ricordi, dist. Hal Leonard) — STRONGEST: each volume includes BOTH a full performance (singer + orchestra) AND the identical orchestra-only backing track (same take).** Book + online audio, typically **$19.99–$29.99** each. Priority volumes:
- Cantolopera: *Puccini Arias for Soprano* Vol. 1 (HL 50484608) — ~$24.99
- Cantolopera: *Puccini Arias for Tenor* Vol. 1 — ~$24.99
- Cantolopera: *Verdi Arias for Soprano* Vol. 1 — ~$22.99
- Cantolopera: *Arias for Soprano* Vol. 1–2 (mixed composers) — ~$19.99 ea
- Cantolopera: *Arias for Tenor* Vol. 1–3 — ~$19.99 ea
- Cantolopera: *Mozart Arias for Soprano* — ~$22.99
- Cantolopera: *Bellini / Donizetti Arias* — ~$22.99
  → ~6–8 volumes ≈ **$150–$200** buys dozens of same-take with/without-voice aria pairs across soprano+tenor, multiple composers.

**② Music Minus One — Opera/Vocal** (dist. Hal Leonard, series MMONE). Mostly the orchestra-only side (pair with a commercial full recording — *not* same-take, so weaker null). Book+audio ~**$19.99–$24.99** (anchor: MMO "Phantom of the Opera – Vocal" $22.99). Useful ones: *Opera Arias for Soprano & Orchestra*, *Verdi Opera Arias*, *Puccini — La Bohème/Tosca arias*.

> Prices are indicative (verify at Hal Leonard / Sheet Music Plus at purchase — their live search is JS-gated to automated fetches). **Recommendation: buy 4–6 Cantolopera volumes first** — they are the only cheap source of *mastered same-take* voice+orchestra pairs, which is the exact tier the anechoic/dry open data cannot certify.

---

## Disk + blockers

- **Disk used this mission:** `~/datasets/works` = **26 GB** downloaded (Cantoría .86 + Choral .94 + ChoralSynth .10 + Dagstuhl 5.1 + ESMUC 2.3 + Jingju 7.1 + 3D-MARCo 2.5 + PHENICX .76 + VocalSet 2.5 + MedleyDB-Sample .42 + MedleyVox .90 + Saraga-Hindustani 4.1). Free on `/`: **2.8 TB (20% used)**. Other missions' dirs (bologna/aalto/spheres/freidi/annas-archive) untouched.
- All 15 files **size-verified** against the Zenodo API byte counts. Cantoría/Choral/ChoralSynth/PHENICX extracted + integrity-checked.

**Blockers / request-gated (noted, not pulled):**
- **MedleyDB full audio** (Zenodo 1649325 / 1715175 — the 6 classical voice+piano multitracks: Mozart *Dies Bildnis*/*Bester Jüngling*, Handel *Tornami a vagheggiar*, Schubert *Erstarrung*, Schumann *Mignon*, Debussy) → **request-gated** (0 downloadable files on Zenodo; audio behind the NYU/Dagstuhl request form; no autonomous form submission / no credentials). The free **MedleyDB Sample** contains only 2 non-classical tracks. **Action for user:** request full MedleyDB at https://medleydb.weebly.com (or the Zenodo "request access").
- **Jingju Multi-Track (JMTRC, voice + jinghu)** (Zenodo 13580365) — **CC BY-NC-ND, 0 downloadable files** (restricted). Record 13863042 the task named is only a thesis PDF. Pulled the CC BY jingju **a-cappella** (6536490) as the OOD-jingju stand-in instead.
- **jaCappella corpus** (HF) — API 401 / agreement-gated (email request at tomohikonakamura.github.io/jaCappella_corpus). Covered functionally by Cantoría/Choral/Dagstuhl/ESMUC/ChoralSynth SATB stems.
- **GTSinger** (HF, not gated but license=None on card; paper says CC BY-NC-SA) — large; VocalSet already gives a clean CC-BY isolated-vocal donor, so deferred.
- **Voice+piano classical** (Zenodo 12928912, 6.2 GB) — **license = None** → skipped per open/legal-only constraint.
- **Saraga Carnatic** (14 GB, CC BY-NC-SA) and **Saraga Audiovisual** (96 GB, CC BY) — pulled the 4.1 GB Hindustani slice only (disk discipline); Carnatic available on demand for a 2nd OOD voice+violin+mridangam work.
- **URMP** (Zenodo 5034983, CC0, 12 GB) — all-instrumental (no voice); usable only as chamber-ensemble donors, deferred.
- **Throttle:** none — `aria2c -x16 -s16` held ~100 MB/s across all Zenodo pulls.
