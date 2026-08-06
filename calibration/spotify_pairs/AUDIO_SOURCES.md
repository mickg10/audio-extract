# AUDIO_SOURCES — resolving our with/without-voice pairs to actual audio

_Reconnaissance + availability-verification for the vocal-separation eval benchmark. Written 2026-08-06._
_Inputs: `pairs_classical.jsonl` (10,344 classical/opera/crossover pairs), `pairs_all.jsonl` (966,395), the Anna's Archive Spotify 2025-07 dump on tt-quietbox2 + NAS._
_Rule of engagement for this work: map + verify routes, do NOT mass-download copyrighted audio. Only small legal/promo clips and file indices were fetched to confirm contents._

---

## TL;DR verdicts

1. **AA `track_files` is NOT audio.** It is `spotify_clean_track_files.sqlite3.zst` (47.12 GB compressed, the single largest file in the 199.9 GB torrent) — a **SQLite manifest** of an audio-acquisition pipeline: per-track `filename` (real `*.ogg` paths), `sha256_original`, `status` (`has_download`/`done`/`error_not_available`/`todo_unk`), `filesize_bytes`, and Spotify CDN **file_ids** per format (`file_id_ogg_vorbis_96/160/320`, `file_id_aac_24`, `file_id_mp3_96`). 47.12e9 / 256M tracks = **184 bytes/track** — arithmetically impossible to be audio. The manifest **proves AA has been downloading the actual audio** (OGG Vorbis) into a `track-popularity-N/…` tree, but **those audio bytes are not in this metadata torrent**. A companion audio payload is implied by the manifest; I could **not** confirm a public AA audio torrent from here (AA domains are blocked in this environment). Either way it is a **shadow-library, research-only, non-redistributable** route.

2. **The dump itself carries Spotify `preview_url`.** The `tracks` table has a `preview_url` column, populated for **17,083 / 19,035 (90%)** of our classical pair track-ids. **8,617 / 10,344 (83%) of classical pairs have BOTH sides' preview.** A sampled `p.scdn.co` clip **still serves live today** (HTTP 200, ~360 KB, ~30 s MP3, Google CDN). BUT coverage is skewed: **only 4 / 20 of the marquee top-20 pairs** (Bocelli/Brightman/Il Volo/Con te partirò) have both Spotify previews — the 83% is carried by the symphonic-metal tail (Nightwish/Within Temptation/Avantasia).

3. **Deezer's public ISRC API fills the marquee gap.** `https://api.deezer.com/track/isrc:<ISRC>` (no auth) resolves our exact ISRCs to a track + 30 s preview. Sampled coverage: **both ISRCs resolve for ~96%** of pairs; **both have a 30 s preview for ~78% overall and ~60% of the marquee top-20** — exactly the pairs Spotify's dump previews miss (verified: Bocelli & Brightman instrumentals return Deezer previews). **Spotify ∪ Deezer previews ≈ 90%+ of pairs** get at least one both-sides 30 s pair, free and legally.

4. **30 s previews are same-master but NOT the same window.** Cross-correlating real base-vs-variant clips: genuine same-master pairs correlate strongly on the accompaniment (envelope peak up to **0.72–0.83**), but the two previews frequently start at different offsets (lags 0 → +11 s) and vary in length (15–30 s). **Usable for exact-reference eval only after a per-pair cross-correlation alignment step**, and overlap can be short when offsets are large.

5. **Legal full-master route = buy by ISRC.** ~49.5% of classical pairs are `same_album=true` (both versions officially released on one single/album; labels confirmed: Sugar, Angel/EMI, Nuclear Blast, Sony/Columbia). These are purchasable full-length (iTunes/Amazon/Qobuz/7digital), keyed by our exact ISRC.

6. **Open datasets don't contain our masters** but give redistributable same-master with/without-voice audio at small scale — MUSDB18 (mix vs vocals+accompaniment) is the closest analog.

---

## 1. Anna's Archive — definitive `track_files` answer + audio holdings

### 1.1 The torrent (parsed from the metainfo we hold)

`annas_archive_spotify_2025_07_metadata` — **199.89 GB, 9 files, no webseeds** (58 trackers, DHT/PEX; 64 MiB pieces, 2,979 pieces). Full file table (exact bytes from the metainfo):

| # | file | bytes | GB | what it is |
|--:|---|---:|---:|---|
| 0 | spotify_artist_redirects.json | 282,155 | 0.00 | id redirects |
| 1 | spotify_audiobook_chapters.jsonl.zst | 1,553,005,565 | 1.55 | audiobook ch. metadata |
| 2 | spotify_audiobooks.jsonl.zst | 1,757,107,274 | 1.76 | audiobook metadata |
| 3 | spotify_clean.sqlite3.zst | 36,653,110,502 | 36.65 | **tracks/artists/albums** (+ `preview_url`) |
| 4 | spotify_clean_audio_features.sqlite3.zst | 17,730,266,497 | 17.73 | per-track audio features |
| 5 | spotify_clean_playlists.sqlite3.zst | 28,516,849,292 | 28.52 | playlists |
| 6 | **spotify_clean_track_files.sqlite3.zst** | **47,115,870,111** | **47.12** | **audio-acquisition manifest (see below)** |
| 7 | spotify_show_episodes.jsonl.zst | 26,964,627,558 | 26.96 | podcast episode metadata |
| 8 | spotify_shows.jsonl.zst | 39,596,247,962 | 39.60 | podcast show metadata |

Every file is `.jsonl.zst` or `.sqlite3.zst` — **there is no audio archive among the 9 files.** This corroborates that the release is metadata (+ an audio manifest), not audio.

### 1.2 What `track_files` actually is (verified schema)

I fetched **only the first 64 MiB piece at the head of file #6** via a one-off libtorrent script on tt-quietbox2 (reading the SQLite header/index to confirm contents — 129 MB real disk, deleted afterward; no bulk download), decompressed the prefix, and read the schema directly:

```
CREATE TABLE "track_files" (
  rowid INTEGER PRIMARY KEY,
  track_id TEXT NOT NULL,              -- Spotify track id (UNIQUE)
  filename TEXT,                       -- real path to a downloaded .ogg
  reencoded_kbit_vbr INTEGER,          -- re-encode bitrate
  fetched_at INTEGER,
  session_country TEXT,                -- which country session pulled it
  sha256_original TEXT,                -- content hash of the audio file
  sha256_with_embedded_meta TEXT,
  status TEXT NOT NULL,                -- has_download / done / error_not_available / error_has_replacement / todo_unk
  isrc_has_download INTEGER,
  track_popularity INTEGER,
  secondary_priority REAL,             -- acquisition-queue priority
  prefixed_ogg_packet BLOB,            -- tiny OGG header packet (NOT the audio)
  alternatives TEXT,
  file_id_ogg_vorbis_96 TEXT,          -- Spotify CDN file_ids (SHA1, per format)
  file_id_ogg_vorbis_160 TEXT,
  file_id_ogg_vorbis_320 TEXT,
  file_id_aac_24 TEXT,
  file_id_mp3_96 TEXT,
  language_of_performance TEXT, artist_roles TEXT, has_lyrics INTEGER,
  licensor TEXT, original_title TEXT, version_title TEXT,
  content_ratings TEXT, filesize_bytes INTEGER);
CREATE UNIQUE INDEX clean_track_files_track_id_unique ON track_files(track_id);
CREATE INDEX track_files_todo ON track_files(track_id) WHERE status='todo_unk';
CREATE INDEX track_files_filesize ON track_files(filesize_bytes);
CREATE INDEX track_files_filesize_todo2 ON track_files(track_popularity, filename)
       WHERE filename IS NOT NULL AND filesize_bytes IS NULL;
```

Actual `filename` values recovered from the prefix (proof the audio is real OGG files in a structured store):
```
track-popularity-0/C/CH/Christian Sinding/2004 Sinding_ Violin Sonatas (3B463rh34zk3cSjgIPj5oq)/01 Christian Sinding - Violin Sonata in F Major, Op. 73_ I. Allegro con brio.ogg
track-popularity-0/C/CH/Christmas Jazz Holiday Music/2022 Christmas Music (Instrumentals) (5bVwV2mhn2ZHIFSP6bUVLz)/06 Christmas Jazz Holiday Music - Frosty Morning.ogg
```
Also present in the prefix: 40-hex Spotify **file_ids** and 64-hex **sha256** hashes; `status` tokens `has_download`, `done`, `error_not_available`, `error_has_replacement`, `todo_unk`.

**Determination (high confidence, schema-level evidence):** `track_files` is **the manifest/ledger of an Anna's Archive Spotify audio-acquisition effort** — it records, per track, where the downloaded `.ogg` lives (`filename`), its hash and size, an acquisition `status`, and the Spotify CDN `file_id`s used to fetch each encoding. It is **not audio, not URLs to audio, and not directly usable as audio.** Two sub-routes to real audio are implied by it, both shadow/gated:
- **AA's own audio store** (`filename` → `*.ogg`, keyed by `track-popularity-N/…`). The audio bytes are **not** in this torrent. If AA releases the payload it would be a separate (likely popularity-sharded) collection. **Not confirmed public from here.**
- **Spotify CDN directly** (the `file_id_*` values). Fetching+decrypting these requires Spotify credentials + PlayPlay/Widevine keys (librespot-style stream-ripping) — ToS-violating, effectively a shadow route.

### 1.3 Does AA host audio generally / an accompanying audio collection?

Anna's Archive is fundamentally a **text/document** shadow library (Library Genesis, Sci-Hub, Z-Library, comics, magazines) plus **metadata scrapes** (this Spotify set, IMDb, WorldCat, etc.). It does **not** run a general music streaming/audio collection. For this Spotify release specifically: the public torrent is **metadata + the audio manifest**; the manifest strongly implies AA has assembled the corresponding OGG audio internally, but **I could not confirm a public AA "Spotify audio" torrent** (annas-archive.org/.se/.gl and web.archive.org are all blocked in this environment — this needs a direct check of `annas-archive.org/torrents` filtered on "spotify"/"music" and the AA blog). **Action item:** verify from an unrestricted network whether an audio payload torrent exists. Even if it does, it is copyrighted commercial audio redistributed without licence → **research-use-only, never redistributable.**

---

## 2. Spotify `preview_url` verdict

**Available in our dump?** Yes. `tracks.preview_url` is populated for **17,083 / 19,035 (89.7%)** of classical pair track-ids. Per pair:

| classical pairs (10,344) | both sides have preview | base-only | neither |
|---|---:|---:|---:|
| count | **8,617 (83.3%)** | 637 | 618 |
| top-50 (ranked) | 32 / 50 | — | — |
| **top-20 marquee** | **4 / 20** | — | — |

**Still live?** Yes — a sampled `https://p.scdn.co/mp3-preview/<hash>?cid=…` returned **HTTP 200, Content-Length 359,795 (~30 s ~96 kbps MP3), served from Google Cloud Storage, `Accept-Ranges: bytes`.** (Spotify removed `preview_url` from the *Get Track* Web-API response in late 2024 — widely reported by developers; not independently re-verifiable in this environment — but the **CDN objects captured in this 2025-07 dump still resolve**, so the dump's URLs are directly fetchable without the API.)

**Same-master / alignment (measured).** I downloaded real base-vs-variant preview clips for 6 genuine same-album instrumental pairs and cross-correlated (waveform + 50 ms RMS envelope):

| pair | env-xcorr peak | best lag | note |
|---|---:|---:|---|
| Within Temptation — Entertain You | 0.78 | ~0 s | same window, near sample-aligned |
| Avantasia — The Witch | 0.72 | 0 s | same window (15 s clips) |
| Avantasia — Here Be Dragons | 0.65 | +1.6 s | same master, offset window |
| Nightwish — Élan | 0.41 | −1.9 s | partial overlap |
| Nightwish — Alpenglow | 0.35 | −0.6 s | different windows, poor overlap |
| Paul Potts — I Believe (marquee) | 0.32 | +4.6 s (wav +11 s) | vocal side lowers wav xcorr; windows offset |

**Verdict:** previews are the **same commercial master** (accompaniment correlates strongly where they overlap), but the base and variant clips are **not guaranteed to be the same 30 s window** (offsets 0→11 s; lengths 15–30 s). For a 30 s exact-reference separation eval they are **usable only after a per-pair cross-correlation alignment + trim to the overlapping region** (align on instrumental sections; the vocal in the base depresses raw-waveform correlation). Where offsets are large, usable overlap can drop to a few seconds. **Sufficient for a quick, zero-cost eval on the aligned overlap; not a drop-in sample-aligned reference.**

---

## 3. ISRC → audio routes

- **Deezer public API (best free ISRC route).** `GET https://api.deezer.com/track/isrc:<ISRC>` — no auth, returns the track + a 30 s preview (`cdnt-preview.dzcdn.net/*.mp3`) when available. Sampled over 162 classical pairs: **both ISRCs resolve ~96%**, **both have a preview ~78% overall / ~60% marquee**. Verified it returns previews for Bocelli "Ama credi e vai (Instrumental)" (ITZ040600046) and Brightman "I Will Be With You … Instrumental" (USNS10700005) — **marquee tracks that have no Spotify preview in the dump.** Same-master (matched by exact ISRC); same 30 s-window/alignment caveat as §2. Some resolve but with empty preview (e.g., Il Volo "Grande amore", ITB001500090/091).
- **Deezer/streaming full track by ISRC → stream-rip.** Deezer resolves the *full* track for ~96% of pairs; full-length audio via the private streaming endpoint (deezer/`deemix`-style) or any streaming client is a **ToS-violating gray route** (full master, but not clean legally).
- **iTunes/Apple.** `itunes.apple.com/lookup?isrc=` was unreliable here; the term-based iTunes Search API returns 30 s AAC previews resolvable by title+artist. Apple Music API resolves ISRC→song but needs a developer token. Lower priority than Deezer.
- **MusicBrainz.** Maps ISRC→recording MBID but **hosts no audio**; useful only to disambiguate/confirm the recording and to hop to purchase/stream links. (WS API not fetchable in this environment.)
- **YouTube / YouTube Music.** No ISRC lookup; match by title+artist. Official instrumentals and "Art Track"/Content-ID uploads are common, but availability is inconsistent and downloading violates YouTube ToS (gray).

---

## 4. Ranked ROUTE TABLE

Legend — licensing: **open-research** (redistributable), **commercial-purchase** (legal, buy), **commercial-stream** (legal to listen, ripping = ToS gray), **shadow** (copyright-infringing / research-only). Same-master: how faithfully both sides come from one master.

| # | route | yields | covers (of 10,344) | licensing | same-master | acquisition steps |
|--:|---|---|---|---|---|---|
| 1 | **Buy by ISRC** (iTunes / Amazon / Qobuz / 7digital) | full-length, hi-qual, exact master | ~49.5% cleanly on one release (5,117 `same_album`); near-all individually | **commercial-purchase** | **Excellent** (the actual master) | ISRC/title→store lookup → purchase → download WAV/FLAC |
| 2 | **Deezer ∪ Spotify 30 s previews** | 30 s clip per side | **~90%+** ≥1 both-sides pair (Spotify dump 8,617; Deezer ~78%; complementary) | **open/legal** (free promo clips) | Good, **after xcorr align** | dump `preview_url` + `api.deezer.com/track/isrc:` → fetch clips → per-pair cross-correlate → trim overlap |
| 3 | **Open-research stem datasets** (MUSDB18/-HQ, Anechoic Opera, MoisesDB, MedleyDB) | mix + vocals + accompaniment stems | **0 of our pairs** (separate corpora) | **open-research** (mostly non-commercial; Opera CC-BY) | Excellent within-corpus | download from Zenodo/host; use for method dev + redistributable eval |
| 4 | **Commercial stream-rip by ISRC** (Deezer/Spotify/Tidal) | full-length, exact master | ~96% (Deezer resolves both) | **commercial-stream (rip = ToS/gray)** | Excellent | resolve ISRC → streaming-client rip (not clean legally) |
| 5 | **Shadow: AA Spotify audio** (manifest = `track_files`) | full-length OGG, exact master | potentially most of catalog incl. our pairs | **shadow / non-redistributable** | Excellent | requires a public AA audio payload (unconfirmed) **or** Spotify `file_id` + keys (librespot rip) |

**For OUR eval:** prefer **#2 this week** (free, ~90% coverage, exact master on the aligned overlap) and **#1** for a clean full-length gold set on the marquee names; use **#3** for redistributable method development. #4/#5 are full-length but not legally clean — research-only, never redistribute.

---

## 5. Open-research datasets (re-confirmed / hunted)

Verified via Zenodo this session:
- **Anechoic recordings of Italian opera** — Zenodo **3628247**, **CC-BY-4.0** (redistributable). Multitrack (soprano/baritone/choir/orchestra), 3 opera excerpts (Donizetti/Verdi/Puccini). Best *open + operatic* source; multitrack, not with/without-voice pairs. https://zenodo.org/records/3628247
- **MUSDB18** — Zenodo **1117372**, non-commercial. 150 songs, 5 stems incl. `vocals` + accompaniment (drums/bass/other). The canonical source-separation benchmark; **mix vs accompaniment = a same-master with/without-voice pair.** https://zenodo.org/records/1117372
- **MUSDB18-HQ** — Zenodo **3338373**, non-commercial. Same 150 tracks, uncompressed WAV stems (`vocals.wav`, `other.wav`, …). https://zenodo.org/records/3338373

From prior work / knowledge (web-restricted this session — verify licences before redistribution):
- **MoisesDB** — 240 tracks, fine-grained stems, research licence. github.com/moises-ai/moises-db
- **MedleyDB** — ~122 (+2.0 set) multitracks incl. isolated vocals; research access on request.
- **Aalto anechoic** (Pätynen/Lokki symphonic anechoic) — multitrack orchestral, research use.
- **FreiDi** (Freischütz Digital, AudioLabs Erlangen) — multitrack opera, research use.
- **Isophonics** — annotations (Beatles/Queen/etc.), **not stems** → not an audio source for us.
- ccMixter / MASS / "The Spheres" — small CC BSS-eval test signals.

**Take-away:** open sets are redistributable but **small (dozens–hundreds of tracks)** and **contain none of our specific commercial masters.** They're for method development and a redistributable eval slice; they do not substitute for acquiring our 10,344 mined pairs. The exact "commercial with/without-voice masters at scale" only exists via routes #1/#4/#5.

---

## 6. Get audio for the TOP-50 classical pairs THIS WEEK (legal first)

The marquee top-20 (Brightman, Bocelli ×several, Paul Potts, Pavarotti, Il Volo, Il Divo, Yoshiki) — Spotify dump previews cover only 4/20, so:

**Step 1 — Free 30 s pairs (today, ~zero cost, legal).**
For all 50 pairs, fetch both sides from **two** sources and keep whichever pair is complete:
- Spotify: use `tracks.preview_url` from `spotify_clean.sqlite3` (already local).
- Deezer: `curl https://api.deezer.com/track/isrc:<base_isrc>` and `…:<variant_isrc>` → take the `preview` field.
Then **per-pair cross-correlate** base vs variant (envelope xcorr as in §2), estimate lag, **trim to the overlapping window**, and use that as the 30 s exact-reference excerpt. Expect ~60% of marquee + ~85% of the metal tail to yield a usable both-sides pair; Deezer specifically covers Bocelli/Brightman that Spotify misses.

**Step 2 — Buy the gaps full-length (this week, small budget, legal, best quality).**
For marquee pairs where a preview side is missing/empty or you want full-length gold references, purchase by ISRC (both versions are official releases — labels: Sugar `Ama credi e vai`, Angel/EMI `I Will Be With You`, Nuclear Blast Nightwish instrumentals, Sony/Columbia Il Volo, Plus Media Paul Potts). Buy on **Qobuz/7digital (FLAC)** or **iTunes/Amazon**; ~£0.79–£1.29/track ⇒ the whole top-50 (100 tracks) is ≈ £80–130. These are the exact masters, sample-accurate, fully legal.
- Caveat: audit `Karaoke`-type variants first — several are third-party "in the style of / Karaoke Version" **re-records (different master)**, e.g., Il Divo "Adagio (In the Style of Il Divo)" and Il Volo's "Eurovision … Karaoke Version" (Universal karaoke compilation). Prefer `variant_type=Instrumental` **and** `same_album=true` (3,464 pairs, 33.5%) for true same-master.

**Step 3 — Redistributable slice.** Add a handful of MUSDB18-HQ tracks (mix vs accompaniment) so at least part of the eval set can be shared publicly.

**Do NOT** for the deliverable: rip full tracks off streaming or pull AA/Spotify-CDN audio — those are gray/shadow and non-redistributable.

---

## 7. Same-master quality signals already in our data

- **`same_album=true` (49.5%)** — both versions on one official release → strongest same-master + cleanly purchasable.
- **`variant_type=Instrumental` + `same_album=true` (33.5%)** — highest same-master confidence.
- **`|duration_delta_ms| ≤ 250` (63.7%)** — tight length match, same-master signal.
- **Distinct ISRCs per side (100%)** — expected (vocal vs instrumental register as different recordings); the ISRC is nonetheless the exact key for purchase/stream/Deezer lookup.
- **Watch-outs:** `Karaoke`/`Accompaniment`/`Backing Track` variants (≈2,300 pairs) skew toward third-party soundalike re-records (different master) — filter or manually verify.

---

## 8. Evidence / method appendix

- Torrent metainfo parsed from `…/spotify_2025_07_metadata/spotify_meta.torrent` (NAS + tt-quietbox2).
- Live `spotify_clean.sqlite3` (125 GB, decompressed) queried on tt-quietbox2 for `preview_url`, album labels, and pair coverage.
- `track_files` schema read by fetching only the head piece of file #6 via libtorrent (129 MB real disk; deleted). No bulk/audio download.
- Preview liveness: `curl -I` on a `p.scdn.co` object → 200, 359,795 bytes.
- Alignment: 14 legal 30 s promo clips fetched, ffmpeg→PCM, numpy FFT cross-correlation.
- Deezer: `api.deezer.com/track/isrc:` sampled over 162 pairs (metadata only; no audio saved).
- Verified URLs: `https://api.deezer.com/track/isrc:ITZ040600046` · `…:USNS10700005` · `…:ITZ040600044` · `https://zenodo.org/records/3628247` · `…/1117372` · `…/3338373` · `https://p.scdn.co/mp3-preview/<hash>?cid=…`.
- Not verifiable in this environment (blocked): annas-archive.org/.se/.gl, web.archive.org, musicbrainz.org WS, raw.githubusercontent.com, en.wikipedia.org — claims relying on these (a public AA audio torrent; the Nov-2024 Web-API `preview_url` removal; librespot `AudioFile{file_id,format}` naming) are flagged inline and should be re-checked from an open network.

### Legality summary (be honest)
- **open-research / legal:** Zenodo/open datasets (redistributable per each licence); free 30 s previews from Spotify `p.scdn.co` and Deezer (promotional clips — fine to fetch for internal research; do not re-host as a music service).
- **commercial-purchase:** buying tracks by ISRC — fully legal; not redistributable.
- **commercial-stream:** listening is legal; **ripping full tracks violates platform ToS** (gray) and the audio is not redistributable.
- **shadow:** AA's implied Spotify audio collection and direct Spotify-CDN `file_id` decryption are **copyright-infringing**; usable at most for internal research, **never** redistribution. `track_files` (the manifest) is metadata and safe to read; the audio it points to is not.
