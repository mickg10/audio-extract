# Spotify 2025-07 Metadata — With/Without-Voice Master-Pair Mining

Anna's Archive `annas_archive_spotify_2025_07_metadata` torrent — metadata dump (no audio).
Mined on tt-quietbox2 (AMD EPYC, 32 threads) with DuckDB 1.5.5 over the decompressed SQLite dbs.

## Dump structure (downloaded subset)

Torrent total = 199.9 GB across 9 files. Only the music-metadata files needed for track pairing were fetched (selective BitTorrent download):

| file | compressed | note |
|---|---:|---|
| spotify_clean.sqlite3.zst | 36.65 GB | core: tracks / artists / albums / track_artists / artist_genres (used) |
| spotify_clean_audio_features.sqlite3.zst | 17.73 GB | per-track audio features incl. instrumentalness (used) |
| spotify_artist_redirects.json | 0.28 MB | id redirects (fetched, unused) |
| (audiobooks / shows / episodes / playlists / track_files) | ~144 GB | 6 files NOT fetched — irrelevant to music track pairs |

`spotify_clean.sqlite3` schema (key tables):
- **tracks**(rowid, id, name, album_rowid, track_number, external_id_isrc, popularity, duration_ms, disc_number, explicit)
- **artists**(rowid, id, name, followers_total, popularity) · **albums**(rowid, id, name, album_type, release_date, label, popularity, total_tracks)
- **track_artists**(track_rowid, artist_rowid) — many-to-many, unordered → "same artist" = shares an artist_rowid
- **artist_genres**(artist_rowid, genre)
- audio features db: **track_audio_features**(track_id, instrumentalness, energy, acousticness, tempo, ...)

## Method

1. Normalize every track title: casefold + strip accents; drop parentheticals/brackets, trailing ` - qualifier`, and the without-voice marker words; strip punctuation; collapse whitespace.
2. Flag a track as a **without-voice variant** if its title contains any marker: `Instrumental / Karaoke / Backing Track / Orchestra Only / Accompaniment / Minus One / Off Vocal / Sing-Along`.
3. Pair each variant to a **non-variant base** that (a) shares an artist_rowid, (b) has the same normalized title, and (c) has **|duration_ms delta| ≤ 2000 ms** (the same-master signal).
4. Keep the best base per variant (same_album desc, |Δdur| asc, base popularity desc).
5. Rank: focus artists (+100) and classical/opera/crossover genres (+40) first, then same_album (+25), tighter duration, popularity, and instrumentalness confirmation.

## Counts

- tracks scanned: **256,039,007**
- without-voice variant tracks (non-empty normalized title): **2,807,404**
- candidate (variant,base) pairs after shared-artist + |Δdur|≤2s join: **2,954,590** (distinct variants paired: 966,395)
- **final pairs (best base per variant): 966,395**  → `pairs_all.jsonl`
  - of which same-album: **549,935**
- **classical/opera/crossover + focus-artist subset: 10,344** → `pairs_classical.jsonl`
- mining wall time: 190.1s

### Variant-type breakdown (all pairs)

| variant type | pairs |
|---|---:|
| Instrumental | 744,504 |
| Karaoke | 187,162 |
| Backing Track | 12,520 |
| Off Vocal | 6,756 |
| Sing-Along | 6,744 |
| Accompaniment | 5,559 |
| Minus One | 2,915 |
| Orchestra Only | 121 |
| Orchestral Version | 114 |

## Duration-delta histogram (final pairs, 250 ms buckets)

| |Δduration| bucket | pairs | bar |
|---|---:|---|
| 0–250 ms | 668,967 | ████████████████████████████████████████ |
| 250–500 ms | 79,433 | █████ |
| 500–750 ms | 53,259 | ███ |
| 750–1000 ms | 39,727 | ██ |
| 1000–1250 ms | 36,486 | ██ |
| 1250–1500 ms | 25,158 | ██ |
| 1500–1750 ms | 23,561 | █ |
| 1750–2000 ms | 20,380 | █ |
| 2000–2250 ms | 19,424 | █ |

## Top 50 classical / focus pairs (ranked)

| # | artist(s) | base title | without-voice variant | type | Δs | same-alb | base→var instr. | score |
|--:|---|---|---|---|--:|:--:|---|--:|
| 1 | Sarah Brightman, Chris Thompson | I Will Be With You (Where The Lost One | I Will Be With You (Where The Lost One | Instrumental | 0.07 | ✓ | 0.00→0.89 | 198.51 |
| 2 | Andrea Bocelli | Ama Credi E Vai (Because We Believe) | Ama credi e vai - Instrumental Version | Instrumental | 0.00 | ✓ | 0.00→0.94 | 196.0 |
| 3 | Andrea Bocelli | Ama credi e vai (because we believe) - | Ama credi e vai (because we believe )  | Instrumental | 0.00 | ✓ | 0.00→0.94 | 195.0 |
| 4 | Paul Potts | I Believe | I Believe - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.91 | 195.0 |
| 5 | Andrea Bocelli | Con te partirò (Orchestra and choir 20 | Con te partirò (Orchestra instrumental | Instrumental | 0.60 | ✓ | 0.13→0.81 | 189.0 |
| 6 | Yoshiki, Sarah Brightman | Miracle - Sarah's Version | Miracle - Sarah's Version / Instrument | Instrumental | 1.12 | ✓ | 0.13→0.94 | 185.63 |
| 7 | Luciano Pavarotti | Ti adoro - DJ Francesco Mix | Ti adoro - Karaoke Version | Karaoke | 0.25 | ✓ | 0.00→0.00 | 182.47 |
| 8 | Luciano Pavarotti | Ti adoro - DJ Francesco Mix | Ti adoro - Instrumental | Instrumental | 0.27 | ✓ | 0.00→0.00 | 182.34 |
| 9 | Sarah Brightman | Running - Single Version | Running - Instrumental | Instrumental | 1.80 | ✓ | 0.01→0.94 | 179.5 |
| 10 | Andrea Bocelli | Dell' amore non si sa | Dell'Amore Non Si Sa - Instrumental | Instrumental | 1.95 | ✓ | 0.00→0.87 | 176.03 |
| 11 | Il Volo | Grande amore - Eurovision Version | Grande Amore - Eurovision 2015 - Italy | Karaoke | 0.34 |  | 0.00→0.75 | 171.93 |
| 12 | Il Divo | Adagio | Adagio (In the Style of Il Divo) [Kara | Karaoke | 0.04 |  | 0.05→0.91 | 170.3 |
| 13 | Il Divo | All By Myself | All By Myself (In the Style of Il Divo | Karaoke | 0.11 |  | 0.00→0.81 | 169.19 |
| 14 | Andrea Bocelli | Con Te Partirò - Orchestra & Choir / 2 | Con te partirò - Instrumental / 2016 V | Instrumental | 0.33 |  | 0.00→0.78 | 167.85 |
| 15 | Andrea Bocelli | Con Te Partirò - Orchestra & Choir / 2 | Con te partirò - Instrumental / 2016 V | Instrumental | 0.33 |  | 0.00→0.77 | 167.84 |
| 16 | Andrea Bocelli | Con Te Partirò - Orchestra & Choir / 2 | Con Te Partirò - Instrumental / 2016 V | Instrumental | 0.33 |  | 0.00→0.77 | 167.84 |
| 17 | Andrea Bocelli | Con Te Partirò - Orchestra & Choir / 2 | Con te partirò - Instrumental / 2016 V | Instrumental | 0.33 |  | 0.00→0.78 | 167.84 |
| 18 | Il Divo, Toni Braxton | The Time of Our Lives (The Official So | The Time of Our Lives (The Official So | Instrumental | 0.25 |  | 0.00→0.66 | 167.46 |
| 19 | Sarah Brightman | Serenade | Serenade ( Instrumental ) | Instrumental | 0.29 |  | 0.99→0.99 | 164.57 |
| 20 | Paul Potts | I Believe | I Believe - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.91 | 155.0 |
| 21 | Jerry Heil, Within Temptation | Sing Like A Siren | Sing Like A Siren - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.70 | 104.83 |
| 22 | Nightwish | Élan | Élan - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.71 | 104.81 |
| 23 | Nightwish | Alpenglow | Alpenglow - Instrumental | Instrumental | 0.01 | ✓ | 0.00→0.54 | 103.92 |
| 24 | Within Temptation | The Reckoning | The Reckoning - Instrumental | Instrumental | 0.09 | ✓ | 0.00→0.84 | 103.9 |
| 25 | Avantasia | The Witch | The Witch - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.78 | 103.67 |
| 26 | Avantasia | Here Be Dragons | Here Be Dragons - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.81 | 103.67 |
| 27 | Avantasia | Avalon | Avalon - Instrumental | Instrumental | 0.00 | ✓ | 0.01→0.95 | 103.5 |
| 28 | Avantasia | Creepshow | Creepshow - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.90 | 103.5 |
| 29 | Within Temptation | Entertain You | Entertain You - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.63 | 103.5 |
| 30 | Avantasia | Bring On The Night | Bring On The Night - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.93 | 103.33 |
| 31 | Avantasia | Phantasmagoria | Phantasmagoria - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.88 | 103.33 |
| 32 | Within Temptation | Wireless | Wireless - Instrumental | Instrumental | 0.00 | ✓ | 0.01→0.90 | 103.17 |
| 33 | Avantasia | Against The Wind | Against The Wind - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.93 | 103.0 |
| 34 | Yes | Long Distance Runaround - 2024 Steven  | Long Distance Runaround (Instrumental) | Instrumental | 0.01 | ✓ | 0.01→0.82 | 102.89 |
| 35 | Avantasia | Everybody's Here Until The End | Everybody's Here Until The End - Instr | Instrumental | 0.00 | ✓ | 0.00→0.84 | 102.83 |
| 36 | Within Temptation | Shed My Skin | Shed My Skin - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.91 | 102.83 |
| 37 | Mono Inc. | Lieb Mich | Lieb Mich - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.88 | 102.83 |
| 38 | Nightwish | Endless Forms Most Beautiful | Endless Forms Most Beautiful - Instrum | Instrumental | 0.00 | ✓ | 0.00→0.59 | 102.79 |
| 39 | Within Temptation | We Go To War | We Go To War - Instrumental | Instrumental | 0.00 | ✓ | 0.25→0.89 | 102.67 |
| 40 | Within Temptation | Ritual | Ritual - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.85 | 102.33 |
| 41 | Sabaton | White Death | White Death - Instrumental | Instrumental | 0.12 | ✓ | 0.00→0.68 | 102.32 |
| 42 | Epica | Crimson Bow and Arrow | Crimson Bow and Arrow - Instrumental | Instrumental | 0.05 | ✓ | 0.00→0.78 | 102.3 |
| 43 | Within Temptation | Mad World | Mad World - Instrumental | Instrumental | 0.03 | ✓ | 0.00→0.90 | 102.23 |
| 44 | Within Temptation | The Fire Within | The Fire Within - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.81 | 102.17 |
| 45 | Equilibrium | Born to Be Epic | Born to Be Epic - Instrumental | Instrumental | 0.10 | ✓ | 0.21→0.77 | 102.12 |
| 46 | Nightwish | Weak Fantasy | Weak Fantasy - Instrumental | Instrumental | 0.01 | ✓ | 0.01→0.89 | 102.04 |
| 47 | Within Temptation | Bleed Out | Bleed Out - Instrumental | Instrumental | 0.00 | ✓ | 0.00→0.86 | 102.0 |
| 48 | Mono Inc. | Louder Than Hell | Louder Than Hell - Instrumental | Instrumental | 0.00 | ✓ | 0.01→0.64 | 102.0 |
| 49 | Equilibrium | Wirtshaus Gaudi | Wirtshaus Gaudi - Instrumental | Instrumental | 0.00 | ✓ | 0.37→0.97 | 101.98 |
| 50 | Nightwish | Edema Ruh | Edema Ruh - Instrumental | Instrumental | 0.01 | ✓ | 0.00→0.81 | 101.88 |

### Full records
Every field (Spotify track ids, album ids, ISRCs, durations, instrumentalness, genres, focus_score) is in the JSONL files. `pairs_classical.jsonl` is the ranked classical/crossover set; `pairs_all.jsonl` is the full general list.
