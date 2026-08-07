# Anna's Archive Spotify AUDIO — Reconnaissance

_Recon run 2026-08-06 from vps2 (root@74.208.17.247, public IP) and tt-quietbox2 (ttuser@100.91.242.69), both with clean internet egress. All AA fetches via `curl`/`aria2c` on those boxes (WebFetch is hard-blocked for annas-archive.* in the local agent env). Reconnaissance + torrent/manifest metadata only — no bulk copyrighted audio was downloaded. The one large fetch performed was the 47 GB `track_files` **metadata** manifest (SQLite), explicitly in scope._

> Access note: `annas-archive.org` and `annas-archive.se` now return **NXDOMAIN** (even via Cloudflare DoH) — casualties of the injunction (below). The **`.gl` mirror is live** (185.178.208.181) and is the working front door used throughout.

---

## 0. TL;DR

- **YES — a public AA Spotify AUDIO collection exists.** AA announced "Backing up Spotify" on **2025-12-20**: a planned **~300 TB / 86 million music files** preservation archive (metadata for 256 M tracks / 186 M ISRCs). The **actual audio began releasing on 2026-02-08** (first batch: **47 music torrents + 1 new metadata torrent, ~2.8 M files, ~6 TB**, AA-Containers format), and is being released **in order of popularity**.
- **Quality tiers:** popularity>0 = **original OGG Vorbis 160 kbit/s** (metadata added, not re-encoded); popularity=0 = **re-encoded OGG Opus 75 kbit/s** (only ~half of pop-0 listens archived). Cutoff 2025-07.
- **Access = direct-URL only, hidden from every public listing** (not in the torrents page, not in the `/dyn/generate_torrents` JSON). No membership needed for the `.torrent` files. This is because **Spotify + major labels sued AA (Dec 2025) and won a US preliminary injunction**; AA lost domains and now releases the audio "quietly." **Legally: copyrighted commercial audio, under active litigation — shadow-library, research-use-only, NON-redistributable.**
- **AA has NO other music/audio holdings.** It is a text/document library; this Spotify set is its entire "audio" story. No stems, no classical-specific set, no other music dump.
- **Upgrade for our pairs (resolved against the real 47 GB `track_files` manifest):**
  - **Classical (10,344 pairs):** **4,207 (40.7%)** have BOTH sides `status='success'` (AA holds both `.ogg`), ≈ **26 GB**; rising to **7,498 (72.5%)** if same-ISRC siblings are counted. Only 1,616 pairs get **both** sides at the good Vorbis-160k tier.
  - **All (966,395 pairs):** **354,154 (36.6%)** both-`success`, ≈ **1.98 TB**; **581,939 (60.2%)** both-obtainable via ISRC siblings.
  - Our set skews hard to **popularity=0** (71% of classical sides), i.e. the Opus-75k tier that releases **last** — so despite the manifest coverage, almost none of our classical audio is *downloadable yet* (the Feb-8 batch is the highest-popularity top only).
- **Non-shadow default stays:** Spotify `preview_url` (dump) ∪ Deezer ISRC previews (~90% both-sides 30 s clips) remains the clean, legal route. The AA audio is a full-length upgrade only under a research/shadow label.

---

## 1. Does a public AA Spotify AUDIO collection exist? — DEFINITIVE: YES

### 1.1 AA's own announcement (authoritative)
**Blog: "Backing up Spotify"**, `https://annas-archive.gl/blog/backing-up-spotify.html` (dated **2025-12-20**, by volunteer "ez"). Verbatim key claims:

- _"We backed up Spotify (metadata and music files). It's distributed in bulk torrents (**~300TB**), grouped by popularity."_
- _"**86 million music files**, representing around **99.6% of listens** … a little under 300TB in total size."_ Metadata for _"an estimated 99.9% of [256 M] tracks"_, **186 M unique ISRCs**.
- _"For **popularity>0**, … the original **OGG Vorbis at 160kbit/s**. Metadata was added without reencoding the audio (and an archive of diff files is available to reconstruct the original files from Spotify)."_
- _"For **popularity=0**, … reencoded to **OGG Opus at 75kbit/s**"_ and only _"files representing about half the number of listens."_
- Data is distributed in the **AAC = "Anna's Archive Containers"** multi-torrent format (not the audio codec).
- The Spotify prepended "invalid OGG packet" is stripped and stored in `track_files.prefixed_ogg_packet` (holds ReplayGain). Cutoff **2025-07**.
- **Staged release checklist (still shows this state on the live page today):**
  - `[X] Metadata (Dec 2025)` ✅
  - `[ ] Music files (releasing in order of popularity)`
  - `[ ] Additional file metadata (torrent paths and checksums)`
  - `[ ] Album art`
  - `[ ] .zstdpatch files (to reconstruct original files before we added embedded metadata)`

### 1.2 The audio actually started shipping — 2026-02-08
Independent reporting (TorrentFreak / Ernesto Van der Sar, relayed by DigitalMusicNews **2026-02-17** and RightsTech), verified by curl from vps2:

- _"we count **47 new music torrents, plus a new metadata torrent** … These releases all contain 60,000 files, except for a smaller batch, bringing the total to roughly **2.8 million files. That's roughly 6 terabytes of music**."_ Plus a **29 GB "seekable" metadata** file that indexes the tracks **by Spotify ID** inside the AAC containers.
- Filenames in the audio torrents are **opaque Spotify track IDs** (no artist/title) — "likely match Spotify's internal cache format."
- This is the **leading (highest-popularity) edge** of the eventual 86 M / 300 TB; more batches follow in popularity order.
- Context: _"Despite a Court-Ordered Injunction, Anna's Archive Releases Millions of Spotify Tracks Online."_ Spotify + labels sued in Dec 2025; AA lost several domains under a **US preliminary injunction**; AA made **no official announcement** of the audio — it simply appeared in the backend index.

### 1.3 Where the torrents live / access gating
- **Metadata torrent (open, direct-URL, fetched anonymously — HTTP 200, 62,837 bytes):**
  `https://annas-archive.gl/dyn/small_file/torrents/other_aa/aa_misc_data/annas_archive_spotify_2025_07_metadata.torrent`
  (magnet btih derivable from it; 186 GiB / 9 files / 199,887,366,916 bytes.)
- **Deliberately hidden from all public listings.** Confirmed:
  - `/dyn/generate_torrents` JSON = **16,770 torrents across 18 groups** (duxiu, gbooks, hathi, ia, libgen_*, magzdb, nexusstc, other_metadata, scihub, upload, worldcat, zlib) — **zero Spotify entries, zero Feb-2026 additions.** Even the metadata torrent (which is fetchable) is not in it.
  - `/torrents`, `/torrents/aa_misc_data`, `/datasets` HTML: **0 "spotify" hits.**
  - `/datasets/spotify`, `/torrents/spotify`, etc. → **404.**
- **Gating verdict:** the `.torrent`/magnet files are **open (no login, no membership, no seedbox)** once you know the direct URL; AA membership only accelerates AA's *HTTP* file mirror, which for Spotify is not offered. The collection is *unlisted*, not *paywalled*. But it is under injunction and served quietly.

---

## 2. Survey of ALL Anna's Archive audio / music holdings

**AA has exactly ONE audio/music holding: this Spotify set. There is no other.**

- `/datasets` lists **17 datasets** — every one is books/papers/text metadata: `duxiu, gbooks, hathi, ia, isbn_ranges, lgli, lgrs, magzdb, nexusstc, oclc, ol, other_metadata, scihub, upload, uploads, zlib, zlibzh`. **None music/audio.**
- `/dyn/generate_torrents` groups (18) — none music/audio.
- No classical/opera set, no other music dump, no vocal+instrumental collection anywhere on AA.

**Relevance to a vocal-separation benchmark (honest):**
- The Spotify archive is **single-mix commercial masters** — **one file per track, NOT stems/multitracks.** It cannot provide ground-truth separated vocal/instrumental stems.
- Its only value for us is exactly the pairs we already mined: Spotify carries many **officially-released "Instrumental Version"** tracks as *separate tracks*. A with-vocal + without-vocal pair is two different Spotify tracks (our pairs). The instrumental is a **separately-mastered release**, not the exact instrumental stem of the vocal mix — same approximate-reference caveat as the preview route, but **full-length and exact-master** if both are archived.
- Net: AA adds no new benchmark *material* beyond our pairs; it potentially upgrades our existing pairs from 30 s lossy previews to full-length masters — for the subset AA actually holds.

---

## 3. The UPGRADE the AA audio unlocks — quantified against the manifest

Method: the metadata torrent's **`spotify_clean_track_files.sqlite3`** (47 GB `.zst` → ~101 GB SQLite; selectively downloaded to tt-quietbox2 via `aria2c --select-file=7` and decompressed) is the per-track archival ledger. Column **`status`** is authoritative: **`status='success'` = AA holds the `.ogg`.** Other statuses mean not held: `error_not_available`, `error_has_replacement`, `skipped_isrc_has_download` (low-pop: only one copy per ISRC fetched), `skipped_low_priority` (pop=0 & secondary_priority<0.351 skipped), `error_no_ogg160`, `error_zero_duration`, `todo_unk`. Quality: `reencoded_kbit_vbr` set ⇒ Opus re-encode (pop=0); null ⇒ original **Vorbis 160k**. There is **no `filesize_bytes` column** — size is estimated as `duration_ms × bitrate` (exact bitrate from `reencoded_kbit_vbr` or 160k).

### 3.1 Classical pairs (10,344) — EXACT from `track_files.status`
19,028 / 19,035 ids present in manifest (7 absent entirely).

| Metric | Pairs | % |
|---|---|---|
| **BOTH sides `status='success'`** (AA directly holds both `.ogg`) | **4,207** | **40.7%** |
| **BOTH sides OBTAINABLE** (`success` OR same-ISRC sibling, see below) | **7,498** | **72.5%** |
| exactly ONE side `success` | 2,804 | 27.1% |
| NEITHER side `success` | 3,333 | 32.2% |

- **Size of the 4,207 both-`success` pairs (both files): ≈ 26 GB.** (Exact `filesize_bytes` where populated; but **93% of our success-sides have `filesize_bytes` null** in this 2025-07 snapshot, so their size is a `duration × bitrate` estimate — Vorbis 4008 KB avg / Opus 1768 KB avg empirically. So 26 GB is a solid estimate, ~±20%.)
- **Quality of the 4,207 both-`success` pairs:** both original **Vorbis 160k = 1,616**; both **Opus 75k (re-encoded) = 1,604**; mixed = 987. So only ~1,616 pairs (16% of all classical) give **both** sides at the good 160k tier.
- **Per-track-side status among our 19,035 ids:** `success` 10,035 (53%) · `skipped_isrc_has_download` **5,001 (26%)** · `skipped_low_priority` 2,156 (11%) · `error_has_replacement` 1,018 (5%) · `error_not_available` 818 (4%) · absent 7.
- **The 5,001 `skipped_isrc_has_download` sides are NOT lost:** that status means AA fetched **one copy per ISRC (the highest-popularity one)** — so the *same recording* is archived under a different (higher-pop) `track_id` sharing that ISRC. Counting those as recoverable lifts both-obtainable to **7,498 (72.5%)**. (Resolving them needs an ISRC→archived-sibling `track_id` lookup, then fetch that sibling.)
- **Shard layout confirmed** (`track_files.filename`): `track-popularity-<N>/<L>/<LL>/<artist>/<year album (albumId)>/<NN artist - title>.ogg` — sharded by **popularity bucket → 1st letter → 2-letter prefix → artist → album**. Real samples:
  - `track-popularity-8/K/KA/Kamelot/2018 The Shadow Theory (Deluxe Bonus Version) (…)/16 Kamelot - Amnesiac (Instrumental Version).ogg`
  - `track-popularity-14/W/WI/Within Temptation/2023 Bleed Out (…)/10 Within Temptation - Shed My Skin - Instrumental.ogg`

_Popularity-model cross-check (from `spotify_clean.tracks`): 14,683 of 20,688 track-sides (71%) are popularity=0 — which is exactly why so many land in the skipped/last-to-release buckets._

### 3.2 All general pairs (966,395) — EXACT from `track_files.status`
1,789,855 / 1,790,304 ids present in manifest (449 absent).

| Metric | Pairs | % |
|---|---|---|
| **BOTH sides `status='success'`** (AA directly holds both) | **354,154** | **36.6%** |
| **BOTH sides OBTAINABLE** (`success` OR same-ISRC sibling) | **581,939** | **60.2%** |
| exactly ONE side `success` | 168,987 | 17.5% |
| NEITHER side `success` | 443,254 | 45.9% |

- **Size of the 354,154 both-`success` pairs (both files): ≈ 1.98 TB** (`filesize_bytes` + `duration × bitrate` fallback for the 766,989 null-size success-sides).
- **Quality of both-`success` pairs:** both Vorbis-160k = 101,595; both Opus-75k = 171,918; mixed = 80,641.
- **Per-side status (1.79 M ids):** `success` 807,070 (45%) · `skipped_low_priority` 596,740 (33%) · `skipped_isrc_has_download` 322,908 (18%) · `error_not_available` 33,280 · `error_has_replacement` 29,853 · absent 449.

### 3.3 Format-tier reality for our audio
Because our pairs (especially the obscure instrumental sides) skew to **popularity=0**, most of the audio AA would give us is the **OGG Opus 75 kbit/s** re-encode, not the 160k Vorbis. At 75 kbit/s the fidelity gain over the free 30 s preview MP3s is marginal (you gain *full length*, not *quality*), and it is well below what a high-fidelity leakage/SDR reference wants. The genuinely valuable subset is therefore the **both-sides-`success` AND both-original-Vorbis160** count: **1,616 classical pairs** (§3.1) and **101,595 general pairs** (§3.2).

---

## 4. Feasibility of a TARGETED subset (just our pairs' audio)

**Can we resolve our pair track-ids → their `.ogg` paths/shards?** **Yes — done** (§3.1). `track_files.filename` gives the human path, sharded by **popularity bucket → first-letter → two-letter prefix → artist → album**, e.g. `track-popularity-14/W/WI/Within Temptation/2023 Bleed Out (…)/10 Within Temptation - Shed My Skin - Instrumental.ogg`.

**Can we fetch JUST those shards?** Partially, and mostly **not yet**:
1. The **audio torrents** use **AAC containers** whose member filenames are **opaque Spotify track IDs**, not the human `filename`. Mapping `track_id → which audio torrent + byte offset` needs the **"Additional file metadata (torrent paths and checksums)"** stage, which is **`[ ]` unreleased** on the blog (the Feb-8 drop's 29 GB "seekable" index is the first piece of this).
2. BitTorrent/AAC **does** support partial fetch (`aria2c --select-file`, exactly how the 47 GB manifest was pulled), so **if** a track's container is in a released torrent, only that file's piece-range need be downloaded.
3. **But** releases go **in popularity order**, and our classical set is overwhelmingly **popularity=0** — the **last** buckets. Our 10,035 classical `success` sides sit at track_popularity **0 (4,991), 1–9 (3,305), 10–29 (1,415), 30–49 (301), 50+ (23)** — i.e. **~83% are popularity ≤ 9**, the tail. The Feb-8 first batch (~2.8 M files) is the highest-popularity **top** of 86 M, so **effectively none of our classical instrumental sides are downloadable today**; they arrive in later batches (timeline unknown, and under injunction pressure).

**Rough size for JUST our classical pairs' audio (both sides), if/when fully released:** ≈ **26 GB** for the 4,207 both-`success` pairs (≈ **46 GB** if the 7,498 ISRC-recoverable pairs are included, both sides, at the same avg). Trivial to store — the constraint is *release timing*, not size.

**Exact steps if/when feasible (research-only):** (a) from `track_files`, take rows where our `track_id` has `status='success'`; (b) obtain the audio-file-metadata index (torrent-path + checksum stage) to map each `track_id`→container+offset; (c) `aria2c --select-file` only those containers, or piece-range fetch; (d) verify `sha256_with_embedded_meta`. This remains a **shadow** route regardless.

---

## 5. Direct Spotify-CDN `file_id` route (honest feasibility)

`track_files` carries `file_id_ogg_vorbis_96/160/320`, `file_id_aac_24`, `file_id_mp3_96` — **Spotify internal CDN keys**, not URLs. Resolving them to bytes requires a **Spotify account + the audio decryption handshake (librespot / PlayPlay key exchange)** — i.e. authenticated stream-ripping. It is **undocumented as open URLs, violates Spotify ToS, and is a credential-based shadow route. Not attempted; do not attempt.**

---

## 6. Legality label on every route

| Route | What you get | Coverage | Legality |
|---|---|---|---|
| **Spotify `preview_url` (dump) ∪ Deezer ISRC previews** | 30 s clip per side | ~90% ≥1 both-sides pair | **OPEN / legal** (free promo clips) — **the non-shadow default** |
| **AA Spotify audio (this recon)** | full-length exact master, Vorbis160 or Opus75 | classical 40.7% both-held / 72.5% w/ISRC; all-pairs 36.6% / 60.2% — but our tail releases **last**, ~none fetchable today | **SHADOW / research-only / NON-redistributable / under active injunction** |
| **Spotify-CDN `file_id` via librespot** | full-length exact master | ~all (if it streams) | **SHADOW + Spotify-ToS violation (needs credentials)** — do not attempt |
| **Commercial stream-rip by ISRC (Deezer/Tidal/Spotify)** | full-length exact master | ~96% | **Gray/ToS** — not clean |

**Reaffirmed default:** for the deliverable, use the **clean Spotify + Deezer 30 s previews** (per-pair xcorr-align, trim to overlap). The AA audio is only a full-length upgrade under an explicit research/shadow label, is largely un-fetchable for our popularity-0 tail today, and would mostly arrive as 75 kbit/s Opus anyway.

---

## 7. Citations (all fetched via curl from vps2/tt-quietbox2)

- AA blog "Backing up Spotify": `https://annas-archive.gl/blog/backing-up-spotify.html` (2025-12-20)
- AA metadata torrent (direct, open): `https://annas-archive.gl/dyn/small_file/torrents/other_aa/aa_misc_data/annas_archive_spotify_2025_07_metadata.torrent`
- AA torrents catalog / JSON (no Spotify): `https://annas-archive.gl/torrents`, `https://annas-archive.gl/dyn/generate_torrents`, `https://annas-archive.gl/datasets`
- DigitalMusicNews (2026-02-17): `https://www.digitalmusicnews.com/2026/02/17/annas-archive-releases-spotify-tracks/`
- CyberInsider (2025-12-22): `https://cyberinsider.com/annas-archive-releases-massive-300tb-spotify-music-scrape/`
- RightsTech: `https://rightstech.com/2026/02/annas-archive-quietly-releases-millions-of-spotify-tracks-despite-legal-pushback/` (source: TorrentFreak)
- Also surfaced: cybernews.com, byteiota.com, korben.info, androidauthority.com, yrbmag.com, ubos.tech (all Dec 2025–Feb 2026, same facts)

_Manifest resolve run against `spotify_clean_track_files.sqlite3` (torrent file idx 7) + `spotify_clean.sqlite3` on tt-quietbox2. Track-id lists from `pairs_classical.jsonl` (19,035 unique ids) and `pairs_all.jsonl` (1,790,304 unique ids)._
