# STATUS — Anna's Archive Spotify 2025-07 METADATA (download + pair mining)

_Last updated: 2026-08-06_

## What / where

Torrent: `annas_archive_spotify_2025_07_metadata` (Anna's Archive, metadata dump — NO audio).
Total torrent = **199.9 GB across 9 files, NO webseeds** (peers/DHT only).

**Primary host = tt-quietbox2** (100.91.242.69 / LAN 10.0.27.212; AMD EPYC, 32c, 503 GB RAM, 3.2 TB free).
Downloaded, decompressed and mined here. (A separate Wormhole/tt-metal compute job was left untouched — this used CPU/disk only.)

### Download (aria2c, selective)
Only the music-metadata files needed for track pairing were fetched via `--select-file=1,4,5`:

| idx | file | compressed bytes | status |
|--:|---|---:|---|
| 1 | spotify_artist_redirects.json | 282,155 | fetched |
| 4 | spotify_clean.sqlite3.zst | 36,653,110,502 | fetched, hash-OK |
| 5 | spotify_clean_audio_features.sqlite3.zst | 17,730,266,497 | fetched, hash-OK |
| 2,3,6,7,8,9 | audiobooks / shows / episodes / playlists / track_files | ~144 GB | NOT fetched (irrelevant to track pairs) |

Selected total ≈ **50 GiB / 55.37 GB**. Observed: 85–98 MiB/s, up to ~140 peer connections, ~80 seeders; completed in ~10 min. All BitTorrent pieces verified `(OK)` by aria2.

### Locations
- Compressed (.zst) on tt-quietbox2: `/home/ttuser/datasets/annas-archive/spotify_2025_07_metadata/annas_archive_spotify_2025_07_metadata/`
- Decompressed SQLite (working copies) same dir: `spotify_clean.sqlite3` (125.4 GB), `spotify_clean_audio_features.sqlite3` (41.5 GB)
- **NAS long-term storage (LANDED):** `10.0.27.98:/share/ZFS19_DATA/datasets/annas-archive/spotify_2025_07_metadata/` — the 3 fetched files (.zst + json) + `spotify_meta.torrent` + `aria2.log`. Byte-exact to metainfo.
- Mining outputs: `…/spotify_2025_07_metadata/out/` → `pairs_all.jsonl`, `pairs_classical.jsonl`, `REPORT.md`, `stats.json`
- Also copied to Mac: `/Users/mickg10/audio-extract/calibration/spotify_pairs/`

### Checksum policy
- Integrity = the torrent's own SHA1 **piece hashing**, verified by aria2 (all pieces `(OK)`); fetched file sizes are byte-exact to the metainfo lengths.
- `zstd -d` verifies each frame's checksum on decompression.
- No separate MD5/SHA is published by AA in this metainfo.

## Root cause of the earlier ZERO PEERS
Prior attempt ran on the **NAS via a VPN-routed qBittorrent container** (`qbittorrentvpnqnap`, `/share/docker/torrent/data/incomplete/` — left **empty**, 0 bytes). A VPN tunnel with no forwarded port / P2P path yields zero peers regardless of the swarm; the NAS is also NAT'd. Moving to **native aria2 on tt-quietbox2** (clean UDP egress; DHT + PEX + the 60 baked-in public trackers) connected to peers within ~10 s. No trackers were added (AA torrents are open; the baked-in set suffices).

## Pair mining (Phase 2)
DuckDB 1.5.5 over the decompressed SQLite. Definition: same artist (shared `artist_rowid`), non-variant base title vs a without-voice variant title (Instrumental/Karaoke/Backing Track/Orchestra Only/Accompaniment/Minus One/Off Vocal/Sing-Along), matching normalized titles, **|duration_ms delta| ≤ 2000 ms** (same-master signal); same_album ranked highest; classical/opera/crossover + focus artists ranked first; instrumentalness used as confirmation. See `REPORT.md`.

## Monitor / stop / re-run

```bash
# --- from this Mac ---
SSH="ssh ttuser@100.91.242.69"       # tt-quietbox2
D=/home/ttuser/datasets/annas-archive/spotify_2025_07_metadata

# mining progress / tmux sessions
$SSH "tail -n 40 $D/mine.log; tmux ls"
# stop mining
$SSH "tmux kill-session -t spotmine"

# re-run mining (idempotent; overwrites out/)
$SSH "cd $D && tmux new-session -d -s spotmine 'python3 mine.py > mine.log 2>&1; echo MINE_RC_\$? >> mine.log'"

# fetch the other 6 torrent files later (if ever needed):
#   aria2c --select-file=2,3,6,7,8,9 --enable-dht=true --seed-time=0 -d $D $D/spotify_meta.torrent
# stop a download:  $SSH "tmux kill-session -t spotdl"   (or: pkill -x aria2c)

# reclaim space: the decompressed .sqlite3 working copies (167 GB) can be deleted; .zst are kept on NAS+box
$SSH "rm -f $D/annas_archive_spotify_2025_07_metadata/*.sqlite3 $D/mine.duckdb; rm -rf $D/duck_tmp"
```
