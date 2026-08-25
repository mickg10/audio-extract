# Reproducing the Listening Lab (upload → render → discuss loop)

Complete recreate instructions for the Tobacco voice-removal setup as of 2026-08-25.
Three machines: **Mac** (web server + orchestrator), **vps2** (public nginx edge),
**research6** (GPU render + models). One human-facing app; one autonomous pipeline.

## Topology

```
phone/laptop ──https──> vps2 nginx (70b.zavulon.com, 443)
                          └─ proxy → Mac tailscale 100.80.193.52:8766  (web/ab.py)
Mac web/ab.py
  ├─ serves pages: /v (variant cards+upload), /w?t=<track> (workspace chat),
  │                /t (picks), /m (mobile ladder), / (desktop)
  ├─ data: ab_pairs/ab_verdicts.json  (ALL votes/comments/threads, append-only)
  │        variants/*.m4a (cards) + variants/.vad_timelines.json (detector bars)
  │        attachments/ (thread files, persistent)  raw_input/ (masters)
  └─ POST /api/upload → saves master → spawns tools/upload_pipeline.py (detached)
tools/upload_pipeline.py (Mac, stdlib)
  ├─ posts step-by-step logs into the track thread via POST /api/vote
  ├─ scp master → research6:/home/mickg/lib_in/
  ├─ ssh research6 auto_track.py <src> <track>   (streams STEP: lines back)
  ├─ scp rendered m4as → variants/ ; scp vad_timelines.json → variants/
  └─ posts final comparator + recommendation to the thread
research6 (mickg@100.73.131.92)
  ├─ /home/mickg/audio-extract        clone of THIS repo (venv: .venv, torch+audio-separator)
  ├─ /home/mickg/models               separator checkpoints (sha256 below)
  ├─ /home/mickg/tobacco_variants/    render_tobacco.py (engine) + auto_track.py (driver)
  │                                   + per-track workdirs with float32 nodes + manifests
  ├─ /mnt/bigdisk/mickg/vad_train/    voice detector: code/ + ckpt/best.pt
  ├─ /mnt/bigdisk/mickg/holefill/     restoration net: code/ + ckpt/best_ema.pt
  └─ /mnt/bigdisk/mickg/vad_scan/     detector scans (wav/, out/, vad_timelines.json)
```

## Pinned artifacts (verify with sha256sum)

| artifact | path (research6) | sha256 |
|---|---|---|
| MDX23C ckpt | models/MDX23C-8KFFT-InstVoc_HQ.ckpt | 49d51472769e34a2501cd1da782346a3212555c3a5619fc2c53507445528d816 |
| BS-Roformer ep317 | models/model_bs_roformer_ep_317_sdr_12.9755.ckpt | 5b84f37eb1b976fbbca65e2442e7e2adfeba57c6e0b998f9f2451684b0a48009 (verify against manifest) |
| MelBand vocals | models/vocals_mel_band_roformer.ckpt | 87201f4d966df484c53c2b486bcd94271906b7a4c8a1c05fcf2521bc4557559e (verify against manifest) |
| float-gate adapter | audio_extract/separate.py @ commit 2c859f6 | 78885d562938521e36d6c198143190d48041cc0a9e5aa8007763f1ac2d6948aa |
| VAD detector ckpt | /mnt/bigdisk/mickg/vad_train/ckpt/best.pt | f26f36cac73f79daf56e44b39c14b0564a232f3eb6fb30ab1b46102ec7fa6d54 |
| hole-filler ckpt | /mnt/bigdisk/mickg/holefill/ckpt/best_ema.pt | 1ac467a859eb3493399761140b9697447e1ba712e5bfbc32193dbd300d06725d |

Exact per-track render provenance (node sha256s, model shas, adapter sha, env pins)
lives in each `tobacco_variants/<track>/manifest.json` — treat those as ground truth;
the BSR/Mel hashes above should match them.

Vendored copies of the research6-side code are in this repo under `tools/research6/`
(render_tobacco.py, auto_track.py, vad/{infer,model,common}.py,
holefill/{infer,model,hf_common}.py) — deploy them to the paths above.
Checkpoints are NOT in git (large); back them up separately (bigdisk + one offsite copy).

## Recreate: Mac

```bash
git clone git@github.com:mickg10/audio-extract.git && cd audio-extract  # branch v2-design
uv sync                              # or: uv venv && uv pip install -e .
brew install ffmpeg                  # ffmpeg/ffprobe on PATH
mkdir -p raw_input input variants attachments ab_pairs
nohup uv run python web/ab.py >/tmp/ab_server.log 2>&1 &   # port 8766, 0.0.0.0
```
Server facts: stdlib-only HTTP/1.1; votes append to `ab_pairs/ab_verdicts.json`
(clear-semantics per user; users self-identify, localStorage `rater`);
`/wav/<name>.m4a` transcodes on demand into `variants/.wav_cache/`.
Restart ONLY on code change — audio/votes/timelines are read per-request,
and restarts kill in-flight listening streams.

## Recreate: vps2 edge (nginx)

`/etc/nginx/sites-enabled/70b` server block (443, certbot-managed cert
70b.zavulon.com):

```nginx
server_name 70b.zavulon.com;
client_max_body_size 512M;              # uploads; default 1M breaks them with HTML 413
location = / { return 302 /m; }
location / {
    proxy_pass http://100.80.193.52:8766;   # Mac tailscale IP
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_read_timeout 120s;
}
```
Alternate mount: same proxy under `mickgnas.zavulon.com/audio_vote/`
(strip prefix with a trailing-slash proxy_pass).

## Recreate: research6

```bash
# repo + venv (torch 2.13 cu130, audio-separator==0.44.5, librosa, soundfile, soxr, scipy, matplotlib)
git clone <repo> /home/mickg/audio-extract && cd /home/mickg/audio-extract && uv sync
# models: place the three checkpoints in /home/mickg/models (verify sha256!)
# deploy drivers from the repo
mkdir -p /home/mickg/tobacco_variants /home/mickg/lib_in
cp tools/research6/{render_tobacco.py,auto_track.py} /home/mickg/tobacco_variants/
mkdir -p /mnt/bigdisk/mickg/vad_train/{code,ckpt} /mnt/bigdisk/mickg/holefill/{code,ckpt} \
         /mnt/bigdisk/mickg/vad_scan/{wav,out}
cp tools/research6/vad/*      /mnt/bigdisk/mickg/vad_train/code/
cp tools/research6/holefill/* /mnt/bigdisk/mickg/holefill/code/
# restore best.pt / best_ema.pt from backup into the ckpt dirs (sha256 above)
```
SSH: Mac must reach `mickg@100.73.131.92` non-interactively (key auth) — the
pipeline shells out to ssh/scp. Docker data-root note: bind-mounted to
/mnt/bigdisk/mickg/docker-root, persisted in /etc/fstab (root disk is 155G).

## The autonomous pipeline contract

`upload_pipeline.py <local_src> <track>`: probe → thread log → scp → run
`auto_track.py` (VAD spans → full-vs-splice at 60% coverage → build_track A–E
via the pinned float-gate adapter → detector timelines + comparator) → pull
m4as + timelines → final thread post. All progress lands as `action_reply`
records (user `pipeline`) in the track's thread; failures post ❌ lines.
Logs: `/tmp/pipeline_<track>.log` (Mac).

## Recipes that decisions rest on (locked knowledge)

- A=mdx23c native instrumental; B=ensemble-max{mdx,bsr} (argmax-magnitude,
  winner-takes-phase, n_fft 4096 hop 1024); C=bs-roformer native; D=N0−vocals
  subtract (consistently worst — keep only as diagnostic); E=melband "(Other)".
- Loud solo soprano ⇒ C (mdx23c leaks ~18 dB more); easy/direct voices ⇒ A
  (ties C without its dullness). Measure per track; never assume.
- Hybrids: time-domain only (frequency blends re-import voice formants).
  Router must be the independent VAD on the CANDIDATE output — never a
  separator's own vocal stem (circularity).
- Splices: original bit-exact outside spans; 80–150 ms amplitude-complementary
  ramps (coherent sources).
```
