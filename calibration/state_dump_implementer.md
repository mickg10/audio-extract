# Implementer STATE DUMP — classical/operatic separator readiness

Host: research6 (RTX PRO 4500, CUDA13/torch 2.13) + tt-quietbox2 (CPU corpus) + NAS (Cantolopera). Repo `audio-extract` @ branch `v2-design`. Mac deliberately NOT used (disk critical, ~1.1 GiB free). Session commits: **6e0425a** (Job 1), **ec99187** (Job 2).

## A. Repo state
| item | value |
|---|---|
| branch / HEAD | v2-design / ec99187 |
| pulled | origin/v2-design 2a97829 (ROLES.md + FROM/TO protocol) then +2 session commits |
| test suite | **222 passed / 0 failed** in 260 s (`./run.sh pytest -q`) — was 219+3F |
| GPU | onnxruntime CUDAExecutionProvider present (restored after uv sync reverted it to CPU) |

## B. Job 1 — repro gate: DONE
Root cause: `delivery.encode_aac` shelled out to a literal `ffmpeg` on PATH → `FileNotFoundError` in envs without a system ffmpeg (gpt56's 3 delivery-test failures). Fix (commit 6e0425a): added `delivery._ffmpeg_exe()` = `shutil.which("ffmpeg")` → `imageio_ffmpeg.get_ffmpeg_exe()` → clear `RuntimeError`; pinned **imageio-ffmpeg** (bundled binary `ffmpeg-linux-x86_64-v7.0.2`). resample.py uses the `soxr` lib (not ffmpeg) — unaffected. → **222/222**.

## C. Job 2 — immutable corpus `datasets/classical-v1.jsonl` (33 works / 18 groups)
Per-row fields: corpus_id, work_id, composer, session_id, singer_id, ensemble_id, room_id, rir_id, mastering_id, integrity_class, lossless, license, sr_hz, channels, duration_s, source_roles, per-role {path, container_sha256, pcm_sha256, sr, channels, frames}, eligible_training_targets, threshold_eligible, group_id, split.

| corpus | works | integrity | lossless | split | threshold-elig |
|---|--:|---|:--:|---|:--:|
| bologna (V/P/D) | 3 | linear_exact | yes | **test-v1** | yes |
| aalto Mozart (dry+hall) | 2→1 grp | linear_exact | no (mp3 src) | **test-v1** | yes |
| freidi (Freischütz 06/08/09) | 3 | same_performance_bleed | yes | train | no |
| cantoria (14 pieces) | 14 | linear_exact (organ by diff) | yes | train | yes |
| choral_singing | 1 pool | same_take (a-cappella) | yes | val | yes |
| vocalset / phenicx / spheres / rir_pori | 4 donor pools | *_donor | yes | train-donor | no |
| **cantolopera (full)** | 6 | **same_take_paired_target** | **yes (LOSSLESS)** | train-weak | **NO** |

- **Hashes computed** (container SHA-256 + decoded PCM SHA-256) for every exact reference file of bologna/aalto (research6) and freidi/cantoria (tt-quietbox2). Sample: `bologna_verdi` M pcm `sha256:3834abf5d5c41c2…`, `aalto_mozart_dry` M `sha256:5d0565e9af3fae8…`. Donor pools + Cantolopera (staging) are cataloged (PCM hashes on materialize).
- 19 threshold-eligible works (all linear_exact); Cantolopera explicitly excluded.

## D. Splits + overlap audit — `datasets/classical-v1-splits.json`, `…-overlap-audit.json`
- **Group = work; ALL derivatives collapse into one group** (gpt56 H2). Verified: `aalto_mozart_dry` + `aalto_mozart_hall` → single group `aalto:aalto_mozart:aalto_mozart_soprano:aalto_orchestra`.
- **Overlap audit: ZERO source-family leakage** (no session/singer/ensemble/room/mastering/rir key appears in two splits). 0 leaks.
- **test-v1 is tuning-contaminated** (gpt56 H3): Bologna V/P/D + Aalto were used across bake-offs/eval — a *legacy* holdout, NOT a clean untouched test. **A genuinely-new exact work is still required for final acceptance.**

## E. Cantolopera full — same_take_paired_target (NEVER linear_exact)
6 valid LOSSLESS works (Aida ×4 mezzo/tenor/soprano/soprano, Andrea Chénier tenor, La Bohème bass). `_voice`=full M, `_orchestra`=strumentale A, same take but **independently mastered → null −20…−29 dB → same_take_paired_target, threshold_eligible=false**. ABORT-guard in the assembler asserts none is linear_exact. Use = **separation fine-tuning + weak refs only**. (User runs the scraper; not built here.)

## F. Job 3 — exact renders + candidate baselines: PENDING
Exact M/A/V for Bologna V/P/D + Aalto already exist and are hashed (research6 truth_pairs; Bologna 92/110/106 s full arias, Aalto 90 s). NOT yet done: one-production-code-path full-length re-render from tt-quietbox2 source multitracks (dry mic first) with sample-grid verification, and the 4 candidate baselines per work (residual:MDX23C, median(MDX23C,MelBand,BS), geometric-median, convex-fusion) with PCM hashes + recipes.

## G. Job 4 — training adapter + smoke test: BLOCKED (fact-based)
**FACT — nothing in the current stack is trainable.** MDX23C / MelBand / BS-Roformer run only through `audio-separator`'s **inference-only** `mdxc_separator` (load checkpoint → demix; no optimizer/loss/backward/export). demucs, torchaudio, asteroid, openunmix, nussl are **NOT installed**. No `configs/train/`, no training loop, no checkpoint-export path in the repo. torch 2.13 + CUDA are available (can train once an arch is wired).
- **The fact-based trainable base is HTDemucs (the `demucs` repo)** — it has a real training implementation AND `audio-separator` ships a `demucs_separator` inference adapter, so a trained HTDemucs checkpoint round-trips back into the production inference path. But `demucs` is not installed and no train/export adapter exists yet.
- **Smoke test (100 steps / 2 works / eval 3rd) is not runnable today** — it requires first installing demucs + implementing the `audio-extract train classical` adapter, the loss family, and the export bridge.

## H. Loss family (per bigoracle) — to implement (NOT one SDR objective)
multi-res complex STFT + waveform L1 + mixture-consistency + no-vocal-false-positive + vocal-false-negative + source-coordinate α/β/R penalties + stereo coherence + event-tail weighting. Do NOT optimize the SHADOW judge (gpt56 H4).

## I. TRAINING-READINESS VERDICT: **NOT training-ready today.**
Jobs 1–2 are green (repro gate 222/222; frozen leak-free corpus + splits), but four hard blockers remain:
1. **No training implementation/adapter/export in-stack** (separators are inference-only; demucs absent). ← the fundamental blocker.
2. **No clean untouched holdout** — test-v1 (Bologna/Aalto) is tuning-contaminated; a fresh exact work is needed for final acceptance.
3. **Job-3 full-length renders + candidate baselines not materialized** (exact M/A/V exist + hashed; candidates + one-path re-render pending).
4. **Loss family not implemented.**

**Concrete path to training-ready (ordered):** (a) `uv add demucs` + implement `audio-extract train classical` (dataset/loss/checkpoint/export adapters) with export into `audio-separator`'s `demucs_separator` for production inference; (b) implement the bigoracle loss family; (c) materialize Job-3 renders + 4 candidate baselines with hashes; (d) acquire/build ONE genuinely-new exact holdout work; (e) run the 100-step/2-work smoke, evaluate the 3rd through the production inference adapter, gate on finite losses + exact frame/channel/rate parity + resumable checkpoint + reproducible hashes.

Artifacts (research6): `datasets/classical-v1.jsonl`, `datasets/classical-v1-splits.json`, `datasets/classical-v1-overlap-audit.json`; commits 6e0425a, ec99187.
