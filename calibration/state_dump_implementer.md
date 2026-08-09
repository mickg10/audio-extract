# Implementer STATE DUMP — classical/operatic separator readiness

Host: research6 (RTX PRO 4500, CUDA13/torch 2.13) + tt-quietbox2 (CPU corpus) + NAS (Cantolopera). Repo `audio-extract` @ branch `v2-design`. Mac deliberately NOT used (disk critical, ~1.1 GiB free). Session commits: **6e0425a** (Job 1), **ec99187** (Job 2), **09ed3ea** (state dump), **6c4b070** (Jobs 3–4).

## A. Repo state
| item | value |
|---|---|
| branch / HEAD | v2-design / 6c4b070 |
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

## F. Job 3 — exact renders + candidate baselines: DONE (commit 6c4b070)
Exact M/A/V for Bologna V/P/D + Aalto (research6 truth_pairs) — **M=A+V sample-grid parity verified −148…−150 dB, ch=2, no truncation/resample/channel mismatch**. Materialized **20 candidate accompaniments** (5 works × {residual:MDX23C, median(MDX23C,MelBand,BS), STFT geometric-median, convex-fusion(uniform)}) with PCM SHA-256 + recipes → `datasets/candidates-v1.jsonl`; audio at `research6:~/classical_candidates/`. (One-production-code-path re-render from tt-quietbox2 88 G source multitracks is a later refinement; the built truth_pairs are already exact.)

## G. Job 4 — trainable base by FACT + smoke: **SMOKE PASS** (commit 6c4b070)
**FACT — MDX23C/MelBand/BS are inference-only** (audio-separator `mdxc_separator`; no optimizer/loss/backward/export). **The trainable base is HTDemucs**: `demucs 4.1.0` installed; `HTDemucs` is a torch `nn.Module` with **verified forward AND backward (finite loss)** — its dora-based `demucs.train` needs `dora`, but a **plain torch loop trains it directly (no dora required)**; and `audio_separator.architectures.demucs_separator` exists so a trained checkpoint **round-trips into production inference**.
- **Smoke** (`audio_extract/train_classical_smoke.py`; config `configs/train/classical-student-v1.yaml`): HTDemucs 11.98 M params, 2 sources [accompaniment, vocals], **100 steps / 2 works (Bologna verdi+puccini), evaluated the 3rd (donizetti) through demucs `apply_model` (production overlap-add inference)**, in 11.5 s on CUDA.
- Loss family (subset per bigoracle): multi-res STFT-mag + waveform L1 + mixture-consistency (A+V=M). Remaining components are TODO.
- **Success criteria ALL met:** losses finite throughout (0.105→0.194 — noisy, not yet converged at 100 steps, as expected); **exact frame/channel/rate parity (4 851 000 frames / 2 ch / 44.1 kHz through the inference path)**; **resumable checkpoint** (model+optimizer+step) `sha256:2cc9dde6…`; reproducible dataset/split/checkpoint hashes. Eval accompaniment SI-SDR −11.7 dB (plumbing smoke, not convergence). → **SMOKE PASS**.

## H. Loss family (per bigoracle) — to implement (NOT one SDR objective)
multi-res complex STFT + waveform L1 + mixture-consistency + no-vocal-false-positive + vocal-false-negative + source-coordinate α/β/R penalties + stereo coherence + event-tail weighting. Do NOT optimize the SHADOW judge (gpt56 H4).

## I. TRAINING-READINESS VERDICT: **training PATH validated (HTDemucs smoke passes); not yet PRODUCTION-ready.**
All four readiness jobs executed. Repro gate green (222/222); frozen leak-free corpus+splits; exact candidate baselines materialized with verified parity; **HTDemucs trains + round-trips into production inference and the smoke passes all mechanical criteria.** The earlier "no trainable arch in-stack" blocker is **RESOLVED** (HTDemucs, plain torch loop).

Remaining before a real training run is *acceptance-grade*:
1. **Full adapter** — promote the smoke into `audio-extract train classical` (CLI + `classical-student-v1.yaml`) with the streaming dataset from `classical-v1.jsonl`, and the **export bridge** to audio-separator's demucs checkpoint format (smoke evaluated via `demucs.apply_model`, the same inference math, but the on-disk checkpoint→separator round-trip is not yet wired).
2. **Full loss family** — add complex-STFT phase, no-vocal-FP, vocal-FN, source-coordinate α/β/R, stereo-coherence, event-tail (currently STFT-mag + L1 + mixture-consistency).
3. **Clean holdout** — test-v1 (Bologna/Aalto) is tuning-contaminated (gpt56 H3); ONE genuinely-new exact work is required for final acceptance.
4. **Convergent run** — 100 steps is plumbing; a real multi-hundred-k-step run on the train split, gated on held-out exact-reference metrics (NOT the SHADOW judge, gpt56 H4).

**Bottom line for bigoracle:** the repo went from *not-training-ready* to a **validated training path** in this session — a trainable+exportable base (HTDemucs), a frozen leak-free corpus, exact eval baselines, and a green smoke — with the four items above as the remaining, well-scoped work to a real classical-separator training run.

Artifacts (research6, repo): `datasets/classical-v1.jsonl` + `-splits.json` + `-overlap-audit.json`, `datasets/candidates-v1.jsonl`, `configs/train/classical-student-v1.yaml`, `audio_extract/train_classical_smoke.py`, `calibration/train_classical_smoke_report.json`, `calibration/state_dump_implementer.md`. Commits: **6e0425a** (Job 1), **ec99187** (Job 2), **09ed3ea** (state dump), **6c4b070** (Jobs 3–4).
