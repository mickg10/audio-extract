# Stage-1 first run — convergence + eval-vs-baseline signal

`audio-extract train classical` on frozen classical-v1 (train split = cantoria×14 + freidi×3, leak-audited, never re-split). HTDemucs 26.9 M params, 9-term asymmetric loss (wav/complex-STFT/mix-consistency/theft-β/vocal-FN/α/stereo + event-tail vocal&tutti). Commit 5d65333. Full artifacts: `calibration/train_classical_run1_report.json`; checkpoint+export at `research6:~/runs/train/classical-v1-run1/`.

## Convergence: YES
loss 1.216 (step0) → 0.976 (100) → 0.503 (200) → 0.355 (750); all finite. Held-out metrics improved fast: e.g. bologna_verdi orchestral-hole p90 40→6.9 dB, SI-SDR −15→+3.2, retained-voice +? →−7 dB over 560 steps.

## Held-out eval (step 559, exact-reference; parity=True on all) vs FROZEN baselines
| work | student rv_db / hole_p90 / SI-SDR | baseline MDX23C | baseline median |
|---|---|---|---|
| bologna_verdi | −6.97 / 6.91 / 3.15 | −18.35 / 1.19 / 18.72 | −21.87 / 0.51 / 21.78 |
| bologna_puccini | −6.63 / 6.92 / 3.10 | −22.86 / 0.29 / 24.61 | −26.0 / 0.09 / 25.97 |
| bologna_donizetti | +8.99 / 10.61 / 0.82 | −29.81 / 2.54 / 25.66 | −32.85 / 0.31 / 27.77 |
| aalto_mozart_dry | +1.13 / 6.36 / 1.81 | −31.81 / 0.28 / 21.46 | −34.9 / 0.19 / 23.67 |

## Verdict: converging + improving fast, but NOT beating baselines — gap is large.
The student trajectory is healthy (hole 40→6-11 dB, SI-SDR −22→+0.8-3.2), but a 27 M HTDemucs at 560 from-scratch steps is far below the pretrained MDX23C/median ensembles (SI-SDR 18-27, hole 0.3-2.5). Two structural reasons + the fixes:
1. **Train/eval domain mismatch (the big one):** Stage-1 train = cantoria (ORGAN) + 1 Weber work — almost NO orchestra — but the product/eval is voice+ORCHESTRA. **Add the orchestral donor constructions (VocalSet×PHENICX/Spheres×RIR, group-atomic) to Stage-1** so the student sees orchestra.
2. **From-scratch + short:** run the full multi-hundred-k schedule; strongly consider **fine-tuning a pretrained HTDemucs** rather than from-scratch to close the gap to the pretrained baselines.

## Export bridge: VERIFIED
Trained HTDemucs serialized to demucs `.th` (`~/runs/train/classical-v1-run1/htdemucs_export.th`, sha256:0a194bf1…) and reloaded via `demucs.states.load_model` + `apply_model` (== audio-separator `demucs_separator` inference math) — round-trip finite. Registry packaging into audio-separator is the final step.

## Infra note
Sustained GPU training was SIGTERM-killed twice (~1-3 min in) under nohup/setsid detachment (not disk/OOM/cgroup — cause unresolved); the attached-ssh finalize completed. The full run needs a robust runner (tmux/systemd-run --user --scope) or an attached session.
