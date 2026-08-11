# Verdict — Zero-init residual correction on the median ensemble (§13.5)

**Result: A_refined TIES the median/MDX champion exactly on held-out Bologna/Aalto. It does NOT
beat it. The learned correction is Δ≈0 (‖Δ‖∞ ~1e-6) — a safe no-op, zero degradation.**

This is the **third** data point, all agreeing that **no trained model beats the delivered median
ensemble** on the held-out exact-reference constrained objective:

1. judge-as-tuner → **SHADOW** (ties median, no selection skill)
2. HTDemucs full fine-tune → **LOST** (unstable, regressed below both baselines)
3. zero-init residual correction (this) → **TIES** (Δ≈0; the ensemble is near-optimal for this approach too)

---

## Experiment

- **Model (§13.5):** `A_refined = A_median + iSTFT(Δθ(STFT(M), STFT(A_median)))`; `V_refined = M − A_refined`.
  Δθ = 265k-param spectral U-Net, **final conv zero-initialized** → Δ=0 at step 0 → `A_refined == A_median`
  bit-exact (hard champion-parity sanity). L1(Δ) regularized (§13.5) so it changes only identified errors.
- **Champion baseline (A_median):** per-sample median of MDX23C / MelBand / BS-Roformer vocal estimates → residual
  (`median_mdx_mel_bs`, the delivered product). Reproduced for all 35 train works (recon M=A_median+V_median at −240 dB).
- **Train set (35 works, group-atomic, frozen splits):** cantoría 14 (organ) + donor 18 (VocalSet×PHENICX bed×Pori RIR,
  real orchestra) + FreiDi 3 (real orchestra). Bologna/Aalto **HELD OUT** (test-v1).
- **Loss:** full asymmetric family — multi-res complex STFT (V̂,Â) + event-weighted waveform L1 + mixture-consistency
  + no-vocal-FP + vocal-FN + differentiable source-coordinate α/β/R (κ-masked, exact-crop-only) + stereo-coherence
  + L1(Δ). NOT the SHADOW judge.
- **Run:** 2000 steps, ~11 min GPU (RTX PRO 4500), all-finite; train loss 0.606→0.538 (learned a small *training-domain* correction).

## Step-0 parity (hard sanity)

`max|A_refined − A_median|` over all 5 held-out works = **0.000e+00** (bit-exact). Product construction correct.

## Gate-2000 — per held-out work, A_refined (R) vs median-champion (C)

| work | retained_voice_p90 R/C | hole_p90 R/C | hole_max R/C | α-err R/C | SI-SDR R/C | ‖Δ‖∞ | call |
|---|---|---|---|---|---|---|---|
| bologna_verdi     | −21.87 / −21.87 | 0.51 / 0.51 | 3.40 / 3.40 | 0.057 / 0.057 | 21.78 / 21.78 | 6.3e-07 | **TIE** |
| bologna_puccini   | −26.00 / −26.00 | 0.09 / 0.09 | 0.53 / 0.53 | 0.011 / 0.011 | 25.97 / 25.97 | 1.2e-06 | **TIE** |
| bologna_donizetti | −32.85 / −32.85 | 0.31 / 0.31 | 4.05 / 4.05 | 0.036 / 0.036 | 27.77 / 27.77 | 1.0e-06 | **TIE** |
| aalto_mozart_dry  | −34.90 / −34.90 | 0.19 / 0.19 | 2.94 / 2.94 | 0.022 / 0.022 | 23.67 / 23.67 | 6.2e-07 | **TIE** |
| aalto_mozart_hall | −15.32 / −15.32 | 2.08 / 2.08 | 7.04 / 7.04 | 0.213 / 0.213 | 12.48 / 12.48 | 8.4e-07 | **TIE** |
| **AGGREGATE mean** | **−26.19 / −26.19** | **0.64 / 0.64** | **3.59 / 3.59** | **0.07 / 0.07** | **22.33 / 22.33** | max 1.2e-06 | **TIE 5/5** |

Gate-500 is identical (all metrics equal; ‖Δ‖∞ max 3.4e-06). **‖Δ‖ on held-out SHRINKS** 3.0e-06 (step500) → 8.6e-07 (step2000).

## Interpretation (with numbers)

- **Δ ≈ 0 on held-out (safe no-op).** ‖Δ‖∞ ~1e-6 → every exact-reference metric is identical to the champion to full
  precision. No work beat the champion; **no work hurt it**. The zero-init + L1 design did exactly its job: it can only
  refine the winner and it correctly declined to change it — unlike HTDemucs full-FT which catastrophically drifted.
- **Why no improvement:** the corrector learned a small *training-domain* correction (loss 0.61→0.54 on cantoría/donor/FreiDi),
  but on the **held-out real-opera** domain it emits essentially zero correction — and the held-out Δ *shrinks* with more
  training. The training domain (organ + synthetic-donor + one Weber) does not transfer a useful correction to real opera,
  and the median ensemble is already strong there (hole_p90 0.09–2.08, SI-SDR 12.5–27.8), leaving no error the corrector
  can generalize to fix.

## Verdict (one line)

**A_refined does not beat the median/MDX champion — it ties it exactly (Δ≈0, no degradation) on held-out Bologna/Aalto.
The delivered ensemble is near-optimal for the residual-correction approach too.**

## Reproduce

- branch `research/ensemble-residual-correction`
- data: `audio_extract/gen_a_median.py` (champion A_median for train works)
- train/eval: `audio_extract/refine_ensemble.py`, config `configs/train/refine-ensemble-v1.yaml`
- artifacts: `runs/refine_run2/{gates.json,run_report.json,ckpt_step0/100/500/2000.pt}`
- `./run.sh python -m audio_extract.refine_ensemble --config configs/train/refine-ensemble-v1.yaml --run-dir runs/refine_run2 --max-step 2000`
