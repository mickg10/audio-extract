# Model research — classical/operatic instrumental extraction (agent, 2026-08-04)

## Highest-value insight
**Most Roformer de-reverb models are VOCAL-trained** (expect a vocal input) and behave
unpredictably on a bare orchestra. For cleaning an *instrumental*, only the general
spectrogram models operate correctly:
- `UVR-De-Echo-Normal.pth` (echo only — safest for classical)
- `UVR-De-Echo-Aggressive.pth` (echo only, stronger)
- `UVR-De-Reverb-aufr33-jarredou.pth` (reverb only)
- `Reverb_HQ_By_FoxJoy.onnx` (reverb only)
- `UVR-DeEcho-DeReverb.pth` (both)
→ our convert.py CLEANUP_MODELS already uses exactly these. Keep de-reverb GENTLE
(`--vr_aggression 2–5`): classical hall ambience is musically wanted; aggressive
de-reverb dries the strings/brass and sounds dead.

## Best instrumental models for opera (add to the ensemble)
1. `melband_roformer_inst_v1e_plus.ckpt` — Mel-Band Roformer, instrumental-primary,
   **highest fullness** (preserves orchestral body + ambience). The #1 opera opener.
2. `model_bs_roformer_ep_317_sdr_12.9755.ckpt` — highest-SDR general split (our default).
3. `mel_band_roformer_instrumental_becruily.ckpt` — instrumental-primary, **bleedless**
   (kills residual voice; pair with #1's fullness).
4. `MDX23C-8KFFT-InstVoc_HQ.ckpt` — different arch → good ensemble diversity (we have it).

**Opera ensemble default: `inst_v1e_plus` + `bs_roformer_ep_317`, Max-Spec instrumental.**

## Pipelines
- **Option A (default):** ensemble instrumental → `UVR-De-Echo-Normal` (aggr 3–5) →
  optional gentle `UVR-DeEcho-DeReverb`. Stop if orchestra sounds dry.
- **Option B (very wet halls):** de-reverb the MIX first (`UVR-DeEcho-DeReverb`, moderate)
  → then separate → less reverb tail lands in the instrumental. Cleanest karaoke.
- **Option C (surgical, preserves orchestral ambience — ideal for opera):**
  1) separate vocals, 2) de-reverb the VOCAL stem with the vocal-trained
  `deverb_bs_roformer_8_384dim_10depth.ckpt` → get the vocal's reverb residue,
  3) `clean_instrumental = instrumental − vocal_reverb_residue` (align + subtract).
  This removes only the *singer's* tail, leaving the orchestra's own ambience.
  → maps directly onto our diff-based pipeline; worth implementing as a variant.

## Ensemble semantics (we implement this ourselves; audio-separator has NO native ensemble)
Max-Spec = fullest, keeps ambience, slightly more bleed (use for opera). Min-Spec =
cleanest/thinner. Avg = balanced.

Full ranked list + HuggingFace sources for non-registry models: see the agent report in
the session log. Registry check: `audio-separator --list_models --list_filter=<term>`.
