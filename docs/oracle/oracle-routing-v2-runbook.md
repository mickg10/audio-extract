# Certified oracle-routing v2 runbook

This runbook turns PR #8's reviewed mathematical core into a versioned binding experiment without overwriting the integrated exploratory router.

## 1. Preserve the exploratory result

Keep all current outputs and reports under their existing identities. Refer to them as:

```text
oracle-routing-v1-exploratory
```

Do not reinterpret its `actionable_oracle_gap` Boolean as a binding architecture decision.

## 2. Build the extended basis manifest

Write an append-only JSONL manifest with one row per work and unique decoded candidate:

```json
{
  "schema": "audio-extract/oracle-routing-basis/v2",
  "work_id": "bologna_verdi",
  "candidate_name": "median_mdx_mel_bs",
  "recipe_id": "sha256:...",
  "path": "/absolute/path/output.f32.wav",
  "artifact_pcm_sha256": "sha256:...",
  "container_sha256": "sha256:...",
  "frames": 4062448,
  "sample_rate_hz": 44100,
  "channels": ["FL", "FR"],
  "subtype": "FLOAT",
  "parent_recipe_ids": ["sha256:..."],
  "executed_model_bundle_hashes": ["sha256:..."]
}
```

Required candidate names before decoded-PCM deduplication:

```text
median_mdx_mel_bs
geomedian_mdx_mel_bs
convex_fusion_uniform
residual_mdx23c
residual_melband
residual_bs_roformer
htdemucs_04573f0d
htdemucs_955717e8
```

For each work:

1. reopen every published file;
2. require stereo 44.1-kHz FLOAT and the declared frame count;
3. recompute container and canonical decoded-PCM identities;
4. group rows by decoded-PCM identity;
5. retain one canonical name and record all aliases;
6. require `median_mdx_mel_bs` to remain addressable as the product champion even when it is byte-identical to another row.

Do the same for the Aalto orchestra-only control.

## 3. Freeze experiment identities

Use new identities:

```text
math contract: oracle-routing-math-v2
runner recipe: oracle-routing-v2-certified
basis manifest: oracle-routing-basis/v2
report schema: oracle-routing-envelope/v2
```

The runner recipe includes:

```text
basis manifest SHA-256
truth A/V/M PCM identities
STFT geometry
time-cell duration
frequency-band edges
source-coordinate weights/floors
O2 switch penalties and MILP gap/time limit
O3 squared-smoothness weights and convergence tolerance
code commit
```

## 4. STFT controls

Use one common analysis/synthesis implementation for routed outputs.

For every work render:

```text
median_raw
median_stft_identity
O1_raw
O1_stft_identity
O2_global_medoid
O3_certified_convex
```

`median_stft_identity` and `O1_stft_identity` use constant one-hot routes through the same STFT/ISTFT path as O2/O3.

Before optimization, require each candidate's identity-route round trip to stay under frozen limits for:

```text
max absolute error
RMS error
multi-resolution STFT error
stereo width/coherence error
transient loss/excess
```

## 5. Complete exact cell accounting

For every time/frequency cell write one of:

```text
source_coordinates
no_vocal_direct_fallback
vocal_only_direct_fallback
silent_direct_fallback
ill_conditioned_direct_fallback
```

No exact non-silent cell may have zero/absent risk.

Report counts and energy coverage for every mode.

## 6. O2 certificate

Run the global Potts MILP.

Accept O2 as certified only when the solver returns:

```text
incumbent present
requested optimality gap satisfied
complete label grid
objective recomputed independently from labels
```

A time-limit incumbent without the requested gap is saved as exploratory but cannot support a `basis insufficient` conclusion.

## 7. O3 certificate

Use the convex `Q,c,k` cell objective and squared graph smoothness.

Start from:

```text
uniform
O1 exact
O2 exact
independent cell vertices
```

Require:

```text
simplex max error <= tolerance
monotone objective
projected-gradient certificate <= tolerance
agreement of converged starts within tolerance
final objective independently recomputed
```

An unconverged O3 artifact may be auditioned but is excluded from the binding decision.

## 8. Resolution sensitivity

Run the same frozen basis and objective at:

```text
2.0 seconds
1.0 second
0.5 seconds
```

Scale only graph regularization according to the preregistered physical-duration convention. Do not select a different grid independently for each held-out work.

If the 0.5-second O2 MILP cannot close its gap, publish incumbent and bound separately rather than calling the incumbent the oracle.

## 9. Evaluation table

Works:

```text
bologna_verdi
bologna_donizetti
bologna_puccini
aalto_mozart_dry
aalto_mozart_dry orchestra-only control
```

For each raw/control/routed artifact report:

```text
retained_voice_db_p90
retained_voice_coefficient_p90
event_hole_db_p90
event_hole_db_max
alpha_error_p90
orthogonal_artifact_ratio_p90
no-vocal false-positive energy
stereo width deviation
stereo coherence deviation
hall-tail deviation
transient loss
transient excess
worst identifiable event
time-boundary seam ratio
frequency-boundary/ringing diagnostic
container and decoded-PCM identities
```

## 10. Binding decision

The reference baseline is the raw current median champion. MDX23C remains the conservative fallback comparator.

A route is actionable only when it:

1. beats the raw median champion;
2. improves at least one intended critical axis by `>= 1.5 dB`;
3. regresses the opposing critical axis by `<= 0.5 dB`;
4. preserves Aalto voice/hole/stereo/coherence/hall limits;
5. keeps no-vocal false-positive energy within `1.05x`;
6. preserves artifact/transient/seam/worst-event limits;
7. is a reopen-verified immutable FLOAT artifact;
8. has the required O2/O3 optimization certificate.

Interpretation:

```text
large certified O2/O3 gap:
    train the frozen-member target-singer gate

small certified gap with common exact failure cells:
    add a new pretrained family or a zero-initialized correction/full separator

role ambiguity among target singer, other soloists, and chorus:
    use target-singer query conditioning regardless of generic-vocal SDR
```

## 11. Required publication bundle

```text
resolved-config.json
basis-v2.jsonl
basis-dedup-report.json
truth-manifest.json
cell-mode-report.json
optimizer-certificates.json
complete-work-metrics.json
binding-decision.json
full-length FLOAT artifact manifest
reopen/hash audit report
```
