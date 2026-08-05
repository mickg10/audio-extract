## Oracle reply — anchored to `2a9a42b`

I agree with the proposed v2 shape: deterministic audio tools, an immutable candidate graph, a bounded per-file conductor, and the job queue as the authoritative lifecycle.

Before the scaffold, I would correct four v1 semantics:

1. `original - estimated_vocal` is a **waveform residual**, not UVR spectral inversion. If `V̂ = V - L + T`, then `M - V̂ = A + L - T`; orchestral material `T` mistakenly included in the vocal estimate becomes a hole in the accompaniment. Remove the “untouched orchestra / no pumping” claim.
2. PCM24 after float processing is quantized, not lossless. Keep candidate/master audio as explicit float32 WAV and make gain, dither, PCM conversion, and AAC separate child transforms.
3. Do not normalize candidates in place. Candidate bytes must be immutable.
4. Bypass or explicitly configure hidden wrapper behavior: the current code resamples through `librosa` to 44.1 kHz, `align2` only truncates, and current `audio-separator` defaults can normalize and pass WAV output through an int16 pydub path.

A detailed design artifact with Mermaid diagrams, schemas, exact seed hashes, thresholds, metrics, and the implementation/test plan is attached as **`audio-extract-v2-design.md`**.

### 1. Candidate hashing

Use schema-normalized [RFC 8785 JCS](https://www.rfc-editor.org/rfc/rfc8785.html), but keep three identities:

- `recipe_id`: logical computation;
- `artifact_pcm_sha256`: actual decoded output PCM;
- `execution_fingerprint`: runtime/backend details.

Materialize effective defaults before hashing; reject non-finite numbers; use integer samples/rates and exact decimal strings or fixed-point integers for dB. Model identity is weight hash + config hash + adapter/code revision, not filename.

`output_construction` must be structured and distinct: native target, mixture-minus-vocal, sum of non-vocal stems, weighted waveform ensemble, spectral ensemble, cleanup, phrase route, etc. Resampler and channel mapping belong in the recipe whenever used. Normalization/dither/encoding are separate child nodes, not fields that mutate a separator candidate.

### 2. Passage miner

Do not make a speech VAD primary. Use the provisional vocal stem and fuse:

- vocal energy relative to a robust local floor;
- PESTO pitch confidence/continuity;
- pYIN fallback/disagreement;
- optional CREPE spot checks;
- accompaniment loudness, band occupancy, and flux;
- phrase boundaries and hall tails.

Seed thresholds:

- active: vocal level ≥ `max(noise floor + 10 dB, track p35)` with pitch confidence ≥ `0.55`, or vocal level ≥ p85;
- high soprano: median voiced F0 ≥ C5 (`523.25 Hz`) or track p80;
- extreme: smoothed peak ≥ F5 (`698.46 Hz`) or track p95;
- vocal/accompaniment ratio groups: `<-9`, `-9..-3`, `-3..+3`, `>+3 dB`;
- dense accompaniment: at least three of loudness ≥ p80, band occupancy ≥ p70, flux/onset density ≥ p70, non-silence.

Use 12 s defaults, 8–20 s allowed, ≥2 s context, interval NMS (`IoU <= .25`, center separation ≥8 s), category quotas, then farthest-point diversity over F0, ratio, density, tail, stereo width, and track position.

### 3. Model panel

Start with a small exact panel:

| Bundle | Weight SHA-256 |
|---|---|
| Viperx-1297 `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | `5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa` |
| Kim Vocal 2 `MelBandRoformer.ckpt` | `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e` |
| unwa `inst_v1e_plus.ckpt` | `6a4ddba739f0352407fb6e18b29206b82318ec427fe37fcedb0f83241e4e15fb` |
| becruily instrumental | `a8da6632a1c25efb1c9be783ce9ea367d226d4b918cd6c3717c8b1d7a396041d` |
| becruily vocal | `a05961310cc55fbb901290c2e8be02682942f73522b6ac76bf2ec11e347ed95a` |
| MDX23C HQ | `49d51472769e34a2501cd1da782346a3212555c3a5619fc2c53507445528d816` |
| BS PolarFormer | `fc8b72c3beb92caad4e14f180979c02ddf18a330182176cf6c7bb0eb6c685e87` |
| HTDemucs FT | bundle-hash its YAML plus all four `.th` files |

Add HyperACE/FNO/Resurrection only through exact code/config adapters. Do **not** add an unnamed “124-band” leaderboard model until the public weight, config, implementation revision, and hash are identified.

### 4. QA judge

Music Flamingo and AF3 are useful experimental annotators, not selectors until locally calibrated. Their published goals are broad music/audio understanding, not fine separation-defect measurement, and independent work has shown serious basic-perception and option-order weaknesses in current audio-language models.

Use:

1. hard technical checks;
2. event-level leakage, fullness, pumping, brightness, transient, hall, and stereo metrics;
3. SAM Audio Judge as the first source-aware learned judge;
4. Pareto filtering;
5. a blinded pairwise Bradley–Terry/logistic ranker trained on human preferences.

Run SAJ on both the instrumental prompt and the vocal prompt. Validate any judge with A/B order reversal, level perturbation, repeats, known injected defects, and 100–200 human-labeled excerpt comparisons.

Brightness should be fidelity to an expected spectral envelope, not maximum centroid. Pumping should be measured as candidate-specific, vocal-event-correlated multiband dips with depth, duration, and recovery.

### 5. ε minimization

Do not default to complex-spectrum maximum for vocals. It can union both models’ false positives and increase orchestral theft.

Compare:

- each vocal residual;
- waveform mean;
- waveform median/trimmed mean;
- learned non-negative weighted vocal average;
- later, a soft TF consensus if needed.

Score the resulting accompaniment, not just the vocal stem. Train weights on synthetic opera-like mixtures with an asymmetric loss that penalizes orchestral holes more strongly than mild bleed.

Direct instrumental versus `M - V̂` is empirical. Generate both where meaningful. Some wrappers already derive the secondary stem as the residual, so detect duplicates. Defer per-band blending because phase, stereo, and transition behavior make it a second-stage feature.

### 6. Harness

Two refinement rounds are enough:

- round 0: 6–8 bundles, defaults, hard excerpts;
- round 1: construction + modest overlap sweep for finalists;
- round 2: at most two evidence-based ensembles or diagnosed repairs.

Suggested cap: ≤30 excerpt candidates, ≤4 full-track renders, ≤2 final ensembles.

The controller exposes typed actions only. DeepSeek V4 Pro receives validated JSON measurements and proposes one-variable experiments; it does not claim to hear absent evidence. Codex implements adapters, metrics, fixtures, UI, and tests. The queue owns persistent state and every transformation is a DAG node.

### 7. Float / dither

- explicit float32 WAV for intermediates and masters;
- float64 accumulation for subtraction/ensemble where useful;
- TPDF only at final integer conversion;
- TPDF default for 16-bit;
- PCM24 may use TPDF, but noise shaping should be off by default;
- no dither for float32;
- encode AAC from the float master;
- never dither twice.

### Implementation order

1. correct residual, sample-rate, alignment, writer, and in-place normalization semantics;
2. immutable JCS recipe DAG + model importer;
3. miner + synthetic metric fixtures;
4. Tier A excerpt panel;
5. objective report + blinded A/B UI;
6. SAJ calibration;
7. bounded conductor;
8. current custom-model adapters and phrase routing.

The core v2 win is not “more models.” It is an auditable mechanism for finding which candidate removes the singer **without stealing orchestral energy on the exact passages that matter**.
