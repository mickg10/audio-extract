# arXiv research — classical/orchestral source separation (agent, 2026-08-04)

## Headline
There is **no published paper applying BS-Roformer / SOTA pop models to real
orchestral-operatic audio and winning.** Classical MSS is a separate, data-starved
sub-field (EU REPERTORIUM + Aalto/Tampere + Georgia Tech) still benchmarking against
X-UMX / Open-Unmix / Conv-TasNet. Real-orchestral SDR is ~0–2 dB vs 9–11 dB for pop
vocals — the wall is the **synthetic→real domain gap**, not architecture.

**Practical implication:** for "remove the operatic voice → instrumental", the pop-trained
Roformers we already use (BS-Roformer + MDX23C ensemble + VR de-echo/de-reverb) ARE the
pragmatic frontier. The academic gains are not plug-and-play — they need training/
fine-tuning or extra inputs (scores). So our pipeline is not leaving an easy win on the table.

## Most relevant papers (with why they matter)
1. **Score-informed MSS in classical** — arXiv 2503.07352 (2025, code public: github.com/ee7u/score-mss).
   Feeding the musical score lifts real-URMP from 1.29→4.48 dB. The single biggest lever
   for real classical audio — but requires the score aligned to audio.
2. **Banquet / hyperellipsoidal queries** — arXiv 2501.16171 (2025, Georgia Tech, code avail).
   Query-based "extract the violin/any instrument" model; strongest architecture if we ever
   want *individual* orchestral instruments rather than just voice-vs-orchestra.
3. **SepACap** — arXiv 2509.26580 (2025, ETH). Multi-singer/choral separation — relevant if
   operatic ensembles/choruses need splitting into individual voices.
4. **Unsupervised vocal dereverb via diffusion** — arXiv 2211.04124 (2022, Sony). Reverb
   removal without paired data; confirms dereverb is hard and best done gently on music.
5. **SynthSOD (2409.10995) + Spheres (2511.21247)** datasets — the training/eval assets
   (Spheres has per-instrument RIRs for dereverb). Relevant only if we fine-tune.

## What this means for our build
- Keep the current ensemble + **gentle** de-echo/de-reverb (research agrees classical hall
  reverb is desirable; aggressive removal deadens the orchestra).
- No model swap needed. The real upgrades (score-informed, query-based, fine-tuned Roformer
  on SynthSOD/Spheres) are research projects, not checkpoint downloads.
- Future direction if wanted: Banquet for per-instrument stems; score-informed if scores exist.

Full paper tables (architectures, datasets, dereverb, SOTA/challenges) are in the session log.
