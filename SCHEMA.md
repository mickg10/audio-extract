# Data contract for the audio-extract web GUI

`convert.py` populates `~/audio-extract/lib/`. The web app is read-only over this tree.
Everything below is the stable contract — build against it; real data appears as the
background batch finishes (handle missing/partial gracefully).

## Layout

```
lib/
  index.json                         # the listing page's data source
  <slug>/                            # one dir per input file (slug = sanitized basename)
    original.mp3  (or .wav)          # HARD LINK to the source audio
    original.png                     # spectrogram of the original
    original.peaks.json              # waveform envelope of the original
    manifest.jsonl                   # one JSON object per line = one variant/experiment
    out/
      <id>.wav                       # each variant's audio
      <id>.png                       # its spectrogram
      <id>.peaks.json                # its waveform envelope
```

## `lib/index.json`

```json
{
  "generated": 1785900000,
  "count": 12,
  "files": [
    {
      "slug": "1_HABANERA_LAST_Tobacco_2026_06_18",
      "title": "1_HABANERA_LAST_Tobacco_2026_06_18",
      "original": "original.mp3",
      "duration_s": 128.4,
      "n_variants": 6,
      "variants": ["roformer_instrumental","roformer_vocals","mdx23c_instrumental",
                   "mdx23c_vocals","ensemble_max_instrumental","residual_voice"]
    }
  ]
}
```

## `manifest.jsonl` (one object per line)

Every entry has: `id`, `kind`, `stem`, `description`, `output` (path relative to the
slug dir), `spectrogram`, `peaks`, `status`, and audio metadata (`duration_s`, `sr`,
`channels`, `peak`, `rms`, `lufs_approx`). `input` names the source(s) this was derived
from (the string `"original"`, another variant `id`, or a 2-element list for diffs).

`kind` ∈ `source | separate | ensemble | diff`. Worked example (one file):

```jsonl
{"id":"original","kind":"source","stem":"original","description":"Original mix","output":"original.mp3","spectrogram":"original.png","peaks":"original.peaks.json","status":"done","duration_s":128.4,"sr":44100,"channels":2,"peak":0.98,"rms":0.12,"lufs_approx":-18.4}
{"id":"roformer_instrumental","kind":"separate","stem":"Instrumental","description":"model_bs_roformer_ep_317_sdr_12.9755.ckpt — instrumental","models":["model_bs_roformer_ep_317_sdr_12.9755.ckpt"],"flags":{"segment_size":256},"input":"original","output":"out/roformer_instrumental.wav","spectrogram":"out/roformer_instrumental.png","peaks":"out/roformer_instrumental.peaks.json","status":"done","duration_s":128.4,"sr":44100,"channels":2,"peak":0.9,"rms":0.10,"lufs_approx":-20.0}
{"id":"mdx23c_instrumental","kind":"separate","stem":"Instrumental","description":"MDX23C-8KFFT-InstVoc_HQ.ckpt — instrumental","models":["MDX23C-8KFFT-InstVoc_HQ.ckpt"],"flags":{"segment_size":256},"input":"original","output":"out/mdx23c_instrumental.wav","spectrogram":"out/mdx23c_instrumental.png","peaks":"out/mdx23c_instrumental.peaks.json","status":"done","duration_s":128.4,"sr":44100,"channels":2,"peak":0.9,"rms":0.10,"lufs_approx":-20.1}
{"id":"ensemble_max_instrumental","kind":"ensemble","stem":"Instrumental","description":"Max-spec ensemble of BS-Roformer + MDX23C instrumentals","models":["model_bs_roformer_ep_317_sdr_12.9755.ckpt","MDX23C-8KFFT-InstVoc_HQ.ckpt"],"flags":{"algo":"max"},"input":["roformer_instrumental","mdx23c_instrumental"],"output":"out/ensemble_max_instrumental.wav","spectrogram":"out/ensemble_max_instrumental.png","peaks":"out/ensemble_max_instrumental.peaks.json","status":"done","duration_s":128.4,"sr":44100,"channels":2,"peak":0.9,"rms":0.10,"lufs_approx":-20.0}
{"id":"residual_voice","kind":"diff","stem":"residual","description":"Original − instrumental ensemble (the removed voice + hall reverb)","flags":{"op":"subtract"},"input":["original","ensemble_max_instrumental"],"output":"out/residual_voice.wav","spectrogram":"out/residual_voice.png","peaks":"out/residual_voice.peaks.json","status":"done","duration_s":128.4,"sr":44100,"channels":2,"peak":0.7,"rms":0.05,"lufs_approx":-26.0}
{"id":"deecho_dereverb_instrumental","kind":"separate","stem":"Instrumental","description":"De-Echo + De-Reverb (removes echo and hall reverb)","models":["UVR-DeEcho-DeReverb.pth"],"input":"ensemble_max_instrumental","output":"out/deecho_dereverb_instrumental.wav","spectrogram":"out/deecho_dereverb_instrumental.png","peaks":"out/deecho_dereverb_instrumental.peaks.json","status":"done","duration_s":128.4,"sr":44100,"channels":2,"peak":0.88,"rms":0.09,"lufs_approx":-20.6}
{"id":"deecho_dereverb_removed","kind":"diff","stem":"residual","description":"What De-Echo + De-Reverb removed (ensemble − cleaned)","flags":{"op":"subtract"},"input":["ensemble_max_instrumental","deecho_dereverb_instrumental"],"output":"out/deecho_dereverb_removed.wav","spectrogram":"out/deecho_dereverb_removed.png","peaks":"out/deecho_dereverb_removed.peaks.json","status":"done","duration_s":128.4,"sr":44100,"channels":2,"peak":0.3,"rms":0.01,"lufs_approx":-34.0}
```

## `*.peaks.json` (waveform envelope for the UI)

```json
{"min":[-0.02,-0.11,...], "max":[0.03,0.14,...]}   // ~1600 buckets, mono, -1..1
```
Draw each bucket as a vertical line from `min[i]` to `max[i]`; index → time is
`i/len * duration_s`. Use for the scrollable waveform + playhead.

## Semantics the UI should lean on

- **Variants to compare** are the `separate` + `ensemble` entries (the instrumentals/vocals).
- **`diff` entries are precomputed comparisons** — `residual_voice` = what the ensemble
  removed; `<tag>_removed` = what a de-echo/de-reverb pass stripped (this is the
  "subtraction vs reverb subtraction" visual). Prefer showing these precomputed diffs
  over computing A−B in the browser.
- Ordering for display: original, then instrumentals (roformer, mdx23c, ensemble,
  deecho*, dereverb*), then vocals, then diffs.
```
