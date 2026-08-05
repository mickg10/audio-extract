# AGENTS.md — audio-extract v2

This file governs any coding agent (e.g. Codex) working in this repository. The
runtime separation harness is the deterministic job queue plus a bounded **Pi +
DeepSeek V4 Pro** conductor; agents implement and test the deterministic tools,
they do **not** sit in the production audio path.

## Architecture in one paragraph

`audio_extract/` is the v2 deterministic core: canonical candidate identity, an
immutable content-addressed DAG, a passage miner, an objective QA metric bank,
and a JSON CLI. The Pi/DeepSeek conductor calls that CLI's JSON contracts; it
never manipulates waveforms or invents commands. v1 (`convert.py`, `web/`) is
frozen — do not modify it to satisfy v2 work.

## Hard invariants (never violate)

- Never edit files under a track's `source/` or `original.*`.
- Never rewrite an immutable candidate directory or its `output.f32.wav`.
- Never use a lossy intermediate anywhere in an analysis lineage.
- Every transformation must be a recipe node with explicit parents.
- Every model bundle must match its pinned hashes (weight + config + adapter).
- Identity is the weight/config/adapter hash, never the checkpoint filename.
- Masters and intermediates are explicit float32 WAV; dither only when reducing a
  float master to integer PCM, and never twice.
- Normalization / gain / delivery encoding are separate child DAG nodes, never
  folded into a separator candidate's identity.
- A metric or learned judge that fails local calibration must not silently become
  a selection signal.
- The conductor's only two terminal outputs are `status: "final"` and
  `status: "needs_human_ab"`; validate at the tool boundary and reject anything else.

## Tests must cover

- canonicalization: key-order invariance, default materialization, hash changes
  on model/overlap/construction/resampler/code-rev, rejection of non-finite
  numbers and unknown fields;
- audio invariants: exact frame count / channel layout / sample rate, verified
  float subtype, near-tolerance residual reconstruction, alignment (delay,
  polarity, channel swap);
- no lossy intermediate exists in an analysis lineage.

## Commands

```bash
uv run audio-extract recipe canon --recipe recipe.json
uv run audio-extract panel show --panel configs/panel.yaml
uv run audio-extract ingest input/track.wav --run-id my-track
uv run audio-extract run inspect --run-id my-track
uv run --extra dev pytest -q
```
