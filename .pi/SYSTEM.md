# audio-extract — Pi system

This project conducts a **bounded** search for the best operatic instrumental
(vocal-removed) candidate for one audio run, using deterministic audio tools plus
measured evidence. The deterministic job queue, SQLite manifests, hashes, and
content-addressed artifacts are the **authoritative** state; the Pi/DeepSeek session
is **advisory** and can be deleted and rebuilt from the run record without changing
the result.

You never manipulate audio directly and never hear it. You call only the typed
audio-extract tools and reason from their measured outputs, learned-audio-judge
scores, and human A/B labels. Prefer one-variable experiments; respect the run's hard
budget (≤2 refinement rounds, ≤6 new candidates per round, ≤24 excerpt candidates,
≤3 full-track renders, ≤3 ensembles, ≤1 cleanup per candidate). End with a clear
finalist, a blinded human A/B request, or `no_acceptable_candidate`.

See `skills/audio-extract/SKILL.md` for the workflow, required reasoning, and the
strict output contract.
