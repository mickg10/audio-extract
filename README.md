# audio-extract

UVR-style music stem separation (vocals / instrumental / …) on the command line,
driven from a **uv** environment, with a faithful **spectrogram ensemble** step.

## Setup (once)

```bash
cd ~/audio-extract
uv sync            # creates .venv and installs audio-separator[cpu] + numpy/soundfile/librosa
```

## Models

Drop your model files into `~/Downloads`, then they get pulled into `models/`:

```bash
./grab_models.sh   # moves *.onnx *.pth *.ckpt *.th *.yaml from ~/Downloads -> models/
```

- `.onnx` → MDX-Net · `.pth` → VR · `.ckpt` → MDX23C · `.th`(+`.yaml`) → Demucs
- For a strong **ensemble**, use two *different architectures* (e.g. one MDX + one Demucs/VR):
  uncorrelated errors are what the ensemble cancels.

## Run

```bash
./separate.sh song.mp3            # isolate Vocals, ensemble = max-spec (fuller)
./separate.sh song.mp3 min        # min-spec (cleaner, less bleed)
STEM=Instrumental ./separate.sh song.mp3 min   # clean karaoke instrumental
```

Per-model stems land in `output/<song>/<model>/`; the merged result is
`output/<song>.<Stem>.ensemble_<algo>.wav`.

### Ensemble algorithm
- **max** — keep the loudest bin across models → fuller, retains more (start here for vocals)
- **min** — keep the quietest → intersection, cleanest/least bleed (best for karaoke instrumentals)
- **avg** — waveform average → safe middle ground

`uvr_ensemble.py` can also be used standalone on any set of aligned stems:
`uv run python uvr_ensemble.py --algo max out.wav a.wav b.wav`

Apple Silicon note: uses the `[cpu]` onnxruntime (CoreML); the `[gpu]` extra is
CUDA-only and useless here.
