#!/usr/bin/env bash
# Drive UVR-style separation from the uv env: grab models, run each on the input,
# ensemble the per-model stems.
#
#   ./separate.sh <audiofile> [min|max|avg]      # default algo: max
#   STEM=Instrumental ./separate.sh song.mp3 min # pick which stem to isolate
#
set -euo pipefail
cd "$(dirname "$0")"

IN="${1:-}"; ALGO="${2:-max}"; STEM="${STEM:-Vocals}"
if [ -z "$IN" ] || [ ! -f "$IN" ]; then
    echo "usage: ./separate.sh <audiofile> [min|max|avg]   (env STEM=Vocals|Instrumental)"
    exit 1
fi

./grab_models.sh
shopt -s nullglob
MODELS=(models/*.onnx models/*.pth models/*.ckpt models/*.th)
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "No models found. Drop your 2 model files in ~/Downloads and re-run."
    exit 1
fi

base="$(basename "${IN%.*}")"
work="output/${base}"; rm -rf "$work"; mkdir -p "$work"
STEMS=()
for m in "${MODELS[@]}"; do
    mb="$(basename "$m")"
    echo ">> separating with ${mb}"
    md="${work}/$(basename "${mb%.*}")"; mkdir -p "$md"
    uv run audio-separator "$IN" \
        --model_filename "$mb" --model_file_dir models \
        --single_stem "$STEM" --output_dir "$md" --output_format WAV
    got=("$md"/*.wav)
    [ "${#got[@]}" -ge 1 ] && STEMS+=("${got[0]}")
done

echo "collected ${#STEMS[@]} '${STEM}' stem(s)"
if [ "${#STEMS[@]}" -ge 2 ]; then
    out="output/${base}.${STEM}.ensemble_${ALGO}.wav"
    uv run python uvr_ensemble.py --algo "$ALGO" "$out" "${STEMS[@]}"
    echo "ENSEMBLE -> $out"
elif [ "${#STEMS[@]}" -eq 1 ]; then
    cp "${STEMS[0]}" "output/${base}.${STEM}.wav"
    echo "single model -> output/${base}.${STEM}.wav  (add a 2nd model for an ensemble)"
fi
