#!/usr/bin/env bash
# Grab UVR model files out of ~/Downloads into this project's models/ dir.
# Moves .onnx (MDX-Net), .pth (VR), .ckpt (MDX23C), .th/.yaml (Demucs).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
shopt -s nullglob
moved=0
for f in "$HOME"/Downloads/*.onnx "$HOME"/Downloads/*.pth \
         "$HOME"/Downloads/*.ckpt "$HOME"/Downloads/*.th "$HOME"/Downloads/*.yaml; do
    mv -n "$f" models/ && { echo "grabbed $(basename "$f")"; moved=$((moved + 1)); }
done
[ "$moved" -eq 0 ] && echo "(no new model files in ~/Downloads)"
have=$(ls models/ 2>/dev/null | grep -vE '\.yaml$' | tr '\n' ' ')
echo "models/: ${have:-<empty>}"
