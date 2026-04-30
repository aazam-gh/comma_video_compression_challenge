#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${HERE}/archive"

echo "Starting full end-to-end compression pipeline..."

# pass along any arguments (e.g., --crf 50)
python3 "${HERE}/compress.py" "$@"

echo "Pipeline complete. Packaging artifacts..."

mkdir -p "$ARCHIVE_DIR"
cd "$ARCHIVE_DIR"

# Include all compressed artifacts: model, masks (both frames), poses, color hints
zip -0 "${HERE}/archive.zip" model.pt.br mask.obu.br pose.npy.br color.npy.br

echo "Done! Final payload saved to: ${HERE}/archive.zip"