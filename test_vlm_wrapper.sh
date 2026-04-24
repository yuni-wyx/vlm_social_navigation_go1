#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 image1.jpg [image2.jpg ...]"
  exit 1
fi

WRAPPER_URL="${WRAPPER_URL:-http://localhost:8000/analyze}"

for image_path in "$@"; do
  if [ ! -f "$image_path" ]; then
    echo "Missing file: $image_path"
    continue
  fi

  echo "===== $image_path ====="
  image_b64=$(base64 -i "$image_path" | tr -d '\n')
  curl -s -X POST "$WRAPPER_URL" \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\":\"${image_b64}\"}"
  echo
  echo
done
