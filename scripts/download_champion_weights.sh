#!/bin/bash
# 下載 FOMO25 冠軍團隊（jbanusco）預訓練權重
# 來源: https://github.com/jbanusco/fomo25/releases

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$(dirname "$SCRIPT_DIR")/weights"
mkdir -p "$WEIGHTS_DIR"

URL="https://github.com/jbanusco/fomo25/releases/download/v1.0.0/fomo25_mmunetvae_pretrained.ckpt"
OUT="$WEIGHTS_DIR/fomo25_mmunetvae_pretrained.ckpt"

echo "Downloading FOMO25 champion (mmunetvae) pretrained weights..."
echo "Source: jbanusco/fomo25 (1st place, Methods track)"
curl -L -o "$OUT" "$URL"
echo "Saved to: $OUT"
echo "Note: This checkpoint is for mmunetvae architecture. For unet_b compatibility, consider integrating jbanusco/fomo25 codebase."
