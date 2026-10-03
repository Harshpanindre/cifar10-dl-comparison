#!/usr/bin/env bash
# Train, evaluate and compare all four models under the common protocol.
set -e
for m in mlp cnn lstm vit; do
  python train.py --model "$m" "$@"
  python evaluate.py --model "$m"
  python error_analysis.py --model "$m"
done
python compare.py
