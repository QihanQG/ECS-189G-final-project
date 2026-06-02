#!/usr/bin/env bash
# Headless runner for the retrieval few-shot notebook on an SSH / GPU box.
# Usage:  bash run_remote.sh
# Tip: run inside tmux so it survives disconnects ->  tmux new -s rag
set -euo pipefail

NB="Qwen3_VL_RAG_fewshot_project.ipynb"

echo "== GPU check =="
nvidia-smi || { echo "ERROR: no GPU visible. This notebook needs a CUDA GPU."; exit 1; }

echo
echo "== Executing $NB  (the inference cell is the long part) =="
# timeout=-1 disables the per-cell timeout so the long inference loop isn't killed.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  "$NB"

echo
echo "== Done. =="
echo "Predictions: qwen_predicted_avm_codes_retrieval_v1.txt"
echo "Filled-in notebook (with outputs) saved in place: $NB"
