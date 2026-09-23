#!/usr/bin/env bash
set -euo pipefail

# Strength sweep for the user's x1 host-concat run.
# The evaluator must have ALIGN_TRAIN_VALIDATION_24X2 = True.

RUN_DIR="${RUN_DIR:-runs/logs/wave_B4_q_full_attn_temp_Lno_state_exit_host1_bf16_batch4}"
EVAL_SCRIPT="${EVAL_SCRIPT:-examples/brushnet/evaluate_brushnet1_wave_align24x2.py}"
OUT_ROOT="${OUT_ROOT:-runs/evaluation_result/strength_sweep_host1_align24x2}"
BRUSHNET_CKPT="${BRUSHNET_CKPT:-data/ckpt/segmentation_mask_brushnet_ckpt}"
BASE_MODEL="${BASE_MODEL:-data/ckpt/realisticVisionV60B1_v51VAE}"
MAPPING_FILE="${MAPPING_FILE:-data/BrushBench/mapping_file.json}"
BASE_DIR="${BASE_DIR:-data/BrushBench}"
METRIC_CKPT="${METRIC_CKPT:-data/ckpt}"

# Same as training validation: 24 samples, 2 repeats are controlled by the Python bool.
# Match training validation batching/seed here.
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-1234}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"

# Override from shell if desired, e.g.:
#   CKPTS="2000 5000" bash strength_sweep_host1_align24x2.sh
CKPTS="${CKPTS:-2000 5000 9000}"
STRENGTHS="${STRENGTHS:-0 0.25 0.5 0.75 1.0}"

# Optional multi-GPU parallelism. Default is conservative/sequential.
# Example: NUM_GPUS=4 GPU_LIST="0,1,2,3" bash strength_sweep_host1_align24x2.sh
NUM_GPUS="${NUM_GPUS:-1}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be >= 1" >&2
  exit 2
fi
if (( NUM_GPUS > ${#GPUS[@]} )); then
  echo "NUM_GPUS=$NUM_GPUS but GPU_LIST only has ${#GPUS[@]} entries" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

common_args=(
  --brushnet_ckpt_path "$BRUSHNET_CKPT"
  --base_model_path "$BASE_MODEL"
  --mapping_file "$MAPPING_FILE"
  --base_dir "$BASE_DIR"
  --metric_ckpt_path "$METRIC_CKPT"
  --batch_size "$BATCH_SIZE"
  --seed "$SEED"
  --num_inference_steps "$NUM_STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --paintingnet_conditioning_scale 1.0
  --overwrite
)

sanitize_strength() {
  local s="$1"
  echo "${s//./p}"
}

run_one() {
  local gpu="$1"
  local kind="$2"
  local ckpt="$3"
  local strength="$4"
  local out_dir="$5"

  echo "[START] gpu=$gpu kind=$kind ckpt=$ckpt strength=$strength -> $out_dir"

  if [[ "$kind" == "baseline" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL_SCRIPT" \
      "${common_args[@]}" \
      --disable_wave \
      --image_save_path "$out_dir"
  else
    local wave_dir="$RUN_DIR/checkpoint-${ckpt}/wave"
    if [[ ! -d "$wave_dir" ]]; then
      echo "Missing wave checkpoint: $wave_dir" >&2
      return 3
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL_SCRIPT" \
      "${common_args[@]}" \
      --wave_path "$wave_dir" \
      --wave_strength "$strength" \
      --image_save_path "$out_dir"
  fi

  echo "[DONE ] gpu=$gpu kind=$kind ckpt=$ckpt strength=$strength"
}

# Simple bounded parallel queue.
pids=()
job_index=0
launch_job() {
  local kind="$1" ckpt="$2" strength="$3" out_dir="$4"
  while (( ${#pids[@]} >= NUM_GPUS )); do
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  done
  local gpu="${GPUS[$((job_index % NUM_GPUS))]}"
  run_one "$gpu" "$kind" "$ckpt" "$strength" "$out_dir" &
  pids+=("$!")
  job_index=$((job_index + 1))
}

# One strict BrushNet baseline under the same 24x2 protocol.
launch_job baseline 0 0 "$OUT_ROOT/brushnet_baseline"

for ckpt in $CKPTS; do
  for strength in $STRENGTHS; do
    label="$(sanitize_strength "$strength")"
    launch_job wave "$ckpt" "$strength" "$OUT_ROOT/ckpt${ckpt}_s${label}"
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

# Gather all evaluation_result_sum.csv files into one compact comparison table.
python - "$OUT_ROOT" <<'PY'
from pathlib import Path
import re
import sys
import pandas as pd

root = Path(sys.argv[1])
rows = []

for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    f = d / "evaluation_result_sum.csv"
    if not f.exists():
        continue

    s = pd.read_csv(f, index_col=0).iloc[:, 0]
    row = {"run": d.name}
    if d.name == "brushnet_baseline":
        row.update(kind="baseline", checkpoint=0, strength=0.0)
    else:
        m = re.fullmatch(r"ckpt(\d+)_s(.+)", d.name)
        if not m:
            continue
        row.update(
            kind="wave",
            checkpoint=int(m.group(1)),
            strength=float(m.group(2).replace("p", ".")),
        )
    row.update({k: float(v) for k, v in s.items()})
    rows.append(row)

if not rows:
    raise SystemExit("No evaluation_result_sum.csv files found")

df = pd.DataFrame(rows).sort_values(["kind", "checkpoint", "strength"], kind="stable")
out = root / "strength_sweep_summary.csv"
df.to_csv(out, index=False)
print("\nCombined summary:")
print(df.to_string(index=False))
print(f"\nSaved: {out}")
PY

echo "All done. Summary: $OUT_ROOT/strength_sweep_summary.csv"
