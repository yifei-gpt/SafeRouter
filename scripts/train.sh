#!/bin/bash
# Full training recipe: 3 safety-head configs x 6 seeds = 18 nets, pooled into
# one ensemble. Reads data/adversarial/probe/labels.jsonl + the cost tensor.
# Run build_cost_tensor and utils.embed first.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../saferouter" && pwd)"
O=../data/checkpoints/k18_retrain; mkdir -p "$O"

# Pre-registered protocol + config flags. Do not tune these to report a result.
DEVICE=cuda
C="--kfold 7 --fold-seed 100 --epochs 100 --n-seeds 6 --cost-train 1.0 --device $DEVICE"
F="--cost-pred-weight 0.5 --use-cost-head --calibrate --calib-mode vector --safety-loss inverse_focal --ckpt-mode constraint --ckpt-asr-target 0.003 --asr-target 0.003 --op-asr-target 0.003 --save-nets"

run(){ local n="$1"; shift; echo "=== $(date '+%F %T') $n START ==="; \
  python train_saferouter.py $C $F "$@" --save-dir "$O/$n" > "$O/$n.log" 2>&1; \
  echo "=== $(date '+%F %T') $n DONE rc=$? ==="; }

run wide_uw3_cf20  --safety-arch flat     --hidden-dim 512 --unsafe-weight 3 --cost-focal-alpha 20
run bilin_uw3_cf20 --safety-arch bilinear --hidden-dim 512 --unsafe-weight 3 --cost-focal-alpha 20
run bilin_uw5_cf20 --safety-arch bilinear --hidden-dim 512 --unsafe-weight 5 --cost-focal-alpha 20

echo "=== $(date '+%F %T') ALL 3 CONFIGS DONE; pooling the 18-net ensemble ==="
python mega_ensemble.py ../data/checkpoints/k18_final \
  "$O/wide_uw3_cf20" "$O/bilin_uw3_cf20" "$O/bilin_uw5_cf20" 2>&1 | tail -8
echo "=== $(date '+%F %T') COMPLETE -> ../data/checkpoints/k18_final/sop_results.json ==="
