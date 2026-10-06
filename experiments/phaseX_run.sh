#!/usr/bin/env bash
# Phase X diagnostic batch: four 5k-step Mode-1 runs, 4 concurrent 1-thread processes.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_optimization/logs/_console
for c in X1_hard_tanh2 X2_hard_tanh2_rc X3_supervised_paperconv X4_supervised_rcconv; do
  python3 -I experiments/run.py --config configs/phaseX/$c.json --budget X5K --threads 1 \
    --log-every 250 --eval-every 500 --ckpt-every 2500 --status DIAGNOSTIC \
    > results_optimization/logs/_console/X_$c.log 2>&1 &
done
wait; echo "X done"; tail -n 1 results_optimization/logs/_console/X_*.log
