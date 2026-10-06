#!/usr/bin/env bash
# Phase Y optimizer diagnostic: four 5k-step Mode-1 runs, 4 concurrent 1-thread processes.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_optimization/logs/_console
for c in Y1_X2_lrsched Y2_X1_lrsched Y3_X4_lrsched Y4_C0_lrsched; do
  python3 -I experiments/run.py --config configs/phaseY/$c.json --budget Y5K --threads 1 \
    --log-every 250 --eval-every 500 --ckpt-every 2500 --status DIAGNOSTIC \
    > results_optimization/logs/_console/Y_$c.log 2>&1 &
done
wait; echo "Y done"; tail -n 1 results_optimization/logs/_console/Y_*.log
