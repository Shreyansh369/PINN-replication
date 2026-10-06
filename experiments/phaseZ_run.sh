#!/usr/bin/env bash
# Phase Z: four 5k-step Mode-1 runs, 4 concurrent 1-thread processes. Seed is taken from each config.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_optimization/logs/_console
for c in Z1_Y1_rad Z2_Y1_seed1235 Z3_Y4_rad Z4_Y1_mb128; do
  python3 -I experiments/run.py --config configs/phaseZ/$c.json --budget Z5K --threads 1 \
    --log-every 250 --eval-every 500 --ckpt-every 2500 --status DIAGNOSTIC \
    > results_optimization/logs/_console/Z_$c.log 2>&1 &
done
wait; echo "Z done"; tail -n 1 results_optimization/logs/_console/Z_*.log
