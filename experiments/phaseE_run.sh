#!/usr/bin/env bash
# Phase E screen: four 5k-step Mode-1 runs, 4 concurrent 1-thread processes.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_optimization/logs/_console
for c in E3_fourier_rad E4_hard_fourier E5_hard_fourier_rad E4b_hard_fourier_rc; do
  python3 -I experiments/run.py --config configs/phaseE/$c.json --budget E5K --threads 1 \
    --log-every 250 --eval-every 500 --ckpt-every 2500 --status CANDIDATE \
    > results_optimization/logs/_console/E_$c.log 2>&1 &
done
wait; echo "E done"; tail -n 1 results_optimization/logs/_console/E_*.log
