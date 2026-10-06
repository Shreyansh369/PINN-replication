#!/usr/bin/env bash
# Phase A / A1: 20k-step (S1) runs of the four Phase-A baselines, 4 concurrent 1-thread processes.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_optimization/logs/_console
for c in BA_vanilla BB_fourier C0_paper C0_rc; do
  python3 -I experiments/run.py --config configs/phaseA/$c.json --budget S1 --threads 1 \
    --log-every 250 --eval-every 1000 --ckpt-every 5000 --status BASELINE \
    > results_optimization/logs/_console/A1_$c.log 2>&1 &
done
wait
echo "A1 done"; tail -n 1 results_optimization/logs/_console/A1_*.log
