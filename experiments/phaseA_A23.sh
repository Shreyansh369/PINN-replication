#!/usr/bin/env bash
# Phase A / A2 + A3: C0 seeds 1235, 1236 at S1 (20k) and C0 seed 1234 at S2 (100k), concurrently,
# 1 thread each. A3 runs in <= 1.9 h segments (resumable); re-invoke this script to continue it.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_optimization/logs/_console
C=configs/phaseA/C0_paper.json
for s in 1235 1236; do
  python3 -I experiments/run.py --config $C --budget S1 --seed $s --threads 1 \
    --log-every 250 --eval-every 1000 --ckpt-every 5000 --status BASELINE \
    >> results_optimization/logs/_console/A2_C0_s$s.log 2>&1 &
done
python3 -I experiments/run.py --config $C --budget S2 --seed 1234 --threads 1 \
  --log-every 500 --eval-every 2000 --ckpt-every 5000 --status BASELINE --max-wall-hours 1.9 \
  >> results_optimization/logs/_console/A3_C0_S2.log 2>&1 &
wait
echo "segment done"; tail -n 1 results_optimization/logs/_console/A2_*.log results_optimization/logs/_console/A3_*.log
