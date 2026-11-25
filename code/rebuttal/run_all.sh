#!/usr/bin/env bash
# Rebuttal experiments on a single A100 (GPU 0). Run from code/rebuttal: nohup bash run_all.sh > ../../results/rebuttal/run_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../../results/rebuttal
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS=ignore

python exp2_degree_positivity.py --B 200 --mc-reps 5000 --n-jobs 32 > ../../results/rebuttal/exp2.log 2>&1 &
for s in out1 out2; do
  python exp1_seeds_conditioning_crossfit.py --setting "$s" --B 50 --seeds 5 --n-jobs 6 > "../../results/rebuttal/exp1_${s}.log" 2>&1
done
wait
python summarize.py > ../../results/rebuttal/summary.txt  # rerun after exp3 finishes (launched separately)
echo "all done"
