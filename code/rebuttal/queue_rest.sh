#!/usr/bin/env bash
# Remaining rebuttal experiments, run in sequence on GPU 1 (GPU 0 is busy with exp1 out2 and exp4).
# nohup bash queue_rest.sh > ../../results/rebuttal/queue_rest.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore

python exp7_semisyn_crossfit.py --B 50 --n-jobs 6 > ../../results/rebuttal/exp7.log 2>&1
python exp6_retune.py --B-tune 20 --B-eval 50 --n-jobs 6 > ../../results/rebuttal/exp6.log 2>&1
for g in BlogCatalog Flickr; do
  python exp3_bc_flickr.py --graphs $g --gnns gat --B 20 --n-jobs 3 --tag _gat > ../../results/rebuttal/exp3_${g}_gat.log 2>&1
done
echo "queue done"
