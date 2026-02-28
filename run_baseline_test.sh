#!/bin/bash
source /opt/anaconda/etc/profile.d/conda.sh
conda activate medical-AI
cd /users/rbao5/codename_goodboidoge
# 使用 -u 禁用 Python 输出缓冲
python -u baseline_diagnosis_test.py --data medqa_train_min2.jsonl --max-cases 200 --model gpt-5-nano --temperature 1.0 --output baseline_results_gpt5nano_200.txt 2>&1 | tee baseline_test_gpt5nano.log
