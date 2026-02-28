#!/bin/bash
source /opt/anaconda/etc/profile.d/conda.sh
conda activate medical-AI
cd /users/rbao5/codename_goodboidoge

# 加载 .env 文件
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 从 checkpoint 恢复
python offline_medical_adaptation.py --data medqa_train_min2.jsonl --max-cases 200 --epochs 3 --resume output/playbooks/run_20251211_050918/checkpoint.pkl 2>&1 | tee -a training_output.log
