#!/bin/bash
# Baseline test with gpt-4o-mini (filtered by guideline for fair comparison)

source /opt/anaconda/etc/profile.d/conda.sh
conda activate medical-AI

cd /users/rbao5/codename_goodboidoge

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Starting Baseline Diagnosis Test with gpt-4o-mini (filtered by guideline)..."

python -u baseline_diagnosis_test.py \
    --data medqa_train_min2.jsonl \
    --max-cases 200 \
    --model gpt-4o-mini \
    --filter-by-guideline \
    --output baseline_gpt4omini_filtered_results.txt \
    2>&1 | tee baseline_gpt4omini_filtered.log
