#!/bin/bash
# ACE + Playbook test with gpt-5-nano (fixed max_output_tokens=4096)

source /opt/anaconda/etc/profile.d/conda.sh
conda activate medical-AI

cd /users/rbao5/codename_goodboidoge

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export OPENAI_MODEL=gpt-5-nano

# Clear previous log to start fresh
> ace_gpt5nano_training.log

echo "Starting ACE + Playbook test with gpt-5-nano (max_output_tokens=4096)..."
echo "Log file: ace_gpt5nano_training.log"

python -u offline_medical_adaptation.py \
    --data medqa_train_min2.jsonl \
    --max-cases 200 \
    --epochs 3 \
    --temperature 1.0 \
    --max-retries 5 \
    2>&1 | tee -a ace_gpt5nano_training.log
