#!/bin/bash
#SBATCH -p aisc                  # Partition name
#SBATCH -N 1                      # Number of nodes
#SBATCH --cpus-per-task=4       # Number of CPU cores
#SBATCH --mem=8G         # Memory allocation
#SBATCH --time=3:00:00           # Max runtime (96 hours)
#SBATCH --job-name=context         # Optional: Name your job
#SBATCH --output=context-%j.log   # Standard output log file
#SBATCH --error=context-%j.log     # Error log file

export UV_CACHE_DIR=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache/.uv_cache
export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache/

cd /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/trl/
source /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/trl/grpo/bin/activate

export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL="gpt-4.1"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选
# python your_script.py


cd /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/OpenCE/
python offline_medical_adaptation.py