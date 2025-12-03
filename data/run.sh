#!/bin/bash
#SBATCH -p aisc                  # Partition name
#SBATCH -N 1                      # Number of nodes
#SBATCH --cpus-per-task=4       # Number of CPU cores
#SBATCH --mem=8G         # Memory allocation
#SBATCH --time=3:00:00           # Max runtime (96 hours)
#SBATCH --job-name=dget         # Optional: Name your job
#SBATCH --output=dget-%j.log   # Standard output log file
#SBATCH --error=dget-%j.log     # Error log file

export UV_CACHE_DIR=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache/.uv_cache
export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache/

cd /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/trl/
source /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/trl/grpo/bin/activate
uv pip install openai==0.28
cd /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/OpenCE/data

python medqa_agentclinic_gen.py
pip install -U openai