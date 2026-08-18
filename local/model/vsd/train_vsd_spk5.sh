#!/bin/bash
#SBATCH -J vsd5-train
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=vsd5-train-%j.out
#SBATCH --error=vsd5-train-%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate avsd

cd /home/speech-audio-research/22b3965/AVSD/local/model/vsd

python train_vsd_spk5.py \
  --epochs 100 \
  --batch-size 16 \
  --num-workers 4 \
  --grad-accum-steps 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --loss-type focal \
  --pos-weight 4.0 \
  --focal-gamma 2.0 \
  --focal-alpha 0.75 \
  --max-speakers 8 \
  --max-session-speakers 5 \
  --checkpoint-dir checkpoints
