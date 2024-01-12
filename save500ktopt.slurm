#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=savecem500kpt
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000M
#SBATCH --time=24:00:00
#SBATCH --output=/home/codee/scratch/mycode/500k_test_%j_%N.txt

source /home/codee/miniconda3/etc/profile.d/conda.sh
conda activate base

log_dir="/home/codee/scratch/servercheckpoint"
echo "start"

python /home/codee/scratch/mycode/savecem500kimage.py

echo "finished"