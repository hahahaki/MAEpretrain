#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unzipcem500k
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=127000M
#SBATCH --time=10:00:00

#export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
#export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

module load python/3.11
module load cuda/11.4
source /home/codee/miniconda3/etc/profile.d/conda.sh
conda activate base

config="default"
config_name=$config.yaml
model_config_dir="/home/codee/scratch/mycode"

unzip /home/codee/scratch/dataset/cem500k.zip -d /home/codee/scratch/dataset
echo "finished"