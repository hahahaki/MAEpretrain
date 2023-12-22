#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=MAESTER_cem500k
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200000M
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

log_dir="/home/codee/scratch/servercheckpoint"
echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir

#echo "$SLURM_NODEID master: $MASTER_ADDR"
echo "$SLURM_NODEID Launching python script"
#echo "$SLURM_NTASKS tasks running"

/home/codee/miniconda3/bin/python /home/codee/scratch/mycode/train.py --logdir $log_dir \
  --model_config_dir $model_config_dir \
  --model_config_name $config_name > $log_dir/output12_22

echo "Python script finished"