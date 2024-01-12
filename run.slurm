#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=MAESTER_cem500k
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000M
#SBATCH --time=1:00:00
#SBATCH --output=//home/codee/scratch/mycode/pretrainMAE_500k_%j_%N.txt

#export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
#export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

#module load python/3.10
module load cuda/11.4
#virtualenv --no-download $SLURM_TMPDIR/env
source /home/codee/envir/bin/activate
#source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index --upgrade pip
#pip install --no-index -r /home/codee/envir/requirements.txt

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
  --world_size $SLURM_NTASKS \
  --model_config_dir $model_config_dir \
  --model_config_name $config_name > $log_dir/test1_11

echo "Python script finished"