#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-480
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 1-0:0  # days-hours:minutes
#SBATCH --output=/mimer/NOBACKUP/groups/inpole/matrees/logs/%x_%A_%a.out

container_path=/mimer/NOBACKUP/groups/inpole/matrees/matrees_env.sif

input_file=$1
eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`

cd ~
rsync -r matrees $TMPDIR
cd $TMPDIR/matrees

apptainer exec --bind "${TMPDIR}:/mnt" $container_path python main.py \
    --estimator_alias "$estimator_alias" \
    --n_train $n_train \
    --n_test $n_test \
    --max_depth $max_depth \
    --alpha $alpha \
    --seed $seed \
    --output_dir $output_dir \
    --task_id $SLURM_ARRAY_TASK_ID
