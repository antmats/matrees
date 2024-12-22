#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-285
#SBATCH -p tetralith
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 1-0:0  # days-hours:minutes
#SBATCH --output=/proj/healthyai/matrees/logs/%x_%A_%a.out

container_path=/proj/healthyai/matrees/matrees_env.sif

input_file=$1
eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`

cd ~
rsync -r matrees $TMPDIR
cd $TMPDIR/matrees

if [ -z "$n_features" ]; then
    n_features=20
fi

apptainer exec --bind "/proj:/proj,${TMPDIR}:/mnt" $container_path python main.py \
    --estimator_alias "$estimator_alias" \
    --n_train $n_train \
    --n_test $n_test \
    --n_features $n_features \
    --max_depth $max_depth \
    --alpha $alpha \
    --seed $seed \
    --output_dir $output_dir \
    --task_id $SLURM_ARRAY_TASK_ID
