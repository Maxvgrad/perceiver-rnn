#!/bin/bash
# NOTE: before running this do `conda activate vista`.
# Example 'sbatch evaluate_slurm.sh --model pilotnet.onnx --traces trace_forward --traces-root ./traces/ --save-video'
# run as `sbatch vista_evaluate.sh`

# Job Details
#SBATCH --partition=gpu
#SBATCH -J vista
#SBATCH --output=./vista_runs/%j-slurm-run.txt 
#SBATCH --error=./vista_runs/%j-slurm-run.txt

# Resources
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2

# Actual job command(s)
module load ffmpeg
srun python -u vista_eval/evaluate.py "$@"
