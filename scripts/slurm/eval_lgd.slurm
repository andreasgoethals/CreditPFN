#!/bin/bash -l
# CreditPFN eval: use the named-config run_experiment.sh launcher.
# GPFS inputs; project weights/results; live DATA shards, consolidated post-run.
# Array sizing, concurrency and walltime are supplied by the launcher.
#SBATCH --job-name=creditpfn-lgd-eval
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --array=0-0%1

set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
source scripts/slurm/_eval_job.sh lgd
