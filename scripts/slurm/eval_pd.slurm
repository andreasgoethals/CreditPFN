#!/bin/bash -l
# CreditPFN eval: use the named-config run_experiment.sh launcher.
# GPFS inputs; project weights/results; live DATA shards, consolidated post-run.
# Array sizing, concurrency and walltime are supplied by the launcher.
#SBATCH --job-name=creditpfn-pd-eval
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --array=0

set -euo pipefail
# The checkout that submitted the job (a second DATA checkout runs its own code).
cd "${SLURM_SUBMIT_DIR:?}"
test -f scripts/slurm/run_experiment.sh || { echo "Submit from the CreditPFN repository root, not $PWD" >&2; exit 2; }
source scripts/slurm/_eval_job.sh pd
