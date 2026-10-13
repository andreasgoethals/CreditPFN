#!/bin/bash -l
#SBATCH --job-name=creditpfn-benchmark-cpu
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
source scripts/slurm/_benchmark_dispatch.sh
export CREDITPFN_USE_SCRATCH=0 CREDITPFN_EVAL_KIND=classical
source scripts/slurm/_eval_job.sh "$TRACK"
