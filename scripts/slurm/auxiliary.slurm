#!/bin/bash -l
# One base/task per GPU, short seed and sampling controls in sequence.
#SBATCH --job-name=creditpfn-auxiliary
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1
source scripts/slurm/_job_log.sh
start_job_log auxiliary
source scripts/slurm/_activate_env.sh
python -u -m src.utils.auxiliary_check --id "${CREDITPFN_FLOW_ID:?}" --task "${SLURM_ARRAY_TASK_ID:?}"
