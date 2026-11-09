#!/bin/bash -l
# Second-campaign evaluation tasks: campaign_eval_gpu.slurm <context|finetune|episode_context>, one task
# per element. Task index = TASK_OFFSET + array index (array indices stay below MaxArraySize 1001).
# Count: python -m src.eval.campaign_tasks count <kind>. Requires the E1 reference checkpoints.
#SBATCH --job-name=creditpfn-campaign-eval
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
kind="${1:?context, finetune or episode_context}"
: "${CREDITPFN_RUNTIME_VERSION:?accepted runtime required}"
case "$kind" in
    context) export CREDITPFN_EXPERIMENT=context ;;
    finetune) export CREDITPFN_EXPERIMENT=target_finetune ;;
    episode_context) export CREDITPFN_EXPERIMENT=episode_size ;;
    *) echo "Unknown evaluation kind: $kind" >&2; exit 2 ;;
esac
task=$(( ${TASK_OFFSET:-0} + ${SLURM_ARRAY_TASK_ID:?} ))
export CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1 PYTHONUNBUFFERED=1
unset CREDITPFN_CONFIG
source scripts/slurm/_job_log.sh
start_job_log "campaign_${kind}_${task}"
source scripts/slurm/_activate_env.sh
python -m src.eval.campaign_tasks run "$kind" --task "$task"
