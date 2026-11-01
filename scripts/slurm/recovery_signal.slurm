#!/bin/bash -l
#SBATCH --job-name=creditpfn-recovery-signal
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
unset CREDITPFN_CONFIG CREDITPFN_LEGACY_ID CREDITPFN_EVAL_REFRESH CREDITPFN_RUNTIME_ACCEPTANCE
source scripts/slurm/_job_log.sh
start_job_log "recovery_signal_${1:?}"
source scripts/slurm/_activate_env.sh
folder="${CREDITPFN_OWNED_CACHE:?}/signal-probe"
mkdir -p "$folder"
source scripts/slurm/_run_train.sh
run_training_python -m src.utils.signal_probe run "$folder" --arm baseline
parent=$$
for warning in USR1 TERM; do
    rm -f -- "$folder/ready"
    (
        for ((i=0; i<2400; i++)); do
            if [[ -f "$folder/ready" ]]; then kill "-$warning" "$parent"; exit 0; fi
            sleep .1
        done
        exit 2
    ) &
    sender=$!
    if run_training_python -m src.utils.signal_probe run "$folder" --arm "$warning"; then rc=0; else rc=$?; fi
    wait "$sender"
    [[ "$rc" == 75 ]] || { echo "Warning did not preserve recovery exit: $rc"; exit 2; }
    run_training_python -m src.utils.signal_probe run "$folder" --arm "$warning"
done
python -m src.utils.signal_probe inspect "$folder" --receipt "${2:?receipt required}"
