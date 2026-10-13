#!/usr/bin/env bash
# Map the combined array index to configuration, dataset partition and evaluation shard.
case "${CREDITPFN_EXPERIMENT:-experiment1}" in
 experiment6) configs=(pd.yaml) ;;
 experiment7) configs=(pd_rows2048.yaml pd_rows26000.yaml) ;;
 *) configs=(pd.yaml lgd.yaml) ;;
esac
total_per_config=$(( ${CREDITPFN_BENCHMARK_SPLITS:?} * ${EVAL_TASKS:?} ))
index="${SLURM_ARRAY_TASK_ID:?}"
(( index >= 0 && index < ${#configs[@]}*total_per_config )) || { echo 'Invalid benchmark array index' >&2; exit 2; }
config="${configs[$(( index / total_per_config ))]}"
TRACK="${config%%[_.]*}"
local_index=$(( index % total_per_config ))
export CREDITPFN_SPLIT_INDEX=$(( local_index / EVAL_TASKS ))
export CREDITPFN_EVAL_TASK_INDEX=$(( local_index % EVAL_TASKS ))
export CREDITPFN_CONFIG="$VSC_DATA/CreditPFN/config/${CREDITPFN_EXPERIMENT:?}/$config"
