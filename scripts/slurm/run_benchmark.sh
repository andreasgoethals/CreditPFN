#!/usr/bin/env bash
# One benchmark phase: submit both resource arrays for both tasks and all partitions.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXPERIMENT="${1:?usage: run_benchmark.sh experiment1|experiment2|experiment3|experiment5|experiment6|experiment7}"
case "$EXPERIMENT" in experiment[1-3]|experiment[5-7]) ;; *) echo 'Expected experiment1–3 or experiment5–7' >&2; exit 2 ;; esac
CONFIG_COUNT=2
[[ "$EXPERIMENT" == experiment6 ]] && CONFIG_COUNT=1
export CREDITPFN_EXPERIMENT="$EXPERIMENT"
export EVAL_TASKS="${EVAL_TASKS:-16}"
export CREDITPFN_USE_SCRATCH="${CREDITPFN_USE_SCRATCH:-1}"
GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-16}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"
MAX_QUEUED="${MAX_QUEUED:-450}"
(( EVAL_TASKS > 0 && EVAL_CONCURRENCY > 0 && EVAL_CONCURRENCY <= GLOBAL_CONCURRENCY )) || exit 2
mapfile -t META < <(python - "$EXPERIMENT" "${DRY:-}" <<'PY'
import sys
from pathlib import Path
import pandas as pd
from src.train.config import load_train_config
from src.utils.prepare_experiment import check_prepared
from src.utils.consolidate_output import latest_trials
from src.utils.paths import manifests_dir, checkpoints_dir
from src.visualize.campaign import planned_trials
experiment, dry = sys.argv[1:]
splits = []
names = (['pd.yaml'] if experiment == 'experiment6' else ['pd_rows2048.yaml', 'pd_rows26000.yaml']
         if experiment == 'experiment7' else ['pd.yaml', 'lgd.yaml'])
paths = [Path('config') / experiment / name for name in names]
# The shared anchor is benchmarked once in experiment 6, not retrained/copied here.
if experiment == 'experiment7' and not dry:
    paths.append(Path('config/experiment6/pd.yaml'))
for path in paths:
    cfg = load_train_config(config_path=str(path))
    track = str(cfg.track)
    owner = str(cfg.experiment.output_group)
    splits.append(int(cfg.corpus.n_splits))
    if dry:
        continue
    check_prepared(path, stage='eval')
    frames = []
    for file in sorted(manifests_dir(owner).glob(f'{cfg.run_name}_s*_{track}.csv')):
        frame = pd.read_csv(file).assign(source_file=file.name)
        frame['source_row'] = range(len(frame))
        frames.append(frame)
    observed = latest_trials(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    if observed.empty:
        raise RuntimeError(f'{track.upper()}: no training outcomes; finish training before benchmarking')
    completed = observed[observed.status.eq('OK')]
    names = {Path(str(p)).stem for p in completed.final_ckpt_path}
    expected = set(planned_trials(cfg).trial_name)
    missing = expected - names
    missing_weights = [n for n in expected if not (checkpoints_dir('trained', owner, track) / (n+'.ckpt')).is_file()]
    if missing or missing_weights:
        raise RuntimeError(f'{track.upper()}: {len(missing)} incomplete trials, {len(missing_weights)} absent checkpoints; finish training first')
if len(set(splits)) != 1:
    raise RuntimeError('All benchmark configurations must have the same partition count')
print(splits[0])
PY
)
[[ ${#META[@]} == 1 ]] || { echo 'Benchmark readiness check failed; no jobs submitted.' >&2; exit 1; }
export CREDITPFN_BENCHMARK_SPLITS="${META[0]%$'\r'}"
N=$(( CONFIG_COUNT * CREDITPFN_BENCHMARK_SPLITS * EVAL_TASKS ))
GPU_CLUSTER="${BENCHMARK_GPU_CLUSTER:-mindwell}"
CPU_CLUSTER="${BENCHMARK_CPU_CLUSTER:-wice}"
# Check room for the complete phase before submitting either resource array.
if [[ -z "${DRY:-}" ]]; then
    for cluster in "$GPU_CLUSTER" "$CPU_CLUSTER"; do
        queued=$(timeout --kill-after=5s 45s squeue -M "$cluster" -u "$USER" -h -r -o '%i' | wc -l)
        needed=$N
        [[ "$GPU_CLUSTER" == "$CPU_CLUSTER" ]] && needed=$(( 2*N ))
        (( queued + needed <= MAX_QUEUED )) || { echo "$cluster lacks queue room for the complete phase. No jobs submitted; retry later." >&2; exit 1; }
    done
fi
for kind in foundation classical; do
    cluster=$GPU_CLUSTER; partition="${BENCHMARK_GPU_PARTITION:-gpu_b200}"
    [[ "$kind" == classical ]] && { cluster=$CPU_CLUSTER; partition="${BENCHMARK_CPU_PARTITION:-batch}"; }
    cmd=(sbatch --parsable --clusters="$cluster" --partition="$partition"
         --account="${CREDITPFN_ACCOUNT:-lp_verbekelab}" --array="0-$((N-1))%$EVAL_CONCURRENCY"
         --time="${EVAL_WALLTIME:-04:00:00}"
         --export="ALL,CREDITPFN_EXPERIMENT=$EXPERIMENT,CREDITPFN_BENCHMARK_SPLITS=$CREDITPFN_BENCHMARK_SPLITS,EVAL_TASKS=$EVAL_TASKS"
         "scripts/slurm/benchmark_${kind}.slurm")
    if [[ -n "${DRY:-}" ]]; then printf 'DRY:'; printf ' %q' "${cmd[@]}"; printf '\n';
    else
        out=$(python -m src.utils.submit_bounded --slots "$GLOBAL_CONCURRENCY" --limit "$EVAL_CONCURRENCY" --cluster "$cluster" -- "${cmd[@]}")
        echo "Submitted $kind benchmark array $out ($N tasks: $CONFIG_COUNT configurations, all partitions)."
    fi
done
