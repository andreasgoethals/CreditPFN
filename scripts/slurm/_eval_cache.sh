#!/usr/bin/env bash
# Share published control artifacts without changing scientific cache identities.
# A hard link costs no second prediction payload; atomic Python writers replace
# directory entries, so future writes cannot modify a donor's published bytes.
share_eval_controls() {
    local cache_dir="${1:-}" root donor source target name linked=0 allowed=0
    local -a donors
    if [[ -z "$cache_dir" ]]; then
        cache_dir="$(python -c 'from src.utils.paths import results_dir; print(results_dir().parent / "evaluation_cache")')" || return
    fi
    cache_dir="${cache_dir%$'\r'}"
    [[ "$(basename "$cache_dir")" == evaluation_cache ]] || {
        echo 'Expected an experiment evaluation_cache directory.' >&2; return 1;
    }
    root="$(dirname "$(dirname "$cache_dir")")"
    # Original-model controls are identical across second-campaign experiments (the slugs of
    # src/utils/paths.py CAMPAIGN2_GROUPS, tested); Experiment 0 stays isolated.
    for donor in reference seed domain_control target_finetune update_size_anchoring scope \
            exposure table_size episode_size context transfer precision; do
        donors+=("$root/$donor/evaluation_cache")
    done
    for donor in "${donors[@]}"; do [[ "$cache_dir" != "$donor" ]] || allowed=1; done
    (( allowed )) || return 0
    mkdir -p -- "$cache_dir" || return
    for donor in "${donors[@]}"; do
        [[ "$donor" != "$cache_dir" && -d "$donor" ]] || continue
        for source in "$donor"/*.json.gz; do
            [[ -f "$source" && ! -L "$source" ]] || continue
            name="$(basename "$source")"
            [[ "$name" =~ ^[0-9a-f]{64}\.json\.gz$ ]] || continue
            target="$cache_dir/$name"
            [[ ! -e "$target" && ! -L "$target" ]] || continue
            if ln -- "$source" "$target" 2>/dev/null; then
                linked=$(( linked + 1 ))
            elif [[ ! -f "$target" ]]; then
                echo "Cannot share control cache on project storage: $target" >&2
                return 1
            fi
        done
    done
    echo "Control cache: shared $linked published artifacts; evaluation still requires matching fingerprints and complete folds/predictions."
}
