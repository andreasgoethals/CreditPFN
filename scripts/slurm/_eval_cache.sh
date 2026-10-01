#!/usr/bin/env bash
# Share published control artifacts without changing scientific cache identities.
# A hard link costs no second prediction payload; atomic Python writers replace
# directory entries, so future writes cannot modify a donor's published bytes.
share_eval_controls() {
    local cache_dir="${1:-}" root donor source target name linked=0
    if [[ -z "$cache_dir" ]]; then
        cache_dir="$(python -c 'from src.utils.paths import results_dir; print(results_dir().parent / "evaluation_cache")')" || return
    fi
    cache_dir="${cache_dir%$'\r'}"
    [[ "$(basename "$cache_dir")" == evaluation_cache ]] || {
        echo 'Expected an experiment evaluation_cache directory.' >&2; return 1;
    }
    root="$(dirname "$(dirname "$cache_dir")")"
    # Experiment 0 has separate controls and is deliberately excluded.
    case "$(basename "$(dirname "$cache_dir")")" in
        experiment1|experiment2|experiment3) ;;
        *) return 0 ;;
    esac
    mkdir -p -- "$cache_dir" || return
    for donor in "$root"/experiment{1,2,3}/evaluation_cache; do
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
