#!/usr/bin/env bash
# Bootstrap mapping before a Python environment exists. Regression-checked against
# config/study.yaml; internal CREDITPFN_EXPERIMENT retains checkpoint campaign IDs.
study_campaign() {
    case "$1" in
        experiment0|experiment1|experiment2|experiment5) echo "$1" ;;
        experiment3|experiment4) echo experiment1 ;;
        experiment6) echo experiment8 ;;
        experiment7) echo experiment3 ;;
        experiment8) echo experiment6 ;;
        experiment9) echo experiment7 ;;
        experiment10) echo experiment4 ;;
        *) echo 'Expected public experiment0–10' >&2; return 2 ;;
    esac
}
study_config_group() {
    case "$1" in
        experiment0|experiment1|experiment2|experiment5) echo "$1" ;;
        experiment3) echo experiment7 ;;
        experiment4) echo experiment10 ;;
        experiment6) echo experiment8 ;;
        experiment7) echo experiment9 ;;
        experiment8) echo experiment6 ;;
        *) echo 'Unknown recorded campaign' >&2; return 2 ;;
    esac
}
