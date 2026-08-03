#!/usr/bin/env bash
# Multi-seed statistical study for the HAC main line reported in the paper
# (Table 3-4's HAC-Fixed row, and the branch-contribution ablation's
# learned-only / heuristic-only rows).
#
# SCOPE NOTE: this repository covers only the HAC main line. The GAC-Fixed
# and SAC-Fixed baseline rows that also appear in the paper's main results
# tables (Tables 3-4) were produced by a separate motivating-experiments
# codebase not included in this release; their numbers are reported in the
# paper verbatim from the original experimental logs. See README.md.
#
# RESUMABLE: if an output .pt or .csv file already exists, that step is
# skipped -- safe to Ctrl-C and re-run this script later to pick up where
# it left off.
#
# Recommended: run inside `tmux` or with `nohup ... &` so it survives an
# SSH disconnect, e.g.:
#   nohup bash scripts/run_full_multiseed_study.sh > multiseed_log.txt 2>&1 &
#
# Usage: bash scripts/run_full_multiseed_study.sh

set -uo pipefail   # no -e: one failed run should not kill the whole sweep
SEEDS="1 2 3 4 5"
DEPLOYMENTS="random uniform"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

run_if_missing () {
    local desc="$1"; shift
    local outfile="$1"; shift
    if [ -f "$outfile" ]; then
        echo "[skip] $desc -> $outfile already exists"
    else
        echo "[run ] $desc -> $outfile"
        "$@"
    fi
}

for dep in $DEPLOYMENTS; do
    for seed in $SEEDS; do

        # --- HAC-Fixed (paper Tables 3-4 main row) ---
        ckpt="$RESULTS_DIR/hac_fixed_${dep}_seed${seed}.pt"
        csv="$RESULTS_DIR/eval_hac_fixed_${dep}_seed${seed}.csv"
        run_if_missing "HAC-Fixed train ($dep, seed=$seed)" "$ckpt" \
            python scripts/train_hybrid.py --variant fixed --deployment "$dep" \
                --seed "$seed" --out "$ckpt" --epochs 300
        run_if_missing "HAC-Fixed eval  ($dep, seed=$seed)" "$csv" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt" --deployment "$dep" --out "$csv"

        # --- Learned-only ablation (force_wT=1.0; branch-contribution ablation) ---
        ckpt="$RESULTS_DIR/learned_only_${dep}_seed${seed}.pt"
        csv="$RESULTS_DIR/eval_learned_only_${dep}_seed${seed}.csv"
        run_if_missing "Learned-only train ($dep, seed=$seed)" "$ckpt" \
            python scripts/train_hybrid.py --variant fixed --deployment "$dep" \
                --force_wT 1.0 --seed "$seed" --out "$ckpt" --epochs 300
        run_if_missing "Learned-only eval  ($dep, seed=$seed)" "$csv" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt" --deployment "$dep" --out "$csv"

        # --- Heuristic-only ablation (no training; branch-contribution ablation) ---
        csv="$RESULTS_DIR/eval_heuristic_only_${dep}_seed${seed}.csv"
        run_if_missing "Heuristic-only eval ($dep, seed=$seed)" "$csv" \
            python scripts/evaluate_heuristic_only.py --deployment "$dep" --seed "$seed" --out "$csv"

    done
done

echo ""
echo "=== HAC main-line multi-seed sweep complete (or resumed to completion). ==="
echo "Run: python scripts/aggregate_multiseed.py   to build the mean +/- std summary table."
echo "(GAC-Fixed / SAC-Fixed rows in the paper's Tables 3-4 require a separate"
echo " motivating-experiments codebase, not included in this repository -- see README.md.)"
