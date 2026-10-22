#!/usr/bin/env bash
# Complete 5-seed reproduction and supplemental experiment suite:
#   1. Main architectures (HAC-Fixed, HAC-Smooth [k=5,s0=0.7], Learned-only, Heuristic-only, SAC-Fixed)
#   2. Schedule steepness 5-seed sweep (Table 6: k in {30, 10, 5}, s0 in {0.125, 0.5, 0.7})
#   3. HSC subnetwork-level counter-experiment 5-seed study (Table 7)
#   4. Train-Eval Schedule Decoupling study (verifying Section 6.2 curriculum hypothesis)

set -uo pipefail

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
        # 1. HAC-Fixed (Main Table)
        ckpt="$RESULTS_DIR/hac_fixed_${dep}_seed${seed}.pt"
        csv="$RESULTS_DIR/eval_hac_fixed_${dep}_seed${seed}.csv"
        run_if_missing "HAC-Fixed train ($dep, seed=$seed)" "$ckpt" \
            python scripts/train_hybrid.py --variant fixed --granularity global --deployment "$dep" \
                --seed "$seed" --out "$ckpt" --epochs 300
        run_if_missing "HAC-Fixed eval ($dep, seed=$seed)" "$csv" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt" --deployment "$dep" --seed "$seed" --out "$csv"

        # 2. HAC-Smooth (k=5, s0=0.7) (Main Table & Schedule Sweep)
        ckpt_smooth="$RESULTS_DIR/hac_smooth_k5_s07_${dep}_seed${seed}.pt"
        csv_smooth="$RESULTS_DIR/eval_hac_smooth_k5_s07_${dep}_seed${seed}.csv"
        run_if_missing "HAC-Smooth train ($dep, seed=$seed)" "$ckpt_smooth" \
            python scripts/train_hybrid.py --variant adaptive --granularity global --deployment "$dep" \
                --awf_k 5.0 --awf_s0 0.7 --seed "$seed" --out "$ckpt_smooth" --epochs 300
        run_if_missing "HAC-Smooth eval ($dep, seed=$seed)" "$csv_smooth" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt_smooth" --deployment "$dep" --seed "$seed" --out "$csv_smooth"

        # 3. Learned-only (force_wT=1.0)
        ckpt_lo="$RESULTS_DIR/learned_only_${dep}_seed${seed}.pt"
        csv_lo="$RESULTS_DIR/eval_learned_only_${dep}_seed${seed}.csv"
        run_if_missing "Learned-only train ($dep, seed=$seed)" "$ckpt_lo" \
            python scripts/train_hybrid.py --variant fixed --granularity global --deployment "$dep" \
                --force_wT 1.0 --seed "$seed" --out "$ckpt_lo" --epochs 300
        run_if_missing "Learned-only eval ($dep, seed=$seed)" "$csv_lo" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt_lo" --deployment "$dep" --seed "$seed" --out "$csv_lo"

        # 4. Heuristic-only
        csv_ho="$RESULTS_DIR/eval_heuristic_only_${dep}_seed${seed}.csv"
        run_if_missing "Heuristic-only eval ($dep, seed=$seed)" "$csv_ho" \
            python scripts/evaluate_heuristic_only.py --deployment "$dep" --seed "$seed" --out "$csv_ho"

        # 5. SAC-Fixed (subnetwork-level learned-only, force_wT=1.0)
        ckpt_sac="$RESULTS_DIR/sac_fixed_${dep}_seed${seed}.pt"
        csv_sac="$RESULTS_DIR/eval_sac_fixed_${dep}_seed${seed}.csv"
        run_if_missing "SAC-Fixed train ($dep, seed=$seed)" "$ckpt_sac" \
            python scripts/train_hybrid.py --variant fixed --granularity subnetwork --deployment "$dep" \
                --force_wT 1.0 --seed "$seed" --out "$ckpt_sac" --epochs 300
        run_if_missing "SAC-Fixed eval ($dep, seed=$seed)" "$csv_sac" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt_sac" --deployment "$dep" --seed "$seed" --out "$csv_sac"

        # 6. HSC Counter-Experiment (fixed, steep default, and smooth)
        ckpt_hsc_fix="$RESULTS_DIR/hsc_fixed_${dep}_seed${seed}.pt"
        csv_hsc_fix="$RESULTS_DIR/eval_hsc_fixed_${dep}_seed${seed}.csv"
        run_if_missing "HSC-Fixed train ($dep, seed=$seed)" "$ckpt_hsc_fix" \
            python scripts/train_hybrid.py --variant fixed --granularity subnetwork --deployment "$dep" \
                --seed "$seed" --out "$ckpt_hsc_fix" --epochs 300
        run_if_missing "HSC-Fixed eval ($dep, seed=$seed)" "$csv_hsc_fix" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt_hsc_fix" --deployment "$dep" --seed "$seed" --out "$csv_hsc_fix"

        ckpt_hsc_def="$RESULTS_DIR/hsc_adaptive_k30_s0125_${dep}_seed${seed}.pt"
        csv_hsc_def="$RESULTS_DIR/eval_hsc_adaptive_k30_s0125_${dep}_seed${seed}.csv"
        run_if_missing "HSC-Adaptive default train ($dep, seed=$seed)" "$ckpt_hsc_def" \
            python scripts/train_hybrid.py --variant adaptive --granularity subnetwork --deployment "$dep" \
                --awf_k 30.0 --awf_s0 0.125 --seed "$seed" --out "$ckpt_hsc_def" --epochs 300
        run_if_missing "HSC-Adaptive default eval ($dep, seed=$seed)" "$csv_hsc_def" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt_hsc_def" --deployment "$dep" --seed "$seed" --out "$csv_hsc_def"

        ckpt_hsc_sm="$RESULTS_DIR/hsc_smooth_k5_s07_${dep}_seed${seed}.pt"
        csv_hsc_sm="$RESULTS_DIR/eval_hsc_smooth_k5_s07_${dep}_seed${seed}.csv"
        run_if_missing "HSC-Smooth train ($dep, seed=$seed)" "$ckpt_hsc_sm" \
            python scripts/train_hybrid.py --variant adaptive --granularity subnetwork --deployment "$dep" \
                --awf_k 5.0 --awf_s0 0.7 --seed "$seed" --out "$ckpt_hsc_sm" --epochs 300
        run_if_missing "HSC-Smooth eval ($dep, seed=$seed)" "$csv_hsc_sm" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt_hsc_sm" --deployment "$dep" --seed "$seed" --out "$csv_hsc_sm"
    done
done

# 7. Remaining Schedule Steepness Sweep on Random Deployment (5 seeds) + Train-Eval Decoupling
for seed in $SEEDS; do
    for cfg_pair in "30.0 0.125 k30_s0125" "10.0 0.125 k10_s0125" "30.0 0.5 k30_s05" "10.0 0.5 k10_s05"; do
        set -- $cfg_pair
        k_val="$1"; s0_val="$2"; tag="$3"
        ckpt="$RESULTS_DIR/hac_adaptive_${tag}_random_seed${seed}.pt"
        csv="$RESULTS_DIR/eval_hac_adaptive_${tag}_random_seed${seed}.csv"
        run_if_missing "HAC-Adaptive $tag train (random, seed=$seed)" "$ckpt" \
            python scripts/train_hybrid.py --variant adaptive --granularity global --deployment random \
                --awf_k "$k_val" --awf_s0 "$s0_val" --seed "$seed" --out "$ckpt" --epochs 300
        run_if_missing "HAC-Adaptive $tag eval (random, seed=$seed)" "$csv" \
            python scripts/evaluate_hybrid.py --ckpt "$ckpt" --deployment random --seed "$seed" --out "$csv"
    done

    # Train-Eval Decoupling: evaluate Steep-trained and Smooth-trained checkpoints with Fixed 0.5 test-time weight
    ckpt_steep="$RESULTS_DIR/hac_adaptive_k30_s0125_random_seed${seed}.pt"
    csv_steep_eval_fix="$RESULTS_DIR/eval_decouple_trainSteep_evalFixed_random_seed${seed}.csv"
    run_if_missing "Decouple Train-Steep Eval-Fixed (seed=$seed)" "$csv_steep_eval_fix" \
        python scripts/evaluate_hybrid.py --ckpt "$ckpt_steep" --deployment random \
            --eval_variant fixed --seed "$seed" --out "$csv_steep_eval_fix"

    ckpt_smooth="$RESULTS_DIR/hac_smooth_k5_s07_random_seed${seed}.pt"
    csv_smooth_eval_fix="$RESULTS_DIR/eval_decouple_trainSmooth_evalFixed_random_seed${seed}.csv"
    run_if_missing "Decouple Train-Smooth Eval-Fixed (seed=$seed)" "$csv_smooth_eval_fix" \
        python scripts/evaluate_hybrid.py --ckpt "$ckpt_smooth" --deployment random \
            --eval_variant fixed --seed "$seed" --out "$csv_smooth_eval_fix"
done

echo "=== Full multi-seed & ablation suite complete. Run: python scripts/aggregate_multiseed.py ==="
