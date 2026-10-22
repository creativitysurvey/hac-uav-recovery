"""
Aggregate multi-seed CSVs into mean +/- std summary tables across all architectures and ablations.
"""

import argparse
import os
import pandas as pd

ARCHS = [
    "hac_fixed",
    "hac_smooth_k5_s07",
    "learned_only",
    "heuristic_only",
    "sac_fixed",
    "hsc_fixed",
    "hsc_adaptive_k30_s0125",
    "hsc_smooth_k5_s07",
    "hac_adaptive_k30_s0125",
    "hac_adaptive_k10_s0125",
    "hac_adaptive_k30_s05",
    "hac_adaptive_k10_s05",
    "decouple_trainSteep_evalFixed",
    "decouple_trainSmooth_evalFixed",
]
DEPLOYMENTS = ["random", "uniform"]


def load_seed_csvs(results_dir, arch, deployment, seeds):
    dfs = []
    for seed in seeds:
        path = os.path.join(results_dir, f"eval_{arch}_{deployment}_seed{seed}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def summarize(df):
    metrics = ["REC_mean", "COV_mean", "AND_mean", "n_valid_of_50"]
    rows = []
    for fr, g in df.groupby("failure_ratio"):
        row = {"failure_ratio": fr, "n_seeds": g["seed"].nunique()}
        for m in metrics:
            vals = g[m].dropna().values
            if len(vals) == 0:
                row[f"{m}_agg"] = "n/a"
            else:
                row[f"{m}_agg"] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main(args):
    combined_rows = []
    for arch in ARCHS:
        for dep in DEPLOYMENTS:
            df = load_seed_csvs(args.results_dir, arch, dep, args.seeds)
            if df is None:
                continue
            summary = summarize(df)
            key = f"{arch}_{dep}"
            print(f"\n=== {arch} / {dep} (n={df['seed'].nunique()} seeds) ===")
            print(summary.to_string(index=False))
            out_path = os.path.join(args.results_dir, f"summary_{key}.csv")
            summary.to_csv(out_path, index=False)

            s = summary.copy()
            s.insert(0, "deployment", dep)
            s.insert(0, "architecture", arch)
            combined_rows.append(s)

    if not combined_rows:
        print("\nNo results found yet. Run scripts/run_full_multiseed_study.sh first.")
        return

    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = os.path.join(args.results_dir, "summary_multiseed_all.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n[saved] combined summary -> {combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = p.parse_args()
    main(args)
