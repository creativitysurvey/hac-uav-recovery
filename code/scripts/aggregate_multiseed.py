"""
Aggregate the multi-seed sweep results (scripts/run_full_multiseed_study.sh) into
mean +/- std summary tables, one per (architecture, deployment), ready to paste
into a paper table.

Usage:
  python scripts/aggregate_multiseed.py
  python scripts/aggregate_multiseed.py --results_dir results --seeds 1 2 3 4 5
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ARCHS = ["hac_fixed", "learned_only", "heuristic_only"]
# NOTE: the paper's main results tables (Tables 3-4) additionally report
# gac_fixed and sac_fixed rows, produced by a separate motivating-experiments
# codebase not included in this repository (see README.md). Their numbers
# are reproduced verbatim in the paper from the original experimental logs.
DEPLOYMENTS = ["random", "uniform"]


def load_seed_csvs(results_dir, arch, deployment, seeds):
    dfs = []
    for seed in seeds:
        path = os.path.join(results_dir, f"eval_{arch}_{deployment}_seed{seed}.csv")
        if not os.path.exists(path):
            print(f"[warn] missing: {path} (skipped -- run scripts/run_full_multiseed_study.sh first)",
                  file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def summarize(df):
    """Group by failure_ratio, compute mean +/- std across seeds for each metric."""
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
    all_summaries = {}
    for arch in ARCHS:
        for dep in DEPLOYMENTS:
            df = load_seed_csvs(args.results_dir, arch, dep, args.seeds)
            if df is None:
                continue
            summary = summarize(df)
            key = f"{arch}_{dep}"
            all_summaries[key] = summary
            print(f"\n=== {arch} / {dep} (n={df['seed'].nunique()} seeds) ===")
            print(summary.to_string(index=False))

            out_path = os.path.join(args.results_dir, f"summary_{key}.csv")
            summary.to_csv(out_path, index=False)

    if not all_summaries:
        print("\nNo results found yet. Run scripts/run_full_multiseed_study.sh first.")
        return

    # combined long-format CSV for easy pivoting/plotting later
    combined_rows = []
    for key, summary in all_summaries.items():
        arch, dep = key.rsplit("_", 1) if key.endswith(("random", "uniform")) else (key, "")
        # arch may itself contain underscores (e.g. "hac_fixed"), so split off the
        # deployment suffix explicitly instead of a naive split
        for dep_name in DEPLOYMENTS:
            if key.endswith(f"_{dep_name}"):
                arch = key[: -(len(dep_name) + 1)]
                dep = dep_name
                break
        s = summary.copy()
        s.insert(0, "deployment", dep)
        s.insert(0, "architecture", arch)
        combined_rows.append(s)
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
