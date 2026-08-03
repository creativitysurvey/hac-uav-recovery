"""
Ablation: heuristic branch ALONE (Eq. 9-12, no learned component, no fusion).
No training needed -- the virtual-force branch has zero learnable parameters.
This is the "wE=1, wT=0" corner of the {heuristic, learned, fusion} ablation set.

Exact run commands (see README.md):
  python scripts/evaluate_heuristic_only.py --deployment random  --out results/eval_heuristic_only_random.csv
  python scripts/evaluate_heuristic_only.py --deployment uniform --out results/eval_heuristic_only_uniform.csv
"""
import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.swarm_env import SimConfig, UAVSwarmEnv
from models.heuristic_force import virtual_force_velocity


@torch.no_grad()
def rollout_policy(env, device):
    cfg = env.cfg
    degrees = []
    for t in range(cfg.T):
        pos_np = env.alive_positions()
        positions = torch.tensor(pos_np, device=device, dtype=torch.float32)
        v = virtual_force_velocity(positions, cfg.comm_radius, cfg.area_size,
                                    v_max=cfg.v_max).cpu().numpy()
        env.step(v)
        degrees.append(env.average_node_degree())
        if env.is_recovered():
            return t + 1, env.coverage_ratio(), float(np.mean(degrees))
    return None, env.coverage_ratio(), float(np.mean(degrees)) if degrees else 0.0


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SimConfig(T=args.T, seed=args.seed)
    rows = []
    for failure_ratio in [0.2, 0.4, 0.6, 0.8]:
        recs, covs, ands, runtimes, n_invalid = [], [], [], [], 0
        for trial in range(args.n_trials):
            env = UAVSwarmEnv(cfg, deployment=args.deployment)
            env.rng = np.random.default_rng(args.seed * 10000 + trial)
            env.positions = env._init_positions()
            env.apply_failure(failure_ratio)

            t0 = time.time()
            t_r, cov, and_ = rollout_policy(env, device)
            runtimes.append(time.time() - t0)

            if t_r is None:
                n_invalid += 1
                continue
            recs.append(env.recovery_time_ratio(t_r))
            covs.append(cov)
            ands.append(and_)

        valid = args.n_trials - n_invalid
        row = {
            "failure_ratio": failure_ratio,
            "REC_mean": np.mean(recs) if recs else np.nan,
            "COV_mean": np.mean(covs) if covs else np.nan,
            "AND_mean": np.mean(ands) if ands else np.nan,
            "runtime_mean_s": np.mean(runtimes),
            "runtime_std_s": np.std(runtimes),
            "n_valid_of_50": valid,
        }
        rows.append(row)
        print(f"[eval] failure={failure_ratio:.1f}  REC={row['REC_mean']:.3f}  "
              f"COV={row['COV_mean']:.3f}  AND={row['AND_mean']:.2f}  valid={valid}/{args.n_trials}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[eval] saved results to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", choices=["random", "uniform"], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--T", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    evaluate(args)
