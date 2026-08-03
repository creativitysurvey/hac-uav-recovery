"""
Evaluate a trained HAC-Fixed / HAC-Adaptive checkpoint using the SAME metrics as
Gap 1 / Gap 3 (REC, COV, AND), so results slot directly into the same comparison
table across all three gaps.

Exact run commands (see README.md):
  python scripts/evaluate_hybrid.py --ckpt results/hac_fixed_random.pt    --deployment random  --out results/eval_hac_fixed_random.csv
  python scripts/evaluate_hybrid.py --ckpt results/hac_adaptive_random.pt --deployment random  --out results/eval_hac_adaptive_random.csv
  python scripts/evaluate_hybrid.py --ckpt results/hac_fixed_uniform.pt   --deployment uniform --out results/eval_hac_fixed_uniform.csv
  python scripts/evaluate_hybrid.py --ckpt results/hac_adaptive_uniform.pt --deployment uniform --out results/eval_hac_adaptive_uniform.csv
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
from models.global_gcn import GlobalGACPolicy, knn_edge_index, node_features, GACFixedWeight, GACAdaptiveWeight
from models.heuristic_force import virtual_force_velocity
from sim.graph_utils import component_ids_from_positions
from scripts.train_hybrid import fuse_velocities


@torch.no_grad()
def rollout_policy(env, policy, weight_fn, device, knn_k=8, force_wT=None):
    cfg = env.cfg
    degrees = []
    for t in range(cfg.T):
        pos_np = env.alive_positions()
        positions = torch.tensor(pos_np, device=device, dtype=torch.float32)

        comp_ids_np = component_ids_from_positions(pos_np, cfg.comm_radius)
        comp_ids = torch.tensor(comp_ids_np, device=device)

        v_heur = virtual_force_velocity(positions, cfg.comm_radius, cfg.area_size, v_max=cfg.v_max)

        edge_index = knn_edge_index(positions, k=knn_k)
        xy_norm, _ = node_features(positions, cfg.area_size, comp_ids)
        deg = torch.zeros(positions.shape[0], device=device)
        src = edge_index[0]
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg_norm = (deg / (deg.max() + 1e-6)).unsqueeze(1)
        x = torch.cat([xy_norm, deg_norm], dim=1)
        v_learn = policy(x, edge_index) * cfg.v_max

        wT = force_wT if force_wT is not None else weight_fn(t, cfg.T)
        wE = 1.0 - wT
        v = fuse_velocities(v_heur, v_learn, wE, wT, cfg.v_max).cpu().numpy()

        env.step(v)
        degrees.append(env.average_node_degree())
        if env.is_recovered():
            return t + 1, env.coverage_ratio(), float(np.mean(degrees))
    return None, env.coverage_ratio(), float(np.mean(degrees)) if degrees else 0.0


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    policy = GlobalGACPolicy(in_dim=3,
                              hidden_dim=ckpt["args"]["hidden_dim"],
                              n_layers=ckpt["args"]["n_layers"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    force_wT = ckpt["args"].get("force_wT", None)   # ablation override, e.g. 1.0 for learned-only
    if ckpt["variant"] == "fixed":
        weight_fn = GACFixedWeight(lambda0=0.5)
    else:
        # read back the SAME k/s0 used at training time (sweepable, see train_hybrid.py --awf_k/--awf_s0)
        weight_fn = GACAdaptiveWeight(k=ckpt["args"].get("awf_k", 30.0),
                                       s0=ckpt["args"].get("awf_s0", 0.125))

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
            t_r, cov, and_ = rollout_policy(env, policy, weight_fn, device,
                                             knn_k=ckpt["args"]["knn_k"], force_wT=force_wT)
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
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--deployment", choices=["random", "uniform"], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--T", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    evaluate(args)
