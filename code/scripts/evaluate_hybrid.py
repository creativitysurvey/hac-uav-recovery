"""
Evaluate a trained HAC / HSC / GAC / Learned-only / SAC checkpoint.
Supports overriding the test-time schedule via --eval_variant, --eval_awf_k, --eval_awf_s0.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.swarm_env import SimConfig, UAVSwarmEnv
from models.global_gcn import (
    GlobalGACPolicy, SubnetSACPolicy, GACFixedWeight, GACAdaptiveWeight
)
from models.heuristic_force import virtual_force_velocity
from sim.graph_utils import component_ids_from_positions
from scripts.train_hybrid import fuse_velocities, compute_learned_velocity


@torch.no_grad()
def rollout_policy(env, policy, weight_fn, device, knn_k=8, force_wT=None,
                   granularity="global", use_subnet_feat=False, no_norm_fusion=False):
    cfg = env.cfg
    degrees = []
    for t in range(cfg.T):
        pos_np = env.alive_positions()
        positions = torch.tensor(pos_np, device=device, dtype=torch.float32)
        comp_ids_np = component_ids_from_positions(pos_np, cfg.comm_radius)
        comp_ids = torch.tensor(comp_ids_np, device=device)

        v_heur = virtual_force_velocity(positions, cfg.comm_radius, cfg.area_size, v_max=cfg.v_max)
        v_learn = compute_learned_velocity(
            policy, positions, comp_ids, cfg.area_size, knn_k, cfg.v_max,
            granularity, use_subnet_feat=use_subnet_feat
        )

        wT = force_wT if force_wT is not None else weight_fn(t, cfg.T)
        wE = 1.0 - wT
        v = fuse_velocities(v_heur, v_learn, wE, wT, cfg.v_max, no_norm_fusion=no_norm_fusion).cpu().numpy()

        env.step(v)
        cur_deg = env.average_node_degree()
        degrees.append(cur_deg)
        if env.is_recovered():
            return t + 1, env.coverage_ratio(), float(np.mean(degrees)), float(cur_deg)
    return None, env.coverage_ratio(), float(np.mean(degrees)) if degrees else 0.0, float(degrees[-1]) if degrees else 0.0


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    granularity = ckpt.get("granularity", ckpt["args"].get("granularity", "global"))
    use_subnet_feat = ckpt["args"].get("use_subnet_feat", False)
    no_norm_fusion = ckpt["args"].get("no_norm_fusion", False)
    in_dim = 4 if use_subnet_feat else 3

    if granularity == "subnetwork":
        policy = SubnetSACPolicy(
            in_dim=3, hidden_dim=ckpt["args"]["hidden_dim"], n_layers=ckpt["args"]["n_layers"]
        ).to(device)
    else:
        policy = GlobalGACPolicy(
            in_dim=in_dim, hidden_dim=ckpt["args"]["hidden_dim"], n_layers=ckpt["args"]["n_layers"]
        ).to(device)

    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    force_wT = args.eval_force_wT if args.eval_force_wT is not None else ckpt["args"].get("force_wT", None)
    variant = args.eval_variant if args.eval_variant is not None else ckpt["variant"]

    if variant == "fixed":
        weight_fn = GACFixedWeight(lambda0=0.5)
    else:
        k_val = args.eval_awf_k if args.eval_awf_k is not None else ckpt["args"].get("awf_k", 30.0)
        s0_val = args.eval_awf_s0 if args.eval_awf_s0 is not None else ckpt["args"].get("awf_s0", 0.125)
        weight_fn = GACAdaptiveWeight(k=k_val, s0=s0_val)

    cfg = SimConfig(T=args.T, seed=args.seed)
    rows = []
    for failure_ratio in [0.2, 0.4, 0.6, 0.8]:
        recs, covs, ands, and_finals, runtimes, n_invalid = [], [], [], [], [], 0
        for trial in range(args.n_trials):
            env = UAVSwarmEnv(cfg, deployment=args.deployment)
            env.rng = np.random.default_rng(args.seed * 10000 + trial)
            env.positions = env._init_positions()
            env.apply_failure(failure_ratio)

            t0 = time.time()
            t_r, cov, and_traj, and_final = rollout_policy(
                env, policy, weight_fn, device,
                knn_k=ckpt["args"]["knn_k"], force_wT=force_wT, granularity=granularity,
                use_subnet_feat=use_subnet_feat, no_norm_fusion=no_norm_fusion
            )
            runtimes.append(time.time() - t0)
            if t_r is None:
                n_invalid += 1
                continue
            recs.append(env.recovery_time_ratio(t_r))
            covs.append(cov)
            ands.append(and_traj)
            and_finals.append(and_final)

        valid = args.n_trials - n_invalid
        row = {
            "failure_ratio": failure_ratio,
            "REC_mean": np.mean(recs) if recs else np.nan,
            "COV_mean": np.mean(covs) if covs else np.nan,
            "AND_mean": np.mean(ands) if ands else np.nan,
            "AND_final_mean": np.mean(and_finals) if and_finals else np.nan,
            "runtime_mean_s": np.mean(runtimes),
            "runtime_std_s": np.std(runtimes),
            "n_valid_of_50": valid,
        }
        rows.append(row)
        print(f"[eval] failure={failure_ratio:.1f} REC={row['REC_mean']:.3f} "
              f"COV={row['COV_mean']:.3f} AND={row['AND_mean']:.2f} valid={valid}/{args.n_trials}")

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
    p.add_argument("--eval_variant", choices=["fixed", "adaptive"], default=None)
    p.add_argument("--eval_awf_k", type=float, default=None)
    p.add_argument("--eval_awf_s0", type=float, default=None)
    p.add_argument("--eval_force_wT", type=float, default=None)
    args = p.parse_args()
    evaluate(args)
