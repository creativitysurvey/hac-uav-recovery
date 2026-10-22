"""
Unified training script for HAC (global hybrid), HSC (subnetwork hybrid),
Learned-only (global learned), and SAC-Fixed (subnetwork learned).
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.swarm_env import SimConfig, UAVSwarmEnv
from models.global_gcn import (
    GlobalGACPolicy, SubnetSACPolicy, knn_edge_index, node_features,
    GACFixedWeight, GACAdaptiveWeight
)
from models.heuristic_force import virtual_force_velocity
from models.losses import connectivity_surrogate_loss, coverage_surrogate_loss, movement_cost_loss
from sim.graph_utils import component_ids_from_positions, sample_grid_points


def fuse_velocities(v_heur, v_learn, wE, wT, v_max, eps=1e-3):
    """Eq. (4) velocity-level fusion: weighted sum renormalized to magnitude v_max."""
    combined = wE * v_heur + wT * v_learn
    norm = combined.norm(dim=-1, keepdim=True)
    return v_max * combined / (norm + eps)


def compute_learned_velocity(policy, positions, comp_ids, area_size, knn_k, v_max, granularity):
    if granularity == "subnetwork":
        return policy(positions, comp_ids, area_size, k=knn_k) * v_max
    edge_index = knn_edge_index(positions, k=knn_k)
    xy_norm, _ = node_features(positions, area_size, comp_ids)
    deg = torch.zeros(positions.shape[0], device=positions.device)
    src = edge_index[0]
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    deg_norm = (deg / (deg.max() + 1e-6)).unsqueeze(1)
    x = torch.cat([xy_norm, deg_norm], dim=1)
    return policy(x, edge_index) * v_max


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_hybrid] device={device} granularity={args.granularity} variant={args.variant} seed={args.seed}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = SimConfig(T=args.rollout_steps, seed=args.seed)
    if args.granularity == "subnetwork":
        policy = SubnetSACPolicy(in_dim=3, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    else:
        policy = GlobalGACPolicy(in_dim=3, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    if args.variant == "fixed":
        weight_fn = GACFixedWeight(lambda0=0.5)
    else:
        weight_fn = GACAdaptiveWeight(k=args.awf_k, s0=args.awf_s0)

    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        env = UAVSwarmEnv(cfg, deployment=args.deployment)
        env.rng = np.random.default_rng(args.seed * 100000 + epoch)
        env.positions = env._init_positions()
        failure_ratio = rng.choice([0.2, 0.4, 0.6, 0.8])
        env.apply_failure(failure_ratio)

        pos_np = env.alive_positions().copy()
        positions = torch.tensor(pos_np, device=device, dtype=torch.float32, requires_grad=False)
        grid_np = sample_grid_points(cfg.area_size, args.coverage_samples, rng)
        grid = torch.tensor(grid_np, device=device, dtype=torch.float32)

        opt.zero_grad()
        total_loss = 0.0
        for t in range(args.rollout_steps):
            comp_ids_np = component_ids_from_positions(positions.detach().cpu().numpy(), cfg.comm_radius)
            comp_ids = torch.tensor(comp_ids_np, device=device)

            with torch.no_grad():
                v_heur = virtual_force_velocity(positions, cfg.comm_radius, cfg.area_size, v_max=cfg.v_max)

            v_learn = compute_learned_velocity(
                policy, positions, comp_ids, cfg.area_size, args.knn_k, cfg.v_max, args.granularity
            )

            wT = args.force_wT if args.force_wT is not None else weight_fn(t, args.rollout_steps)
            wE = 1.0 - wT
            v = fuse_velocities(v_heur, v_learn, wE, wT, cfg.v_max)

            pos_after = torch.clamp(positions + cfg.dt * v, 0, cfg.area_size)

            l_conn = connectivity_surrogate_loss(pos_after, comp_ids)
            l_cov = coverage_surrogate_loss(pos_after, grid, cfg.cov_radius)
            l_move = movement_cost_loss(positions, pos_after)

            step_loss = 0.5 * l_conn + 0.5 * l_cov + args.move_coef * l_move
            total_loss = total_loss + step_loss
            positions = pos_after

        total_loss = total_loss / args.rollout_steps
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if epoch % args.log_every == 0:
            n_comp = len(np.unique(component_ids_from_positions(positions.detach().cpu().numpy(), cfg.comm_radius)))
            print(f"[epoch {epoch:4d}] loss={total_loss.item():.4f} final_components={n_comp} failure={failure_ratio:.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "variant": args.variant,
        "deployment": args.deployment,
        "granularity": args.granularity,
        "paradigm": "learned" if args.force_wT == 1.0 else "hybrid",
        "args": vars(args),
    }, args.out)
    print(f"[train_hybrid] saved checkpoint to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["fixed", "adaptive"], required=True)
    p.add_argument("--granularity", choices=["global", "subnetwork"], default="global",
                   help="Spatial reasoning granularity of the learned branch: 'global' (HAC/GAC) or 'subnetwork' (HSC/SAC).")
    p.add_argument("--deployment", choices=["random", "uniform"], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--rollout_steps", type=int, default=40)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--move_coef", type=float, default=0.01)
    p.add_argument("--coverage_samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--awf_k", type=float, default=30.0)
    p.add_argument("--awf_s0", type=float, default=0.125)
    p.add_argument("--force_wT", type=float, default=None)
    args = p.parse_args()
    train(args)
