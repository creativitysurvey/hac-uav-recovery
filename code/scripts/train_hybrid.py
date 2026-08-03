"""
Train HAC-Fixed / HAC-Adaptive (Gap 2: Hybrid x Global-level x {Fixed,Adaptive}).

Architecture: fuse a deterministic heuristic virtual-force branch (Eq. 9-12, no
learnable params) with a learned GLOBAL-level GCN branch (same GlobalGACPolicy as
Gap 1's GAC, but here trained end-to-end as the "complement" to the heuristic
branch rather than alone), via velocity-level fusion mirroring the paper's own
Eq. (22)-(23) AWF mechanism:
    v_hat = v_max * (wE(t)*v_heuristic + wT(t)*v_learned) / (||...|| + eps)
  - HAC-Fixed:    wE = wT = 0.5 (constant)      -- mirrors the paper's own "w/o AWF" ablation
  - HAC-Adaptive: wT(t) = sigmoid(k*(t/T - s0)), wE(t) = 1 - wT(t)  -- same Eq.(22) constants

Only the learned branch (GlobalGACPolicy) has trainable parameters; the heuristic
branch is fixed and reused unchanged from models/heuristic_force.py.

Exact run commands (see README.md):
  python scripts/train_hybrid.py --variant fixed    --deployment random  --out results/hac_fixed_random.pt
  python scripts/train_hybrid.py --variant adaptive --deployment random  --out results/hac_adaptive_random.pt
  python scripts/train_hybrid.py --variant fixed    --deployment uniform --out results/hac_fixed_uniform.pt
  python scripts/train_hybrid.py --variant adaptive --deployment uniform --out results/hac_adaptive_uniform.pt
"""
import argparse
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.swarm_env import SimConfig, UAVSwarmEnv
from models.global_gcn import (GlobalGACPolicy, knn_edge_index, node_features,
                                GACFixedWeight, GACAdaptiveWeight)
from models.heuristic_force import virtual_force_velocity
from models.losses import connectivity_surrogate_loss, coverage_surrogate_loss, movement_cost_loss
from sim.graph_utils import component_ids_from_positions, sample_grid_points


def fuse_velocities(v_heur, v_learn, wE, wT, v_max, eps=1e-3):
    """Eq.(23)-style fusion: weighted sum, renormalized to magnitude v_max."""
    combined = wE * v_heur + wT * v_learn
    norm = combined.norm(dim=-1, keepdim=True)
    return v_max * combined / (norm + eps)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_hybrid] device = {device}")

    # NOTE (statistical rigor fix): --seed previously only controlled the numpy-side
    # env RNGs (failure sampling, initial deployment), leaving PyTorch's network
    # weight initialization uncontrolled. For multi-seed reproducibility studies, a
    # given --seed value must fully determine the run. Fixed here.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = SimConfig(T=args.rollout_steps, seed=args.seed)
    policy = GlobalGACPolicy(in_dim=3, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    if args.variant == "fixed":
        weight_fn = GACFixedWeight(lambda0=0.5)     # wT(t) = 0.5 constant -> wE = wT = 0.5
    else:
        weight_fn = GACAdaptiveWeight(k=args.awf_k, s0=args.awf_s0)   # sweepable (default = paper's Eq.22 constants)

    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        env = UAVSwarmEnv(cfg, deployment=args.deployment)
        env.rng = np.random.default_rng(args.seed + epoch)
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

            # --- heuristic branch (no grad needed, no learnable params) ---
            with torch.no_grad():
                v_heur = virtual_force_velocity(positions, cfg.comm_radius, cfg.area_size,
                                                 v_max=cfg.v_max)

            # --- learned global branch (same architecture as Gap 1's GAC) ---
            edge_index = knn_edge_index(positions, k=args.knn_k)
            xy_norm, subnet_norm = node_features(positions, cfg.area_size, comp_ids)
            deg = torch.zeros(positions.shape[0], device=device)
            src = edge_index[0]
            deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
            deg_norm = (deg / (deg.max() + 1e-6)).unsqueeze(1)
            x = torch.cat([xy_norm, deg_norm], dim=1)
            v_learn_dir = policy(x, edge_index)
            v_learn = v_learn_dir * cfg.v_max

            # --- Eq.(22)-(23) style adaptive fusion ---
            if args.force_wT is not None:
                wT = args.force_wT           # ablation override: e.g. 1.0 = learned-only, 0.0 = heuristic-only
            else:
                wT = weight_fn(t, args.rollout_steps)
            wE = 1.0 - wT
            v = fuse_velocities(v_heur, v_learn, wE, wT, cfg.v_max)

            pos_after = torch.clamp(positions + cfg.dt * v, 0, cfg.area_size)

            l_conn = connectivity_surrogate_loss(pos_after, comp_ids)
            l_cov = coverage_surrogate_loss(pos_after, grid, cfg.cov_radius)
            l_move = movement_cost_loss(positions, pos_after)
            # equal-weighted total objective (fusion of connectivity/coverage priority
            # already happens at the VELOCITY level via wE/wT, so the loss itself does
            # not need its own separate adaptive schedule here -- this is the key
            # architectural difference from Gap 1)
            step_loss = 0.5 * l_conn + 0.5 * l_cov + args.move_coef * l_move
            total_loss = total_loss + step_loss
            positions = pos_after

        total_loss = total_loss / args.rollout_steps
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if epoch % args.log_every == 0:
            n_comp = len(np.unique(component_ids_from_positions(positions.detach().cpu().numpy(),
                                                                  cfg.comm_radius)))
            print(f"[epoch {epoch:4d}] loss={total_loss.item():.4f}  "
                  f"final_components={n_comp}  failure_ratio={failure_ratio:.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(),
                "variant": args.variant,
                "deployment": args.deployment,
                "granularity": "global",
                "paradigm": "hybrid",
                "args": vars(args)}, args.out)
    print(f"[train_hybrid] saved checkpoint to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["fixed", "adaptive"], required=True)
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
    p.add_argument("--awf_k", type=float, default=30.0,
                    help="AWF sigmoid steepness (Eq.22). Paper default = 30.0. "
                         "Smaller = slower/softer handover from heuristic to learned branch.")
    p.add_argument("--awf_s0", type=float, default=0.125,
                    help="AWF sigmoid transition center as a fraction of T (Eq.22). "
                         "Paper default = 0.125. Larger = later handover from heuristic to learned branch.")
    p.add_argument("--force_wT", type=float, default=None,
                    help="Ablation override: force a constant wT throughout training, ignoring "
                         "--variant's weight_fn entirely. Use 1.0 for 'learned-only' ablation "
                         "(heuristic branch contributes nothing) or 0.0 for 'heuristic-only' "
                         "(though heuristic-only needs no training at all -- see "
                         "scripts/evaluate_heuristic_only.py instead).")
    args = p.parse_args()
    train(args)
