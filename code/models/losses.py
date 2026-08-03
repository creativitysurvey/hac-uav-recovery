"""
Differentiable training-time loss surrogates.

These mirror the *intent* of the paper's Eq. (19)-(21) losses (L_conn, L_time/L_cov-like,
L_move), but had to be re-derived here because:
  (a) the paper's L_conn (Eq. 19) counts connected components, which is non-differentiable,
      and its L_time (Eq. 20) is defined specifically for the subnetwork-translation (IST)
      branch, which does not exist in this Global-level architecture;
  (b) no official code was released, so exact implementation details (e.g., temperature
      constants, surrogate choices) are not specified by the paper.
This is logged as a deviation: results are only comparable to R2C in eval-time metrics
(REC / COV / degree stats computed identically via sim/swarm_env.py), NOT in raw training
loss values.
"""
import math
import torch


def connectivity_surrogate_loss(positions: torch.Tensor, component_ids: torch.Tensor,
                                 tau: float = 50.0) -> torch.Tensor:
    """
    Soft proxy for connectivity restoration: for every pair of currently-disconnected
    components, penalize the distance between their closest pair of nodes using a
    soft-min (log-sum-exp), which is differentiable and decreases as boundary nodes
    of different components are pulled together.
    positions: (N, 2), component_ids: (N,) long tensor from a *pre-move* connectivity check.

    NOTE (bug fix, logged deviation): a plain `-tau * logsumexp(-d/tau)` is NOT a
    count-invariant soft-min -- summing over more pairs (i.e. larger components)
    systematically pulls the estimate more negative regardless of the true minimum
    distance, because logsumexp grows with the number of summands (~ log(N) bias).
    This was caught after the first full training run (see conversation record) where
    it produced spuriously negative total losses and likely rewarded large/populous
    component pairs independent of actual spatial proximity. Fixed by subtracting
    log(n_pairs) so the estimate converges to the true min as tau -> 0, independent
    of how many pairs are being compared.
    """
    device = positions.device
    comps = component_ids.unique()
    if len(comps) <= 1:
        return torch.tensor(0.0, device=device)
    losses = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            pi = positions[component_ids == comps[i]]
            pj = positions[component_ids == comps[j]]
            d = torch.cdist(pi, pj)  # (ni, nj)
            n_pairs = d.numel()
            soft_min = -tau * (torch.logsumexp(-d.flatten() / tau, dim=0) - math.log(n_pairs))
            losses.append(soft_min)
    return torch.stack(losses).mean()


def coverage_surrogate_loss(positions: torch.Tensor, grid_points: torch.Tensor,
                             cov_radius: float, tau: float = 10.0) -> torch.Tensor:
    """
    Soft proxy for (1 - coverage): for a sampled subset of grid points, the "soft coverage"
    of a point is a smooth indicator sigmoid((R_cov - min_dist_to_any_UAV) / tau); averaging
    (1 - soft_covered) over grid points gives a differentiable stand-in for Eq. (7)'s
    (1 - C_cov(t_r)) term in the paper's overall objective P1 (Eq. 8).
    """
    d = torch.cdist(grid_points, positions)          # (M, N)
    min_d, _ = d.min(dim=1)                           # (M,)
    soft_covered = torch.sigmoid((cov_radius - min_d) / tau)
    return (1.0 - soft_covered).mean()


def movement_cost_loss(pos_before: torch.Tensor, pos_after: torch.Tensor) -> torch.Tensor:
    """Direct analogue of the paper's Eq. (21) L_move."""
    return (pos_after - pos_before).norm(dim=1).mean()
