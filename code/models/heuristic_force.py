"""
Deterministic heuristic virtual-force branch, reproducing Eq. (2) buffered-spring
virtual-force mechanism (R2C's ISE module). No learnable parameters.
"""

import torch

DEFAULT_ALPHA = 0.01
DEFAULT_BETA = 0.01
DEFAULT_DELTA = 0.01
DEFAULT_DL_FRAC = 0.8   # d_l = 0.8 * R_c
DEFAULT_DU_FRAC = 0.9   # d_u = 0.9 * R_c


def virtual_force_velocity(positions: torch.Tensor, comm_radius: float, area_size: float,
                            v_max: float = 1.0, dl_frac: float = DEFAULT_DL_FRAC,
                            du_frac: float = DEFAULT_DU_FRAC, alpha: float = DEFAULT_ALPHA,
                            beta: float = DEFAULT_BETA, delta: float = DEFAULT_DELTA,
                            de: float = None, eps: float = 1e-3) -> torch.Tensor:
    """
    Implements Eq. (2): pairwise buffered virtual-spring force (repulsive /
    neutral / attractive zones) + boundary repulsion, summed over neighbors within
    communication range R_c, then normalized to magnitude v_max.
    """
    device = positions.device
    n = positions.shape[0]
    if de is None:
        de = 0.15 * comm_radius
    dl = dl_frac * comm_radius
    du = du_frac * comm_radius

    diff = positions[None, :, :] - positions[:, None, :]        # (N, N, 2): p_j - p_i
    dist = diff.norm(dim=-1)                                     # (N, N)
    dist_safe = dist.clamp(min=1e-6)
    u = diff / dist_safe.unsqueeze(-1)                           # unit vectors u_ij

    neighbor_mask = (dist <= comm_radius) & (dist > 1e-9)        # N_i: within comm range
    repulsive_mask = (dist < dl) & neighbor_mask
    attractive_mask = (dist > du) & (dist <= comm_radius) & neighbor_mask

    force_mag = torch.zeros_like(dist)
    force_mag = torch.where(repulsive_mask, -torch.exp(-alpha * dist_safe / dl), force_mag)
    force_mag = torch.where(
        attractive_mask,
        torch.exp(-beta * (comm_radius - dist_safe) / max(comm_radius - du, 1e-6)),
        force_mag,
    )

    pairwise_force = force_mag.unsqueeze(-1) * u                 # (N, N, 2)
    f_prime = pairwise_force.sum(dim=1)                          # sum over N_i

    # Boundary repulsive force near the four edges of the mission square.
    # Use (d_k >= 0.0) so nodes clamped onto 0 or area_size still receive inward repulsion.
    x, y = positions[:, 0], positions[:, 1]
    d_left, d_right = x, area_size - x
    d_bottom, d_top = y, area_size - y
    f_boundary = torch.zeros(n, 2, device=device)
    for d_k, normal in [
        (d_left, torch.tensor([1.0, 0.0], device=device)),
        (d_right, torch.tensor([-1.0, 0.0], device=device)),
        (d_bottom, torch.tensor([0.0, 1.0], device=device)),
        (d_top, torch.tensor([0.0, -1.0], device=device)),
    ]:
        within = (d_k < de) & (d_k >= 0.0)
        mag = torch.where(within, torch.exp(-delta * d_k.clamp(min=0.0)), torch.zeros_like(d_k))
        f_boundary = f_boundary + mag.unsqueeze(-1) * normal.unsqueeze(0)

    f_total = f_prime + f_boundary
    f_norm = f_total.norm(dim=-1, keepdim=True)
    v = v_max * f_total / (f_norm + eps)
    return v
