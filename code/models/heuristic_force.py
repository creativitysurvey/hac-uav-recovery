"""
Deterministic heuristic virtual-force branch, faithfully reproducing the paper's
Eq. (9)-(12) buffered-spring / virtual-force mechanism (the R2C paper's "ISE" /
Intra-Subnetwork Expansion module). No learnable parameters here -- this is the
"Heuristic" half of Gap 2's Hybrid architecture.

Deviation log: the paper does not give a numeric value for the boundary buffer
width d_e (Eq. 10) -- we set DEFAULT_DE = 0.15 * comm_radius (=18m at R_c=120m)
as a reasonable choice of the same order as the neutral zone width; alpha, beta,
delta all fixed at 0.01 exactly as reported in the paper's "Parameter Settings".
"""
import torch

DEFAULT_ALPHA = 0.01
DEFAULT_BETA = 0.01
DEFAULT_DELTA = 0.01
DEFAULT_DL_FRAC = 0.8   # d_l = 0.8 * R_c, per paper's Parameter Settings
DEFAULT_DU_FRAC = 0.9   # d_u = 0.9 * R_c


def virtual_force_velocity(positions: torch.Tensor, comm_radius: float, area_size: float,
                            v_max: float = 1.0, dl_frac: float = DEFAULT_DL_FRAC,
                            du_frac: float = DEFAULT_DU_FRAC, alpha: float = DEFAULT_ALPHA,
                            beta: float = DEFAULT_BETA, delta: float = DEFAULT_DELTA,
                            de: float = None, eps: float = 1e-3) -> torch.Tensor:
    """
    Implements Eq. (9)-(12): pairwise buffered virtual-spring force (repulsive /
    neutral / attractive zones) + boundary repulsion, summed over neighbors within
    the communication range R_c, then normalized to magnitude v_max.

    positions: (N, 2) tensor of alive-node positions.
    Returns: (N, 2) tensor of per-node velocities.
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
    neutral_mask = (dist >= dl) & (dist <= du) & neighbor_mask   # force = 0, no-op
    attractive_mask = (dist > du) & (dist <= comm_radius) & neighbor_mask

    force_mag = torch.zeros_like(dist)
    force_mag = torch.where(repulsive_mask, -torch.exp(-alpha * dist_safe / dl), force_mag)
    force_mag = torch.where(
        attractive_mask,
        torch.exp(-beta * (comm_radius - dist_safe) / max(comm_radius - du, 1e-6)),
        force_mag,
    )
    pairwise_force = force_mag.unsqueeze(-1) * u                 # (N, N, 2)
    f_prime = pairwise_force.sum(dim=1)                          # Eq.(11) first term, sum over N_i

    # Eq.(10): boundary repulsive force near the four edges of the area
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
        within = (d_k < de) & (d_k > 0)
        mag = torch.where(within, torch.exp(-delta * d_k.clamp(min=0)), torch.zeros_like(d_k))
        f_boundary = f_boundary + mag.unsqueeze(-1) * normal.unsqueeze(0)

    f_total = f_prime + f_boundary                               # Eq.(11)
    f_norm = f_total.norm(dim=-1, keepdim=True)
    v = v_max * f_total / (f_norm + eps)                         # Eq.(12)
    return v
