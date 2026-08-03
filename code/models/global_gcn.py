"""
GAC: Global Adaptive-weighted Convolution policy.

Design rationale (Gap 1 = Learned x Global-level x Adaptive weighting):
- Unlike R2C, there is NO subnetwork partitioning / multipartite graph here.
  A single k-NN graph is built over ALL alive UAVs (Euclidean k-NN, not restricted
  to the R_c communication range) so that information can flow across disconnected
  subnetworks purely through the learned global GCN -- this is the defining feature
  of the "Global-level" cell in Table I, as opposed to R2C's node-level (ISE) +
  subnetwork-level (IST) split.
- A single GCN policy predicts a per-node 2D velocity directly (no separate
  node-level / subnetwork-level branches to fuse).
- Two variants share this exact backbone and differ ONLY in how the
  connectivity-vs-coverage loss weight is scheduled over the recovery horizon:
    * GACFixedWeight:    lambda(t) = lambda0                    (constant)
    * GACAdaptiveWeight: lambda(t) = sigmoid((t/T - s0) * k)     (Eq. 22 in the paper,
                          re-purposed here for a single global policy instead of
                          R2C's two-branch fusion)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GlobalGACPolicy(nn.Module):
    """Shared backbone for both GAC-Fixed and GAC-Adaptive."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        d_in = in_dim
        for _ in range(n_layers):
            self.convs.append(GCNConv(d_in, hidden_dim))
            d_in = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # 2D velocity direction (pre-tanh)
        )

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        v = torch.tanh(self.head(h))   # bounded to [-1, 1]^2, scaled by v_max by the caller
        return v


def knn_edge_index(positions: torch.Tensor, k: int = 8):
    """
    Build a symmetric k-NN graph in Euclidean space over ALL alive nodes
    (deliberately NOT restricted to the communication range R_c), so the
    global GCN can pass messages across disconnected subnetworks.
    positions: (N, 2) tensor
    returns: edge_index (2, E) long tensor
    """
    with torch.no_grad():
        dist = torch.cdist(positions, positions)
        n = positions.shape[0]
        k_eff = min(k + 1, n)  # +1 to include self, removed below
        _, idx = torch.topk(dist, k_eff, largest=False)
        src = torch.arange(n, device=positions.device).unsqueeze(1).repeat(1, k_eff).reshape(-1)
        dst = idx.reshape(-1)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def node_features(positions: torch.Tensor, area_size: float, subnet_ids: torch.Tensor):
    """
    Feature vector per node: normalized (x, y) + normalized subnet-id one-hot-ish scalar
    + local degree within comm range (filled in by caller). Kept simple: (x_norm, y_norm,
    subnet_id_norm, degree_norm) matching the paper's H_t^(0) 'positions and subnetwork IDs'
    feature description (Eq. 15 context), but WITHOUT the multipartite/boundary-node
    construction -- subnet_id here is only used as an input feature, not as a
    partitioning mechanism for separate message passing.
    """
    xy_norm = positions / area_size
    subnet_norm = (subnet_ids.float() / (subnet_ids.max().float() + 1e-6)).unsqueeze(1)
    return xy_norm, subnet_norm


class GACFixedWeight:
    """lambda(t) = lambda0, constant across the recovery horizon."""
    def __init__(self, lambda0: float = 0.5):
        self.lambda0 = lambda0

    def __call__(self, t: int, T: int) -> float:
        return self.lambda0


class GACAdaptiveWeight:
    """lambda(t) = sigmoid(k * (t/T - s0)) -- same functional form as the paper's
    Eq. (22) AWF, re-purposed to schedule the connectivity/coverage loss trade-off
    of a single global policy (rather than fusing two branch velocities)."""
    def __init__(self, k: float = 30.0, s0: float = 0.125):
        self.k = k
        self.s0 = s0

    def __call__(self, t: int, T: int) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-self.k * (t / T - self.s0)))
