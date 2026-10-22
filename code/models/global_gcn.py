"""
Learned GCN policies and fusion weight schedules for HAC, GAC, SAC, and HSC.
- GlobalGACPolicy: Node-level output over a global Euclidean k-NN graph (used by HAC, GAC, Learned-only).
- SubnetSACPolicy: Subnetwork-level rigid translation over a k-NN graph of component centroids (used by SAC, HSC).
"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GlobalGACPolicy(nn.Module):
    """Global GCN backbone predicting per-node 2D velocity directions."""
    def __init__(self, in_dim: int = 3, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        d_in = in_dim
        for _ in range(n_layers):
            self.convs.append(GCNConv(d_in, hidden_dim))
            d_in = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        return torch.tanh(self.head(h))


class SubnetSACPolicy(nn.Module):
    """
    Subnetwork-level GCN backbone (for SAC and HSC).
    Pools each connected component into a centroid embedding (normalized centroid x, y
    and relative component size), runs an L-layer GCN over the centroid k-NN graph,
    and broadcasts the predicted 2D rigid translation velocity to all nodes in that component.
    """
    def __init__(self, in_dim: int = 3, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        d_in = in_dim
        for _ in range(n_layers):
            self.convs.append(GCNConv(d_in, hidden_dim))
            d_in = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, positions: torch.Tensor, comp_ids: torch.Tensor, area_size: float, k: int = 8):
        device = positions.device
        unique_comps, inverse_idx, counts = torch.unique(
            comp_ids, return_inverse=True, return_counts=True
        )
        n_comps = unique_comps.shape[0]
        if n_comps <= 1:
            return torch.zeros_like(positions)

        centroids = torch.zeros(n_comps, 2, device=device, dtype=positions.dtype)
        centroids.scatter_add_(0, inverse_idx.unsqueeze(1).expand(-1, 2), positions)
        centroids = centroids / counts.unsqueeze(1).float()

        size_norm = (counts.float() / positions.shape[0]).unsqueeze(1)
        x_sub = torch.cat([centroids / area_size, size_norm], dim=1)  # (C, 3)
        edge_index_sub = knn_edge_index(centroids, k=min(k, n_comps - 1))

        h = x_sub
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index_sub))
        v_sub = torch.tanh(self.head(h))  # (C, 2)
        return v_sub[inverse_idx]         # Broadcast rigid translation to (N, 2)


def knn_edge_index(positions: torch.Tensor, k: int = 8):
    """Build a symmetric k-NN graph in Euclidean space over all given coordinates."""
    with torch.no_grad():
        n = positions.shape[0]
        if n <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        dist = torch.cdist(positions, positions)
        k_eff = min(k + 1, n)
        _, idx = torch.topk(dist, k_eff, largest=False)
        src = torch.arange(n, device=positions.device).unsqueeze(1).repeat(1, k_eff).reshape(-1)
        dst = idx.reshape(-1)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def node_features(positions: torch.Tensor, area_size: float, comp_ids: torch.Tensor = None):
    """Normalized (x, y) coordinates in [0, 1]^2."""
    xy_norm = positions / area_size
    if comp_ids is None:
        return xy_norm, None
    subnet_norm = (comp_ids.float() / (comp_ids.max().float() + 1e-6)).unsqueeze(1)
    return xy_norm, subnet_norm


class GACFixedWeight:
    """w_T(t) = lambda0, constant across the horizon."""
    def __init__(self, lambda0: float = 0.5):
        self.lambda0 = lambda0

    def __call__(self, t: int, T: int) -> float:
        return self.lambda0


class GACAdaptiveWeight:
    """w_T(t) = sigmoid(k * (t/T - s0)) per Eq. (3)."""
    def __init__(self, k: float = 30.0, s0: float = 0.125):
        self.k = k
        self.s0 = s0

    def __call__(self, t: int, T: int) -> float:
        return 1.0 / (1.0 + math.exp(-self.k * (t / T - self.s0)))
