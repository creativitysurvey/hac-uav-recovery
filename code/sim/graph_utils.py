"""
Small, dependency-light graph/sampling utilities shared by this project's
training and evaluation scripts. Extracted into their own module (rather
than living inside a specific train_*.py script, as they originally did)
so that scripts which need them do not have to import an entire,
paper-specific training script just to get two helper functions.
"""
import numpy as np
import networkx as nx


def component_ids_from_positions(positions_np: np.ndarray, comm_radius: float) -> np.ndarray:
    """
    Assigns each node an integer connected-component id under the
    communication-range adjacency graph (edge iff pairwise distance <= comm_radius).

    positions_np: (N, 2) numpy array.
    Returns: (N,) int64 numpy array of component ids.
    """
    diff = positions_np[:, None, :] - positions_np[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    adj = (dist <= comm_radius) & (dist > 1e-9)
    g = nx.from_numpy_array(adj)
    comp_id = np.zeros(len(positions_np), dtype=np.int64)
    for cid, comp in enumerate(nx.connected_components(g)):
        for node in comp:
            comp_id[node] = cid
    return comp_id


def sample_grid_points(area_size: float, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly samples n_points random points in [0, area_size]^2, used for
    Monte-Carlo estimation of the coverage surrogate loss."""
    return rng.uniform(0, area_size, size=(n_points, 2)).astype(np.float32)
