"""
UAV Swarm Self-Healing Simulation Environment
Implements Definitions 1-5 and Problem Definition from:
"Resilient UAV Swarm with Fast Connectivity Recovery and Extensive Coverage"
(AAAI-26, Peng et al.)

This is an independent re-implementation based on the equations in the paper
(no official code was released), used as the common testbed for Gap 1:
"Learned x Global-level x Adaptive weighting" vs "Learned x Global-level x Fixed weighting".
"""
import numpy as np
import networkx as nx
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    area_size: float = 1000.0          # L_x = L_y, meters
    comm_radius: float = 120.0         # R_c, meters (Definition 3, precomputed)
    cov_radius: float = 120.0          # R_cov, meters (Definition 5)
    v_max: float = 1.0                 # max UAV speed, m/s
    dt: float = 1.0                    # simulation time step, s
    T: int = 400                       # recovery horizon, steps (Definition 4)
    n_uavs_random: int = 200
    grid_side_uniform: int = 14        # 14x14 = 196 UAVs
    coverage_grid_res: float = 5.0     # resolution (m) for Monte-Carlo coverage estimate
    seed: int = 0


class UAVSwarmEnv:
    """
    Discrete-time UAV swarm environment.
    Positions are float32 arrays of shape (N, 2).
    A step takes a per-node 2D velocity command (already clipped to v_max by the policy)
    and advances positions, then recomputes connectivity + coverage.
    """

    def __init__(self, cfg: SimConfig, deployment: str = "random"):
        assert deployment in ("random", "uniform")
        self.cfg = cfg
        self.deployment = deployment
        self.rng = np.random.default_rng(cfg.seed)
        self.positions = self._init_positions()
        self.alive_mask = np.ones(len(self.positions), dtype=bool)
        self.t = 0
        self._coverage_grid = self._build_coverage_grid()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _init_positions(self):
        cfg = self.cfg
        if self.deployment == "random":
            pos = self.rng.uniform(0, cfg.area_size, size=(cfg.n_uavs_random, 2))
        else:
            side = cfg.grid_side_uniform
            xs = np.linspace(0, cfg.area_size, side, endpoint=False) + cfg.area_size / (2 * side)
            ys = np.linspace(0, cfg.area_size, side, endpoint=False) + cfg.area_size / (2 * side)
            gx, gy = np.meshgrid(xs, ys)
            pos = np.stack([gx.ravel(), gy.ravel()], axis=1)
        return pos.astype(np.float32)

    def _build_coverage_grid(self):
        cfg = self.cfg
        n = int(cfg.area_size / cfg.coverage_grid_res)
        xs = (np.arange(n) + 0.5) * cfg.coverage_grid_res
        gx, gy = np.meshgrid(xs, xs)
        return np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M, 2)

    def apply_failure(self, failure_ratio: float):
        """Randomly kill a fraction of UAVs. Matches paper's 'unpredictable random
        failure patterns' description (exact failure-selection algorithm not specified
        in the paper, so uniform-random node removal is used as a documented deviation)."""
        n = len(self.positions)
        n_fail = int(round(failure_ratio * n))
        fail_idx = self.rng.choice(n, size=n_fail, replace=False)
        self.alive_mask[fail_idx] = False
        self.t = 0

    # ------------------------------------------------------------------ #
    # Core quantities (Definitions 1-5)
    # ------------------------------------------------------------------ #
    def alive_positions(self):
        return self.positions[self.alive_mask]

    def adjacency(self):
        """Definition 2/3: link exists iff pairwise distance <= R_c."""
        pos = self.alive_positions()
        if len(pos) == 0:
            return np.zeros((0, 0), dtype=bool)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        adj = (dist <= self.cfg.comm_radius) & (dist > 1e-9)
        return adj

    def num_connected_components(self):
        """Definition 2, Eq.(1): C_t via graph connected components."""
        adj = self.adjacency()
        n = adj.shape[0]
        if n == 0:
            return 0
        g = nx.from_numpy_array(adj)
        return nx.number_connected_components(g)

    def coverage_ratio(self):
        """Definition 5, Eq.(6)-(7): Monte-Carlo grid estimate of union-of-disks coverage."""
        pos = self.alive_positions()
        if len(pos) == 0:
            return 0.0
        grid = self._coverage_grid
        # chunked distance computation to bound memory
        covered = np.zeros(len(grid), dtype=bool)
        chunk = 2000
        for i in range(0, len(pos), chunk):
            sub = pos[i:i + chunk]
            d = np.linalg.norm(grid[:, None, :] - sub[None, :, :], axis=-1)
            covered |= (d <= self.cfg.cov_radius).any(axis=1)
        return covered.mean()

    def average_node_degree(self):
        adj = self.adjacency()
        if adj.shape[0] == 0:
            return 0.0
        return adj.sum(axis=1).mean()

    # ------------------------------------------------------------------ #
    # Step dynamics
    # ------------------------------------------------------------------ #
    def step(self, velocities: np.ndarray):
        """velocities: (N_alive, 2) array, already clipped to v_max by the caller policy."""
        cfg = self.cfg
        idx = np.where(self.alive_mask)[0]
        v = np.clip(velocities, -cfg.v_max, cfg.v_max)
        # position update (C3 in paper's Problem Definition)
        self.positions[idx] = self.positions[idx] + cfg.dt * v
        self.positions[idx] = np.clip(self.positions[idx], 0, cfg.area_size)
        self.t += 1

    def is_recovered(self):
        return self.num_connected_components() <= 1

    def recovery_time_ratio(self, t_r):
        """Definition 4, Eq.(5)."""
        return t_r / self.cfg.T
