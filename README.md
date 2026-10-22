# Simple Hybrids, Smooth Schedules

Code and paper for *"HAC: Simple Hybrids, Smooth Schedules --- Rethinking Adaptive Fusion for Resilient UAV Swarm Connectivity Recovery."*

The paper studies self-healing connectivity recovery for UAV swarms after partial node failure, building on and simplifying [R2C (Peng et al., AAAI-26)](https://ojs.aaai.org). We show that a streamlined hybrid architecture (**HAC**: a parameter-free virtual-force heuristic branch fused at the velocity level with a single global graph-convolution branch) achieves near-perfect connectivity restoration across 20%--80% failure rates and retains 88.7%--90.7% of R2C's reported coverage without subnetwork partitioning or convex-hull boundary extraction. We further diagnose why R2C's steep adaptive fusion schedule degrades robustness when transplanted into a global hybrid, tracing the cause to transition *steepness* ($k$) rather than handover *timing* ($s_0$) via an implicit training-curriculum analysis (`HAC-Smooth`, $k=5, s_0=0.7$), and verify via a subnetwork-level counter-experiment (**HSC**) that schedule smoothing requires node-level velocity expressiveness.

**Paper**: [`paper/main.pdf`](paper/main.pdf) (LaTeX source [`paper/main.tex`](paper/main.tex) and [`paper/references.bib`](paper/references.bib) in the same directory, compilable with `pdflatex` + `bibtex`).

## Repository Structure

```text
paper/              LaTeX source (main.tex), compiled PDF (main.pdf), bibliography (references.bib)
code/
  environment.yml   conda environment specification
  sim/
    swarm_env.py    Simulation environment (temporal graph, connectivity, coverage, failure model)
    graph_utils.py  Shared connected-component and Monte Carlo grid sampling utilities
  models/
    heuristic_force.py  Parameter-free buffered virtual-force branch (Eq. 2)
    global_gcn.py       Global k-NN GCN (GlobalGACPolicy), subnetwork centroid GCN
                        (SubnetSACPolicy), and fusion-weight schedules (Eqs. 3-4)
    losses.py           Count-invariant soft-minimum connectivity surrogate (Eq. 5),
                        coverage surrogate, and movement cost losses
  scripts/
    train_hybrid.py             Unified training for HAC, HSC, Learned-only, and SAC (Algorithm 1)
    evaluate_hybrid.py          Deployment rollout evaluation + train/eval schedule decoupling (Algorithm 2)
    evaluate_heuristic_only.py  Heuristic-only evaluation (zero learnable parameters)
    run_full_multiseed_study.sh Resumable 5-seed suite covering HAC-Fixed, HAC-Smooth,
                                Learned-only, Heuristic-only, SAC-Fixed, HSC, and schedule sweeps
    aggregate_multiseed.py      Aggregates per-seed CSVs into mean +/- std summary tables
```

## Installation

```bash
cd code
conda env create -f environment.yml
conda activate hac-uav-recovery
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start (~10 minutes on GPU)

Train and evaluate **HAC-Fixed** and **HAC-Smooth** ($k=5, s_0=0.7$) for one seed on the random deployment:

```bash
cd code
# 1. HAC-Fixed
python scripts/train_hybrid.py --variant fixed --granularity global --deployment random \
    --seed 1 --epochs 300 --out results/hac_fixed_random_seed1.pt
python scripts/evaluate_hybrid.py --ckpt results/hac_fixed_random_seed1.pt \
    --deployment random --seed 1 --out results/eval_hac_fixed_random_seed1.csv

# 2. HAC-Smooth (k=5, s0=0.7)
python scripts/train_hybrid.py --variant adaptive --granularity global --deployment random \
    --awf_k 5.0 --awf_s0 0.7 --seed 1 --epochs 300 --out results/hac_smooth_k5_s07_random_seed1.pt
python scripts/evaluate_hybrid.py --ckpt results/hac_smooth_k5_s07_random_seed1.pt \
    --deployment random --seed 1 --out results/eval_hac_smooth_k5_s07_random_seed1.csv
```

## Full 5-Seed Reproduction Suite

Reproduces the 5-seed experiments across random and uniform deployments (`HAC-Fixed`, `HAC-Smooth`, `Learned-only`, `Heuristic-only`, `SAC-Fixed`, `HSC` counter-experiments, the Table 6 schedule steepness sweep, and Train-Eval schedule decoupling):

```bash
cd code
nohup bash scripts/run_full_multiseed_study.sh > multiseed_log.txt 2>&1 &
tail -f multiseed_log.txt
```

Once complete (or partially complete), aggregate all runs into `mean +/- std` tables:

```bash
python scripts/aggregate_multiseed.py
```

## License

MIT (see `LICENSE`).
