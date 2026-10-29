# HAC: Node-Level Hybrid Force–Graph Control and Smooth Handover Scheduling for Resilient UAV Swarm Connectivity Recovery

Official repository for the paper **"HAC: Node-Level Hybrid Force–Graph Control and Smooth Handover Scheduling for Resilient UAV Swarm Connectivity Recovery"** (`paper/main.pdf`).

## Overview

Self-healing communication connectivity (SCC) for unmanned aerial vehicle (UAV) swarms requires rapidly reconnecting fractured subnetworks after partial node failures while preserving wide spatial coverage. This repository implements **Hybrid Adaptive Convolution (HAC)**, a lightweight node-level hybrid controller that fuses a parameter-free buffered Hookean virtual-force heuristic with a single global $k$-NN Graph Convolutional Network (GCN) at the velocity level, trained end-to-end via a cardinality-invariant soft-minimum connectivity surrogate loss.

### Key Findings (5 Independent Random Seeds × 50 Trials per Configuration)
1. **Node-Level Hybrid Resilience & Efficiency:** Under constant $50/50$ fusion weights (`HAC-Fixed`), a single 4-layer global GCN fused with a parameter-free virtual-force heuristic achieves near-perfect connectivity recovery ($46.6 \pm 3.9 / 50$ random, $49.4 \pm 0.8 / 50$ uniform at $80\%$ node failure), $1.5\times$–$2.2\times$ faster execution than learned baselines, and retains **$88.4\%$–$91.2\%$** of the coverage reported by complex multipartite boundary-hull architectures (R2C) without subnetwork partitioning or Quickhull extraction.
2. **Smooth vs. Steep Adaptive Handover:** Transplanting a steep adaptive fusion schedule ($k=30, s_0=0.125$) into `HAC` degrades $80\%$-failure recovery to $14.2 \pm 4.3 / 50$, whereas widening the transition window (`HAC-Smooth`, $k=5, s_0=0.7$) restores recovery to $43.6 \pm 2.8 / 50$ (random) and $47.8 \pm 1.3 / 50$ (uniform).
3. **$3 \times 3$ Decoupled Training vs. Inference Matrix:** Cross-evaluating policies trained under `Fixed`, `Smooth`, and `Steep` schedules against all three inference schedules falsifies the training-corruption hypothesis. Policies trained under `Steep` scheduling achieve **$46.6 \pm 0.5 / 50$** (with `Fixed` eval) and **$46.4 \pm 0.5 / 50$** (with `Smooth` eval) and the highest coverage ($0.622$–$0.624$) at $80\%$ failure, proving that steep-schedule collapse is an inference-time phenomenon caused by prematurely extinguishing the heuristic link-retention barrier.
4. **Spatial Granularity Counter-Experiment (`HSC`):** Replacing the node-level global GCN with a subnetwork-level rigid-translation GCN (`HSC`) results in deterministic collapse at $80\%$ failure ($0.0 \pm 0.0$ to $0.4 \pm 0.5 / 50$) regardless of schedule smoothness, separating inference-time handover stability from architectural expressiveness.

---

## Repository Structure

~~text
├── paper/
│   ├── main.tex                  # Manuscript LaTeX source
│   ├── main.pdf                  # Compiled manuscript (22 pages)
│   └── references.bib            # Bibliography database
└── code/
    ├── environment.yml           # Conda environment specification
    ├── models/
    │   ├── global_gcn.py         # Global k-NN GCN & subnetwork-level policies
    │   ├── heuristic_force.py    # Parameter-free 3-zone buffered Hookean force field
    │   └── losses.py             # Cardinality-invariant soft-minimum connectivity & coverage losses
    ├── sim/
    │   ├── graph_utils.py        # Proximity graph, Laplacian components, & k-NN construction
    │   └── swarm_env.py          # 2D UAV swarm deployment, failure injection, & rollout kinematics
    ├── scripts/
    │   ├── train_hybrid.py       # Differentiable multi-step rollout training
    │   ├── evaluate_hybrid.py    # Closed-loop inference evaluation (supports decoupled schedules)
    │   ├── evaluate_heuristic_only.py
    │   ├── aggregate_multiseed.py
    │   └── run_full_multiseed_study.sh
    └── results/                  # 5-seed aggregated CSV logs for all benchmarks & ablations
~~

---

## Main Benchmark Results (5 Seeds × 50 Trials)

### 1. Random Deployment ($N=200$, $80\%$ Node Failure)

| Architecture | Success / 50 ($\uparrow$) | REC ($\downarrow$) | COV ($\uparrow$) | AND | Runtime (s) ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HAC-Fixed** ($50/50$) | **46.6 ± 3.9** | 0.696 ± 0.048 | 0.602 ± 0.009 | 2.578 ± 0.027 | **1.282 ± 0.108** |
| **HAC-Smooth** ($k=5, s_0=0.7$) | 43.6 ± 2.8 | **0.684 ± 0.020** | **0.616 ± 0.006** | **2.479 ± 0.058** | 1.291 ± 0.043 |
| **GAC-Fixed** (Global Learned) | 19.8 ± 5.5 | 0.775 ± 0.017 | 0.310 ± 0.009 | 6.004 ± 0.260 | 1.910 ± 0.080 |
| **Learned-only** ($w_T=1$) | 8.6 ± 3.3 | 0.831 ± 0.023 | 0.347 ± 0.023 | 5.104 ± 0.334 | 1.894 ± 0.057 |
| **SAC-Fixed** (Subnet Rigid) | 0.0 ± 0.0 | n/a | n/a | n/a | 1.778 ± 0.125 |
| **Heuristic-only** ($w_E=1$) | 0.0 ± 0.0 | n/a | n/a | n/a | 0.551 ± 0.021 |

### 2. Decoupled $3 \times 3$ Training vs. Evaluation Schedule Matrix (Random Deployment, $80\%$ Failure)

| Training Schedule $w_T^{\mathrm{train}}$ | Evaluation Schedule $w_T^{\mathrm{eval}}$ | Success / 50 ($\uparrow$) | REC ($\downarrow$) | COV ($\uparrow$) | AND |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fixed** ($50/50$) | **Fixed** ($50/50$) | **46.6 ± 3.9** | 0.696 ± 0.048 | 0.602 ± 0.009 | 2.578 ± 0.027 |
| **Fixed** ($50/50$) | **Smooth** ($5, 0.7$) | 42.6 ± 5.0 | 0.702 ± 0.042 | 0.613 ± 0.008 | 2.483 ± 0.026 |
| **Fixed** ($50/50$) | **Steep** ($30, 0.125$) | 9.8 ± 3.8 | 0.768 ± 0.070 | 0.439 ± 0.054 | 3.828 ± 0.508 |
| **Smooth** ($5, 0.7$) | **Fixed** ($50/50$) | **47.4 ± 1.9** | **0.674 ± 0.027** | 0.602 ± 0.005 | 2.580 ± 0.026 |
| **Smooth** ($5, 0.7$) | **Smooth** ($5, 0.7$) | 43.6 ± 2.8 | 0.684 ± 0.020 | 0.616 ± 0.006 | 2.479 ± 0.058 |
| **Smooth** ($5, 0.7$) | **Steep** ($30, 0.125$) | 12.0 ± 4.6 | 0.797 ± 0.037 | 0.410 ± 0.027 | 4.057 ± 0.220 |
| **Steep** ($30, 0.125$) | **Fixed** ($50/50$) | **46.6 ± 0.5** | 0.679 ± 0.023 | 0.622 ± 0.004 | 2.509 ± 0.048 |
| **Steep** ($30, 0.125$) | **Smooth** ($5, 0.7$) | **46.4 ± 0.5** | 0.706 ± 0.018 | **0.624 ± 0.003** | **2.474 ± 0.043** |
| **Steep** ($30, 0.125$) | **Steep** ($30, 0.125$) | 14.2 ± 4.3 | 0.796 ± 0.032 | 0.413 ± 0.032 | 4.114 ± 0.296 |

---

## Quick Start & Reproduction

### 1. Environment Setup
~~bash
conda env create -f code/environment.yml
conda activate hac-uav
~~

### 2. Run the Full 5-Seed Study & Aggregate Results
~~bash
bash code/scripts/run_full_multiseed_study.sh
python code/scripts/aggregate_multiseed.py
~~

### 3. Compile the Paper PDF
~~bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
~~
