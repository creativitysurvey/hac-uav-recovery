# Simple Hybrids, Smooth Schedules

Code and paper for *"Simple Hybrids, Smooth Schedules: Rethinking Adaptive
Fusion for Resilient UAV Swarm Connectivity Recovery."*

The paper studies self-healing connectivity recovery for UAV swarms after
partial node failure, building on and simplifying
[R2C (Peng et al., AAAI-26)](https://ojs.aaai.org). We show that a
markedly simpler hybrid architecture (HAC: a parameter-free virtual-force
heuristic branch fused with a single global graph-convolution branch)
matches or exceeds the robustness of considerably more complex learned
baselines, and diagnose why R2C's adaptive fusion schedule, when
transplanted into this simplified architecture, initially degrades rather
than improves robustness -- tracing the cause to the *steepness*, not the
*timing*, of the schedule's transition.

**Paper**: [`paper/main.pdf`](paper/main.pdf) (LaTeX source and `.bib` in
the same directory, fully self-contained and compilable with `pdflatex`).

---

## Repository Structure

```
paper/              LaTeX source, compiled PDF, bibliography
code/
  environment.yml    conda environment specification
  sim/
    swarm_env.py      Simulation environment (Definitions 1-5: temporal
                       graph, connectivity, coverage, failure model)
    graph_utils.py     Shared graph/sampling utilities
  models/
    heuristic_force.py Heuristic branch (Eq. 1) -- no learnable parameters
    global_gcn.py       Learned branch: global k-NN graph convolution +
                         fusion-weight schedule classes (Eq. 2)
    losses.py           Differentiable training losses, including the
                         corrected connectivity surrogate (see "Known
                         Deviations" below)
  scripts/
    train_hybrid.py     HAC training (Algorithm 1)
    evaluate_hybrid.py  HAC evaluation / deployment rollout (Algorithm 2)
    evaluate_heuristic_only.py
                         Heuristic-only ablation evaluation (no training)
    run_full_multiseed_study.sh
                         Resumable batch script: 5 seeds x 2 deployments
                         for HAC-Fixed + learned-only + heuristic-only
    aggregate_multiseed.py
                         Aggregates per-seed CSVs into mean +/- std
                         summary tables (reproduces Tables 3-4's HAC-Fixed
                         / Learned-only / Heuristic-only rows)
```

## Scope Note (Read This First)

This repository covers **only the HAC main line** (the paper's proposed
architecture and its direct ablations). The paper's main results tables
(Tables 3-4) also report **GAC-Fixed** and **SAC-Fixed** baseline rows,
produced by a separate motivating-experiments codebase not included here;
their numbers are reported in the paper verbatim from the original
experimental logs (see Appendix A of the paper). This keeps the released
artifact focused and independently reviewable without pulling in an
unrelated codebase's baselines.

## Requirements

- A CUDA-capable GPU is strongly recommended (the full multi-seed study
  is impractical on CPU alone; a single training run is feasible but slow).
- `conda` (or `mamba`) for environment management.

## Installation

```bash
cd code
conda env create -f environment.yml
conda activate hac-uav-recovery
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start (Kick the Tires, ~10 minutes on GPU)

Train and evaluate HAC-Fixed for one seed on the random deployment:

```bash
cd code
python scripts/train_hybrid.py --variant fixed --deployment random \
    --seed 1 --epochs 300 --out results/hac_fixed_random_seed1.pt
python scripts/evaluate_hybrid.py --ckpt results/hac_fixed_random_seed1.pt \
    --deployment random --out results/eval_hac_fixed_random_seed1.csv
cat results/eval_hac_fixed_random_seed1.csv
```

You should see four rows (one per failure rate, 20/40/60/80%) with
`n_valid_of_50` in the high-40s at 20-60% failure, consistent with
Table 3 of the paper.

## Full Reproduction

Reproduces Tables 3-4's HAC-Fixed, Learned-only, and Heuristic-only rows
(5 seeds x 2 deployments x 3 architectures):

```bash
cd code
chmod +x scripts/run_full_multiseed_study.sh
nohup bash scripts/run_full_multiseed_study.sh > multiseed_log.txt 2>&1 &
tail -f multiseed_log.txt
```

This is resumable (safe to interrupt and re-run) and takes several hours
on a single modern GPU. Once complete (or partially complete):

```bash
python scripts/aggregate_multiseed.py
```

This reproduces the mean +/- std summary reported in Tables 3-4 for the
three architectures covered by this repository.

### Reproducing the Schedule-Steepness Sweep (Table 6)

```bash
python scripts/train_hybrid.py --variant adaptive --deployment random \
    --awf_k 5 --awf_s0 0.7 --out results/hac_adaptive_random_k5_s07.pt --epochs 300
python scripts/evaluate_hybrid.py --ckpt results/hac_adaptive_random_k5_s07.pt \
    --deployment random --out results/eval_hac_adaptive_random_k5_s07.csv
```

Vary `--awf_k` / `--awf_s0` to reproduce other rows of Table 6.

### Reproducing the Branch-Contribution Ablation (Section 9.1)

Already covered by the multi-seed study above (`learned_only` uses
`--force_wT 1.0`; `heuristic_only` requires no training). To run either
in isolation at a single seed:

```bash
python scripts/train_hybrid.py --variant fixed --deployment random --force_wT 1.0 \
    --seed 1 --out results/learned_only_random_seed1.pt --epochs 300
python scripts/evaluate_heuristic_only.py --deployment random --seed 1 \
    --out results/eval_heuristic_only_random_seed1.csv
```

## Known Deviations from R2C's Original Formulation

Logged in full in the paper's Appendix A, including: the failure-node
selection procedure (not fully specified by R2C), the differentiable
connectivity-loss surrogate and a count-invariance bug in an earlier
version of it (verified numerically in the paper and fixed in the code
here), the truncated training rollout relative to the full evaluation
horizon, and a weight-initialization seeding fix applied before the
five-seed studies reported in the paper.

## Citation

```bibtex
@misc{hac2026,
  title={Simple Hybrids, Smooth Schedules: Rethinking Adaptive Fusion for Resilient UAV Swarm Connectivity Recovery},
  year={2026}
}
```

This work builds on:

```bibtex
@inproceedings{peng2026r2c,
  title={Resilient {UAV} Swarm with Fast Connectivity Recovery and Extensive Coverage},
  author={Peng, Yabin and Zhou, Chenyu and Cui, Hainan and Duan, Tong and Chen, Haoyang and Zhang, Fan and Liu, Shaoxun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  pages={917--925},
  year={2026}
}
```

## License

MIT (see `LICENSE`).
