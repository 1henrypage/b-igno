<div align="center">

# Bayesian Inverse Generative Neural Operator

**Latent-Space Posterior Formulation for PDE-Constrained Inverse Problems**

Henry Page, MSc Thesis, TU Delft (2026)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![JAX](https://img.shields.io/badge/framework-JAX-orange)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

This repository contains B-IGNO, a JAX implementation that extends the Inverse Generative Neural Operator [Bao & Zang, 2025](https://arxiv.org/abs/2511.03241) with Bayesian posterior inference. Once IGNO's encoders and decoders are trained, B-IGNO places a normalising flow prior over the low-dimensional latent space and samples the posterior using No-U-Turn Sampling. The thesis is available at [repository.tudelft.nl](https://repository.tudelft.nl/record/uuid:742ebfba-6be5-426c-8bdf-fb9689f7f2af).

## Benchmark problems

| Problem | Latent dim | Description |
|---|---|---|
| Darcy Continuous | d = 6 | Steady-state Darcy flow with smooth RBF coefficient function |
| Darcy Piecewise | d = 200 | Darcy flow with binary (low/high) piecewise-constant coefficients |
| EIT | d_a = 6 | Electrical impedance tomography with 20 boundary conditions, 128 sensors |
| Burgers | d = 16 | Time-dependent 1D Burgers equation (viscosity = 0.1/π) |

## Repository structure

```
b-igno/
├── src/                  # Core library (models, problems, solvers, evaluation)
│   ├── components/       # Neural network building blocks (NF, CNN, encoders)
│   ├── problems/         # Problem definitions (Darcy, EIT, Burgers)
│   ├── evaluation/       # Metrics, Laplace approximation
│   └── solver/           # Training loop and configuration
├── experiments/          # Experiment scripts
│   ├── experiment_utils/ # Shared library
│   ├── results/          # Executed notebooks and structured JSON results
│   └── figures/          # Generated figures
├── configs/              # Training configuration files
│   └── training/
├── data/                 # datasets (included, ~several hundred MB)
├── runs/                 # Pre-trained checkpoints
├── training.py           # Training entry point
└── slurm/                # SLURM submission scripts
    ├── slurm_common.sh   # Submission helpers and time limits
    └── run_batch_{1,2,3}.sh # Batch submission scripts
```

## Installation

**Prerequisites:** Python 3.11, a CUDA-capable GPU, and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/1henrypage/b-igno b-igno
cd b-igno
uv sync --extra dev
```

> [!IMPORTANT]
> Every script must `import load_this_before_everything_else` as its **first** import.
> This sets JAX x64 mode and disables TF32. Without it, PDE gradient computations are
> silently numerically unstable — no error, just wrong results.

The clone is approximately 1 GB because pre-trained checkpoints and datasets are included.

## Data

Full datasets are available on [Google Drive](https://drive.google.com/drive/folders/1oIS-415hzStFDywAWu1mdApWfSE70bhq?usp=sharing). Download and place the contents into `data/`.

## Training

Pre-trained checkpoints are included in `runs/`. To train from scratch:

```bash
uv run --extra dev python training.py configs/training/darcy_continuous.yaml
```

Config files for all four problems are in `configs/training/`.

## Experiments

Six experiment types are implemented. Each is a `.py` script that is converted to and executed as a Jupyter notebook:

| Type | Description |
|---|---|
| `baseline` | MCMC posterior in latent space, compared to unconditional prior |
| `map_laplace` | MAP estimate + Laplace approximation baseline |
| `physics` | Data-only vs physics-informed posterior |
| `ood` | Out-of-distribution generalisation |
| `noise_sweep` | Posterior quality vs observation noise level |
| `sensor_sweep` | Posterior quality vs number of sensors |

Scripts follow the naming convention `{type}_{problem}[_variant].py`. The `_5v10`, `_5v100`, `_5v1000` variants apply only to Darcy Piecewise and test higher contrast coefficient ratios.

**Run a single experiment locally:**

```bash
cd experiments
uv run --extra dev python baseline_darcy_continuous.py
```

**Aggregate cross-seed results:**

```bash
uv run --extra dev python experiments/aggregate_seeds.py --all
uv run --extra dev python experiments/aggregate_laplace.py --all
```

Structured JSON results are written to `experiments/results/structured/{experiment}/{problem}_{timestamp}_seed{N}_test{N}.json`.

## SLURM cluster usage

The batch scripts submit job arrays to TU Delft's DAIC cluster via Apptainer:

```bash
./slurm/run_batch_1.sh                    # baseline + map_laplace (14 jobs)
./slurm/run_batch_2.sh                    # ood + noise_sweep + sensor_sweep (12 jobs)
./slurm/run_batch_3.sh                    # physics (4 jobs)
./slurm/run_batch_1.sh baseline_darcy     # pattern match (substring)
```

Each script converts `.py` → `.ipynb`, executes inside the container (`jigno.def`), and saves the executed notebook to `experiments/results/`. See `slurm/slurm_common.sh` for time limits and configuration.

## Citation

```bibtex
@mastersthesis{page2026bayesian,
  title  = {Bayesian Inverse Generative Neural Operator: Latent-Space Posterior
            Formulation for {PDE}-Constrained Inverse Problems},
  author = {Page, Henry},
  school = {Delft University of Technology},
  year   = {2026},
  url    = {https://repository.tudelft.nl/record/uuid:742ebfba-6be5-426c-8bdf-fb9689f7f2af},
}

@article{bao2025igno,
  title   = {A unified physics-informed generative operator framework
             for general inverse problems},
  author  = {Bao, Gang and Zang, Yaohua},
  journal = {arXiv preprint arXiv:2511.03241},
  year    = {2025},
}

@article{zang2025dgenno,
  title   = {{DGenNO}: a novel physics-aware neural operator for solving
             forward and inverse {PDE} problems based on deep, generative
             probabilistic modeling},
  author  = {Zang, Yaohua and Koutsourelakis, Phaedon-Stelios},
  journal = {Journal of Computational Physics},
  volume  = {538},
  pages   = {114137},
  year    = {2025},
  doi     = {10.1016/j.jcp.2025.114137},
}
```

## Acknowledgements

Computational resources were provided by the DAIC cluster at Delft University of Technology (RRID: SCR_025091).

Thesis committee: Dr. Jing Sun, Dr. Alexander Heinlein, Dr. David Tax (responsible advisor), Prof. dr. M.M. de Weerdt.
