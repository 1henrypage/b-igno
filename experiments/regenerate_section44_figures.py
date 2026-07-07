"""Regenerate Section 4.4 figures from structured JSON results.

Produces 8 figures for the writeup:
- 4 noise sweep calibration overlays (one per benchmark)
- 4 sensor sweep CI-width-vs-sensors plots (one per benchmark)

Uses seed_42 results only. Reads from experiments/results/structured/.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    SCIENCE_STYLE = ["science", "no-latex"]
except ImportError:
    SCIENCE_STYLE = None

STRUCTURED_DIR = Path(__file__).parent / "results" / "structured"
OUTPUT_DIR = Path(os.path.expanduser("~/school/writeup/figures"))

SEED = 42

PROBLEMS = [
    ("darcy_continuous", "cal_noise_sweep_darcy_cont.png", "sensor_sweep_darcy_ci.png"),
    ("darcy_piecewise", "cal_noise_sweep_darcy_pw.png", "sensor_sweep_darcy_pw_ci.png"),
    ("eit", "cal_noise_sweep_eit.png", "sensor_sweep_eit_ci.png"),
    ("burgers", "cal_noise_sweep_burgers.png", "sensor_sweep_burgers_ci.png"),
]

NOISE_LABEL_ORDER = ["Clean", "SNR=50dB", "SNR=35dB", "SNR=25dB", "SNR=15dB"]
MARKERS = ["o", "s", "^", "D", "v", "p", "*"]


def load_jsons(sweep_type, problem, seed):
    """Load all JSON results for a given sweep type, problem, and seed."""
    sweep_dir = STRUCTURED_DIR / sweep_type
    results = []
    for f in sorted(sweep_dir.iterdir()):
        if not f.name.endswith(".json"):
            continue
        if f"_seed{seed}.json" not in f.name:
            continue
        if not f.name.startswith(problem + "_"):
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def extract_noise_sweep_data(results):
    """Extract (cal_levels, cal_empirical, label) tuples from noise sweep results.

    Each JSON has either a baseline (label="Clean") or sweep_conditions[0]
    with an SNR label. Returns list sorted by NOISE_LABEL_ORDER.
    """
    entries = {}
    for r in results:
        bl = r.get("baseline")
        scs = r.get("sweep_conditions", [])
        if bl and bl.get("cal_levels"):
            entries[bl["label"]] = (
                np.array(bl["cal_levels"]),
                np.array(bl["cal_empirical"]),
                bl["label"],
            )
        for sc in scs:
            if sc.get("cal_levels"):
                entries[sc["label"]] = (
                    np.array(sc["cal_levels"]),
                    np.array(sc["cal_empirical"]),
                    sc["label"],
                )

    ordered = []
    for label in NOISE_LABEL_ORDER:
        if label in entries:
            ordered.append(entries[label])
    return ordered


def plot_calibration_overlay(calibration_list, save_path):
    """Overlay multiple calibration curves on one plot."""
    ctx = plt.style.context(SCIENCE_STYLE) if SCIENCE_STYLE else _nullcontext()
    with ctx:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")

        for i, (nominal, empirical, label) in enumerate(calibration_list):
            marker = MARKERS[i % len(MARKERS)]
            ax.plot(nominal, empirical, f"{marker}-", markersize=5, label=label)

        ax.set_xlabel("Nominal Coverage", fontsize=14)
        ax.set_ylabel("Empirical Coverage", fontsize=14)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(labelsize=13)
        ax.legend(
            fontsize=9, loc="lower right",
            framealpha=0.3, facecolor="white", edgecolor="black",
        )
        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved calibration overlay -> {save_path}")


def extract_sensor_sweep_data(results):
    """Extract (n_obs, ci_width) pairs from sensor sweep results.

    Each JSON has sweep_conditions[0] with label like "n_obs=25".
    Returns sorted list of (n_obs_int, ci_width).
    """
    points = []
    for r in results:
        for sc in r.get("sweep_conditions", []):
            label = sc.get("label", "")
            if label.startswith("n_obs="):
                n_obs = int(label.split("=")[1])
                ci_width = sc["ci_width"]
                points.append((n_obs, ci_width))
    points.sort(key=lambda x: x[0])
    return points


def plot_ci_width_vs_sensors(points, save_path):
    """Plot CI width vs number of sensors."""
    n_obs_vals = [p[0] for p in points]
    ci_widths = [p[1] for p in points]

    ctx = plt.style.context(SCIENCE_STYLE) if SCIENCE_STYLE else _nullcontext()
    with ctx:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(n_obs_vals, ci_widths, "o-", markersize=7, color="C0", linewidth=1.5)

        ax.set_xlabel("Number of sensors", fontsize=14)
        ax.set_ylabel("Mean CI width", fontsize=14)
        ax.tick_params(labelsize=13)

        ax.set_xticks(n_obs_vals)
        ax.set_xticklabels([str(v) for v in n_obs_vals])

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved CI width plot -> {save_path}")


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for problem, cal_filename, sensor_filename in PROBLEMS:
        print(f"\n{'='*50}")
        print(f"Problem: {problem}")
        print(f"{'='*50}")

        # --- Noise sweep calibration ---
        noise_results = load_jsons("noise_sweep", problem, SEED)
        cal_data = extract_noise_sweep_data(noise_results)
        print(f"  Noise sweep: found {len(cal_data)} conditions: "
              f"{[c[2] for c in cal_data]}")

        if cal_data:
            plot_calibration_overlay(cal_data, OUTPUT_DIR / cal_filename)
        else:
            print(f"  WARNING: No calibration data for {problem}")

        # --- Sensor sweep CI width ---
        sensor_results = load_jsons("sensor_sweep", problem, SEED)
        sensor_data = extract_sensor_sweep_data(sensor_results)
        print(f"  Sensor sweep: found {len(sensor_data)} points: "
              f"{[(n, f'{w:.4f}') for n, w in sensor_data]}")

        if sensor_data:
            plot_ci_width_vs_sensors(sensor_data, OUTPUT_DIR / sensor_filename)
        else:
            print(f"  WARNING: No sensor sweep data for {problem}")


if __name__ == "__main__":
    main()
