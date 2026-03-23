"""Experiment driver for FL parameter sweeps.

Runs flwr simulations with different parameter configurations and saves
results to results/experiments.toml via server_app.py.

Usage:
    python -m scripts.run_experiments                        # run all experiments
    python -m scripts.run_experiments --experiment epochs     # run one sweep
    python -m scripts.run_experiments --experiment fraction_fit --experiment strategy
"""

import argparse
import subprocess
import sys
from itertools import product

# --- Experiment definitions ---
# Each experiment sweeps one parameter while keeping others at defaults.

DEFAULTS = {
    "num-server-rounds": 25,
    "fraction-fit": 0.5,
    "fraction-evaluate": 0.0,
    "local-epochs": 5,
    "learning-rate": 0.01,
    "batch-size": 64,
    "num-clients": 25,
    "strategy": "FedAvg",
    "model": "resnet18",
}

EXPERIMENTS = {
    "epochs": {
        "param": "local-epochs",
        "values": [1, 5, 10, 25],
    },
    "fraction_fit": {
        "param": "fraction-fit",
        "values": [0.1, 0.25, 0.5, 0.75, 1.0],
    },
    "strategy": {
        "param": "strategy",
        "values": ["FedAvg", "FedProx", "FedAdam", "FedAdagrad", "FedAvgM"],
    },
}


def build_run_config(overrides: dict) -> str:
    """Build a --run-config string from overrides."""
    config = {**DEFAULTS, **overrides}
    return " ".join(f"{k}={v}" for k, v in config.items())


def run_single(overrides: dict, dry_run: bool = False) -> int:
    """Run a single flwr simulation with the given config overrides."""
    run_config = build_run_config(overrides)
    cmd = ["flwr", "run", ".", "--run-config", run_config]

    desc = " ".join(f"{k}={v}" for k, v in overrides.items())
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        return 0

    result = subprocess.run(cmd, cwd=".")
    return result.returncode


def run_experiment(name: str, dry_run: bool = False) -> None:
    """Run all configurations for a named experiment."""
    exp = EXPERIMENTS[name]
    param = exp["param"]
    values = exp["values"]

    print(f"\n>>> Experiment: {name} (sweeping {param})")
    print(f">>> Values: {values}\n")

    for val in values:
        rc = run_single({param: val}, dry_run=dry_run)
        if rc != 0:
            print(f"WARNING: run with {param}={val} exited with code {rc}")


def main():
    parser = argparse.ArgumentParser(description="FL parameter sweep experiments")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        action="append",
        help="Which experiment(s) to run. Omit to run all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    experiments = args.experiment or list(EXPERIMENTS.keys())

    print("Experiments to run:", experiments)
    print(f"Results will be saved to results/experiments.toml")

    for name in experiments:
        run_experiment(name, dry_run=args.dry_run)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
