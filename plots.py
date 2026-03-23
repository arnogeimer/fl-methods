"""Plot FL experiment results from results/experiments.toml.

Usage:
    python plots.py epochs          # compare local epoch values (accuracy)
    python plots.py fraction_fit    # compare fraction_fit values
    python plots.py strategy        # compare aggregation strategies
    python plots.py epochs --metric loss
    python plots.py epochs --both   # side-by-side accuracy + loss
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import toml

RESULTS_PATH = Path("results/experiments.toml")

# Map experiment names to the run field used for grouping
EXPERIMENT_FIELDS = {
    "epochs": "local_epochs",
    "fraction_fit": "fraction_fit",
    "strategy": "strategy",
}


def load_runs():
    """Load all runs from the experiments TOML."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run experiments first:\n"
            "  python -m scripts.run_experiments"
        )
    return toml.load(RESULTS_PATH)["runs"]


def filter_runs(runs, experiment: str):
    """Group runs by the swept parameter for a given experiment.

    Returns dict: {parameter_value: run_dict}
    """
    field = EXPERIMENT_FIELDS[experiment]
    grouped = {}
    for run in runs:
        key = run[field]
        # Keep the latest run for each parameter value
        grouped[key] = run
    return grouped


def plot_experiment(experiment: str, metric: str = "accuracy", save: bool = False):
    """Plot convergence curves for one experiment, single metric."""
    runs = load_runs()
    grouped = filter_runs(runs, experiment)
    field = EXPERIMENT_FIELDS[experiment]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, run in sorted(grouped.items(), key=lambda x: str(x[0])):
        values = run[metric]
        rounds = list(range(len(values)))
        ax.plot(rounds, values, label=f"{field}={label}", linewidth=1.5)

    ax.set_xlabel("Round")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"CIFAR-10 FL — {metric.capitalize()} by {field}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        out = Path("plots") / f"{experiment}_{metric}.pdf"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved to {out}")

    plt.show()


def plot_experiment_both(experiment: str, save: bool = False):
    """Plot accuracy and loss side by side for one experiment."""
    runs = load_runs()
    grouped = filter_runs(runs, experiment)
    field = EXPERIMENT_FIELDS[experiment]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, run in sorted(grouped.items(), key=lambda x: str(x[0])):
        rounds = list(range(len(run["accuracy"])))
        ax1.plot(rounds, run["accuracy"], label=f"{field}={label}", linewidth=1.5)
        ax2.plot(rounds, run["loss"], label=f"{field}={label}", linewidth=1.5)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Global Accuracy by {field}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Global Loss by {field}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"CIFAR-10 Federated Learning — {field} sweep", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        out = Path("plots") / f"{experiment}_both.pdf"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved to {out}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot FL experiment results")
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENT_FIELDS.keys()),
        help="Which experiment to plot.",
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "loss"],
        default="accuracy",
        help="Metric to plot (default: accuracy).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Plot accuracy and loss side by side.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plot to plots/ directory as PDF.",
    )
    args = parser.parse_args()

    if args.both:
        plot_experiment_both(args.experiment, save=args.save)
    else:
        plot_experiment(args.experiment, metric=args.metric, save=args.save)


if __name__ == "__main__":
    main()
