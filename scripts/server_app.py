"""pytorchexample: A Flower / PyTorch app."""

import json
import os
from pathlib import Path

import toml
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAdagrad, FedAdam, FedAvg, FedAvgM, FedProx

from task.cifar import Net, load_centralized_dataset, test

STRATEGIES = {
    "FedAvg": FedAvg,
    "FedProx": FedProx,
    "FedAdam": FedAdam,
    "FedAdagrad": FedAdagrad,
    "FedAvgM": FedAvgM,
}

# Create ServerApp
app = ServerApp()

accuracies = []
losses = []


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    strategy_name: str = context.run_config.get("strategy", "FedAvg")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize strategy
    strategy_cls = STRATEGIES.get(strategy_name, FedAvg)
    strategy = strategy_cls(fraction_evaluate=fraction_evaluate)

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save results to TOML
    _save_results(context.run_config, strategy_name)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to("cuda:0")

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader)

    accuracies.append(test_acc)
    losses.append(test_loss)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


def _save_results(run_config, strategy_name: str) -> None:
    """Append run results to the experiments TOML file."""
    results_path = Path("results/experiments.toml")
    results_path.parent.mkdir(exist_ok=True)

    # Load existing results or start fresh
    if results_path.exists():
        data = toml.load(results_path)
    else:
        data = {}

    runs = data.get("runs", [])

    run_entry = {
        "task": run_config.get("model", "resnet18"),
        "strategy": strategy_name,
        "local_epochs": int(run_config["local-epochs"]),
        "fraction_fit": float(run_config["fraction-fit"]),
        "num_server_rounds": int(run_config["num-server-rounds"]),
        "num_clients": int(run_config.get("num-clients", 25)),
        "learning_rate": float(run_config["learning-rate"]),
        "batch_size": int(run_config.get("batch-size", 64)),
        "accuracy": accuracies.copy(),
        "loss": losses.copy(),
    }

    runs.append(run_entry)
    data["runs"] = runs

    with open(results_path, "w") as f:
        toml.dump(data, f)

    print(f"Results saved to {results_path} ({len(accuracies)} rounds)")

    # Reset for next run (in case of sequential runs in same process)
    accuracies.clear()
    losses.clear()