"""quickstart_xgboost: A Flower / XGBoost app."""

import xgboost as xgb
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import numpy as np
from typing import List
import tqdm
from sklearn.metrics import roc_auc_score

PARAMS = {'objective': 'binary:logistic', 'eta': 0.1, 'max_depth': 8, 'eval_metric': 'auc', 'nthread': 16, 'num_parallel_tree': 1, 'subsample': 1, 'tree_method': 'hist'}

def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


fds = None  # Cache FederatedDataset

def load_data(partition_id, num_clients, seed):
    """Load partition HIGGS data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(
            dataset="jxie/higgs",
            partitioners={"train": partitioner},
        )

    # Load the partition for this `partition_id`
    partition = fds.load_partition(partition_id, split="train")
    partition.set_format("numpy")

    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=seed
    )

    # Reformat data to DMatrix for xgboost
    train_dmatrix = transform_dataset_to_dmatrix(train_data)

    return train_dmatrix


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def _local_boost(bst_input, num_local_round, train_dmatrix):
    # Update trees based on local training data.
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    # Bagging: extract the last N=num_local_round trees for sever aggregation
    bst = bst_input[
        bst_input.num_boosted_rounds()
        - num_local_round : bst_input.num_boosted_rounds()
    ]
    return bst


def train(bst, train_dmatrix, epochs: int = 5):
    bst = _local_boost(bst, epochs, train_dmatrix)
    return bst

def ndarrays_from_model(model) -> List:
    """Get model weights as a list of NumPy ndarrays."""
    local_model = model.save_raw("json")
    return np.frombuffer(local_model, dtype=np.uint8)


def ndarrays_to_model(model, ndarrays):
    """Set model weights from a list of NumPy ndarrays."""
    global_model = ndarrays

    # Load global model into booster
    model.load_model(bytearray(global_model.tobytes()))
    return model
