from __future__ import print_function

import os

import numpy as np
import pandas as pd


def load_competition_data(config, require_sample=True):
    paths = config["data"]
    required = [paths["train_path"], paths["test_path"]]
    if require_sample:
        required.append(paths["sample_submission_path"])
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        message = "Missing competition files:\n- " + "\n- ".join(missing)
        message += "\nPut Kaggle CSV files under data/raw or update configs/base.yaml."
        raise FileNotFoundError(message)

    train = pd.read_csv(paths["train_path"])
    test = pd.read_csv(paths["test_path"])
    sample = None
    if os.path.isfile(paths["sample_submission_path"]):
        sample = pd.read_csv(paths["sample_submission_path"])
    validate_schema(train, test, sample, config)
    return train, test, sample


def validate_schema(train, test, sample, config):
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    if target not in train.columns:
        raise ValueError("Target column '{}' is absent from train.csv".format(target))
    if target in test.columns:
        raise ValueError("Target column '{}' unexpectedly exists in test.csv".format(target))
    if id_column not in train.columns or id_column not in test.columns:
        raise ValueError("ID column '{}' must exist in train and test".format(id_column))
    train_features = [column for column in train.columns if column != target]
    if set(train_features) != set(test.columns):
        only_train = sorted(set(train_features) - set(test.columns))
        only_test = sorted(set(test.columns) - set(train_features))
        raise ValueError(
            "Train/test schema mismatch. only_train={}, only_test={}".format(
                only_train, only_test
            )
        )
    values = train[target].dropna().unique()
    if not set(values).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("Target must be binary 0/1; got {}".format(values[:10]))
    if sample is not None:
        if id_column not in sample.columns or target not in sample.columns:
            raise ValueError("sample_submission.csv must contain ID and target columns")
        if len(sample) != len(test):
            raise ValueError("sample_submission.csv row count does not match test.csv")
        if not np.array_equal(sample[id_column].values, test[id_column].values):
            raise ValueError("Sample submission IDs do not match test IDs in row order")


def make_stratified_folds(y, n_splits, seed):
    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(np.zeros(len(y)), y)
    except ImportError:
        from sklearn.cross_validation import StratifiedKFold

        iterator = StratifiedKFold(
            y, n_folds=n_splits, shuffle=True, random_state=seed
        )

    fold_ids = np.full(len(y), -1, dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(iterator):
        fold_ids[valid_idx] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("Failed to assign all rows to a CV fold")
    return fold_ids


def feature_columns(train, config):
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    drop_columns = set(config["features"].get("drop_columns", []))
    unknown = sorted(drop_columns - set(train.columns))
    if unknown:
        raise ValueError("Unknown drop_columns: {}".format(unknown))
    if not config["features"].get("use_id", False):
        drop_columns.add(id_column)
    columns = [
        column
        for column in train.columns
        if column != target and column not in drop_columns
    ]
    if not columns:
        raise ValueError("No features remain after exclusions")
    return columns
