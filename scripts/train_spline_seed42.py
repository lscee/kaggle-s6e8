from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_SHA256 = "a02a968711b0a721eb9ed046b81fafeef3775995e1a6c1548680cb319beed218"
SOURCE_NAME = "contextualized_deep_univariate_spline_transformer_v3.py"
MODEL_NAME = "spline_seed42"


def parser():
    result = argparse.ArgumentParser(
        description="Retrain the pinned public spline architecture on seed-42 folds"
    )
    result.add_argument(
        "--source",
        default=os.path.join(
            ROOT, "data", "external", "public_oof", "spline_transformer_v3", SOURCE_NAME
        ),
    )
    result.add_argument("--train", default=os.path.join(ROOT, "data", "raw", "train.csv"))
    result.add_argument("--test", default=os.path.join(ROOT, "data", "raw", "test.csv"))
    result.add_argument(
        "--output-dir", default=os.path.join(ROOT, "outputs", "spline_seed42")
    )
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=4096)
    result.add_argument("--epochs", type=int, default=35)
    result.add_argument("--patience", type=int, default=7)
    return result


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            "Pinned source patch expected one occurrence of {!r}, found {}".format(
                old, count
            )
        )
    return source.replace(old, new, 1)


def prepare_source(args):
    with open(args.source, "r", encoding="utf-8", newline="") as handle:
        source = handle.read()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            "Refusing to execute unpinned source: expected {}, got {}".format(
                SOURCE_SHA256, digest
            )
        )

    source = replace_once(
        source,
        "TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e8/train.csv'",
        "TRAIN_PATH = {!r}".format(os.path.abspath(args.train).replace("\\", "/")),
    )
    source = replace_once(
        source,
        "TEST_PATH = '/kaggle/input/competitions/playground-series-s6e8/test.csv'",
        "TEST_PATH = {!r}".format(os.path.abspath(args.test).replace("\\", "/")),
    )
    source = replace_once(source, "OUTER_SPLIT_SEED = 21", "OUTER_SPLIT_SEED = 42")
    source = replace_once(source, "SEED = 21", "SEED = 42")
    source = replace_once(
        source, "BATCH_SIZE = 4096", "BATCH_SIZE = {}".format(args.batch_size)
    )
    source = replace_once(source, "EPOCHS = 35", "EPOCHS = {}".format(args.epochs))
    source = replace_once(source, "PATIENCE = 7", "PATIENCE = {}".format(args.patience))
    return source, digest


def assemble_project_outputs(args, namespace, source_digest, started):
    train = namespace["train_raw"]
    test = namespace["test_raw"]
    target = namespace["TARGET"]
    id_column = namespace["ID_COL"]
    oof = np.asarray(namespace["oof_final"], dtype=np.float64)
    test_pred = np.asarray(namespace["test_final"], dtype=np.float64)
    y = train[target].to_numpy(dtype=int)

    folds = np.full(len(train), -1, dtype=np.int16)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[valid_index] = fold
    if (folds < 0).any():
        raise RuntimeError("Failed to reconstruct seed-42 folds")

    pd.DataFrame(
        {id_column: train[id_column].values, target: y, "fold": folds}
    ).to_csv(os.path.join(args.output_dir, "folds.csv"), index=False)
    pd.DataFrame(
        {id_column: train[id_column].values, target: oof}
    ).to_csv(os.path.join(args.output_dir, "oof_{}.csv".format(MODEL_NAME)), index=False)
    pd.DataFrame(
        {id_column: test[id_column].values, target: test_pred}
    ).to_csv(os.path.join(args.output_dir, "test_{}.csv".format(MODEL_NAME)), index=False)

    fold_frame = namespace["fold_df"].copy()
    fold_frame.to_csv(os.path.join(args.output_dir, "fold_metrics.csv"), index=False)
    elapsed = time.time() - started
    metrics = {
        "model": MODEL_NAME,
        "architecture": "contextualized_deep_univariate_spline_transformer",
        "source_owner": "ern711",
        "source_script_version_id": 342607757,
        "source_sha256": source_digest,
        "source_license": "Apache-2.0",
        "cv_seed": 42,
        "n_splits": 5,
        "oof_auc": float(roc_auc_score(y, oof)),
        "fold_metrics": fold_frame.to_dict(orient="records"),
        "elapsed_seconds": elapsed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": str(namespace["DEVICE"]),
        "outer_valid_checkpoint_selection": True,
    }
    with open(
        os.path.join(args.output_dir, "metrics_{}.json".format(MODEL_NAME)),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "base_model": "pytorch",
                "feature_view": "nested_te_deep_univariate_spline",
                "oof_auc": metrics["oof_auc"],
                "elapsed_seconds": elapsed,
                "feature_count": 43,
            }
        ]
    ).to_csv(os.path.join(args.output_dir, "model_summary.csv"), index=False)
    with open(
        os.path.join(args.output_dir, "trained_models.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "models": [MODEL_NAME],
                "target": target,
                "id_column": id_column,
                "n_splits": 5,
                "cv_seed": 42,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("Project OOF: {:.12f}".format(metrics["oof_auc"]))
    print("Project outputs: {}".format(args.output_dir))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch-size, epochs and patience must be positive")
    if not os.path.isfile(args.source):
        raise FileNotFoundError(
            "Pinned source is missing. Run scripts/download_public_spline.py "
            "--include-source first: {}".format(args.source)
        )
    if not os.path.isfile(args.train) or not os.path.isfile(args.test):
        raise FileNotFoundError("Competition train/test CSV files are required")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to fall back to CPU")
    torch.cuda.set_device(0)
    print("Torch: {} CUDA: {}".format(torch.__version__, torch.version.cuda))
    print("GPU: {}".format(torch.cuda.get_device_name(0)))

    source, source_digest = prepare_source(args)
    transformed_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    print("Pinned source: {}".format(source_digest))
    print("Transformed source: {}".format(transformed_digest))

    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()
    previous_directory = os.getcwd()
    namespace = {
        "__name__": "__main__",
        "__file__": os.path.abspath(args.source),
        "display": print,
    }
    try:
        os.chdir(args.output_dir)
        exec(compile(source, args.source, "exec"), namespace, namespace)
    finally:
        os.chdir(previous_directory)
    assemble_project_outputs(args, namespace, source_digest, started)


if __name__ == "__main__":
    main()
