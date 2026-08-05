from __future__ import print_function

import os
import time
import copy

import numpy as np
import pandas as pd

from .data import feature_columns, make_stratified_folds
from .features import encode_for_sklearn, prepare_feature_frames
from .io_utils import write_json
from .metrics import roc_auc
from .models import fit_predict_fold, matrix_kind, model_available


def _save_predictions(path, ids, target_name, predictions, id_column):
    frame = pd.DataFrame(
        {id_column: ids, target_name: np.asarray(predictions, dtype=float)}
    )
    frame.to_csv(path, index=False)


def _experiment_specs(config, requested_models=None):
    configured = config.get("training", {}).get("experiments") or []
    by_name = {item["name"]: item for item in configured}
    if requested_models:
        specs = []
        for requested in requested_models:
            specs.append(
                copy.deepcopy(
                    by_name.get(
                        requested,
                        {"name": requested, "model": requested},
                    )
                )
            )
    elif configured:
        specs = copy.deepcopy(configured)
    else:
        specs = [
            {"name": name, "model": name}
            for name in config["training"]["models"]
        ]
    names = [item.get("name") for item in specs]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Training experiment names must be present and unique")
    default_view = config.get("features", {}).get(
        "default_view",
        "engineered" if config.get("features", {}).get("engineered", True) else "raw",
    )
    for item in specs:
        item.setdefault("model", item["name"])
        item.setdefault("view", default_view)
    return specs


def train_models(train, test, config, output_dir, requested_models=None):
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    y = train[target].astype(int).values
    folds = make_stratified_folds(
        y, config["cv"]["n_splits"], config["cv"]["seed"]
    )
    pd.DataFrame(
        {id_column: train[id_column].values, target: y, "fold": folds}
    ).to_csv(os.path.join(output_dir, "folds.csv"), index=False)

    missing_indicator_columns = feature_columns(train, config)
    feature_cache = {}
    matrix_cache = {}
    experiments = _experiment_specs(config, requested_models)
    skip_missing = config["training"].get("skip_missing_models", True)
    seeds = config["training"].get("seeds", [config["cv"]["seed"]])
    summary_rows = []

    for experiment in experiments:
        experiment_name = experiment["name"]
        model_name = experiment["model"]
        view = experiment["view"]
        experiment_seeds = experiment.get("seeds", seeds)
        if not model_available(model_name):
            message = "Skipping unavailable model: {}".format(model_name)
            if skip_missing:
                print(message)
                continue
            raise ImportError(message)

        if view not in feature_cache:
            feature_cache[view] = prepare_feature_frames(
                train, test, config, view=view
            )
        feature_train_x, feature_test_x = feature_cache[view]
        kind = matrix_kind(model_name)
        cache_key = (view, kind)
        if kind == "encoded_filled":
            if cache_key not in matrix_cache:
                matrix_cache[cache_key] = encode_for_sklearn(
                    feature_train_x,
                    feature_test_x,
                    fill_missing=True,
                    missing_indicator_columns=missing_indicator_columns,
                )
            model_train_x, model_test_x = matrix_cache[cache_key]
        elif kind == "encoded_native_missing":
            if cache_key not in matrix_cache:
                matrix_cache[cache_key] = encode_for_sklearn(
                    feature_train_x,
                    feature_test_x,
                    fill_missing=False,
                    missing_indicator_columns=missing_indicator_columns,
                )
            model_train_x, model_test_x = matrix_cache[cache_key]
        elif kind == "raw":
            model_train_x = feature_train_x
            model_test_x = feature_test_x
        else:
            raise ValueError("Unsupported matrix kind: {}".format(kind))

        params = copy.deepcopy(config["models"][model_name])
        params.update(copy.deepcopy(experiment.get("params", {})))
        oof_sum = np.zeros(len(train), dtype=float)
        test_sum = np.zeros(len(test), dtype=float)
        importance_sum = np.zeros(model_train_x.shape[1], dtype=float)
        importance_count = 0
        fold_metrics = []
        started = time.time()

        for seed_index, seed in enumerate(experiment_seeds):
            seed_oof = np.zeros(len(train), dtype=float)
            for fold in range(config["cv"]["n_splits"]):
                fit_idx = np.where(folds != fold)[0]
                valid_idx = np.where(folds == fold)[0]
                fold_seed = int(seed) + fold * 1009
                valid_pred, test_pred, importance, fit_metadata = fit_predict_fold(
                    model_name,
                    model_train_x.iloc[fit_idx],
                    y[fit_idx],
                    model_train_x.iloc[valid_idx],
                    y[valid_idx],
                    model_test_x,
                    params,
                    fold_seed,
                )
                seed_oof[valid_idx] = valid_pred
                test_sum += test_pred / float(
                    len(experiment_seeds) * config["cv"]["n_splits"]
                )
                fold_auc = roc_auc(y[valid_idx], valid_pred)
                fold_metrics.append(
                    {
                        "seed": int(seed),
                        "seed_index": seed_index,
                        "fold": fold,
                        "auc": fold_auc,
                        "fit_rows": int(len(fit_idx)),
                        "valid_rows": int(len(valid_idx)),
                        "fit_metadata": fit_metadata,
                    }
                )
                print(
                    "{} seed={} fold={} auc={:.6f}".format(
                        experiment_name, seed, fold, fold_auc
                    )
                )
                if importance is not None and len(importance) == len(importance_sum):
                    importance_sum += np.asarray(importance, dtype=float)
                    importance_count += 1
            oof_sum += seed_oof / float(len(experiment_seeds))

        overall_auc = roc_auc(y, oof_sum)
        elapsed = time.time() - started
        print(
            "{} OOF AUC={:.6f} elapsed={:.1f}s".format(
                experiment_name, overall_auc, elapsed
            )
        )

        _save_predictions(
            os.path.join(output_dir, "oof_{}.csv".format(experiment_name)),
            train[id_column].values,
            target,
            oof_sum,
            id_column,
        )
        _save_predictions(
            os.path.join(output_dir, "test_{}.csv".format(experiment_name)),
            test[id_column].values,
            target,
            test_sum,
            id_column,
        )
        metrics_payload = {
            "model": experiment_name,
            "base_model": model_name,
            "feature_view": view,
            "oof_auc": overall_auc,
            "fold_metrics": fold_metrics,
            "elapsed_seconds": elapsed,
            "feature_count": int(model_train_x.shape[1]),
            "seeds": [int(seed) for seed in experiment_seeds],
        }
        write_json(
            os.path.join(output_dir, "metrics_{}.json".format(experiment_name)),
            metrics_payload,
        )
        if importance_count:
            importance_frame = pd.DataFrame(
                {
                    "feature": list(model_train_x.columns),
                    "importance": importance_sum / float(importance_count),
                }
            ).sort_values("importance", ascending=False)
            importance_frame.to_csv(
                os.path.join(
                    output_dir, "feature_importance_{}.csv".format(experiment_name)
                ),
                index=False,
            )
        summary_rows.append(
            {
                "model": experiment_name,
                "base_model": model_name,
                "feature_view": view,
                "oof_auc": overall_auc,
                "elapsed_seconds": elapsed,
                "feature_count": int(model_train_x.shape[1]),
            }
        )

    if not summary_rows:
        raise RuntimeError(
            "No model was trained. Install requirements or request an available model."
        )
    summary = pd.DataFrame(summary_rows).sort_values("oof_auc", ascending=False)
    summary.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    write_json(
        os.path.join(output_dir, "trained_models.json"),
        {
            "models": summary["model"].tolist(),
            "target": target,
            "id_column": id_column,
            "n_splits": int(config["cv"]["n_splits"]),
            "cv_seed": int(config["cv"]["seed"]),
        },
    )
    return summary
