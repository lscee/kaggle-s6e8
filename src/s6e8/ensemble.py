from __future__ import print_function

import glob
import json
import os

import numpy as np
import pandas as pd

from .io_utils import write_json
from .metrics import rank_percentile, roc_auc


def _prediction_files(output_dir, prefix):
    pattern = os.path.join(output_dir, "{}*.csv".format(prefix))
    files = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if "ensemble" not in name:
            files.append(path)
    return files


def _load_prediction_matrix(paths, id_column, target):
    if not paths:
        raise FileNotFoundError("No prediction files were found")
    ids = None
    columns = []
    names = []
    for path in paths:
        frame = pd.read_csv(path)
        if id_column not in frame.columns or target not in frame.columns:
            raise ValueError("Invalid prediction file: {}".format(path))
        if ids is None:
            ids = frame[id_column].values
        elif not np.array_equal(ids, frame[id_column].values):
            raise ValueError("Prediction row order mismatch: {}".format(path))
        columns.append(frame[target].values.astype(float))
        base = os.path.splitext(os.path.basename(path))[0]
        names.append(base.split("_", 1)[1])
    return ids, names, np.column_stack(columns)


def greedy_rank_weights(y, names, matrix, max_steps=40, min_improvement=1e-6):
    ranked = np.column_stack(
        [rank_percentile(matrix[:, index]) for index in range(matrix.shape[1])]
    )
    single_scores = [roc_auc(y, ranked[:, index]) for index in range(ranked.shape[1])]
    best_index = int(np.argmax(single_scores))
    counts = np.zeros(ranked.shape[1], dtype=int)
    counts[best_index] = 1
    blend = ranked[:, best_index].copy()
    best_score = single_scores[best_index]
    history = [
        {
            "step": 1,
            "added_model": names[best_index],
            "auc": best_score,
        }
    ]

    for step in range(2, max_steps + 1):
        candidate_scores = []
        denominator = float(counts.sum() + 1)
        for index in range(ranked.shape[1]):
            candidate = (blend * counts.sum() + ranked[:, index]) / denominator
            candidate_scores.append(roc_auc(y, candidate))
        candidate_index = int(np.argmax(candidate_scores))
        candidate_score = candidate_scores[candidate_index]
        if candidate_score < best_score + min_improvement:
            break
        counts[candidate_index] += 1
        blend = (
            blend * float(counts.sum() - 1) + ranked[:, candidate_index]
        ) / float(counts.sum())
        best_score = candidate_score
        history.append(
            {
                "step": step,
                "added_model": names[candidate_index],
                "auc": best_score,
            }
        )

    weights = counts.astype(float) / float(counts.sum())
    return weights, best_score, history, single_scores


def stable_rank_weights(
    y,
    names,
    matrix,
    max_steps=40,
    min_improvement=1e-6,
    bootstrap_rounds=25,
    bootstrap_fraction=0.80,
    min_selection_frequency=0.15,
    seed=20260803,
    bootstrap_max_rows=120000,
):
    """Average greedy blends over stratified subsamples to reduce weight noise."""

    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    class_indices = [np.where(y == value)[0] for value in np.unique(y)]
    round_weights = []
    round_scores = []
    for _ in range(int(bootstrap_rounds)):
        sampled_parts = []
        for indices in class_indices:
            count = max(2, int(round(len(indices) * float(bootstrap_fraction))))
            if bootstrap_max_rows:
                class_share = float(len(indices)) / float(len(y))
                count = min(
                    count,
                    max(2, int(round(float(bootstrap_max_rows) * class_share))),
                )
            count = min(count, len(indices))
            sampled_parts.append(rng.choice(indices, size=count, replace=False))
        sampled = np.concatenate(sampled_parts)
        rng.shuffle(sampled)
        weights, score, _, _ = greedy_rank_weights(
            y[sampled],
            names,
            matrix[sampled],
            max_steps=max_steps,
            min_improvement=min_improvement,
        )
        round_weights.append(weights)
        round_scores.append(score)

    weight_matrix = np.vstack(round_weights)
    selection_frequency = (weight_matrix > 0).mean(axis=0)
    weights = weight_matrix.mean(axis=0)
    weights[selection_frequency < float(min_selection_frequency)] = 0.0
    if weights.sum() <= 0:
        weights[int(np.argmax(selection_frequency))] = 1.0
    weights /= weights.sum()
    ranked = np.column_stack(
        [rank_percentile(matrix[:, index]) for index in range(matrix.shape[1])]
    )
    full_score = roc_auc(y, np.dot(ranked, weights))
    _, _, full_history, single_scores = greedy_rank_weights(
        y,
        names,
        matrix,
        max_steps=max_steps,
        min_improvement=min_improvement,
    )
    diagnostics = {
        "bootstrap_rounds": int(bootstrap_rounds),
        "bootstrap_fraction": float(bootstrap_fraction),
        "bootstrap_max_rows": (
            int(bootstrap_max_rows) if bootstrap_max_rows else None
        ),
        "bootstrap_auc_mean": float(np.mean(round_scores)),
        "bootstrap_auc_std": float(np.std(round_scores)),
        "selection_frequency": {
            name: float(value) for name, value in zip(names, selection_frequency)
        },
        "weight_std": {
            name: float(value)
            for name, value in zip(names, weight_matrix.std(axis=0))
        },
    }
    return weights, full_score, full_history, single_scores, diagnostics


def run_ensemble(train, test, sample, config, output_dir):
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    manifest_path = os.path.join(output_dir, "trained_models.json")
    included_models = config.get("ensemble", {}).get("include_models")
    if included_models:
        oof_paths = [
            os.path.join(output_dir, "oof_{}.csv".format(name))
            for name in included_models
        ]
        test_paths = [
            os.path.join(output_dir, "test_{}.csv".format(name))
            for name in included_models
        ]
        missing = [path for path in oof_paths + test_paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Prediction files listed by ensemble.include_models are missing: {}".format(
                    missing
                )
            )
    elif os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_models = manifest.get("models", [])
        oof_paths = [
            os.path.join(output_dir, "oof_{}.csv".format(name))
            for name in manifest_models
        ]
        test_paths = [
            os.path.join(output_dir, "test_{}.csv".format(name))
            for name in manifest_models
        ]
        missing = [path for path in oof_paths + test_paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Prediction files listed in trained_models.json are missing: {}".format(
                    missing
                )
            )
    else:
        oof_paths = _prediction_files(output_dir, "oof_")
        test_paths = _prediction_files(output_dir, "test_")
    train_ids, names, oof_matrix = _load_prediction_matrix(
        oof_paths, id_column, target
    )
    test_ids, test_names, test_matrix = _load_prediction_matrix(
        test_paths, id_column, target
    )
    if names != test_names:
        raise ValueError(
            "OOF and test model sets differ: {} versus {}".format(names, test_names)
        )
    if not np.array_equal(train_ids, train[id_column].values):
        raise ValueError("OOF IDs do not match train.csv")
    if not np.array_equal(test_ids, test[id_column].values):
        raise ValueError("Test prediction IDs do not match test.csv")

    settings = config.get("ensemble", {})
    method = settings.get("method", "greedy_rank")
    diagnostics = {}
    if method == "stable_rank":
        weights, blend_auc, history, single_scores, diagnostics = stable_rank_weights(
            train[target].values,
            names,
            oof_matrix,
            max_steps=settings.get("max_steps", 40),
            min_improvement=settings.get("min_improvement", 1e-6),
            bootstrap_rounds=settings.get("bootstrap_rounds", 25),
            bootstrap_fraction=settings.get("bootstrap_fraction", 0.80),
            min_selection_frequency=settings.get("min_selection_frequency", 0.15),
            seed=settings.get("seed", config["cv"]["seed"]),
            bootstrap_max_rows=settings.get("bootstrap_max_rows", 120000),
        )
    elif method == "greedy_rank":
        weights, blend_auc, history, single_scores = greedy_rank_weights(
            train[target].values,
            names,
            oof_matrix,
            max_steps=settings.get("max_steps", 40),
            min_improvement=settings.get("min_improvement", 1e-6),
        )
    else:
        raise ValueError("Unknown ensemble method: {}".format(method))
    ranked_oof = np.column_stack(
        [rank_percentile(oof_matrix[:, index]) for index in range(len(names))]
    )
    ranked_test = np.column_stack(
        [rank_percentile(test_matrix[:, index]) for index in range(len(names))]
    )
    pd.DataFrame(ranked_oof, columns=names).corr().to_csv(
        os.path.join(output_dir, "oof_rank_correlation.csv")
    )
    oof_blend = np.dot(ranked_oof, weights)
    test_blend = np.dot(ranked_test, weights)

    pd.DataFrame(
        {id_column: train_ids, target: oof_blend}
    ).to_csv(os.path.join(output_dir, "oof_ensemble.csv"), index=False)
    pd.DataFrame(
        {id_column: test_ids, target: test_blend}
    ).to_csv(os.path.join(output_dir, "test_ensemble.csv"), index=False)

    submission = sample.copy()
    submission[target] = test_blend
    submission_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    payload = {
        "method": method,
        "ensemble_oof_auc": roc_auc(train[target].values, oof_blend),
        "greedy_search_auc": blend_auc,
        "weights": {name: float(weight) for name, weight in zip(names, weights)},
        "single_model_rank_auc": {
            name: float(score) for name, score in zip(names, single_scores)
        },
        "history": history,
        "stability": diagnostics,
        "submission_path": submission_path,
    }
    write_json(os.path.join(output_dir, "ensemble.json"), payload)
    return payload
