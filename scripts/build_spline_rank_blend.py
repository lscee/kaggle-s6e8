from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from s6e8.config import load_config  # noqa: E402
from s6e8.data import load_competition_data, make_stratified_folds  # noqa: E402


def percentile_rank(values):
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )


def parser():
    result = argparse.ArgumentParser(
        description="Build a fixed-weight rank blend with the public spline model"
    )
    result.add_argument(
        "--config", default=os.path.join(ROOT, "configs", "leaderboard_v3_combined.yaml")
    )
    result.add_argument(
        "--baseline-dir", default=os.path.join(ROOT, "outputs", "leaderboard_v3_combined")
    )
    result.add_argument(
        "--candidate-dir",
        default=os.path.join(
            ROOT, "data", "external", "public_oof", "spline_transformer_v3"
        ),
    )
    result.add_argument("--weight", type=float, default=0.05)
    result.add_argument(
        "--output-dir", default=os.path.join(ROOT, "outputs", "spline_rank_blend")
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0.0 <= args.weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")

    config = load_config(args.config)
    train, test, sample = load_competition_data(config)
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    y = train[target].to_numpy(dtype=int)

    candidate_oof = pd.read_csv(
        os.path.join(args.candidate_dir, "nonlinear_context_5fold_oof.csv")
    )
    candidate_test = pd.read_csv(
        os.path.join(
            args.candidate_dir, "nonlinear_context_5fold_test_predictions.csv"
        )
    )
    if not np.array_equal(candidate_oof[id_column].values, train[id_column].values):
        raise ValueError("Spline OOF IDs do not match train.csv")
    if not np.array_equal(candidate_test[id_column].values, test[id_column].values):
        raise ValueError("Spline test IDs do not match test.csv")
    if not np.array_equal(candidate_oof[target].values, y):
        raise ValueError("Spline OOF labels do not match train.csv")

    baseline_oof = np.load(
        os.path.join(args.baseline_dir, "oof_public_postprocessed.npy")
    ).astype(np.float64)
    baseline_test = np.load(
        os.path.join(args.baseline_dir, "test_public_postprocessed.npy")
    ).astype(np.float64)
    if len(baseline_oof) != len(train) or len(baseline_test) != len(test):
        raise ValueError("Baseline prediction lengths do not match competition data")

    candidate_oof_values = candidate_oof["oof_prediction"].to_numpy(np.float64)
    candidate_test_values = candidate_test["mean_prediction"].to_numpy(np.float64)
    baseline_oof_rank = percentile_rank(baseline_oof)
    baseline_test_rank = percentile_rank(baseline_test)
    candidate_oof_rank = percentile_rank(candidate_oof_values)
    candidate_test_rank = percentile_rank(candidate_test_values)

    blended_oof = (
        (1.0 - args.weight) * baseline_oof_rank
        + args.weight * candidate_oof_rank
    )
    blended_test = (
        (1.0 - args.weight) * baseline_test_rank
        + args.weight * candidate_test_rank
    )

    folds = make_stratified_folds(y, n_splits=5, seed=42)
    fold_rows = []
    for fold in range(5):
        mask = folds == fold
        baseline_auc = roc_auc_score(y[mask], baseline_oof_rank[mask])
        blended_auc = roc_auc_score(y[mask], blended_oof[mask])
        fold_rows.append(
            {
                "fold": fold,
                "baseline_auc": baseline_auc,
                "blend_auc": blended_auc,
                "delta": blended_auc - baseline_auc,
            }
        )

    baseline_auc = roc_auc_score(y, baseline_oof_rank)
    blended_auc = roc_auc_score(y, blended_oof)
    payload = {
        "weight": args.weight,
        "baseline_oof_auc": baseline_auc,
        "candidate_oof_auc": roc_auc_score(y, candidate_oof_values),
        "blend_oof_auc": blended_auc,
        "blend_oof_delta": blended_auc - baseline_auc,
        "oof_spearman": float(
            np.corrcoef(baseline_oof_rank, candidate_oof_rank)[0, 1]
        ),
        "test_spearman": float(
            np.corrcoef(baseline_test_rank, candidate_test_rank)[0, 1]
        ),
        "folds": fold_rows,
        "candidate_outer_fold_seed": 21,
        "blend_evaluation_fold_seed": 42,
        "meta_stacking_allowed": False,
        "reason": "Candidate folds differ from the seed-42 stack; fixed rank blending avoids cross-level fold leakage.",
    }

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "oof_spline_rank_blend.npy"), blended_oof)
    np.save(os.path.join(args.output_dir, "test_spline_rank_blend.npy"), blended_test)
    submission = sample.loc[:, [id_column, target]].copy()
    submission[target] = blended_test
    submission_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    with open(
        os.path.join(args.output_dir, "blend_report.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    print("Submission: {}".format(submission_path))


if __name__ == "__main__":
    main()
