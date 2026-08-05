from __future__ import print_function

import os

import numpy as np
import pandas as pd

from .data import feature_columns, make_stratified_folds
from .features import categorical_columns, encode_for_sklearn, prepare_feature_frames
from .io_utils import write_json
from .metrics import roc_auc


def _safe_univariate_auc(y, values):
    values = pd.Series(values)
    if values.nunique(dropna=False) < 2:
        return None
    ranks = values.rank(method="average", pct=True).fillna(0.5).values
    auc = roc_auc(y, ranks)
    return max(auc, 1.0 - auc)


def _oof_category_auc(y, values, folds):
    values = pd.Series(values).fillna("__MISSING__").astype(str)
    predictions = np.zeros(len(y), dtype=float)
    for fold in sorted(np.unique(folds)):
        fit_mask = folds != fold
        valid_mask = folds == fold
        fold_mean = float(np.mean(y[fit_mask]))
        table = pd.DataFrame({"value": values[fit_mask].values, "target": y[fit_mask]})
        grouped = table.groupby("value")["target"].agg(["mean", "count"])
        smoothed = (grouped["mean"] * grouped["count"] + fold_mean * 20.0) / (
            grouped["count"] + 20.0
        )
        predictions[valid_mask] = (
            values[valid_mask].map(smoothed).fillna(fold_mean).values
        )
    return roc_auc(y, predictions)


def _feature_hash(frame, columns):
    if not columns:
        return pd.Series(np.zeros(len(frame), dtype=np.uint64), index=frame.index)
    return pd.util.hash_pandas_object(frame.loc[:, columns], index=False)


def _duplicate_report(train, test, target, id_column):
    columns = [
        column
        for column in test.columns
        if column != id_column and column in train.columns
    ]
    train_hash = _feature_hash(train, columns)
    test_hash = _feature_hash(test, columns)
    counts = train_hash.value_counts()
    duplicate_mask = train_hash.map(counts).values > 1
    target_by_hash = pd.DataFrame(
        {"hash": train_hash.values, "target": train[target].values}
    ).groupby("hash")["target"].agg(["size", "nunique"])
    conflicting_groups = int(
        ((target_by_hash["size"] > 1) & (target_by_hash["nunique"] > 1)).sum()
    )
    overlap = np.intersect1d(train_hash.unique(), test_hash.unique())
    test_overlap_rows = int(test_hash.isin(overlap).sum())
    return {
        "hash_columns": columns,
        "train_duplicate_rows": int(duplicate_mask.sum()),
        "train_duplicate_groups": int((counts > 1).sum()),
        "train_conflicting_duplicate_groups": conflicting_groups,
        "train_test_matching_groups": int(len(overlap)),
        "test_rows_matching_train": test_overlap_rows,
    }


def _adversarial_auc(train, test, config, max_rows=200000):
    train_x, test_x = prepare_feature_frames(train, test, config)
    if len(train_x) > max_rows:
        train_x = train_x.sample(max_rows, random_state=config["cv"]["seed"])
    if len(test_x) > max_rows:
        test_x = test_x.sample(max_rows, random_state=config["cv"]["seed"])
    train_encoded, test_encoded = encode_for_sklearn(train_x, test_x)
    matrix = pd.concat([train_encoded, test_encoded], ignore_index=True)
    labels = np.concatenate(
        [np.zeros(len(train_encoded), dtype=int), np.ones(len(test_encoded), dtype=int)]
    )
    folds = make_stratified_folds(labels, 3, config["cv"]["seed"])
    predictions = np.zeros(len(labels), dtype=float)
    from sklearn.ensemble import ExtraTreesClassifier

    for fold in range(3):
        fit_idx = np.where(folds != fold)[0]
        valid_idx = np.where(folds == fold)[0]
        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=config["cv"]["seed"] + fold,
        )
        model.fit(matrix.iloc[fit_idx], labels[fit_idx])
        predictions[valid_idx] = model.predict_proba(matrix.iloc[valid_idx])[:, 1]
    return roc_auc(labels, predictions)


def run_audit(train, test, config, output_dir, adversarial=False):
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    y = train[target].astype(int).values
    folds = make_stratified_folds(
        y, config["cv"]["n_splits"], config["cv"]["seed"]
    )

    base_columns = feature_columns(train, config)
    feature_frame = train.loc[:, base_columns]
    categorical = set(categorical_columns(feature_frame))
    rows = []
    for column in base_columns:
        unique_count = int(feature_frame[column].nunique(dropna=False))
        if column in categorical or unique_count <= 30:
            auc = _oof_category_auc(y, feature_frame[column], folds)
            method = "oof_smoothed_target_encoding"
        else:
            auc = _safe_univariate_auc(y, feature_frame[column])
            method = "rank_auc_best_direction"
        rows.append(
            {
                "feature": column,
                "dtype": str(feature_frame[column].dtype),
                "unique_count": unique_count,
                "missing_count": int(feature_frame[column].isnull().sum()),
                "method": method,
                "univariate_auc": auc,
            }
        )
        missing = feature_frame[column].isnull()
        if missing.any() and not missing.all():
            rows.append(
                {
                    "feature": "{}__missing".format(column),
                    "dtype": "missing_indicator",
                    "unique_count": 2,
                    "missing_count": 0,
                    "method": "missing_indicator_auc_best_direction",
                    "univariate_auc": _safe_univariate_auc(y, missing.astype(float)),
                }
            )
    univariate = pd.DataFrame(rows).sort_values(
        "univariate_auc", ascending=False, na_position="last"
    )
    univariate.to_csv(
        os.path.join(output_dir, "audit_univariate_auc.csv"), index=False
    )

    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_mean": float(np.mean(y)),
        "target_counts": {
            str(key): int(value)
            for key, value in train[target].value_counts(dropna=False).to_dict().items()
        },
        "feature_count": len(base_columns),
        "categorical_columns": sorted(categorical),
        "highest_univariate_features": univariate.head(10).to_dict(orient="records"),
        "duplicates": _duplicate_report(train, test, target, id_column),
    }
    if adversarial:
        summary["adversarial_validation_auc"] = _adversarial_auc(
            train, test, config
        )
    write_json(os.path.join(output_dir, "audit_summary.json"), summary)

    fold_frame = pd.DataFrame(
        {
            id_column: train[id_column].values,
            target: y,
            "fold": folds,
        }
    )
    fold_frame.to_csv(os.path.join(output_dir, "folds.csv"), index=False)
    return summary, univariate
