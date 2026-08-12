"""Screen a shallow nonlinear meta-model on the frozen 86-model OOF matrix.

This script is deliberately separate from the production logistic stack.  It trains on
whole frozen folds, reports every fold independently, and writes partial OOF predictions
so a candidate must demonstrate stable residual value before it can be promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]


def build_design(logits: np.ndarray, raw: pd.DataFrame) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    feature_columns = [column for column in raw.columns if column not in ("id", "addicted_label")]
    missing = raw[feature_columns].isna().to_numpy(dtype=np.float32)
    aggregates = np.column_stack(
        [
            logits.mean(axis=1),
            logits.std(axis=1),
            logits.max(axis=1) - logits.min(axis=1),
            np.quantile(logits, 0.25, axis=1),
            np.quantile(logits, 0.50, axis=1),
            np.quantile(logits, 0.75, axis=1),
            missing.sum(axis=1),
        ]
    ).astype(np.float32)
    return np.column_stack([logits, aggregates, missing]).astype(np.float32)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average").to_numpy(np.float64)


def run(args) -> dict:
    train = pd.read_csv(args.train)
    y = train["addicted_label"].to_numpy(dtype=np.int8)
    logits = np.load(args.logits, mmap_mode="r")
    if logits.shape != (len(train), 86):
        raise ValueError("Expected the frozen (n_train, 86) base-logit matrix")
    design = build_design(logits, train)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros(len(y)), y
        )
    )
    selected_folds = args.folds or list(range(5))
    invalid = sorted(set(selected_folds) - set(range(5)))
    if invalid:
        raise ValueError("Invalid folds: {}".format(invalid))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_results = []
    started = time.time()
    for fold_index in selected_folds:
        train_index, valid_index = folds[fold_index]
        if args.backend == "catboost":
            from catboost import CatBoostClassifier

            model_parameters = dict(
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                random_strength=args.random_strength,
                loss_function="Logloss",
                eval_metric="AUC",
                task_type="GPU",
                devices=str(args.gpu),
                border_count=args.border_count,
                bootstrap_type="Bayesian",
                bagging_temperature=args.bagging_temperature,
                random_seed=args.seed + fold_index,
                allow_writing_files=False,
                verbose=args.verbose,
            )
            if args.screen_early_stop:
                model_parameters.update(
                    od_type="Iter", early_stopping_rounds=args.patience
                )
            model = CatBoostClassifier(**model_parameters)
        else:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=args.iterations,
                learning_rate=args.learning_rate,
                max_depth=args.depth,
                min_child_weight=args.min_child_weight,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.l2_leaf_reg,
                gamma=args.gamma,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                device="cuda:{}".format(args.gpu),
                max_bin=args.border_count,
                random_state=args.seed + fold_index,
                early_stopping_rounds=(
                    args.patience if args.screen_early_stop else None
                ),
                n_jobs=-1,
            )
        model.fit(
            design[train_index],
            y[train_index],
            eval_set=[(design[valid_index], y[valid_index])],
            **(
                {"use_best_model": args.screen_early_stop}
                if args.backend == "catboost"
                else {"verbose": args.verbose}
            ),
        )
        prediction = model.predict_proba(design[valid_index])[:, 1]
        oof[valid_index] = prediction
        auc = float(roc_auc_score(y[valid_index], prediction))
        best_iteration = (
            model.get_best_iteration()
            if args.backend == "catboost"
            else getattr(model, "best_iteration", args.iterations - 1)
        )
        result = {
            "fold": int(fold_index),
            "rows": int(len(valid_index)),
            "auc": auc,
            "best_iteration": int(best_iteration),
        }
        fold_results.append(result)
        print(
            "fold={} auc={:.9f} best_iteration={} elapsed={:.0f}s".format(
                fold_index,
                auc,
                result["best_iteration"],
                time.time() - started,
            ),
            flush=True,
        )

    rows = np.flatnonzero(np.isfinite(oof))
    pooled_auc = float(roc_auc_score(y[rows], oof[rows]))
    np.save(output_dir / "oof_{}_meta_partial.npy".format(args.backend), oof)
    np.save(output_dir / "completed_rows.npy", rows)
    payload = {
        "folds": selected_folds,
        "rows": int(len(rows)),
        "design_shape": list(design.shape),
        "pooled_oof_auc": pooled_auc,
        "per_fold": fold_results,
        "parameters": vars(args),
        "checkpoint_selection": (
            "outer-valid early stopping; screening only"
            if args.screen_early_stop
            else "fixed iterations"
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("pooled auc={:.9f} rows={}".format(pooled_auc, len(rows)), flush=True)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train", default=str(ROOT / "data" / "raw" / "train.csv"))
    result.add_argument(
        "--logits",
        default=str(
            ROOT / "outputs" / "leaderboard_v3_gam" / "analysis" / "base86_oof_logits.npy"
        ),
    )
    result.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "nonlinear_meta_screen")
    )
    result.add_argument("--folds", type=int, nargs="*", default=[])
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument(
        "--backend", choices=["xgboost", "catboost"], default="xgboost"
    )
    result.add_argument("--seed", type=int, default=20260812)
    result.add_argument("--iterations", type=int, default=1200)
    result.add_argument("--learning-rate", type=float, default=0.03)
    result.add_argument("--depth", type=int, default=4)
    result.add_argument("--l2-leaf-reg", type=float, default=30.0)
    result.add_argument("--min-child-weight", type=float, default=200.0)
    result.add_argument("--subsample", type=float, default=0.8)
    result.add_argument("--colsample-bytree", type=float, default=0.7)
    result.add_argument("--reg-alpha", type=float, default=1.0)
    result.add_argument("--gamma", type=float, default=0.01)
    result.add_argument("--random-strength", type=float, default=0.5)
    result.add_argument("--bagging-temperature", type=float, default=0.5)
    result.add_argument("--border-count", type=int, default=128)
    result.add_argument("--patience", type=int, default=100)
    result.add_argument(
        "--screen-early-stop",
        action="store_true",
        help="Use outer-valid early stopping for directional screening only.",
    )
    result.add_argument("--verbose", type=int, default=100)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
