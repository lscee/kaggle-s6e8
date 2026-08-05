from __future__ import print_function

import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .io_utils import write_json


EPSILON = 1e-6


def _logit(values):
    clipped = np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def discover_prediction_names(library_dir):
    pattern = os.path.join(library_dir, "oof", "oof_*.npy")
    return sorted(
        os.path.basename(path)[4:-4]
        for path in glob.glob(pattern)
    )


def audit_and_load_library(train, test, library_dir, target, id_column, exclude=None):
    """Verify the public prediction bank and return aligned float64 matrices."""
    library_dir = os.path.abspath(library_dir)
    excluded = set(exclude or ())
    train_keys_path = os.path.join(library_dir, "train_keys.parquet")
    test_keys_path = os.path.join(library_dir, "test_keys.parquet")
    manifest_path = os.path.join(library_dir, "manifest.csv")
    for path in (train_keys_path, test_keys_path, manifest_path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing public-library metadata: {}".format(path))

    train_keys = pd.read_parquet(train_keys_path)
    test_keys = pd.read_parquet(test_keys_path)
    if not np.array_equal(train[id_column].values, train_keys[id_column].values):
        raise ValueError("Public OOF train IDs do not match local train.csv row order")
    if not np.array_equal(test[id_column].values, test_keys[id_column].values):
        raise ValueError("Public test IDs do not match local test.csv row order")
    if not np.array_equal(train[target].values, train_keys[target].values):
        raise ValueError("Public OOF labels do not match local train.csv")

    manifest = pd.read_csv(manifest_path)
    manifest_names = set(manifest["model"].astype(str))
    names = [name for name in discover_prediction_names(library_dir) if name not in excluded]
    if not names:
        raise ValueError("No public OOF predictions remain after exclusions")
    unknown = sorted(set(names) - manifest_names)
    if unknown:
        raise ValueError("Predictions missing from manifest.csv: {}".format(unknown))

    oof_columns = []
    test_columns = []
    rows = []
    y = train[target].to_numpy()
    for name in names:
        oof_path = os.path.join(library_dir, "oof", "oof_{}.npy".format(name))
        test_path = os.path.join(library_dir, "oof", "test_{}.npy".format(name))
        if not os.path.isfile(test_path):
            raise FileNotFoundError("Missing paired test prediction: {}".format(test_path))
        oof = np.load(oof_path, allow_pickle=False)
        test_pred = np.load(test_path, allow_pickle=False)
        if oof.shape != (len(train),):
            raise ValueError("{} OOF shape {} != ({},)".format(name, oof.shape, len(train)))
        if test_pred.shape != (len(test),):
            raise ValueError(
                "{} test shape {} != ({},)".format(name, test_pred.shape, len(test))
            )
        if not (np.isfinite(oof).all() and np.isfinite(test_pred).all()):
            raise ValueError("{} contains NaN or infinite predictions".format(name))
        oof = np.asarray(oof, dtype=np.float64)
        test_pred = np.asarray(test_pred, dtype=np.float64)
        oof_columns.append(oof)
        test_columns.append(test_pred)
        rows.append(
            {
                "model": name,
                "oof_auc": float(roc_auc_score(y, oof)),
                "oof_min": float(oof.min()),
                "oof_max": float(oof.max()),
                "test_min": float(test_pred.min()),
                "test_max": float(test_pred.max()),
                "source_dtype": str(oof.dtype),
            }
        )

    return (
        names,
        np.column_stack(oof_columns),
        np.column_stack(test_columns),
        pd.DataFrame(rows).sort_values("oof_auc", ascending=False),
        manifest,
    )


def load_additional_predictions(
    train,
    test,
    prediction_sources,
    target,
    id_column,
    folds,
):
    """Load project-native CSV predictions after verifying their frozen folds."""
    names = []
    oof_columns = []
    test_columns = []
    rows = []
    expected_fold = np.full(len(train), -1, dtype=np.int16)
    for fold_index, (_, valid_index) in enumerate(folds):
        expected_fold[valid_index] = fold_index

    for source in prediction_sources or ():
        if isinstance(source, str):
            source = {"directory": source}
        directory = os.path.abspath(source["directory"])
        prefix = str(source.get("prefix", "local"))
        folds_path = os.path.join(directory, "folds.csv")
        if not os.path.isfile(folds_path):
            raise FileNotFoundError(
                "Additional prediction source has no folds.csv: {}".format(directory)
            )
        fold_frame = pd.read_csv(folds_path)
        if not np.array_equal(fold_frame[id_column].values, train[id_column].values):
            raise ValueError("Additional OOF IDs do not align: {}".format(directory))
        if target in fold_frame and not np.array_equal(
            fold_frame[target].values, train[target].values
        ):
            raise ValueError("Additional OOF labels do not align: {}".format(directory))
        if not np.array_equal(fold_frame["fold"].values, expected_fold):
            raise ValueError(
                "Additional predictions do not use the configured frozen folds: {}".format(
                    directory
                )
            )

        include = source.get("include_models")
        if include:
            local_names = list(include)
        else:
            local_names = sorted(
                os.path.basename(path)[4:-4]
                for path in glob.glob(os.path.join(directory, "oof_*.csv"))
                if "ensemble" not in os.path.basename(path)
                and "correlation" not in os.path.basename(path)
            )
        for local_name in local_names:
            oof_path = os.path.join(directory, "oof_{}.csv".format(local_name))
            test_path = os.path.join(directory, "test_{}.csv".format(local_name))
            if not (os.path.isfile(oof_path) and os.path.isfile(test_path)):
                raise FileNotFoundError(
                    "Missing additional OOF/test pair for {} in {}".format(
                        local_name, directory
                    )
                )
            oof_frame = pd.read_csv(oof_path, usecols=[id_column, target])
            test_frame = pd.read_csv(test_path, usecols=[id_column, target])
            if not np.array_equal(oof_frame[id_column].values, train[id_column].values):
                raise ValueError("Additional OOF row order mismatch: {}".format(oof_path))
            if not np.array_equal(test_frame[id_column].values, test[id_column].values):
                raise ValueError("Additional test row order mismatch: {}".format(test_path))
            oof = oof_frame[target].to_numpy(np.float64)
            test_pred = test_frame[target].to_numpy(np.float64)
            if not (np.isfinite(oof).all() and np.isfinite(test_pred).all()):
                raise ValueError("Non-finite additional predictions: {}".format(local_name))
            name = "{}__{}".format(prefix, local_name)
            names.append(name)
            oof_columns.append(oof)
            test_columns.append(test_pred)
            rows.append(
                {
                    "model": name,
                    "oof_auc": float(roc_auc_score(train[target].values, oof)),
                    "oof_min": float(oof.min()),
                    "oof_max": float(oof.max()),
                    "test_min": float(test_pred.min()),
                    "test_max": float(test_pred.max()),
                    "source_dtype": str(oof.dtype),
                }
            )
    if not names:
        return names, None, None, pd.DataFrame()
    return (
        names,
        np.column_stack(oof_columns),
        np.column_stack(test_columns),
        pd.DataFrame(rows),
    )


def load_logit_parquet_sources(train, test, sources, target, id_column):
    """Load aligned parquet members that are already stored in log-odds space."""
    names = []
    oof_columns = []
    test_columns = []
    rows = []
    for source in sources or ():
        directory = os.path.abspath(source["directory"])
        prefix = str(source.get("prefix", "external"))
        oof_file = os.path.join(directory, source.get("oof_file", "oof_members.parquet"))
        test_file = os.path.join(
            directory, source.get("test_file", "test_members.parquet")
        )
        oof_frame = pd.read_parquet(oof_file)
        test_frame = pd.read_parquet(test_file)
        if not np.array_equal(oof_frame[id_column].values, train[id_column].values):
            raise ValueError("Logit parquet OOF IDs do not align: {}".format(oof_file))
        if not np.array_equal(test_frame[id_column].values, test[id_column].values):
            raise ValueError("Logit parquet test IDs do not align: {}".format(test_file))
        members = source.get("include_models") or [
            column for column in oof_frame.columns if column != id_column
        ]
        if list(members) != [column for column in members if column in test_frame.columns]:
            raise ValueError("Logit parquet OOF/test members differ in {}".format(directory))
        for member in members:
            oof = oof_frame[member].to_numpy(np.float64)
            test_pred = test_frame[member].to_numpy(np.float64)
            if not (np.isfinite(oof).all() and np.isfinite(test_pred).all()):
                raise ValueError("Non-finite logit member: {}".format(member))
            name = "{}__{}".format(prefix, member)
            names.append(name)
            oof_columns.append(oof)
            test_columns.append(test_pred)
            rows.append(
                {
                    "model": name,
                    "oof_auc": float(roc_auc_score(train[target].values, oof)),
                    "oof_min": float(oof.min()),
                    "oof_max": float(oof.max()),
                    "test_min": float(test_pred.min()),
                    "test_max": float(test_pred.max()),
                    "source_dtype": str(oof.dtype),
                }
            )
    if not names:
        return names, None, None, pd.DataFrame()
    return (
        names,
        np.column_stack(oof_columns),
        np.column_stack(test_columns),
        pd.DataFrame(rows),
    )


def regime_design(logits, missing_count, disagreement_mean=None, disagreement_std=None):
    """Add missingness/reliability interactions to the base-model logits."""
    logits = np.asarray(logits, dtype=np.float64)
    complete = (np.asarray(missing_count) == 0).astype(np.float64)[:, None]
    severe = (np.asarray(missing_count) >= 4).astype(np.float64)[:, None]
    disagreement = logits.std(axis=1, keepdims=True)
    if disagreement_mean is None:
        disagreement_mean = float(disagreement.mean())
    if disagreement_std is None:
        disagreement_std = float(disagreement.std())
    normalized = (disagreement - disagreement_mean) / (disagreement_std + 1e-6)
    aggregates = np.column_stack(
        [
            logits.mean(axis=1),
            logits.std(axis=1),
            logits.max(axis=1) - logits.min(axis=1),
            complete[:, 0],
            severe[:, 0],
        ]
    )
    design = np.column_stack(
        [logits, logits * complete, logits * severe, logits * normalized, aggregates]
    )
    return design, disagreement_mean, disagreement_std


def cross_fitted_logistic(
    design,
    y,
    folds,
    c_value,
    max_iter=5000,
    tolerance=1e-5,
    scale=True,
):
    predictions = np.zeros(len(y), dtype=np.float64)
    iterations = []
    for train_index, valid_index in folds:
        if scale:
            scaler = StandardScaler().fit(design[train_index])
            train_design = scaler.transform(design[train_index])
            valid_design = scaler.transform(design[valid_index])
        else:
            train_design = design[train_index]
            valid_design = design[valid_index]
        model = LogisticRegression(
            C=float(c_value),
            max_iter=int(max_iter),
            solver="lbfgs",
            tol=float(tolerance),
        )
        model.fit(train_design, y[train_index])
        predictions[valid_index] = model.predict_proba(valid_design)[:, 1]
        iterations.append(int(np.max(model.n_iter_)))
    return predictions, float(roc_auc_score(y, predictions)), iterations


def fit_final_logistic(
    design,
    test_design,
    y,
    c_value,
    max_iter=5000,
    tolerance=1e-5,
    scale=True,
):
    if scale:
        scaler = StandardScaler().fit(design)
        fit_design = scaler.transform(design)
        predict_design = scaler.transform(test_design)
    else:
        scaler = None
        fit_design = design
        predict_design = test_design
    model = LogisticRegression(
        C=float(c_value),
        max_iter=int(max_iter),
        solver="lbfgs",
        tol=float(tolerance),
    )
    model.fit(fit_design, y)
    predictions = model.predict_proba(predict_design)[:, 1]
    return predictions, model, scaler


def _validate_and_write_submission(sample, test, predictions, target, id_column, path):
    submission = sample.copy()
    submission[target] = np.asarray(predictions, dtype=np.float64)
    if list(submission.columns) != [id_column, target]:
        raise ValueError("Submission columns must be [{}, {}]".format(id_column, target))
    if not np.array_equal(submission[id_column].values, test[id_column].values):
        raise ValueError("Submission IDs do not match test.csv")
    values = submission[target].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("Submission predictions must be finite probabilities")
    submission.to_csv(path, index=False)


def _percentile_rank(values):
    return pd.Series(np.asarray(values)).rank(method="average").to_numpy(np.float64) / len(
        values
    )


def _reorder_inside_subset(base, indices, local_prediction, weight):
    """Reorder only a subset while preserving its global-score multiset."""
    indices = np.asarray(indices, dtype=np.int64)
    local_rank = _percentile_rank(local_prediction)
    base_rank = _percentile_rank(base[indices])
    order_score = (1.0 - float(weight)) * base_rank + float(weight) * local_rank
    result = np.asarray(base, dtype=np.float64).copy()
    result[indices[np.argsort(order_score, kind="mergesort")]] = np.sort(base[indices])
    return result


def postprocess_meta_predictions(
    train,
    test,
    global_oof,
    regime_oof,
    global_test,
    regime_test,
    settings,
    target,
    id_column,
    project_root,
):
    """Mix meta-model ranks and apply conservative band-local reorderers."""
    regime_weight = float(settings.get("regime_rank_weight", 2.0 / 3.0))
    oof = (
        (1.0 - regime_weight) * _percentile_rank(global_oof)
        + regime_weight * _percentile_rank(regime_oof)
    )
    test_pred = (
        (1.0 - regime_weight) * _percentile_rank(global_test)
        + regime_weight * _percentile_rank(regime_test)
    )
    source_dir = settings.get(
        "band_source_dir", "data/external/public_oof/fm_lattice_v5/extracted"
    )
    if not os.path.isabs(source_dir):
        source_dir = os.path.join(project_root, source_dir)

    band_diagnostics = []
    for band in settings.get("bands", []):
        path = os.path.join(source_dir, band["file"])
        frame = pd.read_parquet(path)
        required = {id_column, "split", "prediction"}
        if not required.issubset(frame.columns):
            raise ValueError("Invalid band prediction file: {}".format(path))
        train_band = frame.loc[frame["split"] == "train"]
        test_band = frame.loc[frame["split"] == "test"]
        train_index = pd.Index(train[id_column]).get_indexer(train_band[id_column])
        test_index = pd.Index(test[id_column]).get_indexer(test_band[id_column])
        if (train_index < 0).any() or (test_index < 0).any():
            raise ValueError("Band IDs are not a subset of local train/test: {}".format(path))
        weight = float(band["weight"])
        before_auc = float(
            roc_auc_score(train[target].values[train_index], oof[train_index])
        )
        local_auc = float(
            roc_auc_score(
                train[target].values[train_index],
                train_band["prediction"].to_numpy(np.float64),
            )
        )
        oof = _reorder_inside_subset(
            oof,
            train_index,
            train_band["prediction"].to_numpy(np.float64),
            weight,
        )
        test_pred = _reorder_inside_subset(
            test_pred,
            test_index,
            test_band["prediction"].to_numpy(np.float64),
            weight,
        )
        band_diagnostics.append(
            {
                "file": band["file"],
                "weight": weight,
                "train_rows": int(len(train_band)),
                "test_rows": int(len(test_band)),
                "base_band_auc": before_auc,
                "local_model_band_auc": local_auc,
            }
        )
    return oof, test_pred, regime_weight, source_dir, band_diagnostics


def run_public_stack(train, test, sample, config, output_dir):
    settings = config.get("public_stack", {})
    root = config.get("_project_root", os.getcwd())
    library_dir = settings.get(
        "library_dir", "data/external/public_oof/library_v6"
    )
    if not os.path.isabs(library_dir):
        library_dir = os.path.join(root, library_dir)
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    exclude = settings.get("exclude_models", [])
    names, oof_matrix, test_matrix, score_table, manifest = audit_and_load_library(
        train, test, library_dir, target, id_column, exclude=exclude
    )

    y = train[target].to_numpy(np.int8)
    n_splits = int(settings.get("n_splits", 5))
    seed = int(settings.get("seed", 42))
    folds = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )
    prediction_sources = []
    for source in settings.get("additional_prediction_sources", []):
        source = dict(source) if isinstance(source, dict) else {"directory": source}
        directory = source["directory"]
        if not os.path.isabs(directory):
            directory = os.path.join(root, directory)
        source["directory"] = directory
        prediction_sources.append(source)
    additional_names, additional_oof, additional_test, additional_scores = (
        load_additional_predictions(
            train,
            test,
            prediction_sources,
            target,
            id_column,
            folds,
        )
    )
    if additional_names:
        names.extend(additional_names)
        oof_matrix = np.column_stack([oof_matrix, additional_oof])
        test_matrix = np.column_stack([test_matrix, additional_test])
        score_table = pd.concat([score_table, additional_scores], ignore_index=True)
        score_table = score_table.sort_values("oof_auc", ascending=False)
    logits = _logit(oof_matrix)
    test_logits = _logit(test_matrix)
    logit_sources = []
    for source in settings.get("additional_logit_parquet_sources", []):
        source = dict(source)
        directory = source["directory"]
        if not os.path.isabs(directory):
            directory = os.path.join(root, directory)
        source["directory"] = directory
        logit_sources.append(source)
    logit_names, logit_oof, logit_test, logit_scores = load_logit_parquet_sources(
        train, test, logit_sources, target, id_column
    )
    if logit_names:
        names.extend(logit_names)
        logits = np.column_stack([logits, logit_oof])
        test_logits = np.column_stack([test_logits, logit_test])
        score_table = pd.concat([score_table, logit_scores], ignore_index=True)
        score_table = score_table.sort_values("oof_auc", ascending=False)
    score_table.to_csv(os.path.join(output_dir, "public_model_scores.csv"), index=False)
    max_iter = int(settings.get("max_iter", 5000))
    tolerance = float(settings.get("tolerance", 1e-5))
    global_c = float(settings.get("global_c", 1.0))

    global_oof, global_auc, global_iterations = cross_fitted_logistic(
        logits,
        y,
        folds,
        global_c,
        max_iter=max_iter,
        tolerance=tolerance,
        scale=False,
    )
    global_test, global_model, _ = fit_final_logistic(
        logits,
        test_logits,
        y,
        global_c,
        max_iter=max_iter,
        tolerance=tolerance,
        scale=False,
    )

    feature_columns = [column for column in test.columns if column != id_column]
    missing_train = train[feature_columns].isna().sum(axis=1).to_numpy(np.int8)
    missing_test = test[feature_columns].isna().sum(axis=1).to_numpy(np.int8)
    regime_matrix, disagreement_mean, disagreement_std = regime_design(
        logits, missing_train
    )
    regime_test_matrix, _, _ = regime_design(
        test_logits,
        missing_test,
        disagreement_mean=disagreement_mean,
        disagreement_std=disagreement_std,
    )

    regime_results = []
    for c_value in settings.get("regime_c_grid", [0.03, 0.1]):
        regime_oof, regime_auc, regime_iterations = cross_fitted_logistic(
            regime_matrix,
            y,
            folds,
            c_value,
            max_iter=max_iter,
            tolerance=tolerance,
            scale=True,
        )
        regime_results.append(
            {
                "c": float(c_value),
                "auc": regime_auc,
                "oof": regime_oof,
                "iterations": regime_iterations,
            }
        )
    best_regime = max(regime_results, key=lambda row: row["auc"])
    regime_test, regime_model, regime_scaler = fit_final_logistic(
        regime_matrix,
        regime_test_matrix,
        y,
        best_regime["c"],
        max_iter=max_iter,
        tolerance=tolerance,
        scale=True,
    )

    safety_margin = float(settings.get("regime_safety_margin", 0.00002))
    regime_gain = float(best_regime["auc"] - global_auc)
    selected = "regime" if regime_gain >= safety_margin else "global"
    selected_oof = best_regime["oof"] if selected == "regime" else global_oof
    selected_test = regime_test if selected == "regime" else global_test

    np.save(os.path.join(output_dir, "oof_public_global.npy"), global_oof)
    np.save(os.path.join(output_dir, "oof_public_regime.npy"), best_regime["oof"])
    np.save(os.path.join(output_dir, "test_public_global.npy"), global_test)
    np.save(os.path.join(output_dir, "test_public_regime.npy"), regime_test)
    pd.DataFrame({id_column: train[id_column], target: selected_oof}).to_csv(
        os.path.join(output_dir, "oof_public_selected.csv"), index=False
    )
    pd.DataFrame({id_column: test[id_column], target: selected_test}).to_csv(
        os.path.join(output_dir, "test_public_selected.csv"), index=False
    )
    _validate_and_write_submission(
        sample,
        test,
        global_test,
        target,
        id_column,
        os.path.join(output_dir, "submission_global.csv"),
    )
    _validate_and_write_submission(
        sample,
        test,
        regime_test,
        target,
        id_column,
        os.path.join(output_dir, "submission_regime.csv"),
    )
    selected_submission = os.path.join(output_dir, "submission.csv")
    _validate_and_write_submission(
        sample, test, selected_test, target, id_column, selected_submission
    )

    postprocess_payload = None
    postprocess_settings = settings.get("postprocess")
    if postprocess_settings and postprocess_settings.get("enabled", True):
        post_oof, post_test, regime_weight, band_source, band_diagnostics = (
            postprocess_meta_predictions(
                train,
                test,
                global_oof,
                best_regime["oof"],
                global_test,
                regime_test,
                postprocess_settings,
                target,
                id_column,
                root,
            )
        )
        post_auc = float(roc_auc_score(y, post_oof))
        post_gain = post_auc - float(roc_auc_score(y, selected_oof))
        min_gain = float(postprocess_settings.get("minimum_oof_gain", 0.000015))
        np.save(os.path.join(output_dir, "oof_public_postprocessed.npy"), post_oof)
        np.save(os.path.join(output_dir, "test_public_postprocessed.npy"), post_test)
        _validate_and_write_submission(
            sample,
            test,
            post_test,
            target,
            id_column,
            os.path.join(output_dir, "submission_postprocessed.csv"),
        )
        adopted = post_gain >= min_gain
        if adopted:
            selected = "band_postprocessed"
            selected_oof = post_oof
            selected_test = post_test
            pd.DataFrame(
                {id_column: train[id_column], target: selected_oof}
            ).to_csv(os.path.join(output_dir, "oof_public_selected.csv"), index=False)
            pd.DataFrame(
                {id_column: test[id_column], target: selected_test}
            ).to_csv(os.path.join(output_dir, "test_public_selected.csv"), index=False)
            _validate_and_write_submission(
                sample, test, selected_test, target, id_column, selected_submission
            )
        postprocess_payload = {
            "oof_auc": post_auc,
            "gain_over_selected_meta": post_gain,
            "minimum_oof_gain": min_gain,
            "adopted": adopted,
            "regime_rank_weight": regime_weight,
            "band_source_dir": band_source,
            "bands": band_diagnostics,
        }

    global_weights = pd.DataFrame(
        {"feature": names, "coefficient": global_model.coef_[0]}
    ).sort_values("coefficient", key=lambda values: values.abs(), ascending=False)
    global_weights.to_csv(os.path.join(output_dir, "global_meta_weights.csv"), index=False)
    regime_feature_names = (
        names
        + ["complete__{}".format(name) for name in names]
        + ["severe__{}".format(name) for name in names]
        + ["disagreement__{}".format(name) for name in names]
        + ["mean", "std", "range", "complete", "severe"]
    )
    # Convert standardized coefficients back to the raw design scale for interpretation.
    regime_raw_coefficients = regime_model.coef_[0] / regime_scaler.scale_
    pd.DataFrame(
        {"feature": regime_feature_names, "coefficient": regime_raw_coefficients}
    ).sort_values("coefficient", key=lambda values: values.abs(), ascending=False).to_csv(
        os.path.join(output_dir, "regime_meta_weights.csv"), index=False
    )

    per_bucket = {}
    for label, mask in (
        ("complete", missing_train == 0),
        ("missing_1_3", (missing_train >= 1) & (missing_train <= 3)),
        ("missing_4_plus", missing_train >= 4),
    ):
        per_bucket[label] = {
            "rows": int(mask.sum()),
            "global_auc": float(roc_auc_score(y[mask], global_oof[mask])),
            "regime_auc": float(roc_auc_score(y[mask], best_regime["oof"][mask])),
        }

    payload = {
        "library_dir": library_dir,
        "library_manifest_rows": int(len(manifest)),
        "models": names,
        "excluded_models": sorted(exclude),
        "additional_prediction_sources": prediction_sources,
        "additional_logit_parquet_sources": logit_sources,
        "matrix_shapes": {
            "oof": list(logits.shape),
            "test": list(test_logits.shape),
            "regime": list(regime_matrix.shape),
        },
        "dtype": str(oof_matrix.dtype),
        "folds": {"n_splits": n_splits, "seed": seed},
        "best_single_auc": float(score_table["oof_auc"].max()),
        "best_single_model": str(score_table.iloc[0]["model"]),
        "global": {
            "c": global_c,
            "oof_auc": global_auc,
            "max_iterations_by_fold": global_iterations,
        },
        "regime_grid": [
            {
                "c": row["c"],
                "oof_auc": row["auc"],
                "max_iterations_by_fold": row["iterations"],
            }
            for row in regime_results
        ],
        "best_regime": {
            "c": best_regime["c"],
            "oof_auc": best_regime["auc"],
            "gain_over_global": regime_gain,
        },
        "regime_safety_margin": safety_margin,
        "selected": selected,
        "selected_oof_auc": float(roc_auc_score(y, selected_oof)),
        "postprocess": postprocess_payload,
        "per_missingness_bucket": per_bucket,
        "submission_path": selected_submission,
    }
    write_json(os.path.join(output_dir, "public_stack.json"), payload)
    return payload
