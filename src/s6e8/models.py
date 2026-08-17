from __future__ import print_function

import copy
import importlib

import numpy as np

from .features import categorical_columns


MODEL_MODULES = {
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "extra_trees": "sklearn",
    "logistic": "sklearn",
}


def model_available(name):
    module = MODEL_MODULES.get(name)
    if module is None:
        return False
    try:
        importlib.import_module(module)
        return True
    except OSError as error:
        if name == "lightgbm" and "libnccl.so" in str(error):
            # The CUDA LightGBM wheel expects NCCL symbols that the PyTorch CUDA
            # runtime loads into the process on this WSL setup.
            importlib.import_module("torch")
            importlib.import_module(module)
            return True
        raise
    except ImportError:
        return False


def matrix_kind(name):
    if name == "catboost":
        return "raw"
    if name in ("lightgbm", "xgboost"):
        return "encoded_native_missing"
    return "encoded_filled"


def _fit_catboost(train_x, y_train, valid_x, y_valid, test_x, params, seed):
    from catboost import CatBoostClassifier

    params = copy.deepcopy(params)
    early_stopping_rounds = params.pop("early_stopping_rounds", 150)
    params["random_seed"] = seed
    categories = categorical_columns(train_x)
    cat_indices = [train_x.columns.get_loc(column) for column in categories]

    def sanitize(frame):
        frame = frame.copy()
        for column in categories:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        return frame

    train_x = sanitize(train_x)
    valid_x = sanitize(valid_x)
    test_x = sanitize(test_x)
    model = CatBoostClassifier(**params)
    model.fit(
        train_x,
        y_train,
        eval_set=(valid_x, y_valid),
        cat_features=cat_indices,
        early_stopping_rounds=early_stopping_rounds,
        verbose=params.get("verbose", False),
    )
    valid_pred = model.predict_proba(valid_x)[:, 1]
    test_pred = model.predict_proba(test_x)[:, 1]
    importance = getattr(model, "feature_importances_", None)
    return valid_pred, test_pred, importance, {
        "best_iteration": int(model.get_best_iteration())
    }


def _fit_lightgbm(train_x, y_train, valid_x, y_valid, test_x, params, seed):
    import lightgbm as lgb

    params = copy.deepcopy(params)
    early_stopping_rounds = params.pop("early_stopping_rounds", 150)
    params["random_state"] = seed
    model = lgb.LGBMClassifier(**params)
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    callbacks.append(lgb.log_evaluation(period=0))
    model.fit(
        train_x,
        y_train,
        eval_set=[(valid_x, y_valid)],
        eval_metric="auc",
        callbacks=callbacks,
    )
    valid_pred = model.predict_proba(valid_x)[:, 1]
    test_pred = model.predict_proba(test_x)[:, 1]
    return valid_pred, test_pred, model.feature_importances_, {
        "best_iteration": int(model.best_iteration_ or params.get("n_estimators", 0))
    }


def _fit_xgboost(train_x, y_train, valid_x, y_valid, test_x, params, seed):
    from xgboost import XGBClassifier

    params = copy.deepcopy(params)
    params["random_state"] = seed
    model = XGBClassifier(**params)
    model.fit(
        train_x,
        y_train,
        eval_set=[(valid_x, y_valid)],
        verbose=False,
    )
    valid_pred = model.predict_proba(valid_x)[:, 1]
    test_pred = model.predict_proba(test_x)[:, 1]
    best_iteration = getattr(model, "best_iteration", None)
    return valid_pred, test_pred, model.feature_importances_, {
        "best_iteration": int(best_iteration) if best_iteration is not None else None
    }


def _fit_extra_trees(train_x, y_train, valid_x, y_valid, test_x, params, seed):
    from sklearn.ensemble import ExtraTreesClassifier

    params = copy.deepcopy(params)
    params["random_state"] = seed
    model = ExtraTreesClassifier(**params)
    model.fit(train_x, y_train)
    valid_pred = model.predict_proba(valid_x)[:, 1]
    test_pred = model.predict_proba(test_x)[:, 1]
    return valid_pred, test_pred, model.feature_importances_, {}


def _fit_logistic(train_x, y_train, valid_x, y_valid, test_x, params, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = copy.deepcopy(params)
    params["random_state"] = seed
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ]
    )
    model.fit(train_x, y_train)
    valid_pred = model.predict_proba(valid_x)[:, 1]
    test_pred = model.predict_proba(test_x)[:, 1]
    coefficients = np.abs(model.named_steps["model"].coef_[0])
    return valid_pred, test_pred, coefficients, {}


def fit_predict_fold(
    name, train_x, y_train, valid_x, y_valid, test_x, params, seed
):
    if name == "catboost":
        return _fit_catboost(
            train_x, y_train, valid_x, y_valid, test_x, params, seed
        )
    if name == "lightgbm":
        return _fit_lightgbm(
            train_x, y_train, valid_x, y_valid, test_x, params, seed
        )
    if name == "xgboost":
        return _fit_xgboost(
            train_x, y_train, valid_x, y_valid, test_x, params, seed
        )
    if name == "extra_trees":
        return _fit_extra_trees(
            train_x, y_train, valid_x, y_valid, test_x, params, seed
        )
    if name == "logistic":
        return _fit_logistic(
            train_x, y_train, valid_x, y_valid, test_x, params, seed
        )
    raise ValueError("Unknown model '{}'".format(name))
