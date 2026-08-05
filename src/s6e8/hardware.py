from __future__ import print_function

import copy

import numpy as np


def run_gpu_check(config):
    """Fit tiny models so a successful check proves both GPU backends work."""
    rng = np.random.RandomState(20260803)
    features = rng.normal(size=(8192, 24)).astype(np.float32)
    logits = (
        1.8 * features[:, 0]
        - 1.2 * features[:, 1]
        + 0.7 * features[:, 2] * features[:, 3]
        + rng.normal(scale=0.8, size=len(features))
    )
    target = (logits > 0).astype(np.int32)

    from catboost import CatBoostClassifier, __version__ as catboost_version

    catboost_params = copy.deepcopy(config["models"]["catboost"])
    catboost_params.pop("early_stopping_rounds", None)
    catboost_params.update(
        {
            "iterations": 25,
            "depth": 6,
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": 20260803,
        }
    )
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(features, target)

    import lightgbm as lgb

    lightgbm_params = copy.deepcopy(config["models"]["lightgbm"])
    lightgbm_params.pop("early_stopping_rounds", None)
    lightgbm_params.update(
        {
            "n_estimators": 25,
            "verbosity": -1,
            "random_state": 20260803,
        }
    )
    lightgbm_model = lgb.LGBMClassifier(**lightgbm_params)
    lightgbm_model.fit(features, target)

    xgboost_result = {"status": "not_configured"}
    try:
        import xgboost
    except ImportError:
        xgboost = None
    if "xgboost" in config.get("models", {}) and xgboost is not None:
        from xgboost import XGBClassifier, __version__ as xgboost_version

        xgboost_params = copy.deepcopy(config["models"]["xgboost"])
        xgboost_params.update(
            {
                "n_estimators": 25,
                "max_depth": 6,
                "random_state": 20260803,
                "verbosity": 0,
            }
        )
        xgboost_params.pop("early_stopping_rounds", None)
        xgboost_model = XGBClassifier(**xgboost_params)
        xgboost_model.fit(features, target)
        xgboost_result = {
            "version": xgboost_version,
            "device": xgboost_params.get("device", "cpu"),
            "tree_method": xgboost_params.get("tree_method", "auto"),
        }

    return {
        "status": "ok",
        "rows": int(features.shape[0]),
        "catboost": {
            "version": catboost_version,
            "task_type": catboost_params.get("task_type", "CPU"),
            "devices": str(catboost_params.get("devices", "")),
        },
        "lightgbm": {
            "version": lgb.__version__,
            "device_type": lightgbm_params.get("device_type", "cpu"),
            "gpu_device_id": int(lightgbm_params.get("gpu_device_id", 0)),
        },
        "xgboost": xgboost_result,
    }
