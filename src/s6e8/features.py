from __future__ import print_function

import numpy as np
import pandas as pd


EPSILON = 1e-6


def _safe_divide(numerator, denominator):
    denominator = denominator.astype(float)
    denominator = denominator.where(denominator.abs() > EPSILON, np.nan)
    result = numerator.astype(float) / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def add_engineered_features(frame):
    result = frame.copy()
    base_columns = list(result.columns)
    base_missing = result.loc[:, base_columns].isnull()
    numeric_base = [
        column
        for column in base_columns
        if pd.api.types.is_numeric_dtype(result[column])
    ]
    categorical_base = [column for column in base_columns if column not in numeric_base]
    result["missing_count"] = base_missing.sum(axis=1).astype(np.float32)
    result["missing_fraction"] = (
        result["missing_count"] / float(max(len(base_columns), 1))
    ).astype(np.float32)
    result["numeric_missing_count"] = (
        result.loc[:, numeric_base].isnull().sum(axis=1).astype(np.float32)
        if numeric_base
        else np.float32(0.0)
    )
    result["categorical_missing_count"] = (
        result.loc[:, categorical_base].isnull().sum(axis=1).astype(np.float32)
        if categorical_base
        else np.float32(0.0)
    )

    def has(*columns):
        return all(column in result.columns for column in columns)

    screen = "daily_screen_time_hours"
    social = "social_media_hours"
    gaming = "gaming_hours"
    work = "work_study_hours"
    sleep = "sleep_hours"
    weekend = "weekend_screen_time"
    notifications = "notifications_per_day"
    opens = "app_opens_per_day"

    if has(social, screen):
        result["social_ratio"] = _safe_divide(result[social], result[screen])
    if has(gaming, screen):
        result["gaming_ratio"] = _safe_divide(result[gaming], result[screen])
    if has(work, screen):
        result["work_ratio"] = _safe_divide(result[work], result[screen])
    if has(social, gaming):
        result["leisure_hours"] = result[social] + result[gaming]
        result["social_minus_gaming"] = result[social] - result[gaming]
    if has(social, gaming, work):
        result["known_usage_hours"] = result[social] + result[gaming] + result[work]
        if has(screen):
            result["other_screen_time"] = (
                result[screen] - result["known_usage_hours"]
            )
    if has(weekend, screen):
        result["weekend_delta"] = result[weekend] - result[screen]
        result["weekend_ratio"] = _safe_divide(result[weekend], result[screen])
    if has(sleep):
        result["sleep_deficit"] = 8.0 - result[sleep]
    if has(screen, sleep):
        result["screen_sleep_sum"] = result[screen] + result[sleep]
        result["screen_sleep_product"] = result[screen] * result[sleep]
        result["screen_per_sleep_hour"] = _safe_divide(
            result[screen], result[sleep]
        )
        result["screen_per_awake_hour"] = _safe_divide(
            result[screen], (24.0 - result[sleep])
        )
    if has(screen):
        result["screen_time_squared"] = result[screen] ** 2
        result["log1p_screen_time"] = np.log1p(result[screen].clip(lower=0))
    if has(weekend):
        result["weekend_time_squared"] = result[weekend] ** 2
    if has(social):
        result["social_time_squared"] = result[social] ** 2
    if has(gaming):
        result["gaming_time_squared"] = result[gaming] ** 2
    if has(notifications, screen):
        result["notifications_per_hour"] = _safe_divide(
            result[notifications], result[screen]
        )
    if has(opens, screen):
        result["app_opens_per_hour"] = _safe_divide(
            result[opens], result[screen]
        )
    if has(notifications, opens):
        result["notifications_per_open"] = _safe_divide(
            result[notifications], result[opens]
        )
        result["digital_interruptions"] = result[notifications] + result[opens]
    if has(notifications):
        result["log1p_notifications"] = np.log1p(result[notifications].clip(lower=0))
    if has(opens):
        result["log1p_app_opens"] = np.log1p(result[opens].clip(lower=0))

    if has("academic_work_impact", screen):
        impact = result["academic_work_impact"].astype(str).str.lower()
        result["impact_yes_x_screen"] = (
            impact.isin(["yes", "1", "true"]).astype(float) * result[screen]
        )
    if has("stress_level", sleep):
        stress_map = {"low": 0.0, "medium": 1.0, "high": 2.0}
        stress = result["stress_level"].astype(str).str.lower().map(stress_map)
        result["stress_x_sleep_deficit"] = stress * (8.0 - result[sleep])
        if has(screen):
            result["stress_x_screen"] = stress * result[screen]

    if has("academic_work_impact", work):
        impact = result["academic_work_impact"].astype(str).str.lower()
        impact_yes = impact.isin(["yes", "1", "true"]).astype(float)
        result["impact_yes_x_work"] = impact_yes * result[work]
    if has(work, screen):
        result["non_work_screen_hours"] = result[screen] - result[work]
        result["screen_minus_work"] = result[screen] - result[work]

    return result


def prepare_feature_frames(train, test, config, view=None):
    from .data import feature_columns

    columns = feature_columns(train, config)
    base_train_x = train.loc[:, columns].copy()
    base_test_x = test.loc[:, columns].copy()
    if view is None:
        view = config["features"].get(
            "default_view",
            "engineered" if config["features"].get("engineered", True) else "raw",
        )
    valid_views = {"raw", "engineered", "raw_parent", "engineered_parent"}
    if view not in valid_views:
        raise ValueError(
            "Unknown feature view '{}'; expected one of {}".format(
                view, sorted(valid_views)
            )
        )

    train_x = base_train_x
    test_x = base_test_x
    if view in ("engineered", "engineered_parent"):
        train_x = add_engineered_features(train_x)
        test_x = add_engineered_features(test_x)
    if view in ("raw_parent", "engineered_parent"):
        from .external import build_parent_neighbor_features

        parent_train, parent_test = build_parent_neighbor_features(
            base_train_x, base_test_x, config
        )
        train_x = pd.concat(
            [train_x.reset_index(drop=True), parent_train], axis=1
        )
        test_x = pd.concat(
            [test_x.reset_index(drop=True), parent_test], axis=1
        )
    if list(train_x.columns) != list(test_x.columns):
        raise RuntimeError("Feature engineering produced inconsistent train/test columns")
    return train_x, test_x


def categorical_columns(frame):
    columns = []
    for column in frame.columns:
        dtype = frame[column].dtype
        if str(dtype) in ("object", "category", "bool"):
            columns.append(column)
    return columns


def encode_for_sklearn(
    train_x,
    test_x,
    fill_missing=True,
    add_missing_indicators=True,
    missing_indicator_columns=None,
    output_dtype=np.float32,
):
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True, sort=False)
    categorical = categorical_columns(combined)
    numeric = [column for column in combined.columns if column not in categorical]
    if add_missing_indicators:
        indicator_candidates = numeric
        if missing_indicator_columns is not None:
            allowed = set(missing_indicator_columns)
            indicator_candidates = [column for column in numeric if column in allowed]
        for column in indicator_candidates:
            if combined[column].isnull().any():
                indicator = "{}__missing".format(column)
                if indicator in combined.columns:
                    raise ValueError("Missing-indicator name collision: {}".format(indicator))
                combined[indicator] = combined[column].isnull().astype(np.float32)
    if categorical:
        combined = pd.get_dummies(
            combined, columns=categorical, dummy_na=True, prefix_sep="__"
        )
    combined = combined.replace([np.inf, -np.inf], np.nan)
    train_encoded = combined.iloc[: len(train_x)].copy()
    test_encoded = combined.iloc[len(train_x) :].copy()
    if fill_missing:
        medians = train_encoded.median(axis=0, numeric_only=True)
        train_encoded = train_encoded.fillna(medians).fillna(0.0)
        test_encoded = test_encoded.fillna(medians).fillna(0.0)
    return train_encoded.astype(output_dtype), test_encoded.astype(output_dtype)
