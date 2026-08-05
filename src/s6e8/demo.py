from __future__ import print_function

import os

import numpy as np
import pandas as pd


def make_demo_data(output_dir, train_rows=1400, test_rows=500, seed=20260803):
    rng = np.random.RandomState(seed)
    total = train_rows + test_rows
    screen = rng.uniform(3.0, 12.0, total)
    social = np.minimum(rng.uniform(0.0, 6.0, total), screen * 0.8)
    gaming = np.minimum(rng.uniform(0.0, 4.0, total), screen * 0.6)
    work = rng.uniform(0.5, 6.0, total)
    sleep = np.clip(rng.normal(7.0, 1.2, total), 3.5, 10.0)
    weekend = np.clip(screen + rng.normal(1.0, 1.4, total), 2.0, 15.0)
    stress_num = rng.choice([0, 1, 2], total, p=[0.3, 0.45, 0.25])
    impact = rng.binomial(1, 1.0 / (1.0 + np.exp(-(screen - 7.0))), total)
    latent = (
        0.55 * (screen - 7.0)
        + 0.45 * (social - 3.0)
        + 0.25 * gaming
        + 0.35 * (weekend - screen)
        + 0.55 * impact
        + 0.25 * stress_num
        - 0.25 * (sleep - 7.0)
        + rng.normal(0.0, 1.2, total)
    )
    threshold = np.quantile(latent[:train_rows], 0.30)
    labels = (latent >= threshold).astype(int)
    cuts = np.quantile(latent, [0.18, 0.38, 0.70])
    levels = np.where(
        latent < cuts[0],
        "None",
        np.where(latent < cuts[1], "Mild", np.where(latent < cuts[2], "Moderate", "Severe")),
    )
    genders = rng.choice(["Female", "Male", "Other"], total)
    stresses = np.asarray(["Low", "Medium", "High"])[stress_num]

    frame = pd.DataFrame(
        {
            "id": np.arange(total),
            "age": rng.randint(18, 36, total),
            "gender": genders,
            "daily_screen_time_hours": np.round(screen, 2),
            "social_media_hours": np.round(social, 2),
            "gaming_hours": np.round(gaming, 2),
            "work_study_hours": np.round(work, 2),
            "sleep_hours": np.round(sleep, 2),
            "notifications_per_day": rng.poisson(45.0 + screen * 12.0),
            "app_opens_per_day": rng.poisson(25.0 + screen * 9.0),
            "weekend_screen_time": np.round(weekend, 2),
            "stress_level": stresses,
            "academic_work_impact": np.where(impact == 1, "Yes", "No"),
            "addiction_level": levels,
        }
    )
    missing_rates = {
        "age": 0.04,
        "daily_screen_time_hours": 0.12,
        "social_media_hours": 0.16,
        "gaming_hours": 0.15,
        "work_study_hours": 0.08,
        "sleep_hours": 0.07,
        "notifications_per_day": 0.10,
        "app_opens_per_day": 0.09,
        "weekend_screen_time": 0.14,
        "gender": 0.04,
        "stress_level": 0.07,
        "academic_work_impact": 0.06,
    }
    for column, rate in missing_rates.items():
        frame.loc[rng.rand(total) < rate, column] = np.nan
    train = frame.iloc[:train_rows].copy()
    train["addicted_label"] = labels[:train_rows]
    test = frame.iloc[train_rows:].copy()
    sample = pd.DataFrame(
        {"id": test["id"].values, "addicted_label": np.full(test_rows, 0.5)}
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    sample.to_csv(os.path.join(output_dir, "sample_submission.csv"), index=False)
    return train, test, sample
