from __future__ import print_function

import numpy as np


def roc_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if not np.isfinite(y_score).all():
        raise ValueError("Predictions contain NaN or infinite values")
    return float(roc_auc_score(y_true, y_score))


def rank_percentile(values):
    import pandas as pd

    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).values
