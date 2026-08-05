from __future__ import print_function

import hashlib
import os

import numpy as np
import pandas as pd


def _cache_signature(train, test, parent_path, feature_columns, settings):
    stat = os.stat(parent_path)
    payload = "|".join(
        [
            os.path.abspath(parent_path),
            str(stat.st_size),
            str(int(stat.st_mtime)),
            str(len(train)),
            str(len(test)),
            str(train.index[0] if len(train) else ""),
            str(test.index[-1] if len(test) else ""),
            ",".join(feature_columns),
            ",".join(str(value) for value in settings.get("neighbors", [])),
            str(settings.get("categorical_weight", 1.25)),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_cached_features(cache_path, signature, train_rows, test_rows):
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
        cached_signature = str(cached["signature"].item())
        if cached_signature != signature:
            return None
        names = [str(value) for value in cached["names"].tolist()]
        train_values = cached["train_values"]
        test_values = cached["test_values"]
        if train_values.shape[0] != train_rows or test_values.shape[0] != test_rows:
            return None
        print("Loaded cached parent-neighbor features: {}".format(cache_path))
        return (
            pd.DataFrame(train_values, columns=names),
            pd.DataFrame(test_values, columns=names),
        )
    except (OSError, ValueError, KeyError):
        return None


def _save_cached_features(cache_path, signature, train_features, test_features):
    if not cache_path:
        return
    directory = os.path.dirname(cache_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    np.savez_compressed(
        cache_path,
        signature=np.asarray(signature),
        names=np.asarray(list(train_features.columns)),
        train_values=train_features.values.astype(np.float32),
        test_values=test_features.values.astype(np.float32),
    )


def _parent_matrix(parent, query, feature_columns, categorical_weight):
    categorical = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(parent[column])
    ]
    numeric = [column for column in feature_columns if column not in categorical]
    parts_parent = []
    parts_query = []

    if numeric:
        parent_numeric = parent.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
        query_numeric = query.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
        medians = parent_numeric.median(axis=0).fillna(0.0)
        scales = parent_numeric.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        parts_parent.append(
            ((parent_numeric.fillna(medians) - medians) / scales).values.astype(
                np.float32
            )
        )
        parts_query.append(
            ((query_numeric.fillna(medians) - medians) / scales).values.astype(
                np.float32
            )
        )

    if categorical:
        for column in categorical:
            parent_values = parent[column].fillna("__MISSING__").astype(str)
            query_values = query[column].fillna("__MISSING__").astype(str)
            categories = sorted(parent_values.unique().tolist())
            parent_codes = np.column_stack(
                [(parent_values == value).values for value in categories]
            ).astype(np.float32)
            query_codes = np.column_stack(
                [(query_values == value).values for value in categories]
            ).astype(np.float32)
            parts_parent.append(parent_codes * float(categorical_weight))
            parts_query.append(query_codes * float(categorical_weight))

    return np.column_stack(parts_parent), np.column_stack(parts_query)


def build_parent_neighbor_features(train_x, test_x, config):
    """Create target-free-to-competition features from the public 7,500-row source.

    The competition labels are never used here.  Each competition row is projected
    into the source-data feature space, then summarized using labelled source-data
    neighbours.  This keeps CV honest while exposing the generator's local rules.
    """

    from sklearn.neighbors import NearestNeighbors

    settings = config.get("features", {}).get("parent_neighbors", {})
    parent_path = config.get("data", {}).get("parent_data_path")
    if not parent_path or not os.path.isfile(parent_path):
        raise FileNotFoundError(
            "Parent-data feature view requested, but parent_data_path is missing: {}".format(
                parent_path
            )
        )
    target = config["project"]["target"]
    parent = pd.read_csv(parent_path)
    feature_columns = [column for column in train_x.columns if column in parent.columns]
    missing = sorted(set(train_x.columns) - set(feature_columns))
    if missing:
        raise ValueError(
            "Parent dataset does not contain competition features: {}".format(missing)
        )
    if target not in parent.columns:
        raise ValueError("Parent dataset is missing target '{}'".format(target))
    parent_target = parent[target].astype(int).values
    if not set(np.unique(parent_target)).issubset({0, 1}):
        raise ValueError("Parent target must be binary 0/1")

    neighbors = sorted(set(int(value) for value in settings.get("neighbors", [1, 3, 5, 10, 25])))
    if not neighbors or neighbors[0] < 1 or neighbors[-1] > len(parent):
        raise ValueError("Invalid parent-neighbor counts: {}".format(neighbors))
    cache_path = settings.get("cache_path")
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(config["_project_root"], cache_path)
    signature = _cache_signature(
        train_x,
        test_x,
        parent_path,
        feature_columns,
        settings,
    )
    cached = _load_cached_features(
        cache_path, signature, len(train_x), len(test_x)
    )
    if cached is not None:
        return cached

    query = pd.concat(
        [train_x.loc[:, feature_columns], test_x.loc[:, feature_columns]],
        axis=0,
        ignore_index=True,
    )
    parent_matrix, query_matrix = _parent_matrix(
        parent,
        query,
        feature_columns,
        settings.get("categorical_weight", 1.25),
    )
    max_neighbors = neighbors[-1]
    index = NearestNeighbors(
        n_neighbors=max_neighbors,
        algorithm=settings.get("algorithm", "kd_tree"),
        leaf_size=int(settings.get("leaf_size", 40)),
        n_jobs=int(settings.get("n_jobs", -1)),
    ).fit(parent_matrix)
    class_indexes = {
        value: NearestNeighbors(
            n_neighbors=1,
            algorithm=settings.get("algorithm", "kd_tree"),
            leaf_size=int(settings.get("leaf_size", 40)),
            n_jobs=int(settings.get("n_jobs", -1)),
        ).fit(parent_matrix[parent_target == value])
        for value in (0, 1)
    }

    feature_names = []
    for count in neighbors:
        feature_names.extend(
            [
                "parent_knn_label_mean_{}".format(count),
                "parent_knn_label_weighted_{}".format(count),
                "parent_knn_distance_mean_{}".format(count),
            ]
        )
    feature_names.extend(
        [
            "parent_knn_distance_min",
            "parent_distance_class_0",
            "parent_distance_class_1",
            "parent_distance_margin",
        ]
    )
    result = np.empty((len(query), len(feature_names)), dtype=np.float32)
    batch_size = int(settings.get("batch_size", 50000))
    prior = float(parent_target.mean())
    smoothing = float(settings.get("smoothing", 1.0))
    for start in range(0, len(query_matrix), batch_size):
        stop = min(start + batch_size, len(query_matrix))
        distances, indices = index.kneighbors(query_matrix[start:stop])
        labels = parent_target[indices]
        columns = []
        for count in neighbors:
            local_distances = distances[:, :count]
            local_labels = labels[:, :count]
            mean = (local_labels.sum(axis=1) + prior * smoothing) / (
                float(count) + smoothing
            )
            weights = 1.0 / np.maximum(local_distances, 1e-4)
            weighted = (local_labels * weights).sum(axis=1) / weights.sum(axis=1)
            columns.extend([mean, weighted, local_distances.mean(axis=1)])
        class_0 = class_indexes[0].kneighbors(
            query_matrix[start:stop], return_distance=True
        )[0][:, 0]
        class_1 = class_indexes[1].kneighbors(
            query_matrix[start:stop], return_distance=True
        )[0][:, 0]
        columns.extend(
            [distances[:, 0], class_0, class_1, class_0 - class_1]
        )
        result[start:stop] = np.column_stack(columns).astype(np.float32)
        print(
            "Parent-neighbor features: {:,}/{:,} rows".format(stop, len(query))
        )

    result_frame = pd.DataFrame(result, columns=feature_names)
    train_features = result_frame.iloc[: len(train_x)].reset_index(drop=True)
    test_features = result_frame.iloc[len(train_x) :].reset_index(drop=True)
    _save_cached_features(
        cache_path, signature, train_features, test_features
    )
    return train_features, test_features
