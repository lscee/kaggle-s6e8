"""Train a GPU factorization machine with synchronized missing-value augmentation.

This is a project-native successor to the public FM lattice members.  It combines:

* exact/coarse value lookup embeddings;
* smooth periodic-linear embeddings for numeric values;
* factorized pair interactions and an optional explicit cross layer;
* a pairwise ranking loss aligned with the competition ROC-AUC metric; and
* synchronized missing-value augmentation across every representation of a field.

Predictions are generated with the frozen public-stack split (5 folds, seed 42).  A
complete run writes both probability CSVs (the checked project-native stack contract)
and raw-logit parquet files (for analysis or direct logit stacking).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s6e8.config import load_config  # noqa: E402
from s6e8.data import load_competition_data  # noqa: E402
from s6e8.io_utils import write_json  # noqa: E402

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised only in a missing GPU env
    raise RuntimeError(
        "PyTorch is required. Install requirements-gpu.txt inside the WSL environment."
    ) from exc


TARGET = "addicted_label"
ID_COLUMN = "id"
NUMERIC_COLUMNS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
CATEGORICAL_COLUMNS = ["gender", "stress_level", "academic_work_impact"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
COMPONENT_COLUMNS = ["social_media_hours", "gaming_hours", "work_study_hours"]
DAILY_COLUMN = "daily_screen_time_hours"


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_lattice_strings(values: pd.Series) -> pd.Series:
    return values.astype(str).where(values.notna(), None)


def build_lattice(frame: pd.DataFrame, min_count: int, coarse: bool):
    """Build field-local lookup codes and retain their raw-field provenance."""
    columns = []
    for source_index, column in enumerate(FEATURE_COLUMNS):
        columns.append((column, _as_lattice_strings(frame[column]), source_index))
    if coarse:
        for source_index, column in enumerate(NUMERIC_COLUMNS):
            values = pd.to_numeric(frame[column], errors="coerce")
            columns.append(
                (
                    "{}_r1".format(column),
                    _as_lattice_strings(values.round(1)),
                    source_index,
                )
            )

    codes = []
    vocab_sizes = []
    names = []
    source_indices = []
    for name, values, source_index in columns:
        counts = values.value_counts()
        keep = counts[counts >= min_count].index
        mapping = {value: index + 2 for index, value in enumerate(keep)}
        code = values.map(mapping)
        code = code.where(values.isna() | code.notna(), 1.0)
        codes.append(code.fillna(0).to_numpy(dtype=np.int64))
        vocab_sizes.append(len(keep) + 2)
        names.append(name)
        source_indices.append(source_index)

    return (
        np.stack(codes, axis=1),
        np.asarray(vocab_sizes, dtype=np.int64),
        names,
        np.asarray(source_indices, dtype=np.int64),
    )


def rank_gauss(frame: pd.DataFrame):
    """Target-free per-column rank Gaussian transform plus an observed-missing mask."""
    values = np.zeros((len(frame), frame.shape[1]), dtype=np.float32)
    missing = np.zeros_like(values)
    for column_index, column in enumerate(frame.columns):
        raw = frame[column].to_numpy(dtype=np.float64)
        observed = ~np.isnan(raw)
        if int(observed.sum()) > 10:
            transform = QuantileTransformer(
                n_quantiles=min(1000, int(observed.sum())),
                output_distribution="normal",
                subsample=min(400_000, int(observed.sum())),
                random_state=0,
            )
            values[observed, column_index] = transform.fit_transform(
                raw[observed].reshape(-1, 1)
            ).ravel().astype(np.float32)
        missing[~observed, column_index] = 1.0
    return values, missing


def build_numeric(frame: pd.DataFrame):
    """Build smooth raw/derived values and raw-field dependencies for augmentation."""
    raw = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    component_sum = frame[COMPONENT_COLUMNS].sum(axis=1, skipna=False)
    other = frame[DAILY_COLUMN] - component_sum
    derived = pd.DataFrame(
        {
            "other_screen": other,
            "component_sum": component_sum,
            "other_fraction": other / frame[DAILY_COLUMN].clip(lower=0.1),
            "component_fraction": component_sum
            / frame[DAILY_COLUMN].clip(lower=0.1),
            "weekend_minus_components": frame["weekend_screen_time"]
            - component_sum,
            "weekend_minus_other": frame["weekend_screen_time"] - other,
            "weekend_minus_daily": frame["weekend_screen_time"]
            - frame[DAILY_COLUMN],
            "notifications_per_open": frame["notifications_per_day"]
            / (frame["app_opens_per_day"] + 1.0),
        }
    )
    index = {name: position for position, name in enumerate(FEATURE_COLUMNS)}
    dependencies = [
        [DAILY_COLUMN] + COMPONENT_COLUMNS,
        COMPONENT_COLUMNS,
        [DAILY_COLUMN] + COMPONENT_COLUMNS,
        [DAILY_COLUMN] + COMPONENT_COLUMNS,
        ["weekend_screen_time"] + COMPONENT_COLUMNS,
        ["weekend_screen_time", DAILY_COLUMN] + COMPONENT_COLUMNS,
        ["weekend_screen_time", DAILY_COLUMN],
        ["notifications_per_day", "app_opens_per_day"],
    ]
    dependency_matrix = np.zeros(
        (len(dependencies), len(FEATURE_COLUMNS)), dtype=np.float32
    )
    for derived_index, source_names in enumerate(dependencies):
        dependency_matrix[derived_index, [index[name] for name in source_names]] = 1.0
    return (
        rank_gauss(raw),
        rank_gauss(derived),
        list(derived.columns),
        dependency_matrix,
    )


class PeriodicLinear(nn.Module):
    """A learned Fourier embedding for one scalar per numeric field."""

    def __init__(self, n_features: int, frequencies: int, width: int, sigma: float = 0.5):
        super().__init__()
        self.frequency = nn.Parameter(torch.randn(n_features, frequencies) * sigma)
        self.projection = nn.Parameter(
            torch.randn(n_features, 2 * frequencies, width)
            / math.sqrt(2 * frequencies)
        )
        self.bias = nn.Parameter(torch.zeros(n_features, width))

    def forward(self, values):
        phase = (
            2.0
            * math.pi
            * values.unsqueeze(-1)
            * self.frequency.unsqueeze(0)
        )
        periodic = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return (
            torch.einsum("bfk,fkd->bfd", periodic, self.projection)
            + self.bias
        )


class MissingAwareRankFM(nn.Module):
    """FM/DeepFM with smooth numeric tokens and optional explicit cross layers."""

    def __init__(
        self,
        total_vocab: int,
        n_lattice: int,
        n_numeric: int,
        n_derived: int,
        width: int,
        frequencies: int,
        deep_width: int,
        dropout: float,
        cross_layers: int,
    ):
        super().__init__()
        self.value_embedding = nn.Embedding(total_vocab, width)
        self.linear_embedding = nn.Embedding(total_vocab, 1)
        nn.init.normal_(self.value_embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        self.bias = nn.Parameter(torch.zeros(1))
        self.raw_numeric = PeriodicLinear(n_numeric, frequencies, width)
        self.derived_numeric = PeriodicLinear(n_derived, frequencies, width)

        self.n_tokens = n_lattice + n_derived
        flat_width = self.n_tokens * width
        self.deep = None
        if deep_width:
            self.deep = nn.Sequential(
                nn.Linear(flat_width, deep_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(deep_width, deep_width // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(deep_width // 2, 1),
            )

        self.cross = nn.ModuleList(
            [nn.Linear(flat_width, flat_width) for _ in range(cross_layers)]
        )
        self.cross_head = nn.Linear(flat_width, 1) if cross_layers else None
        for layer in self.cross:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        if self.cross_head is not None:
            nn.init.zeros_(self.cross_head.weight)
            nn.init.zeros_(self.cross_head.bias)

    def forward(self, lattice, numeric, numeric_missing, derived, derived_missing):
        exact = self.value_embedding(lattice)
        smooth = self.raw_numeric(numeric) * (1.0 - numeric_missing).unsqueeze(-1)
        # FEATURE_COLUMNS starts with NUMERIC_COLUMNS, so the first numeric lookup
        # tokens and the smooth numeric tokens refer to the exact same raw fields.
        exact = torch.cat(
            [exact[:, : smooth.shape[1]] + smooth, exact[:, smooth.shape[1] :]],
            dim=1,
        )
        derived_tokens = self.derived_numeric(derived) * (
            1.0 - derived_missing
        ).unsqueeze(-1)
        tokens = torch.cat([exact, derived_tokens], dim=1)

        summed = tokens.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) - tokens.square().sum(dim=(1, 2))
        )
        output = self.bias + self.linear_embedding(lattice).sum(dim=(1, 2))
        output = output + interaction
        flat = tokens.flatten(start_dim=1)
        if self.deep is not None:
            output = output + self.deep(flat).squeeze(-1)
        if self.cross_head is not None:
            crossed = flat
            for layer in self.cross:
                crossed = flat * layer(crossed) + crossed
            output = output + self.cross_head(crossed).squeeze(-1)
        return output


def _predict(model, tensors, chunk_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensors[0]), chunk_size):
            batch = [value[start : start + chunk_size] for value in tensors]
            chunks.append(model(*batch).float().cpu())
    return torch.cat(chunks).numpy().astype(np.float64, copy=False)


def _copy_parameters(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def _load_parameters(parameters, values) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            parameter.copy_(value)


def train(args) -> dict:
    config = load_config(args.config)
    train_frame, test_frame, _ = load_competition_data(config, require_sample=False)
    target = config["project"]["target"]
    id_column = config["project"]["id_column"]
    if target != TARGET or id_column != ID_COLUMN:
        raise ValueError("The FM rank trainer expects id/addicted_label columns")
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(test_frame.columns))
    if missing_columns:
        raise ValueError("Missing FM input columns: {}".format(missing_columns))

    stack_settings = config.get("public_stack", {})
    n_splits = int(stack_settings.get("n_splits", 5))
    split_seed = int(stack_settings.get("seed", 42))
    if n_splits != 5 or split_seed != 42:
        raise ValueError(
            "Stack compatibility requires public_stack n_splits=5 and seed=42"
        )
    if args.screen_early_stop and args.epochs <= args.warm:
        raise ValueError("epochs must be greater than warm")
    if args.fold_limit < 0 or args.fold_limit > n_splits:
        raise ValueError("fold-limit must be between 0 and {}".format(n_splits))
    if not 0.0 <= args.missing_aug < 1.0:
        raise ValueError("missing-aug must be in [0, 1)")
    if args.device != "cuda":
        raise ValueError("This trainer intentionally requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently train on CPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    y = train_frame[target].to_numpy(dtype=np.float32)
    n_train, n_test = len(train_frame), len(test_frame)
    combined = pd.concat(
        [train_frame[FEATURE_COLUMNS], test_frame[FEATURE_COLUMNS]],
        ignore_index=True,
    )
    lattice, vocab, lattice_names, lattice_sources = build_lattice(
        combined, args.min_count, args.coarse
    )
    offsets = np.concatenate([[0], np.cumsum(vocab)[:-1]]).astype(np.int64)
    lattice = lattice + offsets[None, :]
    total_vocab = int(vocab.sum())
    (numeric, numeric_missing), (derived, derived_missing), derived_names, dependencies = (
        build_numeric(combined)
    )
    if lattice_names[: len(NUMERIC_COLUMNS)] != NUMERIC_COLUMNS:
        raise RuntimeError("Numeric lookup and smooth-token order are not aligned")

    print(
        "{} lattice fields, {} derived fields, {} embeddings, device={}".format(
            len(lattice_names), len(derived_names), total_vocab, device
        ),
        flush=True,
    )
    print(
        "torch={} cuda={} gpu={}".format(
            torch.__version__, torch.version.cuda, torch.cuda.get_device_name(device)
        ),
        flush=True,
    )

    all_tensors = (
        torch.from_numpy(lattice),
        torch.from_numpy(numeric),
        torch.from_numpy(numeric_missing),
        torch.from_numpy(derived),
        torch.from_numpy(derived_missing),
    )
    y_tensor = torch.from_numpy(y)
    missing_codes = torch.from_numpy(offsets).to(device)
    lattice_source_tensor = torch.from_numpy(lattice_sources).to(device)
    dependency_tensor = torch.from_numpy(dependencies).to(device)

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=split_seed
    )
    folds = list(splitter.split(np.zeros(n_train), y))
    fold_ids = np.full(n_train, -1, dtype=np.int16)
    for fold_index, (_, valid_index) in enumerate(folds):
        fold_ids[valid_index] = fold_index
    if (fold_ids < 0).any():
        raise RuntimeError("Frozen folds do not cover every training row")
    pd.DataFrame(
        {
            id_column: train_frame[id_column],
            target: train_frame[target],
            "fold": fold_ids,
        }
    ).to_csv(output_dir / "folds.csv", index=False)

    def move_rows(selection):
        return tuple(value[selection].to(device) for value in all_tensors)

    test_tensors = move_rows(slice(n_train, None))
    oof = np.full(n_train, np.nan, dtype=np.float64)
    test_logits = np.zeros(n_test, dtype=np.float64)
    fold_metrics = []
    started = time.time()
    folds_to_run = args.fold_limit or n_splits

    for fold_index, (train_index, valid_index) in enumerate(folds[:folds_to_run]):
        fold_oof = np.zeros(len(valid_index), dtype=np.float64)
        fold_test = np.zeros(n_test, dtype=np.float64)
        seed_metrics = []
        for seed_index in range(args.seeds):
            model_seed = args.seed + 1000 * seed_index + fold_index
            _set_seed(model_seed)
            model = MissingAwareRankFM(
                total_vocab=total_vocab,
                n_lattice=len(lattice_names),
                n_numeric=numeric.shape[1],
                n_derived=derived.shape[1],
                width=args.width,
                frequencies=args.frequencies,
                deep_width=args.deep_width,
                dropout=args.dropout,
                cross_layers=args.cross_layers,
            ).to(device)
            embedding_parameters = [
                model.value_embedding.weight,
                model.linear_embedding.weight,
            ]
            other_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith(("value_embedding.", "linear_embedding."))
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_parameters, "weight_decay": args.weight_decay},
                    {
                        "params": embedding_parameters,
                        "weight_decay": args.embedding_weight_decay,
                    },
                ],
                lr=args.learning_rate,
            )
            steps_per_epoch = math.ceil(len(train_index) / args.batch_size)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.learning_rate,
                total_steps=steps_per_epoch * args.epochs,
                pct_start=0.2,
            )
            bce = nn.BCEWithLogitsLoss()
            parameters = list(model.parameters())
            ema = _copy_parameters(parameters)
            train_tensors = move_rows(train_index)
            valid_tensors = move_rows(valid_index)
            train_target = y_tensor[train_index].to(device)
            generator = torch.Generator(device=device).manual_seed(model_seed + 17)

            best_auc = -np.inf
            best_epoch = -1
            best_parameters = None
            stale_epochs = 0
            for epoch in range(args.epochs):
                model.train()
                permutation = torch.randperm(
                    len(train_index), generator=generator, device=device
                )
                for start in range(0, len(train_index), args.batch_size):
                    selection = permutation[start : start + args.batch_size]
                    lattice_batch, numeric_batch, numeric_mask, derived_batch, derived_mask = (
                        [value[selection] for value in train_tensors]
                    )
                    target_batch = train_target[selection]

                    if args.missing_aug:
                        # One raw-field mask is shared by its exact lookup, coarse lookup,
                        # smooth token, and every derived token that depends on that field.
                        hide_raw = (
                            torch.rand(
                                (len(selection), len(FEATURE_COLUMNS)),
                                generator=generator,
                                device=device,
                            )
                            < args.missing_aug
                        )
                        hide_lattice = hide_raw[:, lattice_source_tensor]
                        lattice_batch = torch.where(
                            hide_lattice,
                            missing_codes.expand_as(lattice_batch),
                            lattice_batch,
                        )
                        numeric_mask = torch.maximum(
                            numeric_mask,
                            hide_raw[:, : len(NUMERIC_COLUMNS)].float(),
                        )
                        hide_derived = (
                            hide_raw.float() @ dependency_tensor.transpose(0, 1)
                        ) > 0.0
                        derived_mask = torch.maximum(
                            derived_mask, hide_derived.float()
                        )

                    logits = model(
                        lattice_batch,
                        numeric_batch,
                        numeric_mask,
                        derived_batch,
                        derived_mask,
                    )
                    loss = bce(logits, target_batch)
                    if args.rank_weight:
                        positive = torch.nonzero(
                            target_batch > 0.5, as_tuple=True
                        )[0]
                        negative = torch.nonzero(
                            target_batch < 0.5, as_tuple=True
                        )[0]
                        pair_count = min(len(positive), len(negative))
                        if pair_count:
                            positive = positive[
                                torch.randperm(
                                    len(positive), generator=generator, device=device
                                )[:pair_count]
                            ]
                            negative = negative[
                                torch.randperm(
                                    len(negative), generator=generator, device=device
                                )[:pair_count]
                            ]
                            margin = logits[positive] - logits[negative]
                            loss = loss + args.rank_weight * nn.functional.softplus(
                                -margin
                            ).mean()

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
                    optimizer.step()
                    scheduler.step()
                    with torch.no_grad():
                        torch._foreach_mul_(ema, args.ema_decay)
                        torch._foreach_add_(
                            ema,
                            [parameter.detach() for parameter in parameters],
                            alpha=1.0 - args.ema_decay,
                        )

                # Outer-fold checkpoint selection is useful for cheap architecture
                # screening, but it is optimistic if the same fold is reported as OOF.
                # Formal runs therefore keep the final EMA checkpoint by default.
                if args.screen_early_stop and epoch >= args.warm:
                    backup = _copy_parameters(parameters)
                    _load_parameters(parameters, ema)
                    validation_logits = _predict(
                        model, valid_tensors, args.predict_batch_size
                    )
                    validation_auc = float(
                        roc_auc_score(y[valid_index], validation_logits)
                    )
                    if validation_auc > best_auc:
                        best_auc = validation_auc
                        best_epoch = epoch
                        best_parameters = _copy_parameters(parameters)
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                    print(
                        "fold={} seed={} epoch={:02d} auc={:.8f} best={:.8f} "
                        "elapsed={:.0f}s".format(
                            fold_index,
                            seed_index,
                            epoch,
                            validation_auc,
                            best_auc,
                            time.time() - started,
                        ),
                        flush=True,
                    )
                    _load_parameters(parameters, backup)
                    if stale_epochs >= args.patience:
                        break

            if best_parameters is None:
                best_parameters = _copy_parameters(ema)
                best_epoch = args.epochs - 1
            _load_parameters(parameters, best_parameters)
            seed_oof = _predict(model, valid_tensors, args.predict_batch_size)
            seed_test = _predict(model, test_tensors, args.predict_batch_size)
            if not args.screen_early_stop:
                best_auc = float(roc_auc_score(y[valid_index], seed_oof))
            fold_oof += seed_oof / args.seeds
            fold_test += seed_test / args.seeds
            seed_metrics.append(
                {
                    "seed": int(model_seed),
                    "best_epoch": int(best_epoch),
                    "best_validation_auc": float(best_auc),
                }
            )
            del model, train_tensors, valid_tensors, train_target
            torch.cuda.empty_cache()

        oof[valid_index] = fold_oof
        test_logits += fold_test / folds_to_run
        fold_auc = float(roc_auc_score(y[valid_index], fold_oof))
        fold_metrics.append(
            {
                "fold": int(fold_index),
                "rows": int(len(valid_index)),
                "auc": fold_auc,
                "seeds": seed_metrics,
            }
        )
        print(
            "fold={} complete auc={:.8f} elapsed={:.0f}s".format(
                fold_index, fold_auc, time.time() - started
            ),
            flush=True,
        )

    completed_rows = np.flatnonzero(np.isfinite(oof))
    pooled_auc = float(roc_auc_score(y[completed_rows], oof[completed_rows]))
    complete = folds_to_run == n_splits
    np.save(output_dir / "oof_{}.npy".format(args.name), oof)
    np.save(output_dir / "test_{}.npy".format(args.name), test_logits)
    np.save(output_dir / "completed_rows.npy", completed_rows)

    if complete:
        pd.DataFrame(
            {id_column: train_frame[id_column], args.name: oof}
        ).to_parquet(output_dir / "oof_members.parquet", index=False)
        pd.DataFrame(
            {id_column: test_frame[id_column], args.name: test_logits}
        ).to_parquet(output_dir / "test_members.parquet", index=False)
        pd.DataFrame(
            {
                id_column: train_frame[id_column],
                target: _stable_sigmoid(oof),
            }
        ).to_csv(output_dir / "oof_{}.csv".format(args.name), index=False)
        pd.DataFrame(
            {
                id_column: test_frame[id_column],
                target: _stable_sigmoid(test_logits),
            }
        ).to_csv(output_dir / "test_{}.csv".format(args.name), index=False)

    metrics = {
        "name": args.name,
        "complete": complete,
        "n_train": n_train,
        "n_test": n_test,
        "completed_rows": int(len(completed_rows)),
        "pooled_oof_auc": pooled_auc,
        "folds": {"n_splits": n_splits, "seed": split_seed},
        "per_fold": fold_metrics,
        "raw_logit_output": True,
        "checkpoint_selection": (
            "outer-valid early stopping; screening only"
            if args.screen_early_stop
            else "fixed epochs; final EMA checkpoint"
        ),
        "preprocessing_fit": "train+test; target-free",
        "oof_logit_range": [float(np.nanmin(oof)), float(np.nanmax(oof))],
        "test_logit_range": [float(test_logits.min()), float(test_logits.max())],
        "lattice": {
            "names": lattice_names,
            "vocab_sizes": vocab.tolist(),
            "total_vocab": total_vocab,
            "derived_names": derived_names,
        },
        "device": {
            "requested": args.device,
            "gpu_index": args.gpu,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
        },
        "parameters": vars(args),
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(str(output_dir / "metrics.json"), metrics)
    print(
        "OOF AUC over {} rows = {:.9f}; outputs={}".format(
            len(completed_rows), pooled_auc, output_dir
        ),
        flush=True,
    )
    return metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", default=str(ROOT / "configs" / "leaderboard_v3_combined.yaml")
    )
    result.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "leaderboard_v4_fm_rank")
    )
    result.add_argument("--name", default="fm_rank_cross")
    result.add_argument("--device", default="cuda", choices=["cuda"])
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--seed", type=int, default=20260812)
    result.add_argument("--width", type=int, default=8)
    result.add_argument("--frequencies", type=int, default=24)
    result.add_argument("--deep-width", type=int, default=384)
    result.add_argument("--cross-layers", type=int, default=1)
    result.add_argument("--epochs", type=int, default=40)
    result.add_argument("--warm", type=int, default=2)
    result.add_argument("--patience", type=int, default=6)
    result.add_argument(
        "--screen-early-stop",
        action="store_true",
        help=(
            "Select checkpoints on the outer validation fold. This is optimistic "
            "and is allowed only for cheap screening, never formal OOF evidence."
        ),
    )
    result.add_argument("--batch-size", type=int, default=8192)
    result.add_argument("--predict-batch-size", type=int, default=65536)
    result.add_argument("--learning-rate", type=float, default=0.003)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--embedding-weight-decay", type=float, default=3e-4)
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--missing-aug", type=float, default=0.1)
    result.add_argument("--rank-weight", type=float, default=0.2)
    result.add_argument("--gradient-clip", type=float, default=2.0)
    result.add_argument("--ema-decay", type=float, default=0.995)
    result.add_argument("--min-count", type=int, default=15)
    result.add_argument("--seeds", type=int, default=1)
    result.add_argument("--fold-limit", type=int, default=0, help="0 runs all folds")
    result.add_argument("--coarse", action="store_true")
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    print(json.dumps(vars(parsed), indent=2, sort_keys=True), flush=True)
    train(parsed)
