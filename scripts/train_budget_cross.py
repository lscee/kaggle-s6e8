"""Train a constraint-geometry DCNv2 candidate on the frozen seed-42 folds.

Unlike the existing lookup Transformer and factorization machines, this model never
receives an exact numeric value ID.  It sees smooth rank-Gaussian numeric values,
missingness, categorical fields, and target-free feasible-region geometry implied by
the screen-time budget.  Online masking is applied to raw fields first, so all budget
features are recomputed consistently and cannot leak a hidden value through a derived
feature.
"""
from __future__ import annotations

import argparse
import json
import math
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
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install requirements-gpu.txt in WSL before training") from exc


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
TARGET = "addicted_label"
ID_COLUMN = "id"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_gauss(frame: pd.DataFrame) -> np.ndarray:
    """Target-free rank Gaussian transform; missing values remain zero."""
    result = np.zeros((len(frame), frame.shape[1]), dtype=np.float32)
    for index, column in enumerate(frame.columns):
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        observed = np.isfinite(raw)
        if int(observed.sum()) <= 10:
            continue
        transformer = QuantileTransformer(
            n_quantiles=min(1000, int(observed.sum())),
            output_distribution="normal",
            subsample=min(400_000, int(observed.sum())),
            random_state=0,
        )
        result[observed, index] = transformer.fit_transform(
            raw[observed].reshape(-1, 1)
        ).ravel().astype(np.float32)
    return result


def encode_categories(frame: pd.DataFrame):
    codes = []
    vocab_sizes = []
    mappings = {}
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype(str).where(frame[column].notna(), None)
        categories = sorted(value for value in values.dropna().unique())
        mapping = {value: index + 1 for index, value in enumerate(categories)}
        codes.append(values.map(mapping).fillna(0).to_numpy(np.int64))
        vocab_sizes.append(len(categories) + 1)
        mappings[column] = mapping
    return np.column_stack(codes), vocab_sizes, mappings


def _safe_divide(numerator, denominator, valid, epsilon=1e-4):
    safe = np.where(np.abs(denominator) > epsilon, denominator, 1.0)
    return np.where(valid, numerator / safe, 0.0)


def budget_geometry_numpy(values: np.ndarray, observed: np.ndarray, max_daily: float):
    """Mirror the model's target-free budget calculations for scaling statistics."""
    values = np.asarray(values, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    effective = np.where(observed, values, 0.0)
    daily = effective[:, 1]
    daily_seen = observed[:, 1]
    components = effective[:, 2:5]
    component_seen = observed[:, 2:5]
    component_sum = components.sum(axis=1)
    component_count = component_seen.sum(axis=1).astype(np.float32)
    complete_components = component_count == 3
    weekend = effective[:, 8]
    weekend_seen = observed[:, 8]

    daily_upper = np.where(daily_seen, daily, max_daily)
    daily_width = np.maximum(daily_upper - component_sum, 0.0)
    capacity_valid = daily_seen
    capacity = np.where(capacity_valid, np.maximum(daily - component_sum, 0.0), 0.0)
    component_ratio = _safe_divide(
        component_sum, daily, daily_seen & (np.abs(daily) > 1e-4)
    )
    strict_sum = np.where(complete_components, components.sum(axis=1), 0.0)
    other_valid = daily_seen & complete_components
    other_screen = np.where(other_valid, daily - strict_sum, 0.0)

    component_upper = []
    component_width = []
    for index in range(3):
        other_sum = component_sum - components[:, index]
        upper = np.where(daily_seen, np.maximum(daily - other_sum, 0.0), 0.0)
        width = np.where(daily_seen & ~component_seen[:, index], upper, 0.0)
        component_upper.append(upper)
        component_width.append(width)

    screen_indices = [1, 2, 3, 4, 8]
    screen = effective[:, screen_indices]
    screen_seen = observed[:, screen_indices]
    screen_count = screen_seen.sum(axis=1).astype(np.float32)
    screen_sum = screen.sum(axis=1)
    denominator = np.maximum(screen_count, 1.0)
    screen_mean = screen_sum / denominator
    centered = np.where(screen_seen, screen - screen_mean[:, None], 0.0)
    screen_std = np.sqrt((centered * centered).sum(axis=1) / denominator)
    screen_min = np.where(screen_seen, screen, np.inf).min(axis=1)
    screen_max = np.where(screen_seen, screen, -np.inf).max(axis=1)
    any_screen = screen_count > 0
    screen_min = np.where(any_screen, screen_min, 0.0)
    screen_max = np.where(any_screen, screen_max, 0.0)

    weekend_daily_valid = weekend_seen & daily_seen
    notifications_seen = observed[:, 6]
    opens_seen = observed[:, 7]
    notification_ratio_valid = notifications_seen & opens_seen
    daily_sleep_valid = daily_seen & observed[:, 5]

    values_out = np.column_stack(
        [
            component_sum,
            component_count,
            component_sum,
            daily_upper,
            daily_width,
            capacity,
            component_ratio,
            complete_components.astype(np.float32),
            strict_sum,
            other_screen,
            np.where(weekend_daily_valid, weekend - daily, 0.0),
            np.where(weekend_seen, weekend - component_sum, 0.0),
            _safe_divide(
                effective[:, 6],
                effective[:, 7] + 1.0,
                notification_ratio_valid,
            ),
            *component_upper,
            *component_width,
            screen_count,
            screen_sum,
            screen_mean,
            screen_std,
            screen_min,
            screen_max,
            screen_max - screen_min,
            _safe_divide(components[:, 0], daily, daily_seen & component_seen[:, 0]),
            _safe_divide(components[:, 1], daily, daily_seen & component_seen[:, 1]),
            _safe_divide(components[:, 2], daily, daily_seen & component_seen[:, 2]),
            _safe_divide(weekend, daily, weekend_daily_valid),
            np.where(daily_sleep_valid, daily + effective[:, 5], 0.0),
        ]
    ).astype(np.float32)
    valid_out = np.column_stack(
        [
            np.ones(len(values), dtype=bool),
            np.ones(len(values), dtype=bool),
            np.ones(len(values), dtype=bool),
            np.ones(len(values), dtype=bool),
            np.ones(len(values), dtype=bool),
            capacity_valid,
            daily_seen,
            np.ones(len(values), dtype=bool),
            complete_components,
            other_valid,
            weekend_daily_valid,
            weekend_seen,
            notification_ratio_valid,
            daily_seen,
            daily_seen,
            daily_seen,
            daily_seen & ~component_seen[:, 0],
            daily_seen & ~component_seen[:, 1],
            daily_seen & ~component_seen[:, 2],
            np.ones(len(values), dtype=bool),
            any_screen,
            any_screen,
            any_screen,
            any_screen,
            any_screen,
            any_screen,
            daily_seen & component_seen[:, 0],
            daily_seen & component_seen[:, 1],
            daily_seen & component_seen[:, 2],
            weekend_daily_valid,
            daily_sleep_valid,
        ]
    )
    if values_out.shape != valid_out.shape:
        raise RuntimeError("Geometry values and validity masks are misaligned")
    return values_out, valid_out


def geometry_scaler(values: np.ndarray, valid: np.ndarray):
    means = np.zeros(values.shape[1], dtype=np.float32)
    scales = np.ones(values.shape[1], dtype=np.float32)
    for index in range(values.shape[1]):
        selected = values[valid[:, index], index]
        if len(selected):
            means[index] = float(selected.mean())
            scale = float(selected.std())
            scales[index] = scale if scale > 1e-6 else 1.0
    return means, scales


class PeriodicLinear(nn.Module):
    def __init__(self, n_features: int, frequencies: int, width: int):
        super().__init__()
        self.frequency = nn.Parameter(torch.randn(n_features, frequencies) * 0.5)
        self.projection = nn.Parameter(
            torch.randn(n_features, 2 * frequencies, width)
            / math.sqrt(2 * frequencies)
        )
        self.bias = nn.Parameter(torch.zeros(n_features, width))

    def forward(self, values):
        phase = 2.0 * math.pi * values.unsqueeze(-1) * self.frequency.unsqueeze(0)
        periodic = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return torch.einsum("bfk,fkd->bfd", periodic, self.projection) + self.bias


class LowRankCross(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, initial, current):
        return current + initial * self.up(self.down(current))


class BudgetCrossModel(nn.Module):
    """Smooth numeric encoder plus missing-aware constraint geometry and DCNv2."""

    def __init__(
        self,
        category_vocab,
        geometry_mean,
        geometry_scale,
        max_daily,
        numeric_width=16,
        frequencies=16,
        category_width=8,
        pattern_width=16,
        count_width=8,
        hidden_width=256,
        cross_rank=32,
        cross_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.max_daily = float(max_daily)
        self.register_buffer("geometry_mean", torch.as_tensor(geometry_mean))
        self.register_buffer("geometry_scale", torch.as_tensor(geometry_scale))
        self.numeric_encoder = PeriodicLinear(
            len(NUMERIC_COLUMNS), frequencies, numeric_width
        )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(size, category_width) for size in category_vocab]
        )
        self.pattern_embedding = nn.Embedding(1 << len(FEATURE_COLUMNS), pattern_width)
        self.count_embedding = nn.Embedding(len(FEATURE_COLUMNS) + 1, count_width)
        nn.init.normal_(self.pattern_embedding.weight, std=0.02)

        geometry_width = len(geometry_mean)
        input_width = (
            len(NUMERIC_COLUMNS) * numeric_width
            + len(CATEGORICAL_COLUMNS) * category_width
            + pattern_width
            + count_width
            + len(FEATURE_COLUMNS)
            + 2 * geometry_width
        )
        self.project = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_layers = nn.ModuleList(
            [LowRankCross(hidden_width, cross_rank) for _ in range(cross_layers)]
        )
        self.deep = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_width + hidden_width // 2),
            nn.Linear(hidden_width + hidden_width // 2, 1),
        )

    @staticmethod
    def _divide(numerator, denominator, valid):
        safe = torch.where(denominator.abs() > 1e-4, denominator, torch.ones_like(denominator))
        return torch.where(valid, numerator / safe, torch.zeros_like(numerator))

    def geometry(self, physical, observed):
        effective = torch.where(observed, physical, torch.zeros_like(physical))
        daily, daily_seen = effective[:, 1], observed[:, 1]
        components, component_seen = effective[:, 2:5], observed[:, 2:5]
        component_sum = components.sum(dim=1)
        component_count = component_seen.sum(dim=1).float()
        complete_components = component_count == 3
        weekend, weekend_seen = effective[:, 8], observed[:, 8]

        daily_upper = torch.where(
            daily_seen, daily, torch.full_like(daily, self.max_daily)
        )
        daily_width = (daily_upper - component_sum).clamp_min(0.0)
        capacity = torch.where(
            daily_seen, (daily - component_sum).clamp_min(0.0), torch.zeros_like(daily)
        )
        component_ratio = self._divide(component_sum, daily, daily_seen)
        strict_sum = torch.where(
            complete_components, components.sum(dim=1), torch.zeros_like(daily)
        )
        other_valid = daily_seen & complete_components
        other_screen = torch.where(other_valid, daily - strict_sum, torch.zeros_like(daily))

        component_upper = []
        component_width = []
        for index in range(3):
            other_sum = component_sum - components[:, index]
            upper = torch.where(
                daily_seen,
                (daily - other_sum).clamp_min(0.0),
                torch.zeros_like(daily),
            )
            width = torch.where(
                daily_seen & ~component_seen[:, index], upper, torch.zeros_like(daily)
            )
            component_upper.append(upper)
            component_width.append(width)

        screen_indices = torch.as_tensor([1, 2, 3, 4, 8], device=physical.device)
        screen = effective.index_select(1, screen_indices)
        screen_seen = observed.index_select(1, screen_indices)
        screen_count = screen_seen.sum(dim=1).float()
        screen_sum = screen.sum(dim=1)
        denominator = screen_count.clamp_min(1.0)
        screen_mean = screen_sum / denominator
        centered = torch.where(
            screen_seen, screen - screen_mean.unsqueeze(1), torch.zeros_like(screen)
        )
        screen_std = torch.sqrt(centered.square().sum(dim=1) / denominator)
        screen_min = torch.where(
            screen_seen, screen, torch.full_like(screen, float("inf"))
        ).min(dim=1).values
        screen_max = torch.where(
            screen_seen, screen, torch.full_like(screen, float("-inf"))
        ).max(dim=1).values
        any_screen = screen_count > 0
        screen_min = torch.where(any_screen, screen_min, torch.zeros_like(screen_min))
        screen_max = torch.where(any_screen, screen_max, torch.zeros_like(screen_max))

        weekend_daily_valid = weekend_seen & daily_seen
        notification_ratio_valid = observed[:, 6] & observed[:, 7]
        daily_sleep_valid = daily_seen & observed[:, 5]
        geometry = torch.stack(
            [
                component_sum,
                component_count,
                component_sum,
                daily_upper,
                daily_width,
                capacity,
                component_ratio,
                complete_components.float(),
                strict_sum,
                other_screen,
                torch.where(weekend_daily_valid, weekend - daily, torch.zeros_like(daily)),
                torch.where(weekend_seen, weekend - component_sum, torch.zeros_like(daily)),
                self._divide(effective[:, 6], effective[:, 7] + 1.0, notification_ratio_valid),
                *component_upper,
                *component_width,
                screen_count,
                screen_sum,
                screen_mean,
                screen_std,
                screen_min,
                screen_max,
                screen_max - screen_min,
                self._divide(components[:, 0], daily, daily_seen & component_seen[:, 0]),
                self._divide(components[:, 1], daily, daily_seen & component_seen[:, 1]),
                self._divide(components[:, 2], daily, daily_seen & component_seen[:, 2]),
                self._divide(weekend, daily, weekend_daily_valid),
                torch.where(daily_sleep_valid, daily + effective[:, 5], torch.zeros_like(daily)),
            ],
            dim=1,
        )
        valid = torch.stack(
            [
                torch.ones_like(daily_seen),
                torch.ones_like(daily_seen),
                torch.ones_like(daily_seen),
                torch.ones_like(daily_seen),
                torch.ones_like(daily_seen),
                daily_seen,
                daily_seen,
                torch.ones_like(daily_seen),
                complete_components,
                other_valid,
                weekend_daily_valid,
                weekend_seen,
                notification_ratio_valid,
                daily_seen,
                daily_seen,
                daily_seen,
                daily_seen & ~component_seen[:, 0],
                daily_seen & ~component_seen[:, 1],
                daily_seen & ~component_seen[:, 2],
                torch.ones_like(daily_seen),
                any_screen,
                any_screen,
                any_screen,
                any_screen,
                any_screen,
                any_screen,
                daily_seen & component_seen[:, 0],
                daily_seen & component_seen[:, 1],
                daily_seen & component_seen[:, 2],
                weekend_daily_valid,
                daily_sleep_valid,
            ],
            dim=1,
        )
        standardized = (geometry - self.geometry_mean) / self.geometry_scale
        standardized = torch.where(valid, standardized, torch.zeros_like(standardized))
        return standardized, valid.float()

    def forward(self, rank_values, physical, numeric_observed, categories, category_observed, hide=None):
        if hide is not None:
            numeric_observed = numeric_observed & ~hide[:, : len(NUMERIC_COLUMNS)]
            category_observed = category_observed & ~hide[:, len(NUMERIC_COLUMNS) :]
        categories = torch.where(category_observed, categories, torch.zeros_like(categories))
        all_observed = torch.cat([numeric_observed, category_observed], dim=1)

        numeric_tokens = self.numeric_encoder(rank_values) * numeric_observed.unsqueeze(-1)
        category_tokens = torch.cat(
            [
                embedding(categories[:, index])
                for index, embedding in enumerate(self.category_embeddings)
            ],
            dim=1,
        )
        missing = ~all_observed
        bit_weights = (1 << torch.arange(len(FEATURE_COLUMNS), device=missing.device)).long()
        pattern = (missing.long() * bit_weights).sum(dim=1)
        count = missing.sum(dim=1).long()
        geometry, geometry_valid = self.geometry(physical, numeric_observed)
        combined = torch.cat(
            [
                numeric_tokens.flatten(start_dim=1),
                category_tokens,
                self.pattern_embedding(pattern),
                self.count_embedding(count),
                missing.float(),
                geometry,
                geometry_valid,
            ],
            dim=1,
        )
        initial = self.project(combined)
        crossed = initial
        for layer in self.cross_layers:
            crossed = layer(initial, crossed)
        deep = self.deep(initial)
        return self.head(torch.cat([crossed, deep], dim=1)).squeeze(-1)


def _copy_parameters(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def _load_parameters(parameters, saved) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, saved):
            parameter.copy_(value)


def _predict(model, tensors, batch_size, amp_dtype):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            batch = [value[start : start + batch_size] for value in tensors]
            with torch.autocast("cuda", dtype=amp_dtype):
                predictions.append(model(*batch).float().cpu())
    return torch.cat(predictions).numpy().astype(np.float64, copy=False)


def train(args) -> dict:
    config = load_config(args.config)
    train_frame, test_frame, _ = load_competition_data(config, require_sample=False)
    settings = config.get("public_stack", {})
    n_splits = int(settings.get("n_splits", 5))
    fold_seed = int(settings.get("seed", 42))
    if (n_splits, fold_seed) != (5, 42):
        raise ValueError("This candidate must use frozen 5-fold seed-42 OOF")
    if args.fold_limit < 0 or args.fold_limit > n_splits:
        raise ValueError("fold-limit must be in [0, 5]")
    if args.screen_early_stop and args.epochs <= args.warm:
        raise ValueError("epochs must be greater than warm")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to fall back to CPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [train_frame[FEATURE_COLUMNS], test_frame[FEATURE_COLUMNS]], ignore_index=True
    )
    raw_numeric = combined[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric_observed = raw_numeric.notna().to_numpy(bool)
    physical = raw_numeric.fillna(0.0).to_numpy(np.float32)
    rank_values = rank_gauss(raw_numeric)
    categories, category_vocab, category_mappings = encode_categories(combined)
    category_observed = combined[CATEGORICAL_COLUMNS].notna().to_numpy(bool)
    max_daily = float(raw_numeric["daily_screen_time_hours"].max())
    geometry, geometry_valid = budget_geometry_numpy(
        physical, numeric_observed, max_daily
    )
    geometry_mean, geometry_scale = geometry_scaler(geometry, geometry_valid)

    all_tensors = (
        torch.from_numpy(rank_values),
        torch.from_numpy(physical),
        torch.from_numpy(numeric_observed),
        torch.from_numpy(categories),
        torch.from_numpy(category_observed),
    )
    y = train_frame[TARGET].to_numpy(np.float32)
    target_tensor = torch.from_numpy(y)
    n_train, n_test = len(train_frame), len(test_frame)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros(n_train), y
        )
    )
    fold_ids = np.full(n_train, -1, dtype=np.int16)
    for fold_index, (_, valid_index) in enumerate(folds):
        fold_ids[valid_index] = fold_index
    pd.DataFrame(
        {
            ID_COLUMN: train_frame[ID_COLUMN],
            TARGET: train_frame[TARGET],
            "fold": fold_ids,
        }
    ).to_csv(output_dir / "folds.csv", index=False)

    def move_rows(selection):
        return tuple(tensor[selection].to(device) for tensor in all_tensors)

    oof = np.full(n_train, np.nan, dtype=np.float64)
    test_prediction = np.zeros(n_test, dtype=np.float64)
    folds_to_run = args.fold_limit or 5
    test_tensors = move_rows(slice(n_train, None)) if folds_to_run == 5 else None
    fold_metrics = []
    started = time.time()
    print(
        "device={} gpu={} amp={} geometry={} max_daily={:.3f}".format(
            device,
            torch.cuda.get_device_name(device),
            amp_dtype,
            geometry.shape[1],
            max_daily,
        ),
        flush=True,
    )

    for fold_index, (train_index, valid_index) in enumerate(folds[:folds_to_run]):
        _set_seed(args.seed + fold_index)
        model = BudgetCrossModel(
            category_vocab,
            geometry_mean,
            geometry_scale,
            max_daily,
            numeric_width=args.numeric_width,
            frequencies=args.frequencies,
            category_width=args.category_width,
            pattern_width=args.pattern_width,
            count_width=args.count_width,
            hidden_width=args.hidden_width,
            cross_rank=args.cross_rank,
            cross_layers=args.cross_layers,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        steps_per_epoch = math.ceil(len(train_index) / args.batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=steps_per_epoch * args.epochs,
            pct_start=0.15,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        bce = nn.BCEWithLogitsLoss()
        parameters = list(model.parameters())
        ema = _copy_parameters(parameters)
        train_tensors = move_rows(train_index)
        valid_tensors = move_rows(valid_index)
        train_target = target_tensor[train_index].to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + fold_index + 91)
        best_auc = -np.inf
        best_epoch = -1
        best_parameters = None
        stale = 0

        for epoch in range(args.epochs):
            model.train()
            permutation = torch.randperm(
                len(train_index), generator=generator, device=device
            )
            for start in range(0, len(train_index), args.batch_size):
                selection = permutation[start : start + args.batch_size]
                batch = [tensor[selection] for tensor in train_tensors]
                target = train_target[selection]
                hide = None
                if args.missing_aug:
                    observed = torch.cat([batch[2], batch[4]], dim=1)
                    hide = (
                        torch.rand(
                            observed.shape, generator=generator, device=device
                        )
                        < args.missing_aug
                    ) & observed
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(*batch, hide=hide)
                    loss = bce(logits, target)
                    if args.rank_weight:
                        positive = torch.nonzero(target > 0.5, as_tuple=True)[0]
                        negative = torch.nonzero(target < 0.5, as_tuple=True)[0]
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
                            loss = loss + args.rank_weight * nn.functional.softplus(
                                -(logits[positive] - logits[negative])
                            ).mean()
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                with torch.no_grad():
                    torch._foreach_mul_(ema, args.ema_decay)
                    torch._foreach_add_(
                        ema,
                        [parameter.detach() for parameter in parameters],
                        alpha=1.0 - args.ema_decay,
                    )

            # Formal OOF uses the final EMA checkpoint at a pre-registered epoch.
            # Outer-valid early stopping remains an explicit screening-only option.
            if args.screen_early_stop and epoch >= args.warm:
                backup = _copy_parameters(parameters)
                _load_parameters(parameters, ema)
                prediction = _predict(
                    model, valid_tensors, args.predict_batch_size, amp_dtype
                )
                auc = float(roc_auc_score(y[valid_index], prediction))
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    best_parameters = _copy_parameters(parameters)
                    stale = 0
                else:
                    stale += 1
                print(
                    "fold={} epoch={:02d} auc={:.9f} best={:.9f} elapsed={:.0f}s".format(
                        fold_index, epoch, auc, best_auc, time.time() - started
                    ),
                    flush=True,
                )
                _load_parameters(parameters, backup)
                if stale >= args.patience:
                    break

        if best_parameters is None:
            best_parameters = _copy_parameters(ema)
            best_epoch = args.epochs - 1
        _load_parameters(parameters, best_parameters)
        fold_prediction = _predict(
            model, valid_tensors, args.predict_batch_size, amp_dtype
        )
        oof[valid_index] = fold_prediction
        if test_tensors is not None:
            test_prediction += (
                _predict(model, test_tensors, args.predict_batch_size, amp_dtype) / 5.0
            )
        fold_auc = float(roc_auc_score(y[valid_index], fold_prediction))
        fold_metrics.append(
            {
                "fold": fold_index,
                "rows": int(len(valid_index)),
                "auc": fold_auc,
                "best_epoch": best_epoch,
            }
        )
        print(
            "fold={} complete auc={:.9f} best_epoch={} elapsed={:.0f}s".format(
                fold_index, fold_auc, best_epoch, time.time() - started
            ),
            flush=True,
        )
        del model, train_tensors, valid_tensors, train_target
        torch.cuda.empty_cache()

    completed_rows = np.flatnonzero(np.isfinite(oof))
    pooled_auc = float(roc_auc_score(y[completed_rows], oof[completed_rows]))
    complete = folds_to_run == 5
    np.save(output_dir / "oof_{}.npy".format(args.name), oof)
    np.save(output_dir / "completed_rows.npy", completed_rows)
    if complete:
        np.save(output_dir / "test_{}.npy".format(args.name), test_prediction)
        probability_oof = np.exp(-np.logaddexp(0.0, -oof))
        probability_test = np.exp(-np.logaddexp(0.0, -test_prediction))
        pd.DataFrame(
            {ID_COLUMN: train_frame[ID_COLUMN], TARGET: probability_oof}
        ).to_csv(output_dir / "oof_{}.csv".format(args.name), index=False)
        pd.DataFrame(
            {ID_COLUMN: test_frame[ID_COLUMN], TARGET: probability_test}
        ).to_csv(output_dir / "test_{}.csv".format(args.name), index=False)

    payload = {
        "name": args.name,
        "complete": complete,
        "completed_rows": int(len(completed_rows)),
        "pooled_oof_auc": pooled_auc,
        "per_fold": fold_metrics,
        "folds": {"n_splits": 5, "seed": 42},
        "checkpoint_selection": (
            "outer-valid early stopping; screening only"
            if args.screen_early_stop
            else "fixed epochs; final EMA checkpoint"
        ),
        "feature_view": "rank-gauss + missing masks + budget feasible geometry; no exact numeric lookup",
        "preprocessing_fit": "train+test; target-free",
        "geometry_features": int(geometry.shape[1]),
        "category_vocab": category_vocab,
        "category_mappings": category_mappings,
        "device": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "amp": str(amp_dtype),
        },
        "parameters": vars(args),
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(str(output_dir / "metrics.json"), payload)
    print(
        "pooled OOF AUC={:.9f} rows={} output={}".format(
            pooled_auc, len(completed_rows), output_dir
        ),
        flush=True,
    )
    return payload


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", default=str(ROOT / "configs" / "leaderboard_v3_combined.yaml")
    )
    result.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "budget_cross_screen")
    )
    result.add_argument("--name", default="masked_budget_cross")
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--seed", type=int, default=20260812)
    result.add_argument("--fold-limit", type=int, default=2)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--warm", type=int, default=3)
    result.add_argument("--patience", type=int, default=4)
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
    result.add_argument("--learning-rate", type=float, default=0.0015)
    result.add_argument("--weight-decay", type=float, default=0.0002)
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--missing-aug", type=float, default=0.05)
    result.add_argument("--rank-weight", type=float, default=0.2)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--ema-decay", type=float, default=0.995)
    result.add_argument("--numeric-width", type=int, default=16)
    result.add_argument("--frequencies", type=int, default=16)
    result.add_argument("--category-width", type=int, default=8)
    result.add_argument("--pattern-width", type=int, default=16)
    result.add_argument("--count-width", type=int, default=8)
    result.add_argument("--hidden-width", type=int, default=256)
    result.add_argument("--cross-rank", type=int, default=32)
    result.add_argument("--cross-layers", type=int, default=2)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(vars(arguments), indent=2, sort_keys=True), flush=True)
    train(arguments)
