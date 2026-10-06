#!/usr/bin/env python3
"""Precompute the Random-Forest feature-selection G-I-R-S investigation.

The safe default is a complete but deliberately small SMOKE_TEST.  Scientific
outputs, feature-selection evidence, predictions, and provenance are written to
SQLite; fitted sklearn estimators are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sqlite3
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


SMOKE_TEST = "SMOKE_TEST"
FULL_PRECOMPUTE = "FULL_PRECOMPUTE"
SCHEMA_VERSION = 3
RF_SEED = 2024
CLUSTER_SEED = 420
N_JOBS = -1
PERMUTATION_N_JOBS = 1  # avoid nested parallelism
STRATEGIES = ("all", "correlation", "permutation", "correlation_permutation")


def configure_warning_filters() -> None:
    """Prevent a scikit-learn 1.9 false-positive warning flood.

    ``sklearn.utils.parallel`` treats an empty ``warnings.filters`` list as if
    its delayed call were not paired with sklearn's Parallel wrapper.  Nested
    Random-Forest prediction inside permutation importance can then emit one
    warning per delayed task even though this script uses only sklearn's public
    APIs.  Installing an exact default filter gives sklearn a warning context;
    a genuine wrapper mismatch (missing configuration) will still be reported.
    """
    warnings.filterwarnings(
        "default",
        message=(
            r"`sklearn\.utils\.parallel\.delayed` should be used with"
            r" `sklearn\.utils\.parallel\.Parallel`.*"
        ),
        category=UserWarning,
        module=r"sklearn\.utils\.parallel",
    )


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    n_estimators: int
    convergence_estimators: tuple[int, ...]
    convergence_seeds: tuple[int, ...]
    outer_seeds: tuple[int, ...]
    outer_folds: int
    inner_folds: int
    feature_budgets: tuple[float, ...]
    minimum_features: int
    permutation_repeats: int
    correlation_method: str
    representative_budget: float
    robustness_seeds: tuple[int, ...]
    robustness_folds: int
    feature_noise_levels: tuple[float, ...]
    target_noise_levels: tuple[float, ...]
    stability_weight_strengths: tuple[float, ...]
    stability_threshold_strengths: tuple[float, ...]
    stability_weight_repeats: int
    stability_threshold_repeats: int
    stability_threshold_max_depth: int
    run_secondary_sensitivity: bool
    sensitivity_budgets: tuple[float, ...]
    tuning_grid: tuple[tuple[int | None, int, int, float], ...]
    representative_params: tuple[int | None, int, int, float]


SMOKE_CONFIG = ExperimentConfig(
    mode=SMOKE_TEST,
    n_estimators=12,
    convergence_estimators=(4, 8, 12),
    convergence_seeds=(0,),
    outer_seeds=(0,),
    outer_folds=2,
    inner_folds=2,
    feature_budgets=(1.0, 0.5),
    minimum_features=5,
    permutation_repeats=2,
    correlation_method="pearson",
    representative_budget=0.5,
    robustness_seeds=(0,),
    robustness_folds=2,
    feature_noise_levels=(0.0, 0.10),
    target_noise_levels=(0.0, 0.10),
    stability_weight_strengths=(0.0, 0.20),
    stability_threshold_strengths=(0.0, 0.02),
    stability_weight_repeats=3,
    stability_threshold_repeats=2,
    stability_threshold_max_depth=2,
    run_secondary_sensitivity=False,
    sensitivity_budgets=(0.5,),
    tuning_grid=((8, 1, 2, 0.5), (None, 1, 2, 0.5)),
    representative_params=(None, 1, 2, 0.5),
)


FULL_CONFIG = ExperimentConfig(
    mode=FULL_PRECOMPUTE,
    n_estimators=160,
    convergence_estimators=(10, 25, 50, 100, 160, 200),
    convergence_seeds=(0, 1, 2),
    outer_seeds=tuple(range(5)),
    outer_folds=5,
    inner_folds=3,
    feature_budgets=(1.0, 0.75, 0.50, 0.25, 0.10),
    minimum_features=5,
    permutation_repeats=8,
    correlation_method="pearson",
    representative_budget=0.5,
    robustness_seeds=(0, 1, 2),
    robustness_folds=5,
    feature_noise_levels=(0.0, 0.05, 0.10, 0.20),
    target_noise_levels=(0.0, 0.10, 0.20),
    stability_weight_strengths=(0.0, 0.05, 0.10, 0.20, 0.35, 0.50),
    stability_threshold_strengths=(0.0, 0.005, 0.01, 0.02, 0.05),
    stability_weight_repeats=100,
    stability_threshold_repeats=40,
    stability_threshold_max_depth=2,
    run_secondary_sensitivity=True,
    sensitivity_budgets=(0.5,),
    tuning_grid=(
        (8, 1, 2, 0.5), (16, 1, 2, 0.5),
        (16, 2, 4, 0.75), (None, 1, 2, 1.0),
    ),
    representative_params=(16, 1, 2, 0.5),
)


@dataclass
class SelectionEvidence:
    selected: list[str]
    ranking: list[str]
    permutation_rows: list[tuple[int, int, str, float]]
    correlation_rows: list[tuple[int, str, str, float, float, float]]
    inner_assignment_rows: list[tuple[int, int, str]]
    inner_boundary_rows: list[tuple[int, str, str, int, int]]
    diagnostics: dict[str, object]


def find_repository_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "demonstrations" / "random_forest_cv.ipynb").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root")


def identify_target(df: pd.DataFrame) -> str:
    candidates = [
        name for name in df.columns
        if "gap" in name.lower()
        and any(token in name.lower() for token in ("expt", "experiment", "target"))
        and pd.api.types.is_numeric_dtype(df[name])
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one experimental gap target; found {candidates}")
    return candidates[0]


def load_dataset(path: Path):
    df = pd.read_csv(path)
    if df.columns[0] != "composition":
        raise ValueError("The first column must be the composition metadata field")
    target = identify_target(df)
    excluded = [
        ("composition", "metadata", "identifier/domain context; never a model input"),
        (target, "target", "experimental response being predicted"),
    ]
    numeric = [
        c for c in df.columns
        if c not in {"composition", target} and pd.api.types.is_numeric_dtype(df[c])
    ]
    for name in df.columns:
        if name not in {"composition", target} and name not in numeric:
            excluded.append((name, "nonnumeric", "not a raw numeric model feature"))
    missing = int(df[numeric].isna().sum().sum())
    if missing:
        affected = [name for name in numeric if df[name].isna().any()]
        raise ValueError(
            "Missing numeric feature values in " + repr(affected) +
            ". Global imputation is forbidden; add training-fold-only imputation first."
        )
    constant = [name for name in numeric if df[name].nunique(dropna=True) <= 1]
    for name in constant:
        excluded.append((name, "unusable", "constant numeric column"))
    features = [name for name in numeric if name not in set(constant)]
    if df[target].isna().any():
        raise ValueError(f"Target {target!r} contains missing values")
    X = df[features].astype(float).copy()
    y = df[target].astype(float).copy()
    compositions = df["composition"].astype(str).copy()
    assert "composition" not in X and target not in X
    return df, compositions, X, y, target, excluded, missing


def metrics(y_true, y_pred) -> tuple[float, float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred) ** 0.5),
    )


def depth_to_db(value: int | None) -> int:
    return -1 if value is None else int(value)


def stable_hash(values: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in sorted(values))
    return hashlib.sha256(payload.encode()).hexdigest()


def budget_count(n_features: int, fraction: float, minimum: int) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("Feature budgets must be in (0, 1]")
    return min(n_features, max(minimum, int(np.ceil(n_features * fraction))))


def representation_grid(config: ExperimentConfig) -> list[tuple[str, float]]:
    return [("all", 1.0)] + [
        (strategy, budget)
        for strategy in STRATEGIES[1:]
        for budget in config.feature_budgets
    ]


def representative_grid(config: ExperimentConfig) -> list[tuple[str, float]]:
    """The focused grid used for costly robustness and model-stability analyses."""
    return [("all", 1.0)] + [
        (strategy, config.representative_budget) for strategy in STRATEGIES[1:]
    ]


def validate_params(params: tuple[int | None, int, int, float]) -> None:
    depth, leaf, split, max_features = params
    if depth is not None and depth < 1:
        raise ValueError("max_depth must be positive or None")
    if leaf < 1 or split < 2 or not 0 < max_features <= 1:
        raise ValueError(f"Invalid Random Forest parameters: {params}")


def validate_config(config: ExperimentConfig) -> None:
    """Fail early when a downstream paired-control assumption is not available."""
    if config.outer_folds != config.robustness_folds:
        raise ValueError(
            "robustness_folds must equal outer_folds so clean representations can be paired"
        )
    if not set(config.robustness_seeds) <= set(config.outer_seeds):
        raise ValueError("robustness_seeds must be a subset of outer_seeds")
    if 0.0 not in config.feature_noise_levels or 0.0 not in config.target_noise_levels:
        raise ValueError("Both robustness noise grids must include a zero-noise baseline")
    if 1.0 not in config.feature_budgets:
        raise ValueError("feature_budgets must include 1.0 for the paired all-feature control")
    if config.representative_budget not in config.feature_budgets:
        raise ValueError("representative_budget must be present in feature_budgets")


def make_forest(
    config: ExperimentConfig,
    params=None,
    *,
    n_estimators: int | None = None,
) -> RandomForestRegressor:
    params = config.representative_params if params is None else params
    validate_params(params)
    depth, leaf, split, max_features = params
    return RandomForestRegressor(
        n_estimators=config.n_estimators if n_estimators is None else n_estimators,
        max_depth=depth,
        min_samples_leaf=leaf,
        min_samples_split=split,
        max_features=max_features,
        random_state=RF_SEED,
        n_jobs=N_JOBS,
    )


def build_split_cache(X: pd.DataFrame):
    # As in the reference workflow, feature-space LOCO is a pedagogical stress
    # test: its unsupervised clusters use the overall X distribution (never y).
    scaled = StandardScaler().fit_transform(X)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def get_splits(method: str, count: int, seed: int):
        if method == "kfold":
            result = list(KFold(count, shuffle=True, random_state=seed).split(X))
        elif method == "loco":
            key = (count, seed)
            if key not in cache:
                cache[key] = KMeans(
                    n_clusters=count, random_state=CLUSTER_SEED + seed, n_init=10
                ).fit_predict(scaled)
            labels = cache[key]
            result = [
                (np.flatnonzero(labels != cluster), np.flatnonzero(labels == cluster))
                for cluster in range(count)
            ]
        else:
            raise ValueError(f"Unknown CV method: {method}")
        coverage = np.zeros(len(X), dtype=int)
        for train, validation in result:
            if np.intersect1d(train, validation).size:
                raise RuntimeError("Outer training and validation indices overlap")
            coverage[validation] += 1
        if len(result) != count or not np.all(coverage == 1):
            raise RuntimeError("Every observation must appear in exactly one outer fold")
        return result

    return get_splits


def assert_selection_boundary(
    permitted_ids: Sequence[int], forbidden_ids: Sequence[int], observed_ids: Sequence[int]
) -> None:
    permitted, forbidden, observed = map(
        lambda x: set(map(int, x)), (permitted_ids, forbidden_ids, observed_ids)
    )
    if permitted & forbidden:
        raise RuntimeError("Outer train/test boundary is invalid")
    if not observed <= permitted or observed & forbidden:
        raise RuntimeError("Outer-validation observations entered feature selection")


def correlation_pruning_path(
    X_train: pd.DataFrame, *, target_count: int, method: str
) -> tuple[list[str], list[tuple[int, str, str, float, float, float]]]:
    """Remove globally redundant predictors, with name-based deterministic ties."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'")
    retained = list(X_train.columns)
    evidence: list[tuple[int, str, str, float, float, float]] = []
    corr_frame = X_train.corr(method=method).abs().fillna(0.0)
    corr_values = corr_frame.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 0.0)
    corr = pd.DataFrame(
        corr_values, index=corr_frame.index, columns=corr_frame.columns
    )
    step = 0
    while len(retained) > target_count:
        current = corr.loc[retained, retained]
        pairs = []
        for left_index, left in enumerate(retained):
            for right in retained[left_index + 1:]:
                pairs.append((float(current.loc[left, right]), min(left, right), max(left, right)))
        pair_corr, first, second = max(pairs, key=lambda row: (row[0], row[1], row[2]))
        redundancy = current.sum(axis=1) / max(1, len(retained) - 1)
        first_r, second_r = float(redundancy[first]), float(redundancy[second])
        if first_r > second_r:
            removed, survivor = first, second
        elif second_r > first_r:
            removed, survivor = second, first
        else:
            # Column order is never the scientific rule: lexical feature name is.
            removed, survivor = max(first, second), min(first, second)
        evidence.append((step, removed, survivor, pair_corr,
                         float(redundancy[removed]), float(redundancy[survivor])))
        retained.remove(removed)
        step += 1
    return retained, evidence


def inner_permutation_ranking(
    config: ExperimentConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: Sequence[str],
    inner_seed: int,
) -> tuple[
    list[str],
    list[tuple[int, int, str, float]],
    list[tuple[int, int, str]],
    list[tuple[int, str, str, int, int]],
]:
    """Rank with signed importance from held-out inner-validation observations."""
    candidate_features = list(candidate_features)
    splitter = KFold(config.inner_folds, shuffle=True, random_state=inner_seed)
    rows: list[tuple[int, int, str, float]] = []
    assignment_rows: list[tuple[int, int, str]] = []
    boundary_rows: list[tuple[int, str, str, int, int]] = []
    for inner_fold, (development, validation) in enumerate(splitter.split(X_train)):
        if np.intersect1d(development, validation).size:
            raise RuntimeError("Inner training and validation overlap")
        development_ids = X_train.index.to_numpy()[development]
        validation_ids = X_train.index.to_numpy()[validation]
        assignment_rows.extend(
            (int(observation_id), inner_fold, "development")
            for observation_id in development_ids
        )
        assignment_rows.extend(
            (int(observation_id), inner_fold, "validation")
            for observation_id in validation_ids
        )
        boundary_rows.append((
            inner_fold, stable_hash(development_ids), stable_hash(validation_ids),
            len(development_ids), len(validation_ids),
        ))
        forest = make_forest(config)
        forest.fit(X_train.iloc[development][candidate_features], y_train.iloc[development])
        result = permutation_importance(
            forest,
            X_train.iloc[validation][candidate_features],
            y_train.iloc[validation],
            scoring="neg_root_mean_squared_error",
            n_repeats=config.permutation_repeats,
            random_state=inner_seed + 1000 * inner_fold,
            n_jobs=PERMUTATION_N_JOBS,
        )
        for feature_index, feature in enumerate(candidate_features):
            for repeat, value in enumerate(result.importances[feature_index]):
                rows.append((inner_fold, repeat, feature, float(value)))
    mean_importance = {
        feature: float(np.mean([row[3] for row in rows if row[2] == feature]))
        for feature in candidate_features
    }
    ranking = sorted(candidate_features, key=lambda name: (-mean_importance[name], name))
    return ranking, rows, assignment_rows, boundary_rows


def select_features(
    config: ExperimentConfig,
    strategy: str,
    budget: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    outer_train_ids: Sequence[int],
    outer_validation_ids: Sequence[int],
    inner_seed: int,
) -> SelectionEvidence:
    assert_selection_boundary(outer_train_ids, outer_validation_ids, X_train.index)
    if set(X_train.index) != set(map(int, outer_train_ids)):
        raise RuntimeError("Selection did not receive exactly the outer-training rows")
    n_features = X_train.shape[1]
    requested = budget_count(n_features, budget, config.minimum_features)
    permutation_rows: list[tuple[int, int, str, float]] = []
    correlation_rows: list[tuple[int, str, str, float, float, float]] = []
    inner_assignment_rows: list[tuple[int, int, str]] = []
    inner_boundary_rows: list[tuple[int, str, str, int, int]] = []
    if strategy == "all":
        selected = ranking = sorted(X_train.columns)
    elif strategy == "correlation":
        selected, correlation_rows = correlation_pruning_path(
            X_train, target_count=requested, method=config.correlation_method
        )
        ranking = selected + [row[1] for row in reversed(correlation_rows)]
    elif strategy == "permutation":
        (
            ranking, permutation_rows, inner_assignment_rows, inner_boundary_rows
        ) = inner_permutation_ranking(
            config, X_train, y_train, X_train.columns, inner_seed
        )
        selected = ranking[:requested]
    elif strategy == "correlation_permutation":
        # Leave room for both stages: correlation removes half the distance from
        # the full representation to the requested final budget.
        pool_count = requested + int(np.ceil((n_features - requested) / 2))
        pool, correlation_rows = correlation_pruning_path(
            X_train, target_count=pool_count, method=config.correlation_method
        )
        (
            ranking, permutation_rows, inner_assignment_rows, inner_boundary_rows
        ) = inner_permutation_ranking(
            config, X_train, y_train, pool, inner_seed
        )
        selected = ranking[:requested]
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    # Forest random-feature draws operate on column positions.  A canonical
    # source-data order prevents feature rank itself from becoming an accidental
    # fifth experimental variable.  Scientific rank is retained separately.
    selected_set = set(selected)
    selected = [name for name in X_train.columns if name in selected_set]
    if len(selected) != requested or len(set(selected)) != len(selected):
        raise RuntimeError("Feature selection returned an invalid feature set")
    if any(name not in X_train.columns for name in selected):
        raise RuntimeError("Feature selection invented an unknown feature")
    diagnostics = {
        "requested_fraction": budget,
        "requested_count": requested,
        "actual_count": len(selected),
        "candidate_count": n_features,
        "correlation_method": config.correlation_method,
        "correlation_rule": "budget-driven iterative global-redundancy pruning",
        "inner_seed": inner_seed,
        "outer_boundary_assertion": "passed",
        "negative_permutation_values": sum(row[3] < 0 for row in permutation_rows),
    }
    return SelectionEvidence(
        list(selected), list(ranking), permutation_rows, correlation_rows,
        inner_assignment_rows, inner_boundary_rows, diagnostics
    )


SCHEMA_SQL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stage_status (
    stage TEXT PRIMARY KEY, config_hash TEXT NOT NULL, completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id INTEGER PRIMARY KEY, composition TEXT NOT NULL, observed_target REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS features (
    feature_index INTEGER PRIMARY KEY, feature_name TEXT UNIQUE NOT NULL,
    missing_count INTEGER NOT NULL, minimum REAL NOT NULL, maximum REAL NOT NULL,
    mean REAL NOT NULL, standard_deviation REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS excluded_columns (
    column_name TEXT PRIMARY KEY, role TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cluster_assignments (
    cv_seed INTEGER NOT NULL, cluster_count INTEGER NOT NULL,
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
    cluster_id INTEGER NOT NULL, pc1 REAL NOT NULL, pc2 REAL NOT NULL,
    PRIMARY KEY (cv_seed, cluster_count, observation_id)
);
CREATE TABLE IF NOT EXISTS convergence_results (
    cv_seed INTEGER NOT NULL, rf_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    n_estimators INTEGER NOT NULL, r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    PRIMARY KEY (cv_seed, n_estimators)
);
CREATE TABLE IF NOT EXISTS outer_fold_assignments (
    cv_method TEXT NOT NULL, cv_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    observation_id INTEGER NOT NULL, outer_fold INTEGER NOT NULL,
    PRIMARY KEY (cv_method, cv_seed, fold_count, observation_id)
);
CREATE TABLE IF NOT EXISTS selection_runs (
    selection_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow TEXT NOT NULL, cv_method TEXT NOT NULL, cv_seed INTEGER NOT NULL,
    fold_count INTEGER NOT NULL, outer_fold INTEGER NOT NULL, strategy TEXT NOT NULL,
    requested_fraction REAL NOT NULL, requested_count INTEGER NOT NULL,
    actual_count INTEGER NOT NULL, development_hash TEXT NOT NULL,
    outer_validation_hash TEXT NOT NULL, inner_seed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inner_fold_boundaries (
    selection_run_id INTEGER NOT NULL REFERENCES selection_runs(selection_run_id),
    inner_fold INTEGER NOT NULL, development_hash TEXT NOT NULL,
    validation_hash TEXT NOT NULL, development_count INTEGER NOT NULL,
    validation_count INTEGER NOT NULL,
    PRIMARY KEY (selection_run_id, inner_fold)
);
CREATE TABLE IF NOT EXISTS inner_fold_assignments (
    selection_run_id INTEGER NOT NULL REFERENCES selection_runs(selection_run_id),
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
    inner_fold INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('development', 'validation')),
    PRIMARY KEY (selection_run_id, inner_fold, observation_id)
);
CREATE TABLE IF NOT EXISTS selected_features (
    selection_run_id INTEGER NOT NULL REFERENCES selection_runs(selection_run_id),
    feature_name TEXT NOT NULL, selection_rank INTEGER NOT NULL,
    model_position INTEGER NOT NULL,
    PRIMARY KEY (selection_run_id, feature_name)
);
CREATE TABLE IF NOT EXISTS feature_selection_diagnostics (
    selection_run_id INTEGER NOT NULL REFERENCES selection_runs(selection_run_id),
    diagnostic_name TEXT NOT NULL, diagnostic_value TEXT NOT NULL,
    PRIMARY KEY (selection_run_id, diagnostic_name)
);
CREATE TABLE IF NOT EXISTS permutation_importance_evidence (
    selection_run_id INTEGER NOT NULL REFERENCES selection_runs(selection_run_id),
    inner_fold INTEGER NOT NULL, permutation_repeat INTEGER NOT NULL,
    feature_name TEXT NOT NULL, signed_importance REAL NOT NULL,
    PRIMARY KEY (selection_run_id, inner_fold, permutation_repeat, feature_name)
);
CREATE TABLE IF NOT EXISTS correlation_pruning_evidence (
    selection_run_id INTEGER NOT NULL REFERENCES selection_runs(selection_run_id),
    pruning_step INTEGER NOT NULL, removed_feature TEXT NOT NULL,
    surviving_partner TEXT NOT NULL, absolute_pair_correlation REAL NOT NULL,
    removed_global_redundancy REAL NOT NULL, survivor_global_redundancy REAL NOT NULL,
    correlation_method TEXT NOT NULL,
    PRIMARY KEY (selection_run_id, pruning_step)
);
CREATE TABLE IF NOT EXISTS outer_experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    selection_run_id INTEGER NOT NULL REFERENCES selection_runs(selection_run_id),
    workflow TEXT NOT NULL, cv_method TEXT NOT NULL, cv_seed INTEGER NOT NULL,
    fold_count INTEGER NOT NULL, outer_fold INTEGER NOT NULL, strategy TEXT NOT NULL,
    requested_fraction REAL NOT NULL, requested_count INTEGER NOT NULL,
    actual_count INTEGER NOT NULL, rf_seed INTEGER NOT NULL,
    n_estimators INTEGER NOT NULL, max_depth INTEGER NOT NULL,
    min_samples_leaf INTEGER NOT NULL, min_samples_split INTEGER NOT NULL,
    max_features REAL NOT NULL,
    UNIQUE(workflow, cv_method, cv_seed, fold_count, outer_fold, strategy, requested_fraction)
);
CREATE TABLE IF NOT EXISTS generalizability_results (
    experiment_id INTEGER PRIMARY KEY REFERENCES outer_experiments(experiment_id),
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS generalizability_predictions (
    experiment_id INTEGER NOT NULL REFERENCES outer_experiments(experiment_id),
    observation_id INTEGER NOT NULL, outer_fold INTEGER NOT NULL,
    y_true REAL NOT NULL, y_pred REAL NOT NULL,
    PRIMARY KEY (experiment_id, observation_id)
);
CREATE TABLE IF NOT EXISTS generalizability_summary (
    workflow TEXT NOT NULL, cv_method TEXT NOT NULL, cv_seed INTEGER NOT NULL,
    fold_count INTEGER NOT NULL, strategy TEXT NOT NULL,
    requested_fraction REAL NOT NULL, requested_count INTEGER NOT NULL,
    actual_count INTEGER NOT NULL, n_observations INTEGER NOT NULL,
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    PRIMARY KEY (
        workflow, cv_method, cv_seed, fold_count, strategy, requested_fraction
    )
);
CREATE TABLE IF NOT EXISTS generalizability_material_summary (
    workflow TEXT NOT NULL, cv_method TEXT NOT NULL, strategy TEXT NOT NULL,
    requested_fraction REAL NOT NULL,
    observation_id INTEGER NOT NULL REFERENCES observations(observation_id),
    prediction_count INTEGER NOT NULL, mean_prediction REAL NOT NULL,
    prediction_standard_deviation REAL NOT NULL, mean_signed_error REAL NOT NULL,
    mean_absolute_error REAL NOT NULL, rmse REAL NOT NULL,
    maximum_absolute_error REAL NOT NULL,
    PRIMARY KEY (
        workflow, cv_method, strategy, requested_fraction, observation_id
    )
);
CREATE TABLE IF NOT EXISTS posthoc_permutation_evidence (
    experiment_id INTEGER NOT NULL REFERENCES outer_experiments(experiment_id),
    permutation_repeat INTEGER NOT NULL, feature_name TEXT NOT NULL,
    signed_importance REAL NOT NULL,
    PRIMARY KEY (experiment_id, permutation_repeat, feature_name)
);
CREATE TABLE IF NOT EXISTS robustness_results (
    robustness_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cv_method TEXT NOT NULL, cv_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    outer_fold INTEGER NOT NULL, strategy TEXT NOT NULL, requested_fraction REAL NOT NULL,
    process TEXT NOT NULL, feature_noise REAL NOT NULL, target_noise REAL NOT NULL,
    perturbation_seed INTEGER NOT NULL, actual_count INTEGER NOT NULL,
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    UNIQUE(cv_method, cv_seed, outer_fold, strategy, requested_fraction, process,
           feature_noise, target_noise)
);
CREATE TABLE IF NOT EXISTS robustness_predictions (
    robustness_id INTEGER NOT NULL REFERENCES robustness_results(robustness_id),
    observation_id INTEGER NOT NULL, y_true REAL NOT NULL, y_pred REAL NOT NULL,
    PRIMARY KEY (robustness_id, observation_id)
);
CREATE TABLE IF NOT EXISTS robustness_importances (
    robustness_id INTEGER NOT NULL REFERENCES robustness_results(robustness_id),
    feature_name TEXT NOT NULL, importance REAL NOT NULL,
    PRIMARY KEY (robustness_id, feature_name)
);
CREATE TABLE IF NOT EXISTS robustness_selected_features (
    robustness_id INTEGER NOT NULL REFERENCES robustness_results(robustness_id),
    feature_name TEXT NOT NULL, PRIMARY KEY (robustness_id, feature_name)
);
CREATE TABLE IF NOT EXISTS stability_selection_results (
    comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cv_method TEXT NOT NULL, strategy TEXT NOT NULL, requested_fraction REAL NOT NULL,
    selection_run_a INTEGER NOT NULL, selection_run_b INTEGER NOT NULL,
    intersection_count INTEGER NOT NULL, union_count INTEGER NOT NULL,
    jaccard REAL NOT NULL, inclusion_changes INTEGER NOT NULL,
    rank_spearman REAL
);
CREATE TABLE IF NOT EXISTS stability_prediction_results (
    stability_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cv_method TEXT NOT NULL, cv_seed INTEGER NOT NULL, outer_fold INTEGER NOT NULL,
    strategy TEXT NOT NULL, requested_fraction REAL NOT NULL,
    mechanism TEXT NOT NULL, magnitude REAL NOT NULL, realization_count INTEGER NOT NULL,
    baseline_rmse REAL NOT NULL, median_rmse REAL NOT NULL, rmse_movement REAL NOT NULL,
    median_absolute_prediction_movement REAL NOT NULL,
    p95_absolute_prediction_movement REAL NOT NULL,
    median_effective_tree_count REAL, mean_routing_change REAL,
    UNIQUE(cv_method, cv_seed, outer_fold, strategy, requested_fraction, mechanism, magnitude)
);
CREATE TABLE IF NOT EXISTS stability_predictions (
    stability_id INTEGER NOT NULL REFERENCES stability_prediction_results(stability_id),
    observation_id INTEGER NOT NULL, baseline_prediction REAL NOT NULL,
    median_prediction REAL NOT NULL, median_absolute_movement REAL NOT NULL,
    p95_absolute_movement REAL NOT NULL, mean_routing_change REAL,
    prediction_standard_deviation REAL NOT NULL,
    prediction_lower_2_5 REAL NOT NULL, prediction_upper_97_5 REAL NOT NULL,
    prediction_interval_width REAL NOT NULL,
    PRIMARY KEY (stability_id, observation_id)
);
CREATE TABLE IF NOT EXISTS stability_realizations (
    stability_id INTEGER NOT NULL REFERENCES stability_prediction_results(stability_id),
    realization INTEGER NOT NULL, r2 REAL NOT NULL, mae REAL NOT NULL,
    rmse REAL NOT NULL, effective_tree_count REAL, mean_routing_change REAL,
    PRIMARY KEY (stability_id, realization)
);
CREATE INDEX IF NOT EXISTS idx_outer_comparison ON outer_experiments
    (workflow, cv_method, cv_seed, strategy, requested_fraction);
CREATE INDEX IF NOT EXISTS idx_generalization_observation ON generalizability_predictions
    (observation_id, experiment_id);
CREATE INDEX IF NOT EXISTS idx_material_summary_error ON generalizability_material_summary
    (workflow, cv_method, strategy, requested_fraction, mean_absolute_error);
CREATE INDEX IF NOT EXISTS idx_selected_feature ON selected_features
    (feature_name, selection_run_id);
"""


def config_json(config: ExperimentConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def config_hash(config: ExperimentConfig, dataset_sha256: str) -> str:
    payload = (
        f"schema={SCHEMA_VERSION};dataset={dataset_sha256};config={config_json(config)};"
        f"python={sys.version.split()[0]};numpy={np.__version__};"
        f"pandas={pd.__version__};sklearn={sklearn.__version__}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def stage_complete(connection: sqlite3.Connection, stage: str, digest: str) -> bool:
    row = connection.execute(
        "SELECT config_hash FROM stage_status WHERE stage=?", (stage,)
    ).fetchone()
    return row is not None and row[0] == digest


def mark_stage(connection: sqlite3.Connection, stage: str, digest: str) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO stage_status VALUES (?, ?, ?)",
        (stage, digest, datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()


def estimate_fit_count(config: ExperimentConfig) -> dict[str, int]:
    methods = 2
    ensemble_convergence = (
        len(config.convergence_seeds)
        * len(config.convergence_estimators)
        * config.outer_folds
    )
    outer_split_count = methods * len(config.outer_seeds) * config.outer_folds
    representations = len(representation_grid(config))
    representative_representations = len(representative_grid(config))
    supervised_selection = 2 * len(config.feature_budgets) * config.inner_folds
    primary_selection = outer_split_count * supervised_selection
    primary_final = outer_split_count * representations
    robustness_conditions = (
        len(config.feature_noise_levels) * len(config.target_noise_levels)
    )
    robust_splits = methods * len(config.robustness_seeds) * config.robustness_folds
    robustness_final = robust_splits * representative_representations * (
        2 * robustness_conditions - 1  # shared literal zero-noise baseline
    )
    # Frozen selection is computed once; rerun selection is computed per condition.
    representative_supervised_selection = 2 * config.inner_folds
    # Clean representations are reused from primary generalizability; only
    # nonzero-noise reruns require new supervised-selection fits.
    robustness_selection = robust_splits * (
        (robustness_conditions - 1) * representative_supervised_selection
    )
    stability_final = methods * config.outer_folds * representative_representations
    secondary = 0
    if config.run_secondary_sensitivity:
        sensitivity_representations = 1 + 3 * len(config.sensitivity_budgets)
        supervised_representations = 2 * len(config.sensitivity_budgets)
        tuning_model_fits = (
            sensitivity_representations
            * len(config.tuning_grid)
            * config.inner_folds
        )
        tuning_selection_fits = (
            supervised_representations * config.inner_folds * config.inner_folds
        )
        final_selection_fits = supervised_representations * config.inner_folds
        secondary = outer_split_count * (
            tuning_model_fits + tuning_selection_fits
            + sensitivity_representations + final_selection_fits
        )
    counts = {
        "ensemble_convergence": ensemble_convergence,
        "primary_selection_inner": primary_selection,
        "primary_final_models": primary_final,
        "robustness_selection_inner": robustness_selection,
        "robustness_final_models": robustness_final,
        "stability_fitted_models": stability_final,
        "secondary_nested_tuning": secondary,
    }
    counts["total"] = sum(counts.values())
    return counts


def initialize_database(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    dataset_sha256: str,
    dataset_path: Path,
    df: pd.DataFrame,
    compositions: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    excluded: list[tuple[str, str, str]],
    missing: int,
    overwrite: bool,
) -> None:
    if overwrite:
        # Recreate rather than merely emptying tables so --overwrite also handles
        # deliberate schema-version changes.
        existing_tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        drop_order = (
            "stability_realizations", "stability_predictions",
            "stability_prediction_results",
            "stability_selection_results", "robustness_selected_features",
            "robustness_importances",
            "robustness_predictions", "robustness_results",
            "posthoc_permutation_evidence", "generalizability_material_summary",
            "generalizability_summary", "generalizability_predictions",
            "generalizability_results", "outer_experiments", "convergence_results",
            "correlation_pruning_evidence", "permutation_importance_evidence",
            "feature_selection_diagnostics", "selected_features",
            "inner_fold_assignments", "inner_fold_boundaries", "selection_runs",
            "outer_fold_assignments", "cluster_assignments", "excluded_columns",
            "features", "observations",
            "stage_status", "metadata",
        )
        for table in drop_order:
            if table in existing_tables:
                connection.execute(f"DROP TABLE {table}")
        connection.commit()
    connection.executescript(SCHEMA_SQL)
    existing = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }
    if existing and existing.get("config_hash") != digest and not overwrite:
        raise RuntimeError(
            "Existing database has incompatible provenance. Use --overwrite or a "
            "different --database; results from different designs will not be mixed."
        )
    if overwrite:
        # Child tables precede parents so this remains valid with foreign keys on.
        tables = (
            "stability_realizations", "stability_predictions",
            "stability_prediction_results",
            "stability_selection_results", "robustness_selected_features",
            "robustness_importances",
            "robustness_predictions", "robustness_results",
            "posthoc_permutation_evidence", "generalizability_material_summary",
            "generalizability_summary", "generalizability_predictions",
            "generalizability_results", "outer_experiments", "convergence_results",
            "correlation_pruning_evidence", "permutation_importance_evidence",
            "feature_selection_diagnostics", "selected_features",
            "inner_fold_assignments", "inner_fold_boundaries", "selection_runs",
            "outer_fold_assignments", "cluster_assignments", "excluded_columns",
            "features", "observations",
            "stage_status", "metadata",
        )
        for table in tables:
            connection.execute(f"DELETE FROM {table}")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "mode": config.mode,
        "config_hash": digest,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "target_column": target,
        "metadata_columns": ["composition"],
        "feature_columns": list(X.columns),
        "feature_count": X.shape[1],
        "missing_numeric_values": missing,
        "missing_value_policy": (
            "fail fast; any future imputer must be fitted inside the relevant training fold"
        ),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "generated_utc": existing.get(
            "generated_utc", datetime.now(timezone.utc).isoformat()
        ),
        "rf_seed": RF_SEED,
        "cluster_seed": CLUSTER_SEED,
        "convergence_estimators": list(config.convergence_estimators),
        "convergence_seeds": list(config.convergence_seeds),
        "config": asdict(config),
        "fit_estimate": estimate_fit_count(config),
        "primary_question": (
            "With one fixed forest configuration, what does changing the feature "
            "representation do?"
        ),
        "secondary_question": (
            "If each representation receives outer-train-only hyperparameter tuning, "
            "do the conclusions change?"
        ),
        "loco_warning": (
            "Feature-space LOCO is a pedagogical OOD-like stress test. Cluster "
            "construction uses the overall feature distribution, but never targets."
        ),
        "inference_warning": (
            "CV folds and repeated partitions overlap and are not independent. "
            "Stored distributions and paired differences are descriptive, not naive p-values."
        ),
        "selection_rule": (
            "All data-dependent feature decisions use outer-training rows only; supervised "
            "permutation evidence is evaluated on inner-held-out rows."
        ),
        "permutation_scoring": "neg_root_mean_squared_error; signed values preserved",
        "combined_prefilter_rule": (
            "correlation pool count = requested + ceil((all-requested)/2), followed by "
            "inner-held-out permutation ranking"
        ),
        "noise_pairing_rule": (
            "one standardized training-noise direction per CV method/seed/fold is reused "
            "across strategies and every crossed feature-noise x target-noise magnitude; "
            "outer validation X and y remain clean"
        ),
        "stability_pairing_rule": (
            "tree-weight and threshold RNG seeds depend on split/mechanism, not representation"
        ),
        "posthoc_separation": (
            "posthoc_permutation_evidence uses outer validation only after model fitting and "
            "is never used in selected_features"
        ),
        "experiment_definitions": {
            "convergence": (
                "all-feature shuffled-K-fold pooled OOF scores versus n_estimators"
            ),
            "clusters": (
                "global-X-only PCA display coordinates and LOCO assignments for every "
                "seed/fold-count used by the cached experiments"
            ),
            "generalizability_summary": (
                "pooled OOF scores by workflow, validation strategy, partition seed, "
                "feature-selection strategy, and requested budget"
            ),
            "generalizability_material_summary": (
                "material-level recurring prediction/error summaries across CV seeds"
            ),
            "robustness": (
                "full Cartesian feature-noise x target-noise grid with paired directions"
            ),
            "stability": (
                "realization-level performance plus material-level central 95 percent "
                "empirical perturbation intervals"
            ),
        },
        "n_jobs": N_JOBS,
    }
    connection.executemany(
        "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in metadata.items()],
    )
    connection.executemany(
        "INSERT OR REPLACE INTO observations VALUES (?, ?, ?)",
        [(i, compositions.iloc[i], float(y.iloc[i])) for i in range(len(y))],
    )
    feature_rows = []
    for index, name in enumerate(X.columns):
        values = X[name]
        feature_rows.append((
            index, name, int(values.isna().sum()), float(values.min()),
            float(values.max()), float(values.mean()), float(values.std(ddof=0)),
        ))
    connection.executemany(
        "INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?, ?, ?, ?)", feature_rows
    )
    connection.executemany(
        "INSERT OR REPLACE INTO excluded_columns VALUES (?, ?, ?)", excluded
    )
    connection.commit()


def store_fold_assignments(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    get_splits,
) -> None:
    stage = "fold_assignments"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    rows = []
    specifications = {
        (method, seed, count)
        for method in ("kfold", "loco")
        for seed, count in itertools.chain(
            ((seed, config.outer_folds) for seed in config.outer_seeds),
            ((seed, config.robustness_folds) for seed in config.robustness_seeds),
        )
    }
    for method, seed, count in sorted(specifications):
        for fold, (_train, validation) in enumerate(get_splits(method, count, seed)):
            rows.extend((method, seed, count, int(i), fold) for i in validation)
    with connection:
        connection.execute("DELETE FROM outer_fold_assignments")
        connection.executemany(
            "INSERT INTO outer_fold_assignments VALUES (?, ?, ?, ?, ?)", rows
        )
    mark_stage(connection, stage, digest)


def store_cluster_assignments(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    X: pd.DataFrame,
    get_splits,
) -> None:
    """Cache the exact LOCO labels together with one global-X PCA display map.

    PCA and KMeans use predictors only, never the target.  As in the reference
    notebook, this makes LOCO an explicitly labelled pedagogical stress test,
    not a claim that cluster construction simulates a deploy-time boundary.
    """
    stage = "cluster_assignments"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    scaled = StandardScaler().fit_transform(X)
    projection = PCA(n_components=2, random_state=CLUSTER_SEED).fit_transform(scaled)
    specifications = {
        (seed, count)
        for seed, count in itertools.chain(
            ((seed, config.outer_folds) for seed in config.outer_seeds),
            ((seed, config.robustness_folds) for seed in config.robustness_seeds),
        )
    }
    rows = []
    for seed, count in sorted(specifications):
        labels = np.full(len(X), -1, dtype=int)
        for cluster, (_train, validation) in enumerate(get_splits("loco", count, seed)):
            labels[validation] = cluster
        if np.any(labels < 0) or len(np.unique(labels)) != count:
            raise RuntimeError("LOCO cluster assignment coverage is incomplete")
        rows.extend(
            (
                seed, count, observation_id, int(labels[observation_id]),
                float(projection[observation_id, 0]),
                float(projection[observation_id, 1]),
            )
            for observation_id in range(len(X))
        )
    with connection:
        connection.execute("DELETE FROM cluster_assignments")
        connection.executemany(
            "INSERT INTO cluster_assignments VALUES (?, ?, ?, ?, ?, ?)", rows
        )
    mark_stage(connection, stage, digest)


def run_convergence(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    X: pd.DataFrame,
    y: pd.Series,
    get_splits,
) -> None:
    """Cache the reference-style all-feature ensemble-size diagnostic."""
    stage = "convergence"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    rows = []
    for seed in config.convergence_seeds:
        splits = get_splits("kfold", config.outer_folds, seed)
        for trees in config.convergence_estimators:
            if trees < 1:
                raise ValueError("convergence_estimators must be positive")
            prediction = np.full(len(y), np.nan, dtype=float)
            for train, validation in splits:
                forest = make_forest(config, n_estimators=trees)
                forest.fit(X.iloc[train], y.iloc[train])
                prediction[validation] = forest.predict(X.iloc[validation])
            if not np.all(np.isfinite(prediction)):
                raise RuntimeError("Convergence experiment lacks complete OOF predictions")
            rows.append((seed, RF_SEED, len(splits), trees, *metrics(y, prediction)))
    with connection:
        connection.execute("DELETE FROM convergence_results")
        connection.executemany(
            "INSERT INTO convergence_results VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
    mark_stage(connection, stage, digest)


def store_selection_run(
    connection: sqlite3.Connection,
    evidence: SelectionEvidence,
    *,
    workflow: str,
    cv_method: str,
    cv_seed: int,
    fold_count: int,
    outer_fold: int,
    strategy: str,
    budget: float,
    outer_train_ids: Sequence[int],
    validation_ids: Sequence[int],
    inner_seed: int,
) -> int:
    cursor = connection.execute(
        """INSERT INTO selection_runs
        (workflow, cv_method, cv_seed, fold_count, outer_fold, strategy,
         requested_fraction, requested_count, actual_count, development_hash,
         outer_validation_hash, inner_seed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            workflow, cv_method, cv_seed, fold_count, outer_fold, strategy, budget,
            int(evidence.diagnostics["requested_count"]), len(evidence.selected),
            stable_hash(outer_train_ids), stable_hash(validation_ids), inner_seed,
        ),
    )
    run_id = int(cursor.lastrowid)
    if evidence.inner_boundary_rows:
        connection.executemany(
            """INSERT INTO inner_fold_boundaries
            (selection_run_id, inner_fold, development_hash, validation_hash,
             development_count, validation_count)
            VALUES (?, ?, ?, ?, ?, ?)""",
            [(run_id, *row) for row in evidence.inner_boundary_rows],
        )
    if evidence.inner_assignment_rows:
        connection.executemany(
            """INSERT INTO inner_fold_assignments
            (selection_run_id, observation_id, inner_fold, role)
            VALUES (?, ?, ?, ?)""",
            [(run_id, *row) for row in evidence.inner_assignment_rows],
        )
    rank_lookup = {name: rank + 1 for rank, name in enumerate(evidence.ranking)}
    connection.executemany(
        "INSERT INTO selected_features VALUES (?, ?, ?, ?)",
        [(run_id, name, rank_lookup[name], position)
         for position, name in enumerate(evidence.selected)],
    )
    connection.executemany(
        "INSERT INTO feature_selection_diagnostics VALUES (?, ?, ?)",
        [(run_id, key, json.dumps(value, sort_keys=True))
         for key, value in evidence.diagnostics.items()],
    )
    if evidence.permutation_rows:
        connection.executemany(
            "INSERT INTO permutation_importance_evidence VALUES (?, ?, ?, ?, ?)",
            [(run_id, *row) for row in evidence.permutation_rows],
        )
    if evidence.correlation_rows:
        connection.executemany(
            "INSERT INTO correlation_pruning_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, *row, config_value)
             for row in evidence.correlation_rows
             for config_value in [evidence.diagnostics["correlation_method"]]],
        )
    return run_id


def insert_outer_experiment(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    evidence: SelectionEvidence,
    selection_run_id: int,
    workflow: str,
    cv_method: str,
    cv_seed: int,
    outer_fold: int,
    strategy: str,
    budget: float,
    params: tuple[int | None, int, int, float],
    validation: Sequence[int],
    y: pd.Series,
    predictions: np.ndarray,
) -> int:
    depth, leaf, split, max_features = params
    cursor = connection.execute(
        """INSERT INTO outer_experiments
        (selection_run_id, workflow, cv_method, cv_seed, fold_count, outer_fold,
         strategy, requested_fraction, requested_count, actual_count, rf_seed,
         n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            selection_run_id, workflow, cv_method, cv_seed, config.outer_folds,
            outer_fold, strategy, budget, int(evidence.diagnostics["requested_count"]),
            len(evidence.selected), RF_SEED, config.n_estimators, depth_to_db(depth),
            leaf, split, max_features,
        ),
    )
    experiment_id = int(cursor.lastrowid)
    score = metrics(y.iloc[validation], predictions)
    connection.execute(
        "INSERT INTO generalizability_results VALUES (?, ?, ?, ?)",
        (experiment_id, *score),
    )
    connection.executemany(
        "INSERT INTO generalizability_predictions VALUES (?, ?, ?, ?, ?)",
        [(experiment_id, int(obs), outer_fold, float(y.iloc[obs]), float(prediction))
         for obs, prediction in zip(validation, predictions)],
    )
    return experiment_id


def run_generalizability(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    X: pd.DataFrame,
    y: pd.Series,
    get_splits,
) -> None:
    stage = "generalizability"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    with connection:
        connection.execute("DELETE FROM posthoc_permutation_evidence")
        connection.execute("DELETE FROM generalizability_predictions")
        connection.execute("DELETE FROM generalizability_results")
        connection.execute("DELETE FROM outer_experiments")
        connection.execute("DELETE FROM correlation_pruning_evidence")
        connection.execute("DELETE FROM permutation_importance_evidence")
        connection.execute("DELETE FROM feature_selection_diagnostics")
        connection.execute("DELETE FROM selected_features")
        connection.execute("DELETE FROM inner_fold_assignments")
        connection.execute("DELETE FROM inner_fold_boundaries")
        connection.execute("DELETE FROM selection_runs")
    representative_seed = min(config.outer_seeds)
    for method in ("kfold", "loco"):
        for cv_seed in config.outer_seeds:
            splits = get_splits(method, config.outer_folds, cv_seed)
            for outer_fold, (train, validation) in enumerate(splits):
                X_train, y_train = X.iloc[train], y.iloc[train]
                for strategy, budget in representation_grid(config):
                    inner_seed = 10_000 + 10_000 * cv_seed + 100 * outer_fold
                    evidence = select_features(
                        config, strategy, budget, X_train, y_train,
                        train, validation, inner_seed,
                    )
                    forest = make_forest(config)
                    forest.fit(X_train[evidence.selected], y_train)
                    prediction = forest.predict(X.iloc[validation][evidence.selected])
                    with connection:
                        selection_run_id = store_selection_run(
                            connection, evidence, workflow="primary",
                            cv_method=method, cv_seed=cv_seed,
                            fold_count=config.outer_folds, outer_fold=outer_fold,
                            strategy=strategy, budget=budget,
                            outer_train_ids=train, validation_ids=validation,
                            inner_seed=inner_seed,
                        )
                        experiment_id = insert_outer_experiment(
                            connection, config, evidence, selection_run_id, "primary",
                            method, cv_seed, outer_fold, strategy, budget,
                            config.representative_params, validation, y, prediction,
                        )
                        # Outer-fold importance is explanatory evidence only.  It is
                        # intentionally separated from inner selection evidence.
                        representative = (
                            cv_seed == representative_seed
                            and (strategy == "all" or budget == config.representative_budget)
                        )
                        if representative:
                            posthoc = permutation_importance(
                                forest,
                                X.iloc[validation][evidence.selected],
                                y.iloc[validation],
                                scoring="neg_root_mean_squared_error",
                                n_repeats=config.permutation_repeats,
                                random_state=90_000 + 100 * outer_fold,
                                n_jobs=PERMUTATION_N_JOBS,
                            )
                            connection.executemany(
                                "INSERT INTO posthoc_permutation_evidence VALUES (?, ?, ?, ?)",
                                [
                                    (experiment_id, repeat, feature, float(value))
                                    for feature_index, feature in enumerate(evidence.selected)
                                    for repeat, value in enumerate(posthoc.importances[feature_index])
                                ],
                            )
    mark_stage(connection, stage, digest)


def tune_params_inside_outer_training(
    config: ExperimentConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    budget: float,
    outer_training_ids: Sequence[int],
    seed: int,
) -> tuple[int | None, int, int, float]:
    splitter = list(KFold(config.inner_folds, shuffle=True, random_state=seed).split(X_train))
    outer_training_ids = np.asarray(outer_training_ids, dtype=int)
    if not np.array_equal(X_train.index.to_numpy(), outer_training_ids):
        raise RuntimeError("Tuning rows are not aligned with outer-training identifiers")
    tuning_folds = []
    for tuning_fold, (development, validation) in enumerate(splitter):
        development_ids = outer_training_ids[development]
        validation_ids = outer_training_ids[validation]
        evidence = select_features(
            config,
            strategy,
            budget,
            X_train.iloc[development],
            y_train.iloc[development],
            development_ids,
            validation_ids,
            seed + 10_000 + tuning_fold,
        )
        tuning_folds.append((development, validation, evidence.selected))
    scored = []
    for candidate_index, params in enumerate(config.tuning_grid):
        prediction = np.full(len(y_train), np.nan)
        for inner_train, inner_validation, selected in tuning_folds:
            forest = make_forest(config, params)
            forest.fit(X_train.iloc[inner_train][list(selected)], y_train.iloc[inner_train])
            prediction[inner_validation] = forest.predict(
                X_train.iloc[inner_validation][list(selected)]
            )
        scored.append((float(mean_squared_error(y_train, prediction) ** 0.5), candidate_index, params))
    return min(scored, key=lambda row: (row[0], row[1]))[2]


def run_secondary_sensitivity(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    X: pd.DataFrame,
    y: pd.Series,
    get_splits,
) -> None:
    stage = "secondary_sensitivity"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    if not config.run_secondary_sensitivity:
        mark_stage(connection, stage, digest)
        return
    # A failed long FULL_PRECOMPUTE may leave committed tuned folds behind.
    # Clear only this optional workflow so a resume is deterministic and does
    # not collide with its uniqueness constraints.
    with connection:
        connection.execute("DELETE FROM generalizability_material_summary WHERE workflow='tuned'")
        connection.execute("DELETE FROM generalizability_summary WHERE workflow='tuned'")
        connection.execute(
            "DELETE FROM generalizability_predictions WHERE experiment_id IN "
            "(SELECT experiment_id FROM outer_experiments WHERE workflow='tuned')"
        )
        connection.execute(
            "DELETE FROM generalizability_results WHERE experiment_id IN "
            "(SELECT experiment_id FROM outer_experiments WHERE workflow='tuned')"
        )
        connection.execute("DELETE FROM outer_experiments WHERE workflow='tuned'")
        for table in (
            "correlation_pruning_evidence", "permutation_importance_evidence",
            "feature_selection_diagnostics", "selected_features",
            "inner_fold_assignments", "inner_fold_boundaries",
        ):
            connection.execute(
                f"DELETE FROM {table} WHERE selection_run_id IN "
                "(SELECT selection_run_id FROM selection_runs WHERE workflow='tuned')"
            )
        connection.execute("DELETE FROM selection_runs WHERE workflow='tuned'")
        connection.execute(
            "DELETE FROM stage_status WHERE stage='generalizability_summaries'"
        )
    grid = [("all", 1.0)] + [
        (strategy, budget)
        for strategy in STRATEGIES[1:]
        for budget in config.sensitivity_budgets
    ]
    for method in ("kfold", "loco"):
        for cv_seed in config.outer_seeds:
            for outer_fold, (train, validation) in enumerate(
                get_splits(method, config.outer_folds, cv_seed)
            ):
                X_train, y_train = X.iloc[train], y.iloc[train]
                for strategy, budget in grid:
                    inner_seed = 30_000 + 10_000 * cv_seed + 100 * outer_fold
                    params = tune_params_inside_outer_training(
                        config, X_train, y_train, strategy, budget, train,
                        inner_seed + 1,
                    )
                    evidence = select_features(
                        config, strategy, budget, X_train, y_train,
                        train, validation, inner_seed,
                    )
                    forest = make_forest(config, params)
                    forest.fit(X_train[evidence.selected], y_train)
                    prediction = forest.predict(X.iloc[validation][evidence.selected])
                    with connection:
                        run_id = store_selection_run(
                            connection, evidence, workflow="tuned", cv_method=method,
                            cv_seed=cv_seed, fold_count=config.outer_folds,
                            outer_fold=outer_fold, strategy=strategy, budget=budget,
                            outer_train_ids=train, validation_ids=validation,
                            inner_seed=inner_seed,
                        )
                        insert_outer_experiment(
                            connection, config, evidence, run_id, "tuned", method,
                            cv_seed, outer_fold, strategy, budget, params,
                            validation, y, prediction,
                        )
    mark_stage(connection, stage, digest)


def run_generalizability_summaries(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
) -> None:
    """Materialize pooled-OOF and recurring material-level teaching summaries."""
    stage = "generalizability_summaries"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    evidence = pd.read_sql_query(
        """SELECT e.workflow, e.cv_method, e.cv_seed, e.fold_count,
                  e.strategy, e.requested_fraction, e.requested_count,
                  e.actual_count, p.observation_id, p.y_true, p.y_pred
           FROM outer_experiments e
           JOIN generalizability_predictions p USING(experiment_id)
           ORDER BY e.workflow, e.cv_method, e.cv_seed, e.strategy,
                    e.requested_fraction, p.observation_id""",
        connection,
    )
    if evidence.empty:
        raise RuntimeError("Cannot summarize empty generalizability evidence")

    pooled_keys = [
        "workflow", "cv_method", "cv_seed", "fold_count", "strategy",
        "requested_fraction",
    ]
    pooled_rows = []
    for key, group in evidence.groupby(pooled_keys, sort=True, dropna=False):
        if group["observation_id"].nunique() != len(group):
            raise RuntimeError("Pooled generalizability group repeats an observation")
        if group["requested_count"].nunique() != 1 or group["actual_count"].nunique() != 1:
            raise RuntimeError("Feature counts changed within one pooled OOF experiment")
        pooled_rows.append((
            *key,
            int(group["requested_count"].iloc[0]),
            int(group["actual_count"].iloc[0]),
            len(group),
            *metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
        ))

    material_keys = [
        "workflow", "cv_method", "strategy", "requested_fraction", "observation_id",
    ]
    material_rows = []
    for key, group in evidence.groupby(material_keys, sort=True, dropna=False):
        if group["cv_seed"].nunique() != len(group):
            raise RuntimeError("Material summary has repeated predictions for one CV seed")
        if group["y_true"].nunique() != 1:
            raise RuntimeError("Observed target changed across cached predictions")
        predictions = group["y_pred"].to_numpy(dtype=float)
        residual = predictions - group["y_true"].to_numpy(dtype=float)
        material_rows.append((
            *key,
            len(group),
            float(np.mean(predictions)),
            float(np.std(predictions, ddof=0)),
            float(np.mean(residual)),
            float(np.mean(np.abs(residual))),
            float(np.sqrt(np.mean(residual ** 2))),
            float(np.max(np.abs(residual))),
        ))

    with connection:
        connection.execute("DELETE FROM generalizability_material_summary")
        connection.execute("DELETE FROM generalizability_summary")
        connection.executemany(
            """INSERT INTO generalizability_summary
            (workflow, cv_method, cv_seed, fold_count, strategy,
             requested_fraction, requested_count, actual_count, n_observations,
             r2, mae, rmse)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            pooled_rows,
        )
        connection.executemany(
            """INSERT INTO generalizability_material_summary
            (workflow, cv_method, strategy, requested_fraction, observation_id,
             prediction_count, mean_prediction, prediction_standard_deviation,
             mean_signed_error, mean_absolute_error, rmse, maximum_absolute_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            material_rows,
        )
    mark_stage(connection, stage, digest)


def perturbation_seed(method: str, cv_seed: int, outer_fold: int) -> int:
    return 1_000_000 + 100_000_000 * {"kfold": 0, "loco": 1}[method] + 10_000 * cv_seed + outer_fold


def robustness_conditions(config: ExperimentConfig) -> list[tuple[float, float]]:
    # A full crossed surface reveals interactions.  Pairing remains strict:
    # every point for one split scales the same standardized X and y directions.
    return list(itertools.product(
        config.feature_noise_levels, config.target_noise_levels
    ))


def prepare_training_noise(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    feature_scale = X_train.std(axis=0, ddof=0).to_numpy()
    active = feature_scale > 0
    z_feature = np.zeros(X_train.shape)
    z_feature[:, active] = rng.normal(size=(len(X_train), int(active.sum())))
    return {
        "X": X_train,
        "y": y_train,
        "feature_scale": feature_scale,
        "target_scale": float(y_train.std(ddof=0)),
        "z_feature": z_feature,
        "z_target": rng.normal(size=len(y_train)),
        "active": active,
    }


def apply_training_noise(
    prepared: dict[str, object], feature_noise: float, target_noise: float
) -> tuple[pd.DataFrame, pd.Series]:
    X_clean = prepared["X"]
    y_clean = prepared["y"]
    assert isinstance(X_clean, pd.DataFrame) and isinstance(y_clean, pd.Series)
    delta = (
        feature_noise * np.asarray(prepared["z_feature"])
        * np.asarray(prepared["feature_scale"])[None, :]
    )
    X_noisy = X_clean.copy()
    X_noisy.iloc[:, :] = X_clean.to_numpy() + delta
    y_noisy = y_clean + (
        target_noise * float(prepared["target_scale"]) * np.asarray(prepared["z_target"])
    )
    if not np.all(delta[:, ~np.asarray(prepared["active"], dtype=bool)] == 0):
        raise RuntimeError("Constant-within-fold features were perturbed")
    if feature_noise == 0 and not np.array_equal(X_noisy.to_numpy(), X_clean.to_numpy()):
        raise RuntimeError("Zero feature noise changed training X")
    if target_noise == 0 and not np.array_equal(y_noisy.to_numpy(), y_clean.to_numpy()):
        raise RuntimeError("Zero target noise changed training y")
    return X_noisy, y_noisy


def run_robustness(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    X: pd.DataFrame,
    y: pd.Series,
    get_splits,
) -> None:
    if config.robustness_folds != config.outer_folds:
        raise ValueError(
            "robustness_folds must equal outer_folds so clean primary selections "
            "are valid paired baselines"
        )
    if not set(config.robustness_seeds).issubset(config.outer_seeds):
        raise ValueError(
            "robustness_seeds must be a subset of outer_seeds so every robustness "
            "split has a clean primary selection"
        )
    stage = "robustness"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    with connection:
        connection.execute("DELETE FROM robustness_selected_features")
        connection.execute("DELETE FROM robustness_importances")
        connection.execute("DELETE FROM robustness_predictions")
        connection.execute("DELETE FROM robustness_results")
    conditions = robustness_conditions(config)
    for method in ("kfold", "loco"):
        for cv_seed in config.robustness_seeds:
            splits = get_splits(method, config.robustness_folds, cv_seed)
            for outer_fold, (train, validation) in enumerate(splits):
                X_train, y_train = X.iloc[train], y.iloc[train]
                noise_seed = perturbation_seed(method, cv_seed, outer_fold)
                prepared = prepare_training_noise(X_train, y_train, noise_seed)
                for strategy, budget in representative_grid(config):
                    clean_inner_seed = 10_000 + 10_000 * cv_seed + 100 * outer_fold
                    clean_run = connection.execute(
                        """SELECT selection_run_id FROM selection_runs
                           WHERE workflow='primary' AND cv_method=? AND cv_seed=?
                             AND fold_count=? AND outer_fold=? AND strategy=?
                             AND ABS(requested_fraction-?) < 1e-12""",
                        (method, cv_seed, config.robustness_folds, outer_fold,
                         strategy, budget),
                    ).fetchone()
                    if clean_run is None:
                        raise RuntimeError("Missing clean primary selection for robustness")
                    clean_selected = [row[0] for row in connection.execute(
                        """SELECT feature_name FROM selected_features
                           WHERE selection_run_id=? ORDER BY model_position""", clean_run,
                    )]
                    zero_predictions: dict[str, np.ndarray] = {}
                    zero_selections: dict[str, set[str]] = {}
                    zero_importances: dict[str, dict[str, float]] = {}
                    for process in ("frozen_representation", "rerun_selection"):
                        for feature_noise, target_noise in conditions:
                            X_noisy, y_noisy = apply_training_noise(
                                prepared, feature_noise, target_noise
                            )
                            if process == "frozen_representation":
                                selected = clean_selected
                            else:
                                if feature_noise == 0 and target_noise == 0:
                                    selected = clean_selected
                                else:
                                    rerun_seed = clean_inner_seed  # paired inner folds/repeats
                                    rerun = select_features(
                                        config, strategy, budget, X_noisy, y_noisy,
                                        train, validation, rerun_seed,
                                    )
                                    selected = rerun.selected
                            if (
                                process == "rerun_selection"
                                and feature_noise == 0 and target_noise == 0
                            ):
                                if set(selected) != zero_selections["frozen_representation"]:
                                    raise RuntimeError("Zero-noise selection-process check failed")
                                # The two questions share one literal zero-noise baseline.
                                # Reusing it prevents parallel floating-point reduction
                                # jitter from masquerading as a scientific difference.
                                prediction = zero_predictions["frozen_representation"].copy()
                                importance_by_feature = zero_importances[
                                    "frozen_representation"
                                ].copy()
                            else:
                                forest = make_forest(config)
                                forest.fit(X_noisy[selected], y_noisy)
                                prediction = forest.predict(X.iloc[validation][selected])
                                importance_by_feature = {
                                    name: float(value)
                                    for name, value in zip(
                                        selected, forest.feature_importances_, strict=True
                                    )
                                }
                            score = metrics(y.iloc[validation], prediction)
                            with connection:
                                cursor = connection.execute(
                                    """INSERT INTO robustness_results
                                    (cv_method, cv_seed, fold_count, outer_fold, strategy,
                                     requested_fraction, process, feature_noise, target_noise,
                                     perturbation_seed, actual_count, r2, mae, rmse)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (method, cv_seed, config.robustness_folds, outer_fold,
                                     strategy, budget, process, feature_noise, target_noise,
                                     noise_seed, len(selected), *score),
                                )
                                result_id = int(cursor.lastrowid)
                                connection.executemany(
                                    "INSERT INTO robustness_predictions VALUES (?, ?, ?, ?)",
                                    [(result_id, int(obs), float(y.iloc[obs]), float(pred))
                                     for obs, pred in zip(validation, prediction)],
                                )
                                connection.executemany(
                                    "INSERT INTO robustness_selected_features VALUES (?, ?)",
                                    [(result_id, name) for name in selected],
                                )
                                connection.executemany(
                                    "INSERT INTO robustness_importances VALUES (?, ?, ?)",
                                    [
                                        (result_id, name, importance_by_feature[name])
                                        for name in selected
                                    ],
                                )
                            if feature_noise == 0 and target_noise == 0:
                                zero_predictions[process] = prediction
                                zero_selections[process] = set(selected)
                                zero_importances[process] = importance_by_feature
                    if zero_selections["frozen_representation"] != zero_selections["rerun_selection"]:
                        raise RuntimeError("Zero-noise selection-process check failed")
                    if not np.allclose(
                        zero_predictions["frozen_representation"],
                        zero_predictions["rerun_selection"],
                        rtol=0.0, atol=1e-12,
                    ):
                        raise RuntimeError("Zero-noise robustness predictions disagree")
                    if zero_importances["frozen_representation"] != zero_importances[
                        "rerun_selection"
                    ]:
                        raise RuntimeError("Zero-noise robustness importances disagree")
    mark_stage(connection, stage, digest)


def permutation_rank_map(connection: sqlite3.Connection, run_id: int) -> dict[str, float]:
    rows = connection.execute(
        """SELECT feature_name, AVG(signed_importance)
           FROM permutation_importance_evidence WHERE selection_run_id=?
           GROUP BY feature_name""", (run_id,)
    ).fetchall()
    if not rows:
        return {}
    values = pd.Series({name: value for name, value in rows})
    return values.rank(method="average", ascending=False).to_dict()


def run_selection_stability(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
) -> None:
    stage = "selection_stability"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    with connection:
        connection.execute("DELETE FROM stability_selection_results")
    groups = connection.execute(
        """SELECT cv_method, strategy, requested_fraction,
                  GROUP_CONCAT(selection_run_id)
           FROM selection_runs WHERE workflow='primary'
           GROUP BY cv_method, strategy, requested_fraction"""
    ).fetchall()
    rows = []
    for method, strategy, budget, run_ids_text in groups:
        run_ids = [int(value) for value in run_ids_text.split(",")]
        feature_sets = {
            run_id: {row[0] for row in connection.execute(
                "SELECT feature_name FROM selected_features WHERE selection_run_id=?",
                (run_id,),
            )}
            for run_id in run_ids
        }
        ranks = {run_id: permutation_rank_map(connection, run_id) for run_id in run_ids}
        for run_a, run_b in itertools.combinations(run_ids, 2):
            set_a, set_b = feature_sets[run_a], feature_sets[run_b]
            intersection, union = set_a & set_b, set_a | set_b
            rank_correlation = None
            common_ranked = sorted(set(ranks[run_a]) & set(ranks[run_b]))
            if len(common_ranked) >= 2:
                rank_correlation = float(pd.Series(
                    [ranks[run_a][name] for name in common_ranked]
                ).corr(pd.Series(
                    [ranks[run_b][name] for name in common_ranked]
                ), method="spearman"))
            rows.append((
                method, strategy, budget, run_a, run_b, len(intersection), len(union),
                len(intersection) / len(union), len(set_a ^ set_b), rank_correlation,
            ))
    with connection:
        connection.executemany(
            """INSERT INTO stability_selection_results
            (cv_method, strategy, requested_fraction, selection_run_a, selection_run_b,
             intersection_count, union_count, jaccard, inclusion_changes, rank_spearman)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows,
        )
    mark_stage(connection, stage, digest)


def node_depths(tree) -> np.ndarray:
    depths = np.full(tree.node_count, -1, dtype=int)
    depths[0] = 0
    stack = [0]
    while stack:
        node = stack.pop()
        left, right = tree.children_left[node], tree.children_right[node]
        if left != right:
            depths[left] = depths[node] + 1
            depths[right] = depths[node] + 1
            stack.extend((left, right))
    if np.any(depths < 0):
        raise RuntimeError("Could not determine tree-node depths")
    return depths


def custom_tree_predict(cache: dict[str, np.ndarray], X_values: np.ndarray,
                        thresholds: np.ndarray | None = None):
    thresholds = cache["threshold"] if thresholds is None else thresholds
    nodes = np.zeros(len(X_values), dtype=np.int32)
    active = np.ones(len(X_values), dtype=bool)
    while np.any(active):
        rows = np.flatnonzero(active)
        current = nodes[rows]
        features = cache["feature"][current]
        internal = features >= 0
        active[rows[~internal]] = False
        decision_rows = rows[internal]
        if not len(decision_rows):
            continue
        decision_nodes = current[internal]
        decision_features = features[internal]
        go_left = X_values[decision_rows, decision_features] <= thresholds[decision_nodes]
        nodes[decision_rows] = np.where(
            go_left,
            cache["children_left"][decision_nodes],
            cache["children_right"][decision_nodes],
        )
    return cache["value"][nodes], nodes


def prepare_tree_caches(forest, X_train_values, X_test_values):
    caches, tree_predictions = [], []
    for estimator in forest.estimators_:
        tree = estimator.tree_
        feature = tree.feature.copy()
        threshold = tree.threshold.copy()
        local_scale = np.zeros(tree.node_count)
        local_min, local_max = threshold.copy(), threshold.copy()
        paths = estimator.decision_path(X_train_values).tocsc()
        for node in np.flatnonzero(feature >= 0):
            start, stop = paths.indptr[node:node + 2]
            rows = paths.indices[start:stop]
            values = X_train_values[rows, feature[node]].astype(float, copy=False)
            q25, q75 = np.percentile(values, [25.0, 75.0])
            scale = float(q75 - q25)
            if scale == 0:
                scale = float(np.std(values, ddof=0))
            local_scale[node] = scale if np.isfinite(scale) else 0.0
            local_min[node], local_max[node] = float(values.min()), float(values.max())
        cache = {
            "children_left": tree.children_left.copy(),
            "children_right": tree.children_right.copy(),
            "feature": feature,
            "threshold": threshold,
            "value": tree.value[:, 0, 0].copy(),
            "depth": node_depths(tree),
            "local_scale": local_scale,
            "local_min": local_min,
            "local_max": local_max,
        }
        prediction, leaves = custom_tree_predict(cache, X_test_values)
        if not np.allclose(prediction, estimator.predict(X_test_values), atol=1e-12, rtol=0):
            raise RuntimeError("Zero-perturbation custom tree traversal disagrees with sklearn")
        cache["baseline_leaf"] = leaves
        caches.append(cache)
        tree_predictions.append(prediction)
    return caches, np.asarray(tree_predictions)


def prediction_metric_arrays(y_true: np.ndarray, predictions: np.ndarray):
    residual = predictions - y_true[None, :]
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator <= 0:
        raise RuntimeError("Stability validation target has no variance")
    r2 = 1.0 - np.sum(residual ** 2, axis=1) / denominator
    mae = np.mean(np.abs(residual), axis=1)
    rmse = np.sqrt(np.mean(residual ** 2, axis=1))
    return r2, mae, rmse


def stability_seed(method: str, cv_seed: int, fold: int, mechanism: str) -> int:
    return (
        7_000_000 + 1_000_000 * {"kfold": 0, "loco": 1}[method]
        + 10_000 * cv_seed + 100 * fold
        + {"tree_reweighting": 11, "threshold": 29}[mechanism]
    )


def store_stability_condition(
    connection: sqlite3.Connection,
    *, method: str, cv_seed: int, outer_fold: int, strategy: str, budget: float,
    mechanism: str, magnitude: float, baseline: np.ndarray, y_true: np.ndarray,
    predictions: np.ndarray, observation_ids: Sequence[int],
    effective_counts: np.ndarray | None = None,
    routing: np.ndarray | None = None,
) -> None:
    movement = np.abs(predictions - baseline[None, :])
    baseline_rmse = float(np.sqrt(np.mean((baseline - y_true) ** 2)))
    r2_values, mae_values, rmse_values = prediction_metric_arrays(y_true, predictions)
    mean_routing = None if routing is None else float(np.mean(routing))
    median_effective = None if effective_counts is None else float(np.median(effective_counts))
    if effective_counts is not None and len(effective_counts) != len(predictions):
        raise RuntimeError("Effective-tree-count realization coverage is incomplete")
    if routing is not None and routing.shape != predictions.shape:
        raise RuntimeError("Routing-change evidence does not match prediction evidence")
    cursor = connection.execute(
        """INSERT INTO stability_prediction_results
        (cv_method, cv_seed, outer_fold, strategy, requested_fraction, mechanism,
         magnitude, realization_count, baseline_rmse, median_rmse, rmse_movement,
         median_absolute_prediction_movement, p95_absolute_prediction_movement,
         median_effective_tree_count, mean_routing_change)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            method, cv_seed, outer_fold, strategy, budget, mechanism, magnitude,
            predictions.shape[0], baseline_rmse, float(np.median(rmse_values)),
            float(np.median(rmse_values) - baseline_rmse), float(np.median(movement)),
            float(np.percentile(movement, 95)), median_effective, mean_routing,
        ),
    )
    stability_id = int(cursor.lastrowid)
    routing_by_realization = (
        None if routing is None else np.mean(routing, axis=1)
    )
    connection.executemany(
        """INSERT INTO stability_realizations
        (stability_id, realization, r2, mae, rmse, effective_tree_count,
         mean_routing_change)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                stability_id, realization, float(r2_values[realization]),
                float(mae_values[realization]), float(rmse_values[realization]),
                None if effective_counts is None
                else float(effective_counts[realization]),
                None if routing_by_realization is None
                else float(routing_by_realization[realization]),
            )
            for realization in range(len(predictions))
        ],
    )
    connection.executemany(
        """INSERT INTO stability_predictions
        (stability_id, observation_id, baseline_prediction, median_prediction,
         median_absolute_movement, p95_absolute_movement, mean_routing_change,
         prediction_standard_deviation, prediction_lower_2_5,
         prediction_upper_97_5, prediction_interval_width)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                stability_id, int(obs), float(baseline[column]),
                float(np.median(predictions[:, column])),
                float(np.median(movement[:, column])),
                float(np.percentile(movement[:, column], 95)),
                None if routing is None else float(np.mean(routing[:, column])),
                float(np.std(predictions[:, column], ddof=0)),
                float(np.percentile(predictions[:, column], 2.5)),
                float(np.percentile(predictions[:, column], 97.5)),
                float(
                    np.percentile(predictions[:, column], 97.5)
                    - np.percentile(predictions[:, column], 2.5)
                ),
            )
            for column, obs in enumerate(observation_ids)
        ],
    )


def run_prediction_stability(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    X: pd.DataFrame,
    y: pd.Series,
    get_splits,
) -> None:
    stage = "prediction_stability"
    if stage_complete(connection, stage, digest):
        print(f"Skipping completed stage: {stage}")
        return
    with connection:
        connection.execute("DELETE FROM stability_realizations")
        connection.execute("DELETE FROM stability_predictions")
        connection.execute("DELETE FROM stability_prediction_results")
    cv_seed = min(config.outer_seeds)
    grid = [("all", 1.0)] + [
        (strategy, config.representative_budget) for strategy in STRATEGIES[1:]
    ]
    for method in ("kfold", "loco"):
        splits = get_splits(method, config.outer_folds, cv_seed)
        for outer_fold, (train, validation) in enumerate(splits):
            for strategy, budget in grid:
                run_row = connection.execute(
                    """SELECT selection_run_id FROM selection_runs
                       WHERE workflow='primary' AND cv_method=? AND cv_seed=?
                         AND fold_count=? AND outer_fold=? AND strategy=?
                         AND ABS(requested_fraction-?) < 1e-12""",
                    (method, cv_seed, config.outer_folds, outer_fold, strategy, budget),
                ).fetchone()
                if run_row is None:
                    raise RuntimeError("Missing primary representation for stability")
                selected = [row[0] for row in connection.execute(
                    """SELECT feature_name FROM selected_features
                       WHERE selection_run_id=? ORDER BY model_position""", run_row,
                )]
                forest = make_forest(config)
                forest.fit(X.iloc[train][selected], y.iloc[train])
                X_train_values = np.asarray(X.iloc[train][selected], dtype=np.float32)
                X_test_values = np.asarray(X.iloc[validation][selected], dtype=np.float32)
                baseline = forest.predict(X.iloc[validation][selected])
                caches, tree_predictions = prepare_tree_caches(
                    forest, X_train_values, X_test_values
                )
                if not np.allclose(tree_predictions.mean(axis=0), baseline, atol=1e-12, rtol=0):
                    raise RuntimeError("Equal tree weights do not reproduce the forest")

                weight_rng = np.random.default_rng(
                    stability_seed(method, cv_seed, outer_fold, "tree_reweighting")
                )
                weight_z = weight_rng.normal(
                    size=(config.stability_weight_repeats, config.n_estimators)
                )
                for strength in config.stability_weight_strengths:
                    log_weight = strength * weight_z
                    log_weight -= log_weight.max(axis=1, keepdims=True)
                    weights = np.exp(log_weight)
                    weights /= weights.sum(axis=1, keepdims=True)
                    predictions = weights @ tree_predictions
                    effective = 1.0 / np.sum(weights ** 2, axis=1)
                    if strength == 0 and not np.allclose(
                        predictions, baseline[None, :], atol=1e-12, rtol=0
                    ):
                        raise RuntimeError("Zero tree-reweighting perturbation changed predictions")
                    with connection:
                        store_stability_condition(
                            connection, method=method, cv_seed=cv_seed,
                            outer_fold=outer_fold, strategy=strategy, budget=budget,
                            mechanism="tree_reweighting", magnitude=strength,
                            baseline=baseline, y_true=y.iloc[validation].to_numpy(),
                            predictions=predictions, observation_ids=validation,
                            effective_counts=effective,
                        )

                for strength in config.stability_threshold_strengths:
                    predictions = np.empty((config.stability_threshold_repeats, len(validation)))
                    routing = np.zeros_like(predictions)
                    for realization in range(config.stability_threshold_repeats):
                        rng = np.random.default_rng(
                            stability_seed(method, cv_seed, outer_fold, "threshold")
                            + realization
                        )
                        prediction_sum = np.zeros(len(validation))
                        changed_sum = np.zeros(len(validation))
                        for cache in caches:
                            eligible = (
                                (cache["feature"] >= 0)
                                & (cache["local_scale"] > 0)
                                & (cache["depth"] <= config.stability_threshold_max_depth)
                            )
                            perturbed = cache["threshold"].copy()
                            nodes = np.flatnonzero(eligible)
                            if len(nodes):
                                perturbed[nodes] += (
                                    strength * rng.normal(size=len(nodes))
                                    * cache["local_scale"][nodes]
                                )
                                perturbed[nodes] = np.clip(
                                    perturbed[nodes], cache["local_min"][nodes],
                                    cache["local_max"][nodes],
                                )
                            tree_prediction, leaves = custom_tree_predict(
                                cache, X_test_values, perturbed
                            )
                            prediction_sum += tree_prediction
                            changed_sum += leaves != cache["baseline_leaf"]
                        predictions[realization] = prediction_sum / config.n_estimators
                        routing[realization] = changed_sum / config.n_estimators
                    if strength == 0:
                        if not np.allclose(predictions, baseline[None, :], atol=1e-12, rtol=0):
                            raise RuntimeError("Zero threshold perturbation changed predictions")
                        if np.count_nonzero(routing):
                            raise RuntimeError("Zero threshold perturbation changed routing")
                    with connection:
                        store_stability_condition(
                            connection, method=method, cv_seed=cv_seed,
                            outer_fold=outer_fold, strategy=strategy, budget=budget,
                            mechanism="threshold", magnitude=strength,
                            baseline=baseline, y_true=y.iloc[validation].to_numpy(),
                            predictions=predictions, observation_ids=validation,
                            routing=routing,
                        )
    mark_stage(connection, stage, digest)


def verify_database(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    digest: str,
    dataset_sha256: str,
    n_observations: int,
    n_features: int,
) -> dict[str, int]:
    required = {
        "metadata", "stage_status", "observations", "features", "excluded_columns",
        "cluster_assignments", "convergence_results", "outer_fold_assignments",
        "selection_runs", "inner_fold_assignments", "inner_fold_boundaries",
        "selected_features",
        "feature_selection_diagnostics", "permutation_importance_evidence",
        "correlation_pruning_evidence", "outer_experiments",
        "generalizability_results", "generalizability_predictions",
        "generalizability_summary", "generalizability_material_summary",
        "posthoc_permutation_evidence", "robustness_results",
        "robustness_predictions", "robustness_selected_features",
        "robustness_importances",
        "stability_selection_results", "stability_prediction_results",
        "stability_predictions", "stability_realizations",
    }
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if required - tables:
        raise RuntimeError(f"Missing database tables: {sorted(required - tables)}")
    counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in sorted(required)
    }
    for table, count in counts.items():
        if count == 0:
            raise RuntimeError(f"SMOKE_TEST/full pipeline did not exercise table {table}")
    if counts["observations"] != n_observations or counts["features"] != n_features:
        raise RuntimeError("Dataset dimension tables are incomplete")

    cluster_specifications = {
        (seed, count)
        for seed, count in itertools.chain(
            ((seed, config.outer_folds) for seed in config.outer_seeds),
            ((seed, config.robustness_folds) for seed in config.robustness_seeds),
        )
    }
    expected_clusters = len(cluster_specifications) * n_observations
    if counts["cluster_assignments"] != expected_clusters:
        raise RuntimeError(
            f"Cluster-assignment coverage incomplete: expected {expected_clusters}, "
            f"found {counts['cluster_assignments']}"
        )
    mismatched_clusters = connection.execute(
        """SELECT COUNT(*) FROM cluster_assignments c
           JOIN outer_fold_assignments f
             ON f.cv_method='loco' AND f.cv_seed=c.cv_seed
            AND f.fold_count=c.cluster_count
            AND f.observation_id=c.observation_id
           WHERE c.cluster_id != f.outer_fold"""
    ).fetchone()[0]
    if mismatched_clusters:
        raise RuntimeError("Cached PCA/cluster rows disagree with LOCO folds")
    expected_convergence = (
        len(config.convergence_seeds) * len(config.convergence_estimators)
    )
    if counts["convergence_results"] != expected_convergence:
        raise RuntimeError("Ensemble-convergence coverage is incomplete")
    if connection.execute(
        "SELECT COUNT(*) FROM features WHERE feature_name IN ('composition', ?)",
        (json.loads(connection.execute(
            "SELECT value FROM metadata WHERE key='target_column'"
        ).fetchone()[0]),),
    ).fetchone()[0]:
        raise RuntimeError("Target or composition entered the feature matrix")

    representations = len(representation_grid(config))
    primary_experiments = (
        2 * len(config.outer_seeds) * config.outer_folds * representations
    )
    secondary_representations = 1 + 3 * len(config.sensitivity_budgets)
    secondary_experiments = (
        2 * len(config.outer_seeds) * config.outer_folds * secondary_representations
        if config.run_secondary_sensitivity else 0
    )
    expected_experiments = primary_experiments + secondary_experiments
    for table in ("selection_runs", "outer_experiments", "generalizability_results"):
        if counts[table] != expected_experiments:
            raise RuntimeError(
                f"{table} coverage incomplete: expected {expected_experiments}, found {counts[table]}"
            )
    expected_predictions = (
        2 * len(config.outer_seeds) * representations * n_observations
        + (2 * len(config.outer_seeds) * secondary_representations * n_observations
           if config.run_secondary_sensitivity else 0)
    )
    if counts["generalizability_predictions"] != expected_predictions:
        raise RuntimeError("Generalizability OOF prediction coverage is incomplete")
    expected_summaries = (
        2 * len(config.outer_seeds) * representations
        + (2 * len(config.outer_seeds) * secondary_representations
           if config.run_secondary_sensitivity else 0)
    )
    if counts["generalizability_summary"] != expected_summaries:
        raise RuntimeError("Pooled OOF generalizability summary coverage is incomplete")
    expected_material_summaries = (
        2 * representations * n_observations
        + (2 * secondary_representations * n_observations
           if config.run_secondary_sensitivity else 0)
    )
    if counts["generalizability_material_summary"] != expected_material_summaries:
        raise RuntimeError("Material-level generalizability summary coverage is incomplete")
    bad_summary_coverage = connection.execute(
        """SELECT COUNT(*) FROM generalizability_summary
           WHERE n_observations != ?""", (n_observations,)
    ).fetchone()[0]
    if bad_summary_coverage:
        raise RuntimeError("A pooled OOF summary does not cover every observation")

    invalid_boundaries = 0
    invalid_inner_boundaries = 0
    expected_inner_boundaries = 0
    expected_inner_assignments = 0
    for (
        run_id, method, seed, fold_count, fold, strategy,
        development_hash, validation_hash,
    ) in connection.execute(
        """SELECT selection_run_id, cv_method, cv_seed, fold_count, outer_fold,
                  strategy, development_hash, outer_validation_hash
           FROM selection_runs"""
    ):
        validation = [row[0] for row in connection.execute(
            """SELECT observation_id FROM outer_fold_assignments
               WHERE cv_method=? AND cv_seed=? AND fold_count=? AND outer_fold=?""",
            (method, seed, fold_count, fold),
        )]
        training = sorted(set(range(n_observations)) - set(validation))
        invalid_boundaries += int(
            stable_hash(training) != development_hash
            or stable_hash(validation) != validation_hash
        )
        boundaries = {
            int(inner_fold): (dev_hash, val_hash, int(dev_count), int(val_count))
            for inner_fold, dev_hash, val_hash, dev_count, val_count
            in connection.execute(
                """SELECT inner_fold, development_hash, validation_hash,
                          development_count, validation_count
                   FROM inner_fold_boundaries WHERE selection_run_id=?""",
                (run_id,),
            )
        }
        assignments = list(connection.execute(
            """SELECT observation_id, inner_fold, role
               FROM inner_fold_assignments WHERE selection_run_id=?""",
            (run_id,),
        ))
        if strategy in {"permutation", "correlation_permutation"}:
            expected_inner_boundaries += config.inner_folds
            expected_inner_assignments += config.inner_folds * len(training)
            invalid_inner_boundaries += int(len(boundaries) != config.inner_folds)
            for inner_fold in range(config.inner_folds):
                development_ids = sorted(
                    int(observation_id)
                    for observation_id, stored_fold, role in assignments
                    if stored_fold == inner_fold and role == "development"
                )
                validation_ids = sorted(
                    int(observation_id)
                    for observation_id, stored_fold, role in assignments
                    if stored_fold == inner_fold and role == "validation"
                )
                boundary = boundaries.get(inner_fold)
                invalid_inner_boundaries += int(
                    boundary is None
                    or set(development_ids) & set(validation_ids) != set()
                    or set(development_ids) | set(validation_ids) != set(training)
                    or set(development_ids) & set(validation) != set()
                    or set(validation_ids) & set(validation) != set()
                    or (
                        boundary is not None
                        and (
                            boundary[0] != stable_hash(development_ids)
                            or boundary[1] != stable_hash(validation_ids)
                            or boundary[2] != len(development_ids)
                            or boundary[3] != len(validation_ids)
                        )
                    )
                )
        else:
            invalid_inner_boundaries += int(bool(boundaries) or bool(assignments))
    if invalid_boundaries:
        raise RuntimeError(f"{invalid_boundaries} selection runs violate outer boundaries")
    if invalid_inner_boundaries:
        raise RuntimeError(
            f"{invalid_inner_boundaries} selection runs violate inner boundaries"
        )
    if counts["inner_fold_boundaries"] != expected_inner_boundaries:
        raise RuntimeError("Inner-fold boundary coverage is incomplete")
    if counts["inner_fold_assignments"] != expected_inner_assignments:
        raise RuntimeError("Inner-fold assignment coverage is incomplete")

    bad_selection_counts = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT r.selection_run_id, r.actual_count, COUNT(s.feature_name) AS stored
               FROM selection_runs r JOIN selected_features s USING(selection_run_id)
               GROUP BY r.selection_run_id HAVING stored != r.actual_count
           )"""
    ).fetchone()[0]
    if bad_selection_counts:
        raise RuntimeError("Selected-feature identity coverage is incomplete")
    if connection.execute(
        """SELECT COUNT(*) FROM selected_features
           WHERE feature_name='composition' OR feature_name=(
               SELECT json_extract(value, '$') FROM metadata WHERE key='target_column'
           )"""
    ).fetchone()[0]:
        raise RuntimeError("Metadata or target appeared in a selected feature set")
    supervised_without_evidence = connection.execute(
        """SELECT COUNT(*) FROM selection_runs r
           WHERE r.strategy IN ('permutation', 'correlation_permutation')
             AND NOT EXISTS (
                 SELECT 1 FROM permutation_importance_evidence p
                 WHERE p.selection_run_id=r.selection_run_id
             )"""
    ).fetchone()[0]
    if supervised_without_evidence:
        raise RuntimeError("A supervised selection run lacks inner-held-out evidence")
    if connection.execute(
        "SELECT COUNT(*) FROM permutation_importance_evidence WHERE signed_importance IS NULL"
    ).fetchone()[0]:
        raise RuntimeError("Permutation importance contains missing values")

    bad_oof = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT e.workflow, e.cv_method, e.cv_seed, e.strategy,
                      e.requested_fraction, COUNT(p.observation_id) AS rows_seen,
                      COUNT(DISTINCT p.observation_id) AS observations_seen
               FROM outer_experiments e
               JOIN generalizability_predictions p USING(experiment_id)
               GROUP BY e.workflow, e.cv_method, e.cv_seed, e.strategy, e.requested_fraction
               HAVING rows_seen != ? OR observations_seen != ?
           )""", (n_observations, n_observations),
    ).fetchone()[0]
    if bad_oof:
        raise RuntimeError("An experiment lacks exactly one OOF prediction per observation")
    primary_configurations = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT DISTINCT rf_seed, n_estimators, max_depth, min_samples_leaf,
                               min_samples_split, max_features
               FROM outer_experiments WHERE workflow='primary'
           )"""
    ).fetchone()[0]
    if primary_configurations != 1:
        raise RuntimeError("Primary representations did not share one forest configuration")
    # At a 100% budget, every strategy is the same ordered representation; this
    # provides a strong paired-control assertion for the whole selection path.
    for method in ("kfold", "loco"):
        for seed in config.outer_seeds:
            for fold in range(config.outer_folds):
                experiments = dict(connection.execute(
                    """SELECT strategy, experiment_id FROM outer_experiments
                       WHERE workflow='primary' AND cv_method=? AND cv_seed=?
                         AND outer_fold=? AND ABS(requested_fraction-1.0)<1e-12""",
                    (method, seed, fold),
                ))
                baseline = np.asarray([row[0] for row in connection.execute(
                    """SELECT y_pred FROM generalizability_predictions
                       WHERE experiment_id=? ORDER BY observation_id""",
                    (experiments["all"],),
                )])
                for strategy in STRATEGIES[1:]:
                    comparison = np.asarray([row[0] for row in connection.execute(
                        """SELECT y_pred FROM generalizability_predictions
                           WHERE experiment_id=? ORDER BY observation_id""",
                        (experiments[strategy],),
                    )])
                    if not np.allclose(baseline, comparison, rtol=0, atol=1e-12):
                        raise RuntimeError(
                            f"100% {strategy} is not identical to all-feature baseline"
                        )
    wrong_fold_predictions = connection.execute(
        """SELECT COUNT(*) FROM generalizability_predictions p
           JOIN outer_experiments e USING(experiment_id)
           JOIN outer_fold_assignments f
             ON f.cv_method=e.cv_method AND f.cv_seed=e.cv_seed
            AND f.fold_count=e.fold_count AND f.observation_id=p.observation_id
           WHERE f.outer_fold != e.outer_fold OR p.outer_fold != e.outer_fold"""
    ).fetchone()[0]
    if wrong_fold_predictions:
        raise RuntimeError("OOF predictions are attached to the wrong outer fold")

    conditions = len(robustness_conditions(config))
    expected_robustness = (
        2 * len(config.robustness_seeds) * config.robustness_folds
        * len(representative_grid(config)) * 2 * conditions
    )
    if counts["robustness_results"] != expected_robustness:
        raise RuntimeError(
            f"Robustness coverage incomplete: expected {expected_robustness}, "
            f"found {counts['robustness_results']}"
        )
    expected_robustness_predictions = (
        2 * len(config.robustness_seeds) * len(representative_grid(config))
        * 2 * conditions * n_observations
    )
    if counts["robustness_predictions"] != expected_robustness_predictions:
        raise RuntimeError("Observation-level robustness prediction coverage is incomplete")
    expected_robustness_features = connection.execute(
        "SELECT COALESCE(SUM(actual_count), 0) FROM robustness_results"
    ).fetchone()[0]
    if counts["robustness_selected_features"] != expected_robustness_features:
        raise RuntimeError("Robustness selected-feature coverage is incomplete")
    if counts["robustness_importances"] != expected_robustness_features:
        raise RuntimeError("Robustness impurity-importance coverage is incomplete")
    bad_robustness_importance = connection.execute(
        """SELECT COUNT(*) FROM robustness_importances i
           LEFT JOIN robustness_selected_features s
             ON s.robustness_id=i.robustness_id AND s.feature_name=i.feature_name
           WHERE s.feature_name IS NULL OR i.importance < 0"""
    ).fetchone()[0]
    if bad_robustness_importance:
        raise RuntimeError("Robustness importance is invalid or lacks a selected feature")
    incomplete_crossed_grids = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT cv_method, cv_seed, outer_fold, strategy,
                      requested_fraction, process,
                      COUNT(DISTINCT printf('%.15g|%.15g', feature_noise, target_noise))
                          AS condition_count
               FROM robustness_results
               GROUP BY cv_method, cv_seed, outer_fold, strategy,
                        requested_fraction, process
               HAVING condition_count != ?
           )""", (conditions,),
    ).fetchone()[0]
    if incomplete_crossed_grids:
        raise RuntimeError("A robustness result group lacks the full crossed noise grid")
    zero_mismatch = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT cv_method, cv_seed, outer_fold, strategy, requested_fraction,
                      COUNT(DISTINCT printf('%.15g|%.15g|%.15g', r2, mae, rmse)) AS versions
               FROM robustness_results
               WHERE feature_noise=0 AND target_noise=0
               GROUP BY cv_method, cv_seed, outer_fold, strategy, requested_fraction
               HAVING versions != 1
           )"""
    ).fetchone()[0]
    if zero_mismatch:
        raise RuntimeError("Zero-noise frozen/rerun robustness conditions disagree")

    expected_stability_results = (
        2 * config.outer_folds * len(representative_grid(config))
        * (len(config.stability_weight_strengths) + len(config.stability_threshold_strengths))
    )
    if counts["stability_prediction_results"] != expected_stability_results:
        raise RuntimeError("Prediction-stability coverage is incomplete")
    expected_stability_realizations = (
        2 * config.outer_folds * len(representative_grid(config))
        * (
            len(config.stability_weight_strengths) * config.stability_weight_repeats
            + len(config.stability_threshold_strengths)
            * config.stability_threshold_repeats
        )
    )
    if counts["stability_realizations"] != expected_stability_realizations:
        raise RuntimeError("Realization-level stability coverage is incomplete")
    expected_stability_predictions = (
        2 * len(representative_grid(config))
        * (len(config.stability_weight_strengths) + len(config.stability_threshold_strengths))
        * n_observations
    )
    if counts["stability_predictions"] != expected_stability_predictions:
        raise RuntimeError("Observation-level stability prediction coverage is incomplete")
    nonzero_zero_stability = connection.execute(
        """SELECT COUNT(*) FROM stability_prediction_results
           WHERE magnitude=0 AND
             (ABS(rmse_movement)>1e-12 OR ABS(median_absolute_prediction_movement)>1e-12
              OR ABS(p95_absolute_prediction_movement)>1e-12
              OR (mechanism='threshold' AND ABS(mean_routing_change)>1e-12))"""
    ).fetchone()[0]
    if nonzero_zero_stability:
        raise RuntimeError("A zero-strength model perturbation changed the model")
    invalid_intervals = connection.execute(
        """SELECT COUNT(*) FROM stability_predictions
           WHERE prediction_interval_width < -1e-12
              OR prediction_lower_2_5 > median_prediction + 1e-12
              OR median_prediction > prediction_upper_97_5 + 1e-12
              OR ABS(prediction_interval_width
                     - (prediction_upper_97_5-prediction_lower_2_5)) > 1e-10"""
    ).fetchone()[0]
    if invalid_intervals:
        raise RuntimeError("A material-level stability interval is internally inconsistent")

    metadata = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }
    for key in (
        "config_hash", "dataset_sha256", "python_version", "numpy_version",
        "pandas_version", "sklearn_version", "selection_rule", "loco_warning",
        "noise_pairing_rule", "stability_pairing_rule", "posthoc_separation",
    ):
        if key not in metadata:
            raise RuntimeError(f"Missing provenance field: {key}")
    if metadata["config_hash"] != digest or metadata["dataset_sha256"] != dataset_sha256:
        raise RuntimeError("Stored provenance does not match this run")
    expected_stages = {
        "fold_assignments", "cluster_assignments", "convergence",
        "generalizability", "secondary_sensitivity", "generalizability_summaries",
        "robustness", "selection_stability", "prediction_stability",
    }
    stages = dict(connection.execute("SELECT stage, config_hash FROM stage_status"))
    if set(stages) != expected_stages or any(value != digest for value in stages.values()):
        raise RuntimeError("Not every precompute stage completed under this configuration")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise RuntimeError("SQLite foreign-key verification failed")
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "full"), default="smoke",
        help="safe default: smoke",
    )
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--yes", action="store_true",
        help="required for non-interactive FULL_PRECOMPUTE",
    )
    return parser.parse_args()


def main() -> None:
    configure_warning_filters()
    args = parse_args()
    config = SMOKE_CONFIG if args.mode == "smoke" else FULL_CONFIG
    validate_config(config)
    root = find_repository_root(Path.cwd())
    dataset_path = root / "demonstrations" / "data" / "featurized_matbench_expt_gap.csv"
    database_path = args.database or (
        root / "demonstrations" / "precomputed" /
        ("rfr_feature_selection_smoke.sqlite" if args.mode == "smoke"
         else "rfr_feature_selection_precomputed.sqlite")
    )
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    fits = estimate_fit_count(config)
    print(f"Mode: {config.mode}")
    print("Estimated RandomForestRegressor fits:")
    for name, count in fits.items():
        print(f"  {name:30s} {count:,}")
    print(f"Ensemble size: {config.n_estimators} trees")
    print(f"RF n_jobs: {N_JOBS}; permutation n_jobs: {PERMUTATION_N_JOBS}")
    if config.mode == FULL_PRECOMPUTE and not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("FULL_PRECOMPUTE requires --yes in non-interactive use")
        answer = input(f"Proceed with approximately {fits['total']:,} fits? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Cancelled")

    df, compositions, X, y, target, excluded, missing = load_dataset(dataset_path)
    print(
        f"Dataset: {len(df):,} observations x {len(df.columns):,} columns | "
        f"target={target!r} | model features={X.shape[1]} | composition excluded"
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    digest = config_hash(config, dataset_sha256)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        initialize_database(
            connection, config, digest, dataset_sha256, dataset_path, df,
            compositions, X, y, target, excluded, missing, args.overwrite,
        )
        get_splits = build_split_cache(X)
        stages = (
            (store_fold_assignments, (get_splits,)),
            (store_cluster_assignments, (X, get_splits)),
            (run_convergence, (X, y, get_splits)),
            (run_generalizability, (X, y, get_splits)),
            (run_secondary_sensitivity, (X, y, get_splits)),
            (run_generalizability_summaries, ()),
            (run_robustness, (X, y, get_splits)),
            (run_selection_stability, ()),
            (run_prediction_stability, (X, y, get_splits)),
        )
        for function, stage_args in stages:
            print(f"Running {function.__name__} ...", flush=True)
            function(connection, config, digest, *stage_args)
        counts = verify_database(
            connection, config, digest, dataset_sha256, len(y), X.shape[1]
        )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    print(f"Database complete: {database_path}")
    print("Verified row counts:")
    for table, count in counts.items():
        print(f"  {table:38s} {count:,}")


if __name__ == "__main__":
    main()
