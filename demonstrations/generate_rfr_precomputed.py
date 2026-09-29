#!/usr/bin/env python3
"""Precompute Sections 4--10 of the Random Forest teaching notebook.

The expensive experiment suite is stored as scientific outputs in SQLite; no
fitted sklearn estimators are serialized.  The safe default is SMOKE_TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

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


# ---------------------------------------------------------------------------
# All computational choices live here.  SMOKE_TEST checks the complete data
# flow quickly; FULL_PRECOMPUTE creates the intended teaching database.
# ---------------------------------------------------------------------------
SMOKE_TEST = "SMOKE_TEST"
FULL_PRECOMPUTE = "FULL_PRECOMPUTE"
SCHEMA_VERSION = 3
RF_SEED = 2024
CLUSTER_SEED = 420
N_JOBS = -1  # experiment loops are serial; parallelism occurs inside one RF
PERMUTATION_N_JOBS = 1  # prevents nested parallelism


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    fixed_n_estimators: int
    convergence_estimators: tuple[int, ...]
    convergence_seeds: tuple[int, ...]
    selection_depths: tuple[int | None, ...]
    selection_outer_seeds: tuple[int, ...]
    selection_outer_folds: int
    selection_inner_folds: int
    validation_seeds: tuple[int, ...]
    kfold_counts: tuple[int, ...]
    loco_counts: tuple[int, ...]
    hyperparameter_seeds: tuple[int, ...]
    hyper_kfold_counts: tuple[int, ...]
    hyper_loco_counts: tuple[int, ...]
    max_depth_grid: tuple[int | None, ...]
    min_samples_leaf_grid: tuple[int, ...]
    min_samples_split_grid: tuple[int, ...]
    max_features_grid: tuple[float, ...]
    representative_params: tuple[int | None, int, int, float]
    importance_cv_seed: int
    importance_fold_count: int
    permutation_repeats: int
    feature_noise_levels: tuple[float, ...]
    target_noise_levels: tuple[float, ...]
    robustness_seeds: tuple[int, ...]
    robustness_fold_count: int
    selection_leaf_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    selection_split_sizes: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


SMOKE_CONFIG = ExperimentConfig(
    mode=SMOKE_TEST,
    fixed_n_estimators=12,
    convergence_estimators=(4, 8, 12),
    convergence_seeds=(0,),
    selection_depths=(6, None),
    selection_outer_seeds=(0,),
    selection_outer_folds=3,
    selection_inner_folds=2,
    validation_seeds=(0, 1),
    kfold_counts=(3, 4),
    loco_counts=(3, 4),
    hyperparameter_seeds=(0,),
    hyper_kfold_counts=(3, 4),
    hyper_loco_counts=(3, 4),
    max_depth_grid=(8, None),
    min_samples_leaf_grid=(1, 4),
    min_samples_split_grid=(2, 8),
    max_features_grid=(0.5, 1.0),
    representative_params=(None, 1, 2, 0.5),
    importance_cv_seed=0,
    importance_fold_count=3,
    permutation_repeats=2,
    feature_noise_levels=(0.0, 0.10),
    target_noise_levels=(0.0, 0.10),
    robustness_seeds=(0,),
    robustness_fold_count=3,
    selection_leaf_sizes=(1, 4),
    selection_split_sizes=(2, 8),
)

FULL_CONFIG = ExperimentConfig(
    mode=FULL_PRECOMPUTE,
    fixed_n_estimators=120,
    convergence_estimators=(10, 25, 50, 100, 120, 200),
    convergence_seeds=(0, 1, 2),
    selection_depths=(4, 8, 16, None),
    selection_outer_seeds=tuple(range(5)),
    selection_outer_folds=5,
    selection_inner_folds=3,
    validation_seeds=tuple(range(30)),
    kfold_counts=(5, 10),
    loco_counts=(5, 10),
    hyperparameter_seeds=tuple(range(5)),
    hyper_kfold_counts=(5, 10),
    hyper_loco_counts=(5, 10),
    max_depth_grid=(8, 16, None),
    min_samples_leaf_grid=(1, 2, 4),
    min_samples_split_grid=(2, 4, 8),
    max_features_grid=(0.5, 0.75, 1.0),
    representative_params=(16, 1, 2, 0.5),
    importance_cv_seed=0,
    importance_fold_count=5,
    permutation_repeats=8,
    feature_noise_levels=(0.0, 0.05, 0.10, 0.20),
    target_noise_levels=(0.0, 0.10, 0.20),
    robustness_seeds=tuple(range(5)),
    robustness_fold_count=5,
)


def find_repository_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "demonstrations" / "lasso_cv.ipynb").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root")


def identify_target(df: pd.DataFrame) -> str:
    """Identify the one experimental band-gap column without relying on position."""
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
    excluded: list[tuple[str, str, str]] = [
        ("composition", "metadata", "identifier/domain context; never a model input"),
        (target, "target", "experimental response being predicted"),
    ]
    numeric_candidates = [
        c for c in df.columns
        if c not in {"composition", target} and pd.api.types.is_numeric_dtype(df[c])
    ]
    nonnumeric = [
        c for c in df.columns
        if c not in {"composition", target} and not pd.api.types.is_numeric_dtype(df[c])
    ]
    for name in nonnumeric:
        excluded.append((name, "nonnumeric", "not a raw numeric model feature"))
    missing_before = int(df[numeric_candidates].isna().sum().sum())
    if missing_before:
        missing_columns = [
            name for name in numeric_candidates if df[name].isna().any()
        ]
        raise ValueError(
            "Numeric feature values are missing in columns "
            f"{missing_columns}. This generator intentionally refuses global "
            "imputation because medians calculated before cross-validation would "
            "leak validation information. Add fold-safe training-median imputation "
            "to every fitting workflow before using a dataset with missing values."
        )
    constant = [c for c in numeric_candidates if df[c].nunique(dropna=True) <= 1]
    for name in constant:
        excluded.append((name, "unusable", "constant numeric column"))
    features = [c for c in numeric_candidates if c not in set(constant)]
    X = df[features].copy()
    if df[target].isna().any():
        raise ValueError(
            f"Target column {target!r} contains missing values; target cleaning must "
            "be resolved explicitly before modeling."
        )
    compositions = df["composition"].astype(str).copy()
    y = df[target].astype(float).copy()
    assert "composition" not in X.columns
    assert target not in X.columns
    assert list(X.columns) == features
    return df, compositions, X, y, target, excluded, missing_before


def depth_to_db(value: int | None) -> int:
    return -1 if value is None else int(value)


def depth_from_db(value: int) -> int | None:
    return None if int(value) == -1 else int(value)


def metrics(y_true, y_pred) -> tuple[float, float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred) ** 0.5),
    )


def make_forest(config: ExperimentConfig, *, n_estimators=None, max_depth=None,
                min_samples_leaf=1, min_samples_split=2, max_features=1.0):
    return RandomForestRegressor(
        n_estimators=n_estimators or config.fixed_n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=RF_SEED,
        n_jobs=N_JOBS,
    )


def validate_param_tuple(max_depth, min_samples_leaf, min_samples_split, max_features):
    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be positive or None")
    if min_samples_leaf < 1 or min_samples_split < 2:
        raise ValueError("Invalid sample-count hyperparameter")
    if not 0 < max_features <= 1:
        raise ValueError("max_features must be in (0, 1]")


def build_split_cache(X: pd.DataFrame):
    scaled = StandardScaler().fit_transform(X)
    assert scaled.shape[1] == X.shape[1]
    labels: dict[tuple[int, int], np.ndarray] = {}

    def get_splits(method: str, count: int, seed: int):
        if method == "kfold":
            splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
            result = list(splitter.split(X))
        elif method == "loco":
            key = (count, seed)
            if key not in labels:
                labels[key] = KMeans(
                    n_clusters=count, random_state=seed, n_init=10
                ).fit_predict(scaled)
                if len(np.unique(labels[key])) != count:
                    raise RuntimeError("KMeans produced an empty cluster")
            result = []
            for cluster_id in range(count):
                validation = np.flatnonzero(labels[key] == cluster_id)
                training = np.flatnonzero(labels[key] != cluster_id)
                result.append((training, validation))
        else:
            raise ValueError(f"Unknown CV method: {method}")
        coverage = np.zeros(len(X), dtype=int)
        for train, validation in result:
            if np.intersect1d(train, validation).size:
                raise RuntimeError("Training and validation indices overlap")
            coverage[validation] += 1
        if len(result) != count or not np.all(coverage == 1):
            raise RuntimeError("Invalid out-of-fold coverage")
        return result

    return scaled, get_splits


def oof_predictions(config, X, y, splits, params, collect_importance=False,
                    permutation_repeats=0):
    max_depth, min_leaf, min_split, max_features = params
    validate_param_tuple(*params)
    predictions = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    importance_rows = []
    for fold, (train, validation) in enumerate(splits):
        forest = make_forest(
            config, max_depth=max_depth, min_samples_leaf=min_leaf,
            min_samples_split=min_split, max_features=max_features,
        )
        forest.fit(X.iloc[train], y.iloc[train])
        predictions[validation] = forest.predict(X.iloc[validation])
        fold_ids[validation] = fold
        if collect_importance:
            for feature, value in zip(X.columns, forest.feature_importances_):
                importance_rows.append((fold, feature, "impurity", float(value), 0.0))
            if permutation_repeats:
                perm = permutation_importance(
                    forest, X.iloc[validation], y.iloc[validation],
                    scoring="r2", n_repeats=permutation_repeats,
                    random_state=RF_SEED + fold, n_jobs=PERMUTATION_N_JOBS,
                )
                for feature, mean, std in zip(
                    X.columns, perm.importances_mean, perm.importances_std
                ):
                    importance_rows.append(
                        (fold, feature, "permutation", float(mean), float(std))
                    )
    if not np.all(np.isfinite(predictions)) or not np.all(fold_ids >= 0):
        raise RuntimeError("Every observation must receive one finite OOF prediction")
    return predictions, fold_ids, importance_rows


SCHEMA_SQL = """
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
    cluster_count INTEGER NOT NULL, observation_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL, pc1 REAL NOT NULL, pc2 REAL NOT NULL,
    PRIMARY KEY (cluster_count, observation_id)
);
CREATE TABLE IF NOT EXISTS convergence_results (
    cv_seed INTEGER NOT NULL, rf_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    n_estimators INTEGER NOT NULL, r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    PRIMARY KEY (cv_seed, n_estimators)
);
CREATE TABLE IF NOT EXISTS model_selection_summary (
    workflow TEXT NOT NULL, outer_seed INTEGER NOT NULL, rf_seed INTEGER NOT NULL,
    fold_count INTEGER NOT NULL, selected_max_depth TEXT NOT NULL,
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    PRIMARY KEY (workflow, outer_seed)
);
CREATE TABLE IF NOT EXISTS model_selection_depth_choices (
    workflow TEXT NOT NULL, outer_seed INTEGER NOT NULL, outer_fold INTEGER NOT NULL,
    selected_max_depth INTEGER NOT NULL,
    PRIMARY KEY (workflow, outer_seed, outer_fold)
);
CREATE TABLE IF NOT EXISTS model_selection_predictions (
    workflow TEXT NOT NULL, outer_seed INTEGER NOT NULL, fold INTEGER NOT NULL,
    observation_id INTEGER NOT NULL, composition TEXT NOT NULL,
    y_true REAL NOT NULL, y_pred REAL NOT NULL,
    PRIMARY KEY (workflow, outer_seed, observation_id)
);
CREATE TABLE IF NOT EXISTS validation_results (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT, cv_method TEXT NOT NULL,
    cv_seed INTEGER NOT NULL, rf_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    max_depth INTEGER NOT NULL, min_samples_leaf INTEGER NOT NULL,
    min_samples_split INTEGER NOT NULL, max_features REAL NOT NULL,
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    UNIQUE (cv_method, cv_seed, fold_count)
);
CREATE TABLE IF NOT EXISTS validation_predictions (
    experiment_id INTEGER NOT NULL, observation_id INTEGER NOT NULL, fold INTEGER NOT NULL,
    composition TEXT NOT NULL, y_true REAL NOT NULL, y_pred REAL NOT NULL,
    PRIMARY KEY (experiment_id, observation_id)
);
CREATE TABLE IF NOT EXISTS hyperparameter_results (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT, cv_method TEXT NOT NULL,
    cv_seed INTEGER NOT NULL, rf_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    max_depth INTEGER NOT NULL, min_samples_leaf INTEGER NOT NULL,
    min_samples_split INTEGER NOT NULL, max_features REAL NOT NULL,
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    UNIQUE (cv_method, cv_seed, fold_count, max_depth, min_samples_leaf,
            min_samples_split, max_features)
);
CREATE TABLE IF NOT EXISTS importance_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, cv_method TEXT NOT NULL,
    cv_seed INTEGER NOT NULL, rf_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    max_depth INTEGER NOT NULL, min_samples_leaf INTEGER NOT NULL,
    min_samples_split INTEGER NOT NULL, max_features REAL NOT NULL,
    UNIQUE (cv_method, cv_seed, fold_count)
);
CREATE TABLE IF NOT EXISTS feature_importances (
    run_id INTEGER NOT NULL, fold INTEGER NOT NULL, feature_name TEXT NOT NULL,
    method TEXT NOT NULL, importance REAL NOT NULL, importance_std REAL NOT NULL,
    PRIMARY KEY (run_id, fold, feature_name, method)
);
CREATE TABLE IF NOT EXISTS robustness_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT, cv_method TEXT NOT NULL,
    cv_seed INTEGER NOT NULL, rf_seed INTEGER NOT NULL, fold_count INTEGER NOT NULL,
    feature_noise REAL NOT NULL, target_noise REAL NOT NULL,
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    UNIQUE (cv_method, cv_seed, fold_count, feature_noise, target_noise)
);
CREATE TABLE IF NOT EXISTS robustness_importances (
    result_id INTEGER NOT NULL, feature_name TEXT NOT NULL, importance REAL NOT NULL,
    PRIMARY KEY (result_id, feature_name)
);
CREATE TABLE IF NOT EXISTS robustness_predictions (
    result_id INTEGER NOT NULL, observation_id INTEGER NOT NULL, composition TEXT NOT NULL,
    y_true_clean REAL NOT NULL, y_pred REAL NOT NULL,
    PRIMARY KEY (result_id, observation_id)
);
CREATE INDEX IF NOT EXISTS idx_hyper_view ON hyperparameter_results
    (cv_method, fold_count, max_depth, min_samples_leaf, min_samples_split, max_features);
CREATE INDEX IF NOT EXISTS idx_validation_observation ON validation_predictions
    (observation_id, experiment_id);
CREATE INDEX IF NOT EXISTS idx_robust_view ON robustness_results
    (cv_method, fold_count, feature_noise, target_noise);
"""


APPEND_SCHEMA = """
CREATE TABLE IF NOT EXISTS generation_history (
    digest TEXT PRIMARY KEY, config TEXT NOT NULL, runtime TEXT NOT NULL, started_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_provenance (
    table_name TEXT NOT NULL, row_key TEXT NOT NULL, digest TEXT NOT NULL,
    PRIMARY KEY (table_name, row_key)
);
CREATE TABLE IF NOT EXISTS selection_experiments (
    experiment TEXT PRIMARY KEY, parameter TEXT NOT NULL, candidates TEXT NOT NULL,
    outer_folds INTEGER NOT NULL, inner_folds INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS selection_scores (
    experiment TEXT NOT NULL, workflow TEXT NOT NULL, outer_seed INTEGER NOT NULL,
    r2 REAL NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
    PRIMARY KEY (experiment, workflow, outer_seed)
);
CREATE TABLE IF NOT EXISTS selection_choices (
    experiment TEXT NOT NULL, workflow TEXT NOT NULL, outer_seed INTEGER NOT NULL,
    outer_fold INTEGER NOT NULL, selected_value TEXT NOT NULL,
    PRIMARY KEY (experiment, workflow, outer_seed, outer_fold)
);
CREATE TABLE IF NOT EXISTS selection_predictions (
    experiment TEXT NOT NULL, workflow TEXT NOT NULL, outer_seed INTEGER NOT NULL,
    observation_id INTEGER NOT NULL, fold INTEGER NOT NULL, y_true REAL NOT NULL, y_pred REAL NOT NULL,
    PRIMARY KEY (experiment, workflow, outer_seed, observation_id)
);
"""


def exists(connection, table, **keys):
    where = " AND ".join(f"{key}=?" for key in keys)
    return connection.execute(f"SELECT 1 FROM {table} WHERE {where} LIMIT 1", tuple(keys.values())).fetchone() is not None


def record_generation(connection, config, digest):
    runtime = dict(python=sys.version.split()[0], numpy=np.__version__,
                   pandas=pd.__version__, sklearn=sklearn.__version__,
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with connection:
        connection.execute("INSERT OR IGNORE INTO generation_history VALUES (?, ?, ?, ?)",
                           (digest, config_json(config), json.dumps(runtime), datetime.now(timezone.utc).isoformat()))
    # Triggers attach provenance in the same transaction as each result insertion.
    natural_keys = {
        "cluster_assignments": ("cluster_count", "observation_id"),
        "convergence_results": ("cv_seed", "n_estimators"),
        "validation_results": ("experiment_id",),
        "hyperparameter_results": ("experiment_id",),
        "importance_runs": ("run_id",),
        "robustness_results": ("result_id",),
        "selection_scores": ("experiment", "workflow", "outer_seed"),
    }
    for table, columns in natural_keys.items():
        key = "json_array(" + ", ".join(f"NEW.{column}" for column in columns) + ")"
        connection.execute(f"DROP TRIGGER IF EXISTS provenance_{table}")
        connection.execute(f"""CREATE TEMP TRIGGER provenance_{table} AFTER INSERT ON main.{table}
            BEGIN INSERT INTO experiment_provenance VALUES ('{table}', {key}, '{digest}'); END""")
    connection.commit()


def config_json(config: ExperimentConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def config_hash(config: ExperimentConfig, dataset_sha256: str) -> str:
    """Digest the dataset, experiment choices, schema, and software runtime."""
    payload = (
        f"schema={SCHEMA_VERSION};dataset_sha256={dataset_sha256};"
        f"config={config_json(config)};python={sys.version.split()[0]};"
        f"numpy={np.__version__};pandas={pd.__version__};"
        f"sklearn={sklearn.__version__};generator={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def mark_stage(connection, stage, digest):
    connection.execute(
        "INSERT OR REPLACE INTO stage_status VALUES (?, ?, ?)",
        (stage, digest, datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()


def estimate_fit_count(config: ExperimentConfig) -> dict[str, int]:
    combinations = (
        len(config.max_depth_grid) * len(config.min_samples_leaf_grid)
        * len(config.min_samples_split_grid) * len(config.max_features_grid)
    )
    counts = {
        "ensemble_convergence": len(config.convergence_seeds)
        * len(config.convergence_estimators) * config.selection_outer_folds,
        "selection_non_nested": len(config.selection_outer_seeds)
        * sum(map(len, selection_grids(config).values())) * config.selection_outer_folds,
        "selection_nested": len(config.selection_outer_seeds)
        * config.selection_outer_folds
        * (sum(map(len, selection_grids(config).values())) * config.selection_inner_folds + 3),
        "validation_seeds": len(config.validation_seeds)
        * (sum(config.kfold_counts) + sum(config.loco_counts)),
        "hyperparameter_landscape": combinations
        * len(config.hyperparameter_seeds)
        * (sum(config.hyper_kfold_counts) + sum(config.hyper_loco_counts)),
        "explanations": 2 * config.importance_fold_count,
        "robustness": len(config.feature_noise_levels)
        * len(config.target_noise_levels) * len(config.robustness_seeds)
        * 2 * config.robustness_fold_count,
    }
    counts["total"] = sum(counts.values())
    return counts


def initialize_database(connection, config, digest, dataset_sha256, dataset_path,
                        df, compositions, X, y, target, excluded, missing_before,
                        overwrite):
    connection.executescript(SCHEMA_SQL)
    connection.executescript(APPEND_SCHEMA)
    existing = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }
    if existing and not overwrite:
        invariant_fields = (
            "fixed_n_estimators", "selection_outer_folds", "selection_inner_folds",
            "representative_params", "permutation_repeats",
        )
        changed = [key for key in invariant_fields
                   if json.dumps(existing["config"][key]) != json.dumps(asdict(config)[key])]
        if (existing.get("schema_version") != SCHEMA_VERSION
                or existing.get("dataset_sha256") != dataset_sha256
                or existing.get("rf_seed") != RF_SEED
                or existing.get("cluster_seed") != CLUSTER_SEED or changed):
            raise ValueError(f"Incompatible base experiment settings: {changed}. Use a separate --database.")
        migrate_depth_results(connection, existing)
        record_generation(connection, config, digest)
        return  # Preserve original provenance and observations; requests live in generation_history.
    if overwrite:
        for table in (
            "selection_experiments", "selection_scores", "selection_choices",
            "selection_predictions", "generation_history", "experiment_provenance",
            "stage_status", "observations", "features", "excluded_columns",
            "cluster_assignments", "convergence_results", "model_selection_depth_choices",
            "model_selection_summary",
            "model_selection_predictions", "validation_predictions", "validation_results",
            "hyperparameter_results", "feature_importances", "importance_runs",
            "robustness_importances", "robustness_predictions", "robustness_results",
        ):
            connection.execute(f"DELETE FROM {table}")
        connection.execute("DELETE FROM metadata")
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
        "missing_numeric_values": missing_before,
        "missing_value_policy": (
            "fail fast; no global imputation. Any future imputation must be fitted "
            "inside each training fold and applied to its validation fold."
        ),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rf_seed": RF_SEED,
        "cluster_seed": CLUSTER_SEED,
        "validation_cv_seeds": list(config.validation_seeds),
        "representative_hyperparameters": {
            "max_depth": config.representative_params[0],
            "min_samples_leaf": config.representative_params[1],
            "min_samples_split": config.representative_params[2],
            "max_features": config.representative_params[3],
            "n_estimators": config.fixed_n_estimators,
        },
        "inner_cv_seed_rule": "10000 + outer_seed + outer_fold",
        "perturbation_seed_rule": (
            "1000000 + 100000000*method_code + 10000*cv_seed + 100*fold; "
            "method_code: kfold=0, loco=1"
        ),
        "perturbation_reuse_rule": (
            "Within each (CV method, CV seed, outer fold), draw one standard-normal "
            "feature matrix Z_X and one target vector Z_y, then reuse them at every "
            "feature_noise x target_noise magnitude. Scales come only from that "
            "training fold; validation X and y remain clean."
        ),
        "n_jobs": N_JOBS,
        "config": asdict(config),
        "fit_estimate": estimate_fit_count(config),
        "experiment_definitions": {
            "convergence": "OOF shuffled K-fold scores versus n_estimators",
            "selection": "same-evidence depth selection versus nested depth selection",
            "validation": (
                "OOF metrics and observation-level predictions for every CV "
                "partition seed, with the RF seed fixed"
            ),
            "hyperparameters": "four-dimensional cached validation landscape",
            "importance": "fold-level impurity and held-out permutation importance",
            "robustness": (
                "training-fold-scaled feature and target noise; one standardized "
                "direction reused across all magnitudes within each method/seed/fold"
            ),
        },
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
    for i, name in enumerate(X.columns):
        series = X[name]
        feature_rows.append((
            i, name, int(df[name].isna().sum()), float(series.min()),
            float(series.max()), float(series.mean()), float(series.std(ddof=0)),
        ))
    connection.executemany(
        "INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?, ?, ?, ?)", feature_rows
    )
    connection.executemany(
        "INSERT OR REPLACE INTO excluded_columns VALUES (?, ?, ?)", excluded
    )
    connection.commit()
    record_generation(connection, config, digest)


def run_clusters(connection, config, digest, X, scaled):
    stage = "clusters"
    counts = sorted(set(config.loco_counts + config.hyper_loco_counts +
                        (config.importance_fold_count, config.robustness_fold_count)))
    projection = PCA(n_components=2, random_state=CLUSTER_SEED).fit_transform(scaled)
    for count in counts:
        if exists(connection, "cluster_assignments", cluster_count=count):
            continue
        labels = KMeans(n_clusters=count, random_state=CLUSTER_SEED, n_init=10).fit_predict(scaled)
        if len(np.unique(labels)) != count:
            raise RuntimeError("KMeans cluster occupancy check failed")
        rows = [
            (count, i, int(labels[i]), float(projection[i, 0]), float(projection[i, 1]))
            for i in range(len(X))
        ]
        with connection:
            connection.executemany(
                "INSERT INTO cluster_assignments VALUES (?, ?, ?, ?, ?)", rows
            )
    mark_stage(connection, stage, digest)


def run_convergence(connection, config, digest, X, y, get_splits):
    stage = "convergence"
    for seed in config.convergence_seeds:
        splits = get_splits("kfold", config.selection_outer_folds, seed)
        for trees in config.convergence_estimators:
            if exists(connection, "convergence_results", cv_seed=seed, n_estimators=trees):
                continue
            pred = np.full(len(y), np.nan)
            for train, validation in splits:
                forest = make_forest(config, n_estimators=trees, max_features=0.5)
                forest.fit(X.iloc[train], y.iloc[train])
                pred[validation] = forest.predict(X.iloc[validation])
            score = metrics(y, pred)
            with connection:
                connection.execute(
                    "INSERT INTO convergence_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (seed, RF_SEED, len(splits), trees, *score),
                )
    mark_stage(connection, stage, digest)


def selection_id(parameter, candidates, config):
    return hashlib.sha256(json.dumps([
        parameter, list(candidates), config.fixed_n_estimators,
        config.selection_outer_folds, config.selection_inner_folds, RF_SEED,
    ]).encode()).hexdigest()


def migrate_depth_results(connection, metadata):
    """Register legacy depth results without changing any original table or refitting."""
    saved = metadata["config"]
    config = replace(FULL_CONFIG, **{key: tuple(value) if isinstance(value, list) else value
                                   for key, value in saved.items()})
    experiment = selection_id("max_depth", config.selection_depths, config)
    if exists(connection, "selection_experiments", experiment=experiment):
        return
    if not connection.execute("SELECT 1 FROM model_selection_summary LIMIT 1").fetchone():
        return
    with connection:
        connection.execute("INSERT INTO selection_experiments VALUES (?, ?, ?, ?, ?)",
                           (experiment, "max_depth", json.dumps(list(config.selection_depths)),
                            config.selection_outer_folds, config.selection_inner_folds))
        for workflow, seed, r2, mae, rmse in connection.execute(
                "SELECT workflow, outer_seed, r2, mae, rmse FROM model_selection_summary").fetchall():
            connection.execute("INSERT INTO selection_scores VALUES (?, ?, ?, ?, ?, ?)",
                                        (experiment, workflow, seed, r2, mae, rmse))
            connection.execute("INSERT OR IGNORE INTO experiment_provenance VALUES (?, ?, ?)",
                               ("selection_scores", json.dumps([experiment, workflow, seed], separators=(",", ":")), metadata["config_hash"]))
        connection.execute("""INSERT INTO selection_predictions
            SELECT ?, workflow, outer_seed, observation_id, fold, y_true, y_pred
            FROM model_selection_predictions""", (experiment,))
        for workflow, seed, fold, depth in connection.execute("SELECT * FROM model_selection_depth_choices").fetchall():
            connection.execute("INSERT INTO selection_choices VALUES (?, ?, ?, ?, ?)",
                               (experiment, workflow, seed, fold, json.dumps(depth_from_db(depth))))
        connection.execute("INSERT OR IGNORE INTO generation_history VALUES (?, ?, ?, ?)",
                           (metadata["config_hash"], json.dumps(saved), json.dumps({
                               key: metadata.get(key) for key in
                               ("python_version", "numpy_version", "pandas_version", "sklearn_version")}),
                            metadata.get("generated_utc", "legacy")))


def selection_grids(config):
    return {"max_depth": config.selection_depths,
            "min_samples_leaf": config.selection_leaf_sizes,
            "min_samples_split": config.selection_split_sizes}


def selection_params(parameter, value):
    params = dict(max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=0.5)
    params[parameter] = value
    return tuple(params.values())


def run_model_selection(connection, config, digest, X, y, compositions, get_splits):
    for parameter, candidates in selection_grids(config).items():
        candidate_json = json.dumps(list(candidates))
        # A new candidate set defines a different selection experiment; preserve the old one.
        experiment = selection_id(parameter, candidates, config)
        with connection:
            connection.execute("INSERT OR IGNORE INTO selection_experiments VALUES (?, ?, ?, ?, ?)",
                               (experiment, parameter, candidate_json,
                                config.selection_outer_folds, config.selection_inner_folds))
        for seed in config.selection_outer_seeds:
            if exists(connection, "selection_scores", experiment=experiment, outer_seed=seed, workflow="nested"):
                continue
            print(f"  Selection {parameter}, candidates={candidates}, seed={seed}", flush=True)
            splits = get_splits("kfold", config.selection_outer_folds, seed)
            candidate_predictions = {
                value: oof_predictions(config, X, y, splits, selection_params(parameter, value))[0]
                for value in candidates
            }
            selected = min(candidates, key=lambda value: mean_squared_error(y, candidate_predictions[value]))
            choices = [(experiment, "same_evidence", seed, -1, json.dumps(selected))]
            nested = np.full(len(y), np.nan)
            fold_ids = np.full(len(y), -1, dtype=int)
            for fold, (train, validation) in enumerate(splits):
                inner_X, inner_y = X.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True)
                inner_splits = list(KFold(n_splits=config.selection_inner_folds, shuffle=True,
                                        random_state=10_000 + seed + fold).split(inner_X))
                scores = {
                    value: mean_squared_error(inner_y, oof_predictions(
                        config, inner_X, inner_y, inner_splits, selection_params(parameter, value))[0])
                    for value in candidates
                }
                chosen = min(candidates, key=scores.get)
                choices.append((experiment, "nested", seed, fold, json.dumps(chosen)))
                params = selection_params(parameter, chosen)
                forest = make_forest(config, max_depth=params[0], min_samples_leaf=params[1],
                                     min_samples_split=params[2], max_features=params[3])
                forest.fit(X.iloc[train], y.iloc[train])
                nested[validation] = forest.predict(X.iloc[validation])
                fold_ids[validation] = fold
            # Publish both workflows and their children atomically, once the seed finishes.
            with connection:
                connection.executemany("INSERT INTO selection_choices VALUES (?, ?, ?, ?, ?)", choices)
                for workflow, pred in [("same_evidence", candidate_predictions[selected]), ("nested", nested)]:
                    connection.execute("INSERT INTO selection_scores VALUES (?, ?, ?, ?, ?, ?)",
                                       (experiment, workflow, seed, *metrics(y, pred)))
                    connection.executemany("INSERT INTO selection_predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [(experiment, workflow, seed, i, int(fold_ids[i]), float(y.iloc[i]), float(pred[i]))
                         for i in range(len(y))])
    mark_stage(connection, "model_selection", digest)


def run_validation(connection, config, digest, X, y, compositions, get_splits):
    stage = "validation"
    params = config.representative_params
    for method, counts in (("kfold", config.kfold_counts), ("loco", config.loco_counts)):
        for count in counts:
            for seed in config.validation_seeds:
                if exists(connection, "validation_results", cv_method=method, cv_seed=seed, fold_count=count):
                    continue
                splits = get_splits(method, count, seed)
                pred, folds, _ = oof_predictions(config, X, y, splits, params)
                score = metrics(y, pred)
                with connection:
                    cursor = connection.execute(
                        """INSERT INTO validation_results
                        (cv_method, cv_seed, rf_seed, fold_count, max_depth,
                         min_samples_leaf, min_samples_split, max_features, r2, mae, rmse)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (method, seed, RF_SEED, count, depth_to_db(params[0]),
                         params[1], params[2], params[3], *score),
                    )
                    experiment_id = cursor.lastrowid
                    connection.executemany(
                        "INSERT INTO validation_predictions VALUES (?, ?, ?, ?, ?, ?)",
                        [(experiment_id, i, int(folds[i]), compositions.iloc[i],
                          float(y.iloc[i]), float(pred[i])) for i in range(len(y))],
                    )
    mark_stage(connection, stage, digest)


def run_hyperparameters(connection, config, digest, X, y, get_splits):
    stage = "hyperparameters"
    grid = itertools.product(
        config.max_depth_grid, config.min_samples_leaf_grid,
        config.min_samples_split_grid, config.max_features_grid,
    )
    grid = list(grid)
    for method, counts in (
        ("kfold", config.hyper_kfold_counts), ("loco", config.hyper_loco_counts)
    ):
        for count in counts:
            for seed in config.hyperparameter_seeds:
                splits = get_splits(method, count, seed)
                rows = []
                for params in grid:
                    if exists(connection, "hyperparameter_results", cv_method=method,
                              cv_seed=seed, fold_count=count, max_depth=depth_to_db(params[0]),
                              min_samples_leaf=params[1], min_samples_split=params[2], max_features=params[3]):
                        continue
                    pred, _, _ = oof_predictions(config, X, y, splits, params)
                    rows.append((
                        method, seed, RF_SEED, count, depth_to_db(params[0]),
                        params[1], params[2], params[3], *metrics(y, pred),
                    ))
                with connection:
                    connection.executemany(
                        """INSERT INTO hyperparameter_results
                        (cv_method, cv_seed, rf_seed, fold_count, max_depth,
                         min_samples_leaf, min_samples_split, max_features, r2, mae, rmse)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows,
                    )
    mark_stage(connection, stage, digest)


def run_importance(connection, config, digest, X, y, get_splits):
    stage = "importance"
    params = config.representative_params
    for method in ("kfold", "loco"):
        if exists(connection, "importance_runs", cv_method=method,
                  cv_seed=config.importance_cv_seed, fold_count=config.importance_fold_count):
            continue
        splits = get_splits(method, config.importance_fold_count, config.importance_cv_seed)
        _, _, rows = oof_predictions(
            config, X, y, splits, params, collect_importance=True,
            permutation_repeats=config.permutation_repeats,
        )
        with connection:
            cursor = connection.execute(
                """INSERT INTO importance_runs
                (cv_method, cv_seed, rf_seed, fold_count, max_depth,
                 min_samples_leaf, min_samples_split, max_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (method, config.importance_cv_seed, RF_SEED,
                 config.importance_fold_count, depth_to_db(params[0]),
                 params[1], params[2], params[3]),
            )
            run_id = cursor.lastrowid
            connection.executemany(
                "INSERT INTO feature_importances VALUES (?, ?, ?, ?, ?, ?)",
                [(run_id, *row) for row in rows],
            )
    mark_stage(connection, stage, digest)


def prepare_fold_perturbations(X, y, train, seed):
    """Draw one standardized noise realization and training-fold scales."""
    rng = np.random.default_rng(seed)
    X_train = X.iloc[train].copy()
    y_train = y.iloc[train].copy()
    feature_scale = X_train.std(axis=0, ddof=0).to_numpy()
    active = feature_scale > 0
    z_feature = np.zeros(X_train.shape, dtype=float)
    z_feature[:, active] = rng.normal(size=(len(train), int(active.sum())))
    target_scale = float(y_train.std(ddof=0))
    z_target = rng.normal(size=len(train))
    return {
        "X_train": X_train,
        "y_train": y_train,
        "feature_scale": feature_scale,
        "target_scale": target_scale,
        "z_feature": z_feature,
        "z_target": z_target,
        "active_features": active,
    }


def apply_fold_perturbations(prepared, feature_noise, target_noise):
    """Scale a fixed fold-level direction without drawing new noise."""
    X_noisy = prepared["X_train"].copy()
    feature_delta = (
        float(feature_noise)
        * prepared["z_feature"]
        * prepared["feature_scale"][None, :]
    )
    X_noisy.iloc[:, :] = X_noisy.to_numpy() + feature_delta
    y_noisy = prepared["y_train"] + (
        float(target_noise) * prepared["target_scale"] * prepared["z_target"]
    )
    if not np.all(feature_delta[:, ~prepared["active_features"]] == 0.0):
        raise RuntimeError("Constant-within-fold features were perturbed")
    if feature_noise == 0 and not np.array_equal(
        X_noisy.to_numpy(), prepared["X_train"].to_numpy()
    ):
        raise RuntimeError("Zero feature noise changed the training features")
    if target_noise == 0 and not np.array_equal(
        y_noisy.to_numpy(), prepared["y_train"].to_numpy()
    ):
        raise RuntimeError("Zero target noise changed the training target")
    return X_noisy, y_noisy


def perturbation_seed(cv_method, cv_seed, fold):
    method_code = {"kfold": 0, "loco": 1}[cv_method]
    return 1_000_000 + 100_000_000 * method_code + 10_000 * cv_seed + 100 * fold


def run_robustness(connection, config, digest, X, y, compositions, get_splits):
    stage = "robustness"
    params = config.representative_params
    for method in ("kfold", "loco"):
        for cv_seed in config.robustness_seeds:
            splits = get_splits(method, config.robustness_fold_count, cv_seed)
            fold_perturbations = [
                prepare_fold_perturbations(
                    X, y, train, perturbation_seed(method, cv_seed, fold)
                )
                for fold, (train, _validation) in enumerate(splits)
            ]
            for feature_noise in config.feature_noise_levels:
                for target_noise in config.target_noise_levels:
                    if exists(connection, "robustness_results", cv_method=method, cv_seed=cv_seed,
                              fold_count=config.robustness_fold_count,
                              feature_noise=feature_noise, target_noise=target_noise):
                        continue
                    pred = np.full(len(y), np.nan)
                    importances = []
                    for fold, (train, validation) in enumerate(splits):
                        X_train, y_train = apply_fold_perturbations(
                            fold_perturbations[fold], feature_noise, target_noise
                        )
                        forest = make_forest(
                            config, max_depth=params[0], min_samples_leaf=params[1],
                            min_samples_split=params[2], max_features=params[3],
                        )
                        forest.fit(X_train, y_train)
                        pred[validation] = forest.predict(X.iloc[validation])
                        importances.append(forest.feature_importances_)
                    if not np.all(np.isfinite(pred)):
                        raise RuntimeError("Missing robustness predictions")
                    score = metrics(y, pred)
                    with connection:
                        cursor = connection.execute(
                            """INSERT INTO robustness_results
                            (cv_method, cv_seed, rf_seed, fold_count, feature_noise,
                             target_noise, r2, mae, rmse)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (method, cv_seed, RF_SEED, config.robustness_fold_count,
                             feature_noise, target_noise, *score),
                        )
                        result_id = cursor.lastrowid
                        mean_importance = np.mean(importances, axis=0)
                        connection.executemany(
                            "INSERT INTO robustness_importances VALUES (?, ?, ?)",
                            [(result_id, name, float(value))
                             for name, value in zip(X.columns, mean_importance)],
                        )
                        connection.executemany(
                            "INSERT INTO robustness_predictions VALUES (?, ?, ?, ?, ?)",
                            [(result_id, i, compositions.iloc[i], float(y.iloc[i]),
                              float(pred[i])) for i in range(len(y))],
                        )
    mark_stage(connection, stage, digest)


def verify_database(connection, config, digest, dataset_sha256,
                    n_observations, n_features, studies=None):
    """Validate requested keys and child coverage; supersets are intentionally allowed."""
    studies = set(studies or STUDIES)
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if json.loads(metadata["dataset_sha256"]) != dataset_sha256:
        raise RuntimeError("Dataset hash mismatch")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("SQLite integrity check failed")
    def require(table, **keys):
        if not exists(connection, table, **keys):
            raise RuntimeError(f"Missing {table} experiment: {keys}")
    if "clusters" in studies:
        for count in set(config.loco_counts + config.hyper_loco_counts +
                         (config.importance_fold_count, config.robustness_fold_count)):
            total = connection.execute("SELECT COUNT(*) FROM cluster_assignments WHERE cluster_count=?", (count,)).fetchone()[0]
            if total != n_observations:
                raise RuntimeError(f"Incomplete cluster assignment: {count}")
    if "convergence" in studies:
        for seed, trees in itertools.product(config.convergence_seeds, config.convergence_estimators):
            require("convergence_results", cv_seed=seed, n_estimators=trees)
    if "validation" in studies:
        for method, counts in [("kfold", config.kfold_counts), ("loco", config.loco_counts)]:
            for seed, count in itertools.product(config.validation_seeds, counts):
                require("validation_results", cv_method=method, cv_seed=seed, fold_count=count)
    if "hyperparameters" in studies:
        for method, counts in [("kfold", config.hyper_kfold_counts), ("loco", config.hyper_loco_counts)]:
            for seed, count, depth, leaf, split, features in itertools.product(
                    config.hyperparameter_seeds, counts, config.max_depth_grid,
                    config.min_samples_leaf_grid, config.min_samples_split_grid, config.max_features_grid):
                require("hyperparameter_results", cv_method=method, cv_seed=seed, fold_count=count,
                        max_depth=depth_to_db(depth), min_samples_leaf=leaf,
                        min_samples_split=split, max_features=features)
    if "importance" in studies:
        for method in ("kfold", "loco"):
            require("importance_runs", cv_method=method, cv_seed=config.importance_cv_seed,
                    fold_count=config.importance_fold_count)
    if "robustness" in studies:
        for method, seed, feature, target in itertools.product(
                ("kfold", "loco"), config.robustness_seeds, config.feature_noise_levels, config.target_noise_levels):
            require("robustness_results", cv_method=method, cv_seed=seed,
                    fold_count=config.robustness_fold_count, feature_noise=feature, target_noise=target)
    if "model_selection" in studies:
        for parameter, candidates in selection_grids(config).items():
            experiment = connection.execute(
                "SELECT experiment FROM selection_experiments WHERE parameter=? AND candidates=? AND outer_folds=? AND inner_folds=?",
                (parameter, json.dumps(list(candidates)), config.selection_outer_folds, config.selection_inner_folds),
            ).fetchone()
            if experiment is None:
                raise RuntimeError(f"Missing selection study: {parameter}")
            for seed, workflow in itertools.product(config.selection_outer_seeds, ("nested", "same_evidence")):
                require("selection_scores", experiment=experiment[0], outer_seed=seed, workflow=workflow)
                rows = connection.execute("SELECT COUNT(*) FROM selection_predictions WHERE experiment=? AND outer_seed=? AND workflow=?",
                                          (experiment[0], seed, workflow)).fetchone()[0]
                if rows != n_observations:
                    raise RuntimeError("Incomplete selection predictions")
    # Parent/child records are saved atomically. Validate complete stored runs, not just totals.
    for parent, child, key, expected in [
        ("validation_results", "validation_predictions", "experiment_id", n_observations),
        ("robustness_results", "robustness_importances", "result_id", n_features),
    ]:
        for identifier, count in connection.execute(
                f"SELECT p.{key}, COUNT(c.{key}) FROM {parent} p LEFT JOIN {child} c USING ({key}) GROUP BY p.{key}"):
            if count != expected:
                raise RuntimeError(f"Incomplete {child}: {identifier}")
    return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for (table,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence'").fetchall()}


STUDIES = ("clusters", "convergence", "model_selection", "validation", "hyperparameters", "importance", "robustness")


def validate_config(config):
    for name, value in asdict(config).items():
        if isinstance(value, tuple) and (not value or len(set(value)) != len(value)):
            raise ValueError(f"{name} must be nonempty and contain no duplicates")
    for parameter, values in selection_grids(config).items():
        for value in values:
            validate_param_tuple(*selection_params(parameter, value))
    for params in itertools.product(config.max_depth_grid, config.min_samples_leaf_grid,
                                    config.min_samples_split_grid, config.max_features_grid):
        validate_param_tuple(*params)
    for name in ("feature_noise_levels", "target_noise_levels"):
        values = getattr(config, name)
        if 0 not in values or any(v < 0 for v in values):
            raise ValueError(f"{name} must include zero and contain only nonnegative values")
    for name, value in asdict(config).items():
        values = value if isinstance(value, tuple) else (value,)
        if "fold" in name or name.endswith("counts"):
            if any(not isinstance(v, int) or v < 2 for v in values):
                raise ValueError(f"{name} requires integers >= 2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "full"), default=None,
        help="default: use existing database config, or smoke for a new database",
    )
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="explicitly erase all saved studies")
    parser.add_argument("--config", type=Path, help="JSON object overriding selected ExperimentConfig fields")
    parser.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument("--jobs", type=int, default=-1, help="parallel trees per forest; experiment loop stays serial")
    parser.add_argument(
        "--yes", action="store_true",
        help="required for non-interactive FULL_PRECOMPUTE",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global N_JOBS
    N_JOBS = args.jobs
    if N_JOBS == 0:
        raise ValueError("--jobs cannot be zero")
    root = find_repository_root(Path.cwd())
    dataset_path = root / "demonstrations" / "data" / "featurized_matbench_expt_gap.csv"
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    database_path = args.database or (
        root / "demonstrations" / "precomputed" / "rfr_precomputed_results.sqlite"
    )
    config = FULL_CONFIG if args.mode == "full" else SMOKE_CONFIG
    if args.mode is None and database_path.exists() and not args.overwrite:
        with sqlite3.connect(database_path) as saved:
            row = saved.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
            if row:
                config = FULL_CONFIG if json.loads(row[0])["mode"] == FULL_PRECOMPUTE else SMOKE_CONFIG
                config = replace(config, **{key: tuple(value) if isinstance(value, list) else value
                                           for key, value in json.loads(row[0]).items()})
    if args.config:
        overrides = json.loads(args.config.read_text())
        config = replace(config, **{key: tuple(value) if isinstance(value, list) else value
                                   for key, value in overrides.items()})
    validate_config(config)
    fits = estimate_fit_count(config)
    print(f"Mode: {config.mode}")
    print("Upper-bound fits for all studies before cache reuse (selected studies: " + ", ".join(args.studies) + "):")
    for name, count in fits.items():
        print(f"  {name:28s} {count:,}")
    print(f"Main ensemble size: {config.fixed_n_estimators} trees")
    print(f"RF n_jobs: {N_JOBS}; permutation n_jobs: {PERMUTATION_N_JOBS}")
    if config.mode == FULL_PRECOMPUTE and not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("FULL_PRECOMPUTE requires --yes in non-interactive use")
        answer = input(f"Proceed with {fits['total']:,} fits? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Cancelled")

    df, compositions, X, y, target, excluded, missing_before = load_dataset(dataset_path)
    print(
        f"Dataset: {df.shape[0]:,} observations x {df.shape[1]:,} columns | "
        f"target={target!r} | features={X.shape[1]} | metadata=['composition']"
    )
    print(f"Missing numeric feature values: {missing_before}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    digest = config_hash(config, dataset_sha256)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        initialize_database(
            connection, config, digest, dataset_sha256, dataset_path,
            df, compositions, X, y,
            target, excluded, missing_before, args.overwrite,
        )
        scaled, get_splits = build_split_cache(X)
        stages = (
            (run_clusters, (X, scaled)),
            (run_convergence, (X, y, get_splits)),
            (run_model_selection, (X, y, compositions, get_splits)),
            (run_validation, (X, y, compositions, get_splits)),
            (run_hyperparameters, (X, y, get_splits)),
            (run_importance, (X, y, get_splits)),
            (run_robustness, (X, y, compositions, get_splits)),
        )
        for function, stage_args in stages:
            if function.__name__.removeprefix("run_") not in args.studies:
                continue
            print(f"Running {function.__name__} ...", flush=True)
            function(connection, config, digest, *stage_args)
        counts = verify_database(
            connection, config, digest, dataset_sha256, len(y), X.shape[1], args.studies
        )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    print(f"Database complete: {database_path}")
    print("Row counts:")
    for table, count in counts.items():
        print(f"  {table:30s} {count:,}")


if __name__ == "__main__":
    main()
