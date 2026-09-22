"""Precompute the fold-fit prediction envelopes used by Section 11."""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCHEMA_VERSION = 2
TRUE_COEFFICIENTS = np.array([1.0, 1.5, 1.0, -1.0, 4.0])
DATA_X_RANGE = (-3.0, 2.2)
PREDICTION_X_RANGE = (-5.0, 5.0)
N_POINTS = 60
DATA_GENERATION_SEED = 2024
OBSERVATION_NOISE_STD = 1.0

DEGREES = np.array([4, 5])
CV_TYPES = ("k-fold", "leave-one-cluster-out")
SPLIT_COUNTS = np.arange(2, 13)
FIRST_SEED = 420
N_SEEDS = 100
LAMBDA_VALUE = 1e-2
LASSO_MAX_ITER = 1_000_000
LASSO_TOL = 1e-5
N_CURVE_POINTS = 400

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "lasso_cv_stability_results.npz"
)


def generate_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(DATA_GENERATION_SEED)
    x = rng.uniform(DATA_X_RANGE[0], DATA_X_RANGE[1], size=N_POINTS)
    y = np.polyval(TRUE_COEFFICIENTS, x)
    y = y + rng.normal(0.0, OBSERVATION_NOISE_STD, size=N_POINTS)
    order = np.argsort(x)
    return x[order], y[order]


def selected_splits(
    cv_type: str, x: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if cv_type == "k-fold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(x))

    labels = KMeans(
        n_clusters=n_splits, random_state=seed, n_init=10
    ).fit_predict(x.reshape(-1, 1))
    return list(LeaveOneGroupOut().split(x, groups=labels))


def fit_fold_curve(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    x_curve: np.ndarray,
    degree: int,
) -> np.ndarray:
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(
        polynomial.fit_transform(x[train_indices].reshape(-1, 1))
    )
    x_curve_scaled = scaler.transform(
        polynomial.transform(x_curve.reshape(-1, 1))
    )
    y_train = y[train_indices]
    _, coefficient_path, _ = lasso_path(
        x_train,
        y_train - y_train.mean(),
        alphas=[LAMBDA_VALUE],
        max_iter=LASSO_MAX_ITER,
        tol=LASSO_TOL,
    )
    return x_curve_scaled @ coefficient_path[:, 0] + y_train.mean()


def compute_split_count(n_splits: int) -> tuple[int, dict[str, np.ndarray]]:
    x, y = generate_dataset()
    x_curve = np.linspace(*PREDICTION_X_RANGE, N_CURVE_POINTS)
    result_shape = (len(DEGREES), len(CV_TYPES), len(x_curve))
    prediction_mean = np.empty(result_shape)
    prediction_median = np.empty(result_shape)
    prediction_lower = np.empty(result_shape)
    prediction_upper = np.empty(result_shape)

    split_sets = {
        cv_type: [
            selected_splits(cv_type, x, n_splits, seed)
            for seed in range(FIRST_SEED, FIRST_SEED + N_SEEDS)
        ]
        for cv_type in CV_TYPES
    }

    for degree_index, degree_value in enumerate(DEGREES):
        degree = int(degree_value)
        for cv_index, cv_type in enumerate(CV_TYPES):
            fold_curves = []
            curve_cache = {}
            for splits in split_sets[cv_type]:
                for train_indices, validation_indices in splits:
                    fold_signature = tuple(validation_indices.tolist())
                    if fold_signature not in curve_cache:
                        curve_cache[fold_signature] = fit_fold_curve(
                            x, y, train_indices, x_curve, degree
                        )
                    fold_curves.append(curve_cache[fold_signature])

            fold_curves = np.stack(fold_curves)
            expected_count = N_SEEDS * n_splits
            if fold_curves.shape != (expected_count, N_CURVE_POINTS):
                raise RuntimeError(
                    f"Expected {(expected_count, N_CURVE_POINTS)}, "
                    f"received {fold_curves.shape}"
                )
            index = (degree_index, cv_index)
            prediction_mean[index] = fold_curves.mean(axis=0)
            prediction_median[index] = np.median(fold_curves, axis=0)
            prediction_lower[index] = np.percentile(
                fold_curves, 2.5, axis=0
            )
            prediction_upper[index] = np.percentile(
                fold_curves, 97.5, axis=0
            )

    return n_splits, {
        "prediction_mean": prediction_mean,
        "prediction_median": prediction_median,
        "prediction_lower": prediction_lower,
        "prediction_upper": prediction_upper,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jobs", type=int, default=min(4, os.cpu_count() or 1),
        help="Number of split-count configurations to compute in parallel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    warnings.simplefilter("error", ConvergenceWarning)

    completed = {}
    if args.jobs == 1:
        for n_splits in SPLIT_COUNTS:
            split_count, result = compute_split_count(int(n_splits))
            completed[split_count] = result
            print(f"completed {split_count} folds/clusters", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(compute_split_count, int(n_splits)): int(n_splits)
                for n_splits in SPLIT_COUNTS
            }
            for future in as_completed(futures):
                split_count, result = future.result()
                completed[split_count] = result
                print(f"completed {split_count} folds/clusters", flush=True)

    ordered_results = [completed[int(value)] for value in SPLIT_COUNTS]
    summary_names = (
        "prediction_mean", "prediction_median",
        "prediction_lower", "prediction_upper",
    )
    summaries = {
        name: np.stack([result[name] for result in ordered_results])
        for name in summary_names
    }
    x, y = generate_dataset()
    x_curve = np.linspace(*PREDICTION_X_RANGE, N_CURVE_POINTS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_output,
        schema_version=np.array(SCHEMA_VERSION),
        split_counts=SPLIT_COUNTS,
        degrees=DEGREES,
        cv_types=np.asarray(CV_TYPES),
        first_seed=np.array(FIRST_SEED),
        n_seeds=np.array(N_SEEDS),
        lambda_value=np.array(LAMBDA_VALUE),
        observation_noise_std=np.array(OBSERVATION_NOISE_STD),
        data_x_range=np.asarray(DATA_X_RANGE),
        prediction_x_range=np.asarray(PREDICTION_X_RANGE),
        interval_percentiles=np.array([2.5, 97.5]),
        fold_fit_counts=N_SEEDS * SPLIT_COUNTS,
        x_observed=x,
        y_observed=y,
        x_curve=x_curve,
        y_ground_truth=np.polyval(TRUE_COEFFICIENTS, x_curve),
        **summaries,
    )
    temporary_output.replace(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
