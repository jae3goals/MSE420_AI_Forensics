"""Precompute the robustness surfaces used by Sections 10.1 and 10.2.

Run this script from the repository root after intentionally changing the
experiment definition. The notebook only reads the resulting NPZ bundle.
"""

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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCHEMA_VERSION = 4
TRUE_COEFFICIENTS = np.array([1.0, 1.5, 1.0, -1.0, 4.0])
X_RANGE = (-3.0, 2.2)
N_POINTS = 60
DATA_GENERATION_SEED = 2024

DEGREES = np.array([4, 5])
CV_TYPES = ("k-fold", "leave-one-cluster-out")
SPLIT_COUNTS = np.arange(2, 13)
FEATURE_RELATIVE_NOISE_LEVELS = np.round(np.arange(6) * 0.04, 2)
OUTPUT_RELATIVE_NOISE_LEVELS = np.round(np.arange(6) * 0.1, 1)
LAMBDA_VALUE = 1e-2
FIRST_SEED = 420
N_R2_SEEDS = 100
N_PARAMETER_SEEDS = 30
MAX_PARAMETER_SAMPLES = N_PARAMETER_SEEDS * int(SPLIT_COUNTS.max())
NOISE_SEED = 314159
LASSO_MAX_ITER = 1_000_000
LASSO_TOL = 1e-5
N_UNIVERSAL_PARAMETERS = 6  # x, x^2, x^3, x^4, x^5, intercept

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "lasso_cv_robustness_results.npz"
)


def generate_clean_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(DATA_GENERATION_SEED)
    x = rng.uniform(X_RANGE[0], X_RANGE[1], size=N_POINTS)
    y = np.polyval(TRUE_COEFFICIENTS, x)
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


def fit_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    x_column = x.reshape(-1, 1)
    x_train = scaler.fit_transform(
        polynomial.fit_transform(x_column[train_indices])
    )
    x_validation = scaler.transform(
        polynomial.transform(x_column[validation_indices])
    )
    y_train = y[train_indices]
    _, coefficient_path, _ = lasso_path(
        x_train,
        y_train - y_train.mean(),
        alphas=[LAMBDA_VALUE],
        max_iter=LASSO_MAX_ITER,
        tol=LASSO_TOL,
    )
    standardized_coefficients = coefficient_path[:, 0]
    predictions = x_validation @ standardized_coefficients + y_train.mean()

    original_coefficients = standardized_coefficients / scaler.scale_
    original_intercept = (
        y_train.mean()
        - standardized_coefficients @ (scaler.mean_ / scaler.scale_)
    )
    universal_parameters = np.full(N_UNIVERSAL_PARAMETERS, np.nan)
    universal_parameters[:degree] = original_coefficients
    universal_parameters[-1] = original_intercept
    return predictions, universal_parameters


def compute_split_count(n_splits: int) -> tuple[int, dict[str, np.ndarray]]:
    x_clean, y_clean = generate_clean_dataset()
    feature_reference_scale = np.std(x_clean)
    output_reference_scale = np.std(y_clean)
    noise_rng = np.random.default_rng(NOISE_SEED)
    feature_noise = noise_rng.normal(size=N_POINTS)
    output_noise = noise_rng.normal(size=N_POINTS)
    noisy_x_values = [
        x_clean + relative_noise * feature_reference_scale * feature_noise
        for relative_noise in FEATURE_RELATIVE_NOISE_LEVELS
    ]

    split_sets = {}
    for feature_index, x_noisy in enumerate(noisy_x_values):
        for cv_type in CV_TYPES:
            split_sets[(feature_index, cv_type)] = [
                selected_splits(cv_type, x_noisy, n_splits, seed)
                for seed in range(FIRST_SEED, FIRST_SEED + N_R2_SEEDS)
            ]

    grid_shape = (
        len(DEGREES), len(CV_TYPES),
        len(OUTPUT_RELATIVE_NOISE_LEVELS),
        len(FEATURE_RELATIVE_NOISE_LEVELS),
    )
    r2_mean = np.empty(grid_shape)
    r2_std = np.empty(grid_shape)
    r2_values = np.empty(grid_shape + (N_R2_SEEDS,))
    parameter_mean = np.full(grid_shape + (N_UNIVERSAL_PARAMETERS,), np.nan)
    parameter_std = np.full_like(parameter_mean, np.nan)
    parameter_values = np.full(
        grid_shape + (N_UNIVERSAL_PARAMETERS, MAX_PARAMETER_SAMPLES),
        np.nan,
    )

    for output_index, output_relative_noise in enumerate(
        OUTPUT_RELATIVE_NOISE_LEVELS
    ):
        y_noisy = (
            y_clean
            + output_relative_noise * output_reference_scale * output_noise
        )
        for feature_index, x_noisy in enumerate(noisy_x_values):
            for degree_index, degree_value in enumerate(DEGREES):
                degree = int(degree_value)
                for cv_index, cv_type in enumerate(CV_TYPES):
                    seed_scores = []
                    parameter_samples = []
                    fold_fit_cache = {}
                    all_seed_splits = split_sets[(feature_index, cv_type)]

                    for seed_offset, splits in enumerate(all_seed_splits):
                        predictions = np.full(N_POINTS, np.nan)
                        for train_indices, validation_indices in splits:
                            fold_signature = tuple(validation_indices.tolist())
                            if fold_signature not in fold_fit_cache:
                                fold_fit_cache[fold_signature] = fit_fold(
                                    x_noisy,
                                    y_noisy,
                                    train_indices,
                                    validation_indices,
                                    degree,
                                )
                            fold_predictions, fold_parameters = (
                                fold_fit_cache[fold_signature]
                            )
                            predictions[validation_indices] = fold_predictions
                            if seed_offset < N_PARAMETER_SEEDS:
                                parameter_samples.append(fold_parameters)

                        if not np.all(np.isfinite(predictions)):
                            raise RuntimeError("Incomplete out-of-fold predictions")
                        seed_scores.append(r2_score(y_noisy, predictions))

                    parameter_samples_array = np.stack(parameter_samples)
                    index = (
                        degree_index, cv_index, output_index, feature_index
                    )
                    r2_mean[index] = np.mean(seed_scores)
                    r2_std[index] = np.std(seed_scores)
                    r2_values[index] = seed_scores
                    available_parameters = np.any(
                        np.isfinite(parameter_samples_array), axis=0
                    )
                    parameter_mean[index][available_parameters] = np.mean(
                        parameter_samples_array[:, available_parameters], axis=0
                    )
                    parameter_std[index][available_parameters] = np.std(
                        parameter_samples_array[:, available_parameters], axis=0
                    )
                    sample_count = len(parameter_samples_array)
                    parameter_values[index][:, :sample_count] = (
                        parameter_samples_array.T
                    )

    return n_splits, {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "r2_values": r2_values,
        "parameter_mean": parameter_mean,
        "parameter_std": parameter_std,
        "parameter_values": parameter_values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output NPZ path (default: {DEFAULT_OUTPUT})",
    )
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
    output_arrays = {
        name: np.stack([result[name] for result in ordered_results])
        for name in (
            "r2_mean", "r2_std", "r2_values",
            "parameter_mean", "parameter_std", "parameter_values",
        )
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_output,
        schema_version=np.array(SCHEMA_VERSION),
        split_counts=SPLIT_COUNTS,
        degrees=DEGREES,
        cv_types=np.asarray(CV_TYPES),
        feature_relative_noise_levels=FEATURE_RELATIVE_NOISE_LEVELS,
        output_relative_noise_levels=OUTPUT_RELATIVE_NOISE_LEVELS,
        feature_reference_scale=np.array(
            generate_clean_dataset()[0].std()
        ),
        output_reference_scale=np.array(
            generate_clean_dataset()[1].std()
        ),
        noise_scale_definition=np.asarray(
            "relative noise = Gaussian noise standard deviation / "
            "population standard deviation of the clean sampled variable"
        ),
        lambda_value=np.array(LAMBDA_VALUE),
        first_seed=np.array(FIRST_SEED),
        n_r2_seeds=np.array(N_R2_SEEDS),
        n_parameter_seeds=np.array(N_PARAMETER_SEEDS),
        parameter_sample_counts=N_PARAMETER_SEEDS * SPLIT_COUNTS,
        noise_seed=np.array(NOISE_SEED),
        parameter_labels=np.asarray(["x", "x^2", "x^3", "x^4", "x^5", "intercept"]),
        **output_arrays,
    )
    temporary_output.replace(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
