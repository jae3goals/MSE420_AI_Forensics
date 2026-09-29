"""Small real-forest regression tests for append-only experiment storage."""
import sqlite3
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from demonstrations import generate_rfr_precomputed as g


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(24, 3)), columns=list('abc'))
        self.y = pd.Series(self.X.a ** 2 + self.X.b)
        self.compositions = pd.Series([f'Material{i}' for i in range(24)])
        self.df = self.X.assign(composition=self.compositions, expt_gap=self.y)
        self.config = replace(g.SMOKE_CONFIG, fixed_n_estimators=2,
            convergence_estimators=(2,), convergence_seeds=(0,),
            selection_depths=(2, None), selection_leaf_sizes=(1, 4), selection_split_sizes=(2, 8),
            selection_outer_folds=2, selection_inner_folds=2,
            validation_seeds=(0,), kfold_counts=(2,), loco_counts=(2,),
            hyper_kfold_counts=(2,), hyper_loco_counts=(2,), max_depth_grid=(None,),
            min_samples_leaf_grid=(1,), min_samples_split_grid=(2,), max_features_grid=(0.5,),
            importance_fold_count=2, robustness_fold_count=2, permutation_repeats=1)
        g.N_JOBS = 1
        self.scaled, self.splits = g.build_split_cache(self.X)

    def tearDown(self):
        self.db.close()

    def initialize(self, config):
        digest = g.config_hash(config, 'dataset')
        g.initialize_database(self.db, config, digest, 'dataset', Path('synthetic.csv'),
                              self.df, self.compositions, self.X, self.y, 'expt_gap', [], 0, False)
        return digest

    def run_all(self, config):
        digest = self.initialize(config)
        for fn, args in [
            (g.run_clusters, (self.X, self.scaled)),
            (g.run_convergence, (self.X, self.y, self.splits)),
            (g.run_model_selection, (self.X, self.y, self.compositions, self.splits)),
            (g.run_validation, (self.X, self.y, self.compositions, self.splits)),
            (g.run_hyperparameters, (self.X, self.y, self.splits)),
            (g.run_importance, (self.X, self.y, self.splits)),
            (g.run_robustness, (self.X, self.y, self.compositions, self.splits)),
        ]:
            fn(self.db, config, digest, *args)
        g.verify_database(self.db, config, digest, 'dataset', len(self.y), self.X.shape[1])

    def snapshot(self):
        tables = [r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: set(self.db.execute(f'SELECT * FROM {t}')) for t in tables
                if t not in {'stage_status', 'sqlite_sequence'}}

    def test_append_resume_and_candidate_sets(self):
        self.run_all(self.config)
        original = self.snapshot()
        with patch.object(g, 'make_forest', side_effect=AssertionError('cached result refitted')):
            self.run_all(self.config)
        self.assertEqual(original, self.snapshot())
        extended = replace(self.config, convergence_seeds=(0, 1), convergence_estimators=(2, 3), validation_seeds=(0, 1),
                           selection_outer_seeds=(0, 1), selection_leaf_sizes=(1, 4, 8),
                           min_samples_leaf_grid=(1, 3), min_samples_split_grid=(2, 4), robustness_seeds=(0, 1),
                           feature_noise_levels=(0.0, 0.1, 0.2), robustness_fold_count=3,
                           loco_counts=(2, 3), importance_cv_seed=1, importance_fold_count=3)
        self.run_all(extended)
        after = self.snapshot()
        for table, rows in original.items():
            self.assertTrue(rows <= after[table], table)
        with patch.object(g, 'make_forest', side_effect=AssertionError('cached extension refitted')):
            self.run_all(extended)
        self.assertEqual(after, self.snapshot())
        with self.assertRaises(ValueError):
            self.initialize(replace(extended, fixed_n_estimators=3))

    def test_failed_fit_can_resume(self):
        digest = self.initialize(self.config)
        with patch.object(g, 'make_forest', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                g.run_model_selection(self.db, self.config, digest, self.X, self.y, self.compositions, self.splits)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM selection_scores').fetchone()[0], 0)
        self.run_all(self.config)

    def test_split_one_rejected(self):
        with self.assertRaises(ValueError):
            g.validate_config(replace(self.config, selection_split_sizes=(1, 2)))


if __name__ == '__main__':
    unittest.main()
