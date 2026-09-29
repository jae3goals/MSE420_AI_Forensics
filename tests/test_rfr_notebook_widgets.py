"""Exercise the real robustness widget callbacks, including NumPy SQL parameters."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import Dropdown, FloatSlider, HBox, IntSlider, SelectionSlider, VBox, interactive_output

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'demonstrations/random_forest_cv.ipynb').read_text())


class RobustnessWidgetTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(np=np, pd=pd, sqlite3=sqlite3, plt=plt,
            DATABASE_PATH=ROOT / 'demonstrations/precomputed/rfr_precomputed_results.sqlite',
            Dropdown=Dropdown, FloatSlider=FloatSlider, HBox=HBox, IntSlider=IntSlider,
            SelectionSlider=SelectionSlider, VBox=VBox, display=lambda value: None,
            CV_LABELS={'kfold': 'Shuffled K-fold', 'loco': 'Feature-space LOCO'})
        setup = ast.parse(''.join(NOTEBOOK['cells'][5]['source']))
        query_node = next(node for node in setup.body if isinstance(node, ast.FunctionDef) and node.name == 'query')
        exec(compile(ast.Module(body=[query_node], type_ignores=[]), '<query>', 'exec'), self.ns)
        self.errors = []
        self.calls = []
        def checked_output(function, controls):
            def checked(**kwargs):
                self.calls.append(kwargs)
                try:
                    return function(**kwargs)
                except Exception as exc:
                    self.errors.append(exc)
                    raise
            return interactive_output(checked, controls)
        self.ns['interactive_output'] = checked_output
        self.ns['robustness'] = self.ns['query']('SELECT * FROM robustness_results')
        # Reproduce the original widget's NumPy scalar, even after options are normalized.
        self.ns['robust_count_widget'] = SelectionSlider(options=[np.int64(5)])

    def tearDown(self):
        plt.close('all')
        for value in self.ns.values():
            if hasattr(value, 'close') and isinstance(value, (Dropdown, FloatSlider, HBox, IntSlider, SelectionSlider, VBox)):
                value.close()

    def test_numpy_parameters_are_sql_numbers(self):
        rows = self.ns['query']('SELECT typeof(?) AS kind, typeof(?) AS real_kind', (np.int64(5), np.float64(0.2)))
        self.assertEqual(rows.kind.iloc[0], 'integer')
        self.assertEqual(rows.real_kind.iloc[0], 'real')

    def test_real_widget_callbacks_and_empty_setting(self):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(NOTEBOOK['cells'][23]['source']), self.ns)
            self.ns['robust_explain_cv_widget'].value = 'loco'
            self.ns['example_feature_noise_widget'].value = 0.0
            self.ns['example_target_noise_widget'].value = 0.0
            pairs = self.ns['robustness_prediction_changes']('loco', np.float64(0), np.float64(0), np.int64(5))
        self.assertGreaterEqual(len(self.calls), 4)
        self.assertEqual(self.errors, [])
        self.assertFalse(pairs.empty)
        self.assertTrue((pairs.absolute_prediction_change == 0).all())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.ns['render_robustness_explanations']('kfold', 6, 0.2, 0.2, 0.25, 999)
        self.assertIn('No saved robustness importances', output.getvalue())


if __name__ == '__main__':
    unittest.main()
