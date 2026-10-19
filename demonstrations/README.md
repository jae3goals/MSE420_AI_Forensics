# In-Class Demonstrations

Place instructor-led notebooks and examples here. Demonstrations should be runnable from the repository root and introduce concepts related to trust in AI, such as evaluation, bias, interpretability, robustness, and documentation.

## Extending Random Forest experiments

From the repository root, generate the three saved nested-CV comparisons for Section 6:

```bash
python demonstrations/generate_rfr_precomputed.py --studies model_selection --jobs -1 --yes
```

An existing database supplies the base configuration. The generator reuses completed
experiments, including the original depth-selection results, and adds missing ones.
Leaf candidates default to `1, 2, 4, 8, 16, 32`; split candidates to
`2, 4, 8, 16, 32, 64` (scikit-learn requires split size at least 2).
The notebook's Section 6 dropdown switches among saved parameters and candidate sets.

To extend another study, pass a JSON configuration override:

```json
{
  "validation_seeds": [30, 31],
  "loco_counts": [5, 10, 15],
  "min_samples_leaf_grid": [1, 2, 4, 8]
}
```

```bash
python demonstrations/generate_rfr_precomputed.py --config experiments.json --studies validation hyperparameters clusters --yes
```

JSON lists specify requested combinations, not removals. Completed rows remain,
and interrupted runs can be restarted with the same command. Available studies are
`clusters`, `convergence`, `model_selection`, `validation`, `hyperparameters`,
`importance`, and `robustness`. Changing a selection candidate set creates a new
nested-CV experiment; it does not overwrite the old candidate set.

The original metadata is retained. `generation_history` records each request and
its runtime versions; `experiment_provenance` links newly inserted result rows to
that history. Legacy rows without a provenance link use the original metadata.
Runtime changes are recorded rather than forcing a rebuild; keep them in mind when
comparing old and new results. New robustness runs store individual predictions
for every seed. Fold-count controls keep importance and robustness results from
different fold counts separate.

Use a separate `--database` when changing the dataset, fixed tree count,
representative model, selection fold counts, or permutation repeat count.
`--overwrite` remains an explicit destructive reset, not a prerequisite for adding
experiments. New databases default to smoke settings; pass `--mode full` for full
settings. Restart/rerun notebook setup and study cells after updating the database.

Regression checks:

```bash
python -m unittest discover -s tests -v
```

## SAM robustness and stability

[`sam_sem_robustness_stability_analysis.ipynb`](sam_sem_robustness_stability_analysis.ipynb)
studies how SEM-image degradations change pooled SAM ViT-B representations and
how local mask-decoder parameter perturbations affect fixed-prompt predictions.
Its ten input images are stored in `data/trajectory images/`.

Create the dedicated environment from the repository root:

```bash
conda env create -f environment-sam.yml
conda activate mse420-sam
python -m ipykernel install --user --name mse420-sam --display-name "MSE420 SAM"
```

Download Meta's `sam_vit_b_01ec64.pth` checkpoint into
`weights/segment-anything/`. Model checkpoints and generated analysis caches are
intentionally ignored by Git.
