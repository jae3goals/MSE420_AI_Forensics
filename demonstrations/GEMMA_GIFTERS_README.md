# GIFTERS sampling and robustness update

Open `gemma_gifters_local.ipynb` using `/Users/jae3goals/anaconda3/envs/Agentic_Workflow_Build/bin/python`. Run from this demonstrations directory. Qwen3-0.6B and the frozen local MiniLM encoder are unchanged. The generalizability section is unchanged. No downloads or remote inference occur.

## Running and resuming

Select `cfg = settings_for('fast')` or `cfg = settings_for('extended')` in the first code cell, then restart the kernel and run all cells.

| Setting | Fast | Extended (currently selected) |
|---|---:|---:|
| Responses per ensemble | 8 | 12 |
| Independent baseline pairs per original prompt | 20 | 32 |
| Baseline ensembles per original prompt | 40 | 64 |
| Requested variants per type per prompt | 2 | 3 |
| Weight grid | 3×3 | 3×3 |

Override temperature, seed, selected edit types, magnitudes, response lengths or grid size in the same cell. Increase ensemble sampling before increasing grid resolution. A cold-run estimate is displayed before expensive execution; a 3.8-hour session budget stops additional generation if exceeded. A single in-flight response and restoration can extend slightly beyond that budget.

Results go to a new `gifters_results/<timestamp>-<mode>/` directory. Individual responses and embeddings are saved atomically under `gifters_results/checkpoints_v2/`. Restarting and running all cells resumes compatible samples, even partway through an ensemble or grid point. Temperature, prompt, random seed, generation length, model weights, software versions, device and weight coordinates participate in generation identities; encoder identity and response texts participate in embedding identities. Editing settings therefore recomputes affected quantities without confusing old and new results. Do not run multiple notebook kernels writing the shared cache concurrently.

## Interpretation

The baseline consists of distances between disjoint independent pairs of original-input ensembles. A separate anchor is used for treatment distances. Stability reuses the original prompt's matching baseline and anchor, but its center is independently sampled and is not forced to zero.

Tables report empirical baseline percentile, distance/baseline-median ratio, a descriptive smoothed upper-tail fraction, and whether the distance exceeds baseline q95. These are exploratory diagnostics, not formal p-values or multiple-comparison-adjusted significance claims. With 20 reference distances, the upper-tail resolution is 1/21; uncertainty around q95 is displayed via a bootstrap interval. Treatment distances share an anchor and are correlated. More samples reduce noise but do not establish semantic equivalence. Optional `paired_sampling=True` adds a same-seed secondary displacement while retaining the independent main comparison and baseline; it adds generation cost.

## Generic perturbations

Qwen generates paraphrases with bounded retries and a conservative request/question-form check. Python generates transposition typos, punctuation edits, whitespace edits and adjacent sentence swaps. Requested magnitude is method-specific; measured character edit ratio is saved separately. Variants must differ from the original and be unique within each prompt. Unchanged, duplicate, inapplicable and suspicious paraphrases are recorded in `perturbation_attempts.json`; shortfalls are explicit. No scientific-question replacements remain.

Two sentences have only one distinct reordering; one sentence cannot be reordered. Sentence splitting and the English paraphrase heuristic have limitations. Meaning is never guaranteed. Inspect the table and set `include=False` for unsuitable rows before interpreting results.

## Measured runtime summary

The main end-to-end execution took **28.0 minutes** (24.5 minutes for robustness and baselines, 3.3 minutes for stability). An earlier deliberate interruption test took about 1.5 minutes. The final validation replay, including one additional successful paraphrase and its eight response samples, took **31.7 seconds**. Total measured execution across these stages was about **30 minutes**, comfortably below four hours. The final replay reused all nine grid points and 872 generation requests; it made nine new generation requests. The initial cold-workload estimate was 56.2 minutes. At that stage, extended mode was only estimated. It subsequently completed in 42.1 minutes before padding was added; the new padding validation is recorded below.

## Actual validation

Validated run: `gifters_results/20261105-084141-fast`.

- 872 response samples, in ensembles of 8.
- 20 disjoint baseline pairs per original prompt.
- 18 accepted unique input variants; 9 weight-plane points.
- Seven saved plots and finite numerical results.
- Exact model-weight restoration verified at every grid point.
- Independently sampled center displacement: 0.020546.
- Execution time for the validated session: 0.51 minutes (cache statistics below distinguish resumed work).
- Local profiling estimated 56.2 minutes for the upper-bound default workload before execution.

Timings (seconds):

```json
{
  "model_and_encoder_load_seconds": 4.180033874930814,
  "perturbation_generation_seconds": 0.8016669999342412,
  "robustness_including_baselines_seconds": 21.786085749976337,
  "stability_seconds": 1.8133935411460698,
  "session_total_seconds": 30.514720333041623
}
```

Cache statistics:

```json
{
  "generation_hits": 872,
  "generation_misses": 9,
  "embedding_hits": 112
}
```

Variant shortfalls:

```json
[
  {
    "prompt_id": 0,
    "method": "reorder",
    "requested": 2,
    "accepted": 1
  },
  {
    "prompt_id": 1,
    "method": "reorder",
    "requested": 2,
    "accepted": 1
  }
]
```

A real baseline run was deliberately interrupted and resumed; `gifters_results/interruption_test_v2.json` records the saved sample count. Implementation tests additionally simulate interruption and check cache invalidation, generic edit uniqueness, baseline-seed independence, teacher-forcing alignment, zero-centroid handling, direction normalization, exact restoration after an injected failure, and plot creation.

Run checks with:

```sh
MPLCONFIGDIR=/tmp/gifters-matplotlib /Users/jae3goals/anaconda3/envs/Agentic_Workflow_Build/bin/python validate_gifters.py
```

The timings above document the earlier fast-mode validation. Extended mode has subsequently completed, including the padding addition below. The optional paired-sampling full workload remains unvalidated end to end. Saved results remain exploratory teaching examples, not a benchmark.


## Separate prompt-padding experiment (Section 2b)

Inspired by [Silent Tokens, Loud Effects: Padding in LLMs](https://arxiv.org/html/2510.01238v2), Section 2b prepends actual tokenizer pad IDs before the full chat template. It compares correctly masked pads with intentionally attended pads and an independently sampled zero-padding control. It retains the existing response-centroid cosine-distance metric and frozen MiniLM encoder; it is not a replication of the paper's other evaluations.

Your selected `extended` preset is preserved: 12 responses per ensemble and padding counts `(0,1,2,4,8,16,32,128)`. This adds 360 responses for the two default prompts. `fast` uses `(0,1,8,32)` and adds 112 responses. Set `cfg.padding_enabled=False` to skip the addition. Both mask modes run by default, and baseline/anchor samples are reused only when their original prompt and settings match. The total runtime estimate includes padding.

Validation run: `gifters_results/20261105-111003-extended`. All 360 padding responses and 30 conditions completed, with 5 unusable samples. Padding took 16.52 minutes. The existing extended results were reused from cache; this validates the extended notebook with the new padding section without regenerating its prior experiments. Properly masked padding at 128 tokens preserved input positions and passed the next-token-logit equivalence check for both prompts.

`padding.csv` records distances, baseline percentiles, effect ratios and failure rates. `padding_responses.json` records complete outputs, seeds and failure flags. `padding_alignment.json` records the masking checks; `padding_token_info.json` records the actual token IDs. Four padding plots compare displacement against pad count and baseline variability, with failed-sample fractions shown separately. Empty, generation-capped and encoder-overlength samples are retained as failures; a condition with any such sample has no full-ensemble semantic distance. No failed sample is silently resampled.

Run `validate_gifters_padding.py` using the same Python environment for tokenizer/mask/position/cache/failure checks. Padding checkpoints are stored alongside the existing cache with a separate identity that includes pad count, mask mode and token ID. A repeat Run All reuses completed padding samples.

Final padding replay: `gifters_results/20261105-112738-extended`, 21.3 seconds for the whole notebook, zero new generation calls. All 360 padding responses, including failures, were reused and all padding result tables matched exactly. The five unusable responses all reached the 256-token generation cap; four also exceeded the semantic encoder context. No full-ensemble distance is reported for that condition.
