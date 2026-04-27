# Urban Renovation

Current project contract as of 2026-04-27:

- Runtime: Python `3.13`
- Main entry: `scripts/pipeline/main_py313.py`
- Stable pipeline entry: `scripts/pipeline/run_stable_release.py`
- Legacy compatibility entries: `scripts/main.py` and root-level `scripts/*.py` wrappers
- Stable configuration: `three_stage_hybrid --hybrid-llm-assist on`
- Stable model: `deepseek-v4-flash`
- Primary task shape:
  - `topic_final` is the main output
  - `urban_flag` / `final_label` are derived from `topic_final`
  - topic space is `U1-U15 / N1-N10 / Unknown`
- BERTopic is auxiliary only:
  - dynamic topic discovery
  - `Unknown` review support
  - rule and label iteration support
  - not an online primary decision source
- LLM is precision-constrained:
  - only used to collect a family hint for `Unknown`
  - does not overwrite `topic_final`
  - `llm_used` must remain `0` in the stable release

## Experiment governance

Three tracks are now enforced:

- `stable_release`
  - only the current hybrid mainline
  - dataset fixed to `Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407`
  - main task only: `urban_renewal`
- `research_matrix`
  - method comparison for the main task
  - long-context order-sensitivity comparison
  - `spatial` should be evaluated here as a separate report track
- `legacy_archive`
  - historical scripts, reports, and heuristic truth binding only
  - no new stable conclusions may cite this track

## Directory Layout

Canonical project data layout:

```text
Data/
  <dataset_id>/
    input/
      labels/              # read-only truth and labeled input workbooks
    runs/
      <experiment_track>/
        <run_tag>/
          predictions/     # model or pipeline prediction workbooks
          reports/         # Eval_*.xlsx and Eval_Summary.xlsx
          reviews/         # Unknown_Review and manual review workbooks
          logs/            # run logs
          Stable_Run_Summary.json
  train/                   # research and development training workbooks
output/
  models/                  # local model artifacts used by the current code
history/
  sessions/                # optional prompt/session audit traces
```

Compatibility note: older `Data/<dataset_id>/labels`, `Data/<dataset_id>/output`, and `Data/<dataset_id>/Result` folders are retained as historical archives only. New stable runs must use `Data/<dataset_id>/runs/<track>/<tag>/...`.

Truth and data contract:

- stable release uses only `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/input/labels`
- `test1-test7_merged` is historical baseline only
- labels are read-only truth sources
- official summary conclusions must come from `scripts/evaluation/evaluate.py` and `Eval_Summary.xlsx`

Metric scale contract:

- `Accuracy` is stored and reported as `0-100`
- `Precision`, `Recall`, `F1` stay in `0-1`
- any `Accuracy > 100` is an error

Long-context comparison contract:

- long-context results belong to `research_matrix`, not the stable gate
- use fixed orders:
  - `canonical_title_order`
  - `shuffle_seed_20260415_a`
  - `shuffle_seed_20260415_b`
- cite only aggregated results, never a single long-context run
- `Long Context Stability` in `Eval_Summary.xlsx` is the authority for order sensitivity

## Stable release lock

Locked performance-optimal stable release:

- Output:
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260427_deepseek_v4_flash_stable/predictions/urban_renewal_three_stage_hybrid_few_llm_on_20260427_deepseek_v4_flash_stable.xlsx`
- Summary:
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260427_deepseek_v4_flash_stable/reports/Eval_Summary.xlsx`
- Unknown review pool:
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260427_deepseek_v4_flash_stable/reviews/Unknown_Review_hybrid_llm_on_20260427_deepseek_v4_flash_stable.xlsx`

Stable release metrics:

- `Accuracy = 92.2`
- `Precision = 0.959900`
- `Recall = 0.943350`
- `F1 = 0.951553`
- `FP = 32`
- `FN = 46`
- `Predicted Unknown Count = 38`
- `unknown_hint_resolution Accuracy = 94.8980`
- `llm_attempted = 137`
- `llm_used = 0`

Stable pipeline command:

```powershell
.venv-bertopic313\Scripts\python.exe scripts\pipeline\run_stable_release.py --skip-classification
```

Use `--force` only when intentionally re-running the live 1000-sample classification and overwriting the locked prediction workbook.

Reference full-matrix baseline for comparison:

- historical archive: `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/Result/baseline_20260409_finalstable`

## Release gates

Use the same labeled dataset, same prompt family, and the same evaluator for every release check.

Required matrix:

1. `local_topic_classifier`
2. `three_stage_hybrid --hybrid-llm-assist off`
3. `three_stage_hybrid --hybrid-llm-assist on`

Required checks:

- bootstrap environment first:
  - `python -m pip install -e .[dev]`
  - `python -m pytest -q`
- `evaluate.py` output must include:
  - `All Metrics`
  - `Run Metadata`
  - `Protocol`
  - `Comparability`
  - `Long Context Stability`
  - `Decision Source Metrics`
  - `Unknown Rate`
  - `Theme Metrics`
  - `Theme Confusion`
  - `U-N Family Metrics`
  - `Topic Distribution`
  - `Explainability Quality`
  - `Evidence Balance Metrics`
- official narrative report generation is owned by `scripts/reporting/generate_stage_report.py`

Stable release acceptance thresholds:

- `hybrid + LLM on` accuracy `>= 88.0`
- `hybrid + LLM on` precision `>= 0.956`
- `hybrid + LLM on` recall `>= 0.940`
- `hybrid + LLM on` F1 `>= 0.948`
- `FP <= 34`
- `FN <= 48`
- `Predicted Unknown Count <= 38`
- `llm_used == 0`
- `unknown_hint_resolution` subset accuracy `>= 92.0%`
- explanation coverage `>= 100%`
- decision rule stack coverage `>= 100%`
- binary decision evidence coverage `>= 100%`

## Label inputs

Binary evaluation works with the current label workbook.

Optional theme-review columns are supported when present:

- `theme_gold`
- `theme_gold_source`
- `review_status`

Theme evaluation is only computed for rows where `theme_gold` is populated. The current stable release still has empty `Theme Metrics`, so 25-class theme accuracy is not yet considered closed.
