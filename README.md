# Urban Renovation

Current project contract as of 2026-04-27:

- Runtime: Python `3.13`
- Python version launcher: `scripts/pipeline/main.py`
- Main entry: `scripts/pipeline/main_py313.py`
- Stable pipeline entry: `scripts/pipeline/run_stable_release.py`
- Stable configuration: `three_stage_hybrid --hybrid-llm-assist on`
- Stable model: `deepseek-v4-flash`
- Primary task shape:
  - `topic_final` is the main output
  - `urban_flag` / `final_label` are written by the stable evidence strategy
  - topic space is `U1-U15 / N1-N10 / Unknown`
- Urban-renewal decision definition:
  - positive class requires an existing urban object, a renewal/redevelopment action, and action-as-main-subject
  - hard exclusions protect rural revitalization, greenfield expansion, method-only/background usage, and terminology misuse
  - auxiliary model signals may support evidence, but they do not redefine what counts as urban renewal
- BERTopic is auxiliary only:
  - dynamic topic discovery
  - `Unknown` review support
  - rule and label iteration support
  - not an online primary decision source
- LLM is precision-constrained:
  - only used for semantic evidence on difficult or ambiguous samples
  - positive evidence requires an existing urban object, renewal action, and action-as-main-subject
  - may support the final decision only when the structured semantic triplet is satisfied
  - every used LLM adjudication must write `llm_attempted`, `llm_used`, and `llm_semantic_evidence`
- Stable decision chain is a four-step rule, audited through strategy fields:
  - extract core object, renewal action, main-subject, and risk evidence
  - reject hard exclusions before any model promotion
  - call LLM only for boundary semantics when rule/model evidence is unclear
  - write final labels once through the stable strategy output builder

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
  train/                   # read-only input workbooks selected by users and scripts
    <dataset_id>.xlsx
  output/
    <dataset_id>/
      runs/
        <experiment_track>/
          <run_tag>/
            predictions/   # model or pipeline prediction workbooks
              urban_renewal/
              spatial/
              merged/
            reports/       # eval__<prediction_stem>.xlsx and Eval_Summary.xlsx
            reviews/       # unknown_review__*.xlsx and manual review workbooks
            logs/          # run logs
            Stable_Run_Summary.json
      analysis/            # consolidated historical analysis outputs
      legacy_output/       # consolidated files from obsolete Data/<dataset_id>/output paths
      legacy_result/       # consolidated files from obsolete Data/<dataset_id>/Result paths
      bundles/             # consolidated deliverable/progress bundles
```

New managed run paths must use `Data/output/<dataset_id>/runs/<track>/<tag>/...`. Prediction workbooks are partitioned by task under `predictions/urban_renewal/`, `predictions/spatial/`, and `predictions/merged/`.

Root-level `output/` remains only for non-data runtime artifacts such as local models.

Compatibility note: obsolete `Data/<dataset_id>/labels`, `Data/<dataset_id>/output`, and `Data/<dataset_id>/Result` paths are historical read-only references only. The current `Data` top level should contain only `train/` and `output/`.

Truth and data contract:

- input workbooks live in `Data/train/`
- canonical labeled input path is `Data/train/<dataset_id>.xlsx`
- `--input "Urban Renovation V2.0.xlsx"` resolves to `Data/train/Urban Renovation V2.0.xlsx`
- `--input Data/train/<file>.xlsx` is allowed
- `--input` outside `Data/train` is rejected
- custom run outputs must be under `Data/output`
- stable release uses `Data/train/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx`
- `test1-test7_merged` is historical baseline only
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
  - `Data/output/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260427_deepseek_v4_flash_stable/predictions/urban_renewal/urban_renovation_v2_0_20260407__urban_renewal__three_stage_hybrid_few_llm_on__20260427_deepseek_v4_flash_stable.xlsx`
- Summary:
  - `Data/output/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260427_deepseek_v4_flash_stable/reports/Eval_Summary.xlsx`
- Unknown review pool:
  - `Data/output/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260427_deepseek_v4_flash_stable/reviews/unknown_review__urban_renovation_v2_0_20260407__urban_renewal__20260427_deepseek_v4_flash_stable.xlsx`

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
- `llm_used = 0` in this historical locked artifact

Stable pipeline command:

```powershell
.venv-bertopic313\Scripts\python.exe scripts\pipeline\run_stable_release.py --skip-classification
```

Use `--force` only when intentionally re-running the live 1000-sample classification and overwriting the locked prediction workbook.

Reference full-matrix baseline for comparison:

- historical archive: `Data/output/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/legacy_result/baseline_20260409_finalstable`

## Release gates

Use the same labeled dataset, same prompt family, and the same evaluator for every release check.

Environment contract:

- recommended install: `uv sync --all-extras`
- lock verification: `uv lock --check`
- local environment verification: `.venv-bertopic313\Scripts\python.exe scripts\dev\check_environment.py`
- optional dependency audit: `.venv-bertopic313\Scripts\python.exe -m pip_audit`

Required matrix:

1. `local_topic_classifier`
2. `three_stage_hybrid --hybrid-llm-assist off`
3. `three_stage_hybrid --hybrid-llm-assist on`

Required checks:

- bootstrap environment first:
  - `uv sync --all-extras`
  - `python scripts\dev\check_environment.py`
  - `python -m pytest -q`
- `evaluate.py` output must include:
  - `All Metrics`
  - `Run Metadata`
  - `Protocol`
  - `Comparability`
  - `Long Context Stability`
  - `Theme Metrics`
  - `Theme Confusion`
  - `U-N Family Metrics`
  - `Decision Source Metrics`
  - `Unknown Rate`
  - `Topic Distribution`
  - `Boundary Bucket Metrics`
  - `Unknown Conflict Analysis`
  - `Explainability Quality`
  - `Evidence Balance Metrics`
  - `Bootstrap CI`
  - `McNemar`
- official narrative report generation is owned by `scripts/reporting/generate_stage_report.py`

Stable release acceptance thresholds:

- `hybrid + LLM on` accuracy `>= 88.0`
- `hybrid + LLM on` precision `>= 0.959900`
- `hybrid + LLM on` recall `> 0.943350`
- `hybrid + LLM on` F1 `>= 0.951553`
- `FP <= 34`
- `FN < 46`
- `Predicted Unknown Count <= 38`
- `llm_used` is allowed only for structured boundary adjudication and must be covered by `llm_attempted` audit records
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
