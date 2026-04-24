# Stable Release and Test Plan

This document supersedes the earlier 2026-04-08 baseline interpretation and locks the current performance-optimal stable release as of 2026-04-17.

## Governance update

Current experiment governance is split into three tracks:

1. `stable_release`
   - current hybrid mainline only
   - dataset fixed to `Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407`
   - main entry fixed to `scripts/main_py313.py`
2. `research_matrix`
   - method comparison
   - long-context order-stability protocol
   - `spatial` reported separately from the stable gate
3. `legacy_archive`
   - historical scripts, reports, and heuristic truth matching only

Current evaluation contract:

- only `scripts/evaluate.py` may generate official acceptance summaries
- `Accuracy` remains on a `0-100` scale
- `Precision`, `Recall`, `F1` remain on a `0-1` scale
- any `Accuracy > 100` is invalid
- strict tracks allow truth binding only by explicit `--truth` or a unique label workbook
- `scripts/main.py` is now legacy compatibility only

## Canonical directory layout

All new stable and research outputs should use one run directory per dataset, track, and tag:

```text
Data/<dataset_id>/
  input/labels/<dataset_id>.xlsx
  runs/<experiment_track>/<run_tag>/
    predictions/
    reports/
    reviews/
    logs/
    Stable_Run_Summary.json
```

Canonical labeled input path: `Data/<dataset_id>/input/labels/<dataset_id>.xlsx`.

Legacy folders are retained for historical comparison only:

- `Data/<dataset_id>/labels`
- `Data/<dataset_id>/output`
- `Data/<dataset_id>/Result`

## Locked stable release

- Runtime: Python `3.13`
- Main entry: `scripts/main_py313.py`
- Stable pipeline entry: `scripts/run_stable_release.py`
- Stable mode: `three_stage_hybrid --hybrid-llm-assist on`
- Stable result directory:
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/reports`
- Stable output:
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/predictions/urban_renewal_three_stage_hybrid_few_llm_on_20260417_unknown_recovery.xlsx`
- Stable review pool:
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/reviews/Unknown_Review_hybrid_llm_on_20260417_unknown_recovery.xlsx`

## Locked metrics

Urban Renewal metrics for the stable release:

| Run | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `three_stage_hybrid + LLM on` | 88.0% | 0.956743 | 0.940000 | 0.948298 | 752 | 128 | 34 | 48 | 38 |

Unknown recovery diagnostics for the stable release:

| Decision source | Total | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| `rule_model_fusion` | 846 | 91.1348 | 0.960452 | 0.935351 | 0.947735 |
| `stage1_rule` | 17 | 100.0000 | 0.000000 | 0.000000 | 0.000000 |
| `unknown_hint_resolution` | 99 | 92.9293 | 0.923077 | 0.986301 | 0.953642 |
| `unknown_review` | 38 | 0.0000 | 0.000000 | 0.000000 | 0.000000 |

Reference comparison baselines:

| Baseline dir | Run | Precision | Recall | F1 | FP | Unknown |
|---|---|---:|---:|---:|---:|---:|
| `baseline_20260409_finalstable` | `three_stage_hybrid + LLM on` | 0.949266 | 0.913882 | 0.931238 | 38 | 56 |
| `baseline_20260408_precision_round2` | `three_stage_hybrid + LLM on` | 0.947441 | 0.910904 | 0.928814 | 38 | 88 |
| `baseline_20260408_precision` | `three_stage_hybrid + LLM on` | 0.946839 | 0.901505 | 0.923616 | 37 | 111 |

## Release gates

The following are required before calling any new version stable:

1. Run `python -m pytest -q`
   - First install with `python -m pip install -e .[dev]`
   - Must remain fully green in a clean Python `3.13` environment
2. Re-run the fixed experiment matrix under Python `3.13`
   - `local_topic_classifier`
   - `three_stage_hybrid --hybrid-llm-assist off`
   - `three_stage_hybrid --hybrid-llm-assist on`
3. Evaluate all outputs with `scripts/evaluate.py`
   - Required sheets:
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

Single-run stable pipeline:

```powershell
.venv-bertopic313\Scripts\python.exe scripts\run_stable_release.py --skip-classification
```

Use `--force` only when intentionally re-running and overwriting the locked 1000-sample prediction workbook.

Long-context comparison rule:

- any `stepwise_long` style long-context claim must aggregate three fixed orders
- main conclusion tables may only cite the aggregated mean
- if `max_delta_accuracy > 1.5` or `max_delta_f1 > 0.015`, the method is `order_sensitive`

Acceptance thresholds for the stable run:

- `Accuracy >= 88.0`
- `Precision >= 0.956`
- `Recall >= 0.940`
- `F1 >= 0.948`
- `FP <= 34`
- `FN <= 48`
- `Predicted Unknown Count <= 38`
- `llm_used == 0`
- `unknown_hint_resolution` accuracy `>= 92.0%`
- explanation coverage `>= 100%`
- decision rule stack coverage `>= 100%`
- binary decision evidence coverage `>= 100%`

## Functional regression coverage

Required regression coverage:

- `stage1_rule`
  - math term misuse
  - rural non-urban
  - greenfield / new town
- `rule_model_fusion`
  - same-label agreement
  - same-group preference
  - cross-group strong rule override
  - cross-group strong local override
- `Unknown` conservative recovery
  - `N3/N8 -> U*`
  - `U12/U4/U9/U1 -> N*`
  - `local Unknown + llm family hint`
- BERTopic auxiliary-only contract
  - BERTopic may emit hints
  - BERTopic must not rewrite `topic_final`
- Output contract
  - `urban_flag` must be derived from `topic_final`
  - deterministic explanation fields must be populated for every row
  - `decision_source` must stay within:
    - `rule_model_fusion`
    - `stage1_rule`
    - `stage2_classifier`
    - `unknown_hint_resolution`
    - `unknown_review`

Required topic-boundary samples:

- general governance / policy / discourse must not become positive on weak renewal wording
- `TIF / PPP / compensation / redevelopment finance` must stay positive
- `brownfield redevelopment / adaptive reuse / urban village / inner-city regeneration` must stay positive
- general `gentrification / neighborhood change` must not be auto-positive
- gentrification explicitly tied to renewal process or consequences must stay positive

## Known limitation

- `theme_gold` is not yet populated at scale
- `Theme Metrics` and `Theme Confusion` are expected to remain structurally empty in the locked stable release
- next-phase work is limited to:
  - filling `theme_gold`
  - high-precision recovery on the remaining `Unknown` pool
