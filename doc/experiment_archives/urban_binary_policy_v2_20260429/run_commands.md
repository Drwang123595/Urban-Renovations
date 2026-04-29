# 城市更新二分类规则 V2 复现实验命令

## 1000 篇带标签 no-LLM 回归

```powershell
.\.venv-bertopic313\Scripts\python.exe scripts\pipeline\main_py313.py --task urban_renewal --experiment-track research_matrix --input "Data\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407\input\labels\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx" --truth-file "Data\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407\input\labels\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx" --dataset-id "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407" --urban-method three_stage_hybrid --hybrid-llm-assist off --dynamic-topics on --dynamic-binary-refine on --dynamic-binary-allow-flip --non-interactive --output "<run>/predictions/urban_renewal_three_stage_hybrid_policy_v2_recall_no_llm_20260428.xlsx"
```

## 1000 篇带标签 LLM 困难样本裁决

```powershell
.\.venv-bertopic313\Scripts\python.exe scripts\pipeline\main_py313.py --task urban_renewal --experiment-track research_matrix --input "Data\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407\input\labels\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx" --truth-file "Data\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407\input\labels\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx" --dataset-id "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407" --urban-method three_stage_hybrid --hybrid-llm-assist on --dynamic-topics on --dynamic-binary-refine on --dynamic-binary-allow-flip --non-interactive --output "<run>/predictions/urban_renewal_three_stage_hybrid_policy_v2_recall_llm_20260428.xlsx"
```

## 10000 篇重新抽样 no-LLM 全流程

```powershell
.\.venv-bertopic313\Scripts\python.exe scripts\pipeline\main_py313.py --task urban_renewal --experiment-track research_matrix --input "Data\Urban Renovation V2.0\runs\research_matrix\20260428_policy_v2_resample10000_no_llm_full_on_seed2026042802\input\Urban_Renovation_V2_policy_v2_resample10000_seed2026042802.xlsx" --dataset-id "Urban Renovation V2.0_policy_v2_resample10000_seed2026042802" --urban-method three_stage_hybrid --hybrid-llm-assist off --dynamic-topics on --dynamic-topics-full-corpus --dynamic-binary-refine on --dynamic-binary-allow-flip --non-interactive --output "Data\Urban Renovation V2.0\runs\research_matrix\20260428_policy_v2_resample10000_no_llm_full_on_seed2026042802\predictions\urban_renewal_three_stage_hybrid_policy_v2_resample10000_no_llm_full_on_20260428.xlsx"
```

## 40276 篇全量 no-LLM 完整运行

```powershell
.\.venv-bertopic313\Scripts\python.exe scripts\pipeline\main_py313.py --task urban_renewal --experiment-track research_matrix --input "Data\Urban Renovation V2.0\input\labels\Urban Renovation V2.0.xlsx" --dataset-id "Urban Renovation V2.0_policy_v2_full_no_llm_full_on_complete_20260429" --urban-method three_stage_hybrid --hybrid-llm-assist off --dynamic-topics on --dynamic-topics-full-corpus --dynamic-binary-refine on --dynamic-binary-allow-flip --non-interactive --output "Data\Urban Renovation V2.0\runs\research_matrix\20260429_policy_v2_full_no_llm_full_on_complete\predictions\urban_renewal_three_stage_hybrid_policy_v2_full_no_llm_full_on_complete_20260429.xlsx"
```
