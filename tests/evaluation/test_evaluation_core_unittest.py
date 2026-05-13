import unittest

import numpy as np
import pandas as pd

from src.config import Schema
from src.evaluation_core import (
    align_truth_pred,
    evaluate_merged,
    summarize_bootstrap_ci,
    summarize_boundary_bucket_metrics,
    summarize_decision_source_metrics,
    summarize_dynamic_binary_recommendations,
    summarize_dynamic_fixed_crosswalk,
    summarize_dynamic_topic_candidates,
    summarize_dynamic_topic_distribution,
    summarize_dynamic_topic_quality,
    summarize_metrics,
    summarize_mcnemar,
    summarize_evidence_balance_metrics,
    summarize_explainability_quality,
    summarize_theme_confusion,
    summarize_theme_family_metrics,
    summarize_theme_metrics,
    summarize_unknown_conflict_analysis,
    summarize_unknown_rate,
    validate_accuracy_bounds,
)


class EvaluationCoreTests(unittest.TestCase):
    def test_alignment_summary_and_unmatched(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C"],
                "是否属于城市更新研究": [1, 0, 1],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "D"],
                "是否属于城市更新研究": [1, 1, 0],
            }
        )
        result = align_truth_pred(truth_df, pred_df, strict=False)
        self.assertEqual(result.summary["match_count"], 2.0)
        self.assertEqual(len(result.diagnostics["truth_unmatched"]), 1)
        self.assertEqual(len(result.diagnostics["pred_unmatched"]), 1)

    def test_strict_mode_duplicate_key_fail(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "A"],
                "是否属于城市更新研究": [1, 0],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A"],
                "是否属于城市更新研究": [1],
            }
        )
        with self.assertRaises(ValueError):
            align_truth_pred(truth_df, pred_df, strict=True)

    def test_binary_metrics_precision_recall_f1(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                "是否属于城市更新研究": [1, 1, 0, 0],
                "空间研究/非空间研究": [1, 0, 1, 0],
                "空间等级": ["7", "7", "7", "7"],
                "具体空间描述": ["x", "x", "x", "x"],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                "是否属于城市更新研究": [1, 0, 1, 0],
                "空间研究/非空间研究": [1, 0, 0, 0],
                "空间等级": ["7", "7", "7", "7"],
                "具体空间描述": ["x", "x", "x", "x"],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)
        metrics_df, _ = evaluate_merged(aligned.merged, "demo")
        urban = metrics_df[metrics_df["Metric"] == "Urban Renewal"].iloc[0]
        self.assertEqual(int(urban["TP"]), 1)
        self.assertEqual(int(urban["FP"]), 1)
        self.assertEqual(int(urban["FN"]), 1)
        self.assertAlmostEqual(float(urban["Precision"]), 0.5, places=6)
        self.assertAlmostEqual(float(urban["Recall"]), 0.5, places=6)
        self.assertAlmostEqual(float(urban["F1"]), 0.5, places=6)

    def test_missing_non_target_fields_are_skipped_not_zero_score(self):
        truth_df = pd.DataFrame(
            {
                Schema.TITLE: ["A", "B"],
                Schema.IS_URBAN_RENEWAL: [1, 0],
            }
        )
        pred_df = pd.DataFrame(
            {
                Schema.TITLE: ["A", "B"],
                Schema.IS_URBAN_RENEWAL: [1, 0],
            }
        )

        aligned = align_truth_pred(truth_df, pred_df, strict=True)
        metrics_df, detail_df = evaluate_merged(aligned.merged, "urban_only")

        skipped = metrics_df[metrics_df["Metric"].isin(["Spatial Study", "Spatial Level", "Spatial Desc"])]
        self.assertEqual(set(skipped["Total"]), {0})
        self.assertEqual(set(skipped["Correct"]), {0})
        self.assertTrue(skipped["Accuracy"].isna().all())
        self.assertTrue(skipped["Precision"].isna().all())
        self.assertTrue(detail_df["Diff_Spatial Study"].isna().all())
        self.assertTrue(detail_df["Diff_Spatial Level"].isna().all())
        self.assertTrue(detail_df["Diff_Spatial Desc"].isna().all())

        summary = summarize_metrics(metrics_df)
        spatial_summary = summary[summary["Metric"] == "Spatial Study"].iloc[0]
        self.assertEqual(int(spatial_summary["Total"]), 0)
        self.assertTrue(pd.isna(spatial_summary["Accuracy"]))

    def test_global_summary(self):
        raw = pd.DataFrame(
            [
                {"File": "f1", "Metric": "Urban Renewal", "Accuracy": 50.0, "Correct": 1, "Total": 2, "TP": 1, "TN": 0, "FP": 1, "FN": 0, "Precision": 0.5, "Recall": 1.0, "F1": 0.666667},
                {"File": "f2", "Metric": "Urban Renewal", "Accuracy": 100.0, "Correct": 2, "Total": 2, "TP": 1, "TN": 1, "FP": 0, "FN": 0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0},
            ]
        )
        summary = summarize_metrics(raw)
        row = summary.iloc[0]
        self.assertEqual(row["File"], "__GLOBAL__")
        self.assertEqual(int(row["Correct"]), 3)
        self.assertEqual(int(row["Total"]), 4)

    def test_detail_only_core_fields(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A"],
                "Abstract": ["abs"],
                "是否属于城市更新研究": [1],
                "空间研究/非空间研究": [1],
                "空间等级": ["7"],
                "具体空间描述": ["beijing"],
                "other_truth": ["t"],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A"],
                "是否属于城市更新研究": [1],
                "空间研究/非空间研究": [1],
                "空间等级": ["7"],
                "具体空间描述": ["beijing"],
                "other_pred": ["p"],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)
        _, detail_df = evaluate_merged(aligned.merged, "demo")
        self.assertEqual(
            list(detail_df.columns),
            [
                "Article Title",
                "Abstract",
                "是否属于城市更新研究_truth",
                "是否属于城市更新研究_pred",
                "Diff_Urban Renewal",
                "空间研究/非空间研究_truth",
                "空间研究/非空间研究_pred",
                "Diff_Spatial Study",
                "空间等级_truth",
                "空间等级_pred",
                "Diff_Spatial Level",
                "具体空间描述_truth",
                "具体空间描述_pred",
                "Diff_Spatial Desc",
            ],
        )

    def test_non_binary_precision_recall_f1_are_nan(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A"],
                "是否属于城市更新研究": [1],
                "空间研究/非空间研究": [1],
                "空间等级": ["7"],
                "具体空间描述": ["beijing"],
            }
        )
        pred_df = truth_df.copy()
        aligned = align_truth_pred(truth_df, pred_df, strict=True)
        metrics_df, _ = evaluate_merged(aligned.merged, "demo")
        spatial_level = metrics_df[metrics_df["Metric"] == "Spatial Level"].iloc[0]
        self.assertTrue(np.isnan(spatial_level["Precision"]))
        self.assertTrue(np.isnan(spatial_level["Recall"]))
        self.assertTrue(np.isnan(spatial_level["F1"]))
        self.assertEqual(
            list(metrics_df.columns),
            [
                "File",
                "Metric",
                "Accuracy",
                "Correct",
                "Total",
                "TP",
                "TN",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
            ],
        )

    def test_local_v2_truth_column_does_not_self_compare_with_prediction(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B"],
                f"{Schema.IS_URBAN_RENEWAL}_local_v2": [1, 0],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B"],
                Schema.IS_URBAN_RENEWAL: [1, 1],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)
        metrics_df, detail_df = evaluate_merged(aligned.merged, "demo")
        urban = metrics_df[metrics_df["Metric"] == "Urban Renewal"].iloc[0]
        self.assertEqual(int(urban["Correct"]), 1)
        self.assertEqual(int(urban["FP"]), 1)
        self.assertEqual(int(urban["FN"]), 0)
        self.assertEqual(list(detail_df["Diff_Urban Renewal"]), [1, 0])

    def test_theme_metrics_are_emitted_when_theme_gold_exists(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C"],
                Schema.IS_URBAN_RENEWAL: [1, 0, 1],
                "theme_gold": ["U10", "N3", "U12"],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C"],
                Schema.IS_URBAN_RENEWAL: [1, 0, 0],
                "topic_final": ["U10", "N3", "Unknown"],
                "decision_source": ["rule_model_fusion", "stage2_classifier", "unknown_review"],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)
        theme_metrics = summarize_theme_metrics(aligned.merged, "demo")
        overall = theme_metrics[theme_metrics["Theme"] == "__OVERALL__"].iloc[0]
        self.assertEqual(int(overall["Correct"]), 2)
        self.assertEqual(int(overall["Total"]), 3)

        confusion = summarize_theme_confusion(aligned.merged, "demo")
        self.assertTrue(
            ((confusion["Truth Theme"] == "U12") & (confusion["Pred Theme"] == "Unknown")).any()
        )

        family = summarize_theme_family_metrics(aligned.merged, "demo")
        self.assertEqual(int(family.iloc[0]["TP"]), 1)
        self.assertEqual(int(family.iloc[0]["Total"]), 2)

        unknown = summarize_unknown_rate(aligned.merged, "demo")
        self.assertEqual(int(unknown.iloc[0]["Predicted Unknown Count"]), 1)

    def test_decision_source_metrics_split_binary_results(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, 0, 1, 0],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, 1, 0, 0],
                "decision_source": [
                    "stage2_classifier",
                    "stage2_classifier",
                    "unknown_review",
                    "unknown_review",
                ],
                "topic_final": ["U9", "U9", "Unknown", "Unknown"],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)
        metrics = summarize_decision_source_metrics(aligned.merged, "demo")
        stage2 = metrics[metrics["Decision Source"] == "stage2_classifier"].iloc[0]
        self.assertEqual(int(stage2["TP"]), 1)
        self.assertEqual(int(stage2["FP"]), 1)
        unknown = metrics[metrics["Decision Source"] == "unknown_review"].iloc[0]
        self.assertEqual(int(unknown["FN"]), 1)
        self.assertAlmostEqual(float(unknown["Unknown Topic Rate"]), 1.0, places=6)

    def test_explainability_quality_and_evidence_balance_metrics(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, 0, 1, 0],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, 1, 0, 0],
                "decision_explanation": ["final=1; score=0.8", "final=1; score=0.5", "", "final=0; score=0.2"],
                "decision_rule_stack": ["route=a", "route=b", "", "route=d"],
                "binary_decision_evidence": ["score=a", "score=b", "", "score=d"],
                "primary_positive_evidence": ["topic=U9", "topic=U10", "", ""],
                "primary_negative_evidence": ["", "", "topic=Unknown", "topic=N3"],
                "evidence_balance": ["strong_positive", "low_confidence_positive", "conflict_negative", ""],
                "review_flag": [0, 1, 0, 1],
                "review_reason": ["", "binary_near_threshold", "binary_topic_inconsistency", ""],
                "binary_topic_consistency_flag": [0, 0, 1, 0],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)

        quality = summarize_explainability_quality(aligned.merged, "demo").iloc[0]
        self.assertEqual(int(quality["Total"]), 4)
        self.assertAlmostEqual(float(quality["Decision Explanation Coverage"]), 0.75, places=6)
        self.assertAlmostEqual(float(quality["Rule Stack Coverage"]), 0.75, places=6)
        self.assertAlmostEqual(float(quality["Binary Evidence Coverage"]), 0.75, places=6)
        self.assertEqual(int(quality["Review Trigger Count"]), 2)
        self.assertAlmostEqual(float(quality["Review Trigger Rate"]), 0.5, places=6)
        self.assertEqual(int(quality["Near Threshold Count"]), 1)
        self.assertEqual(int(quality["Conflict Count"]), 1)

        evidence = summarize_evidence_balance_metrics(aligned.merged, "demo")
        self.assertEqual(set(evidence["Evidence Balance"]), {
            "strong_positive",
            "low_confidence_positive",
            "conflict_negative",
            "missing",
        })
        low_conf = evidence[evidence["Evidence Balance"] == "low_confidence_positive"].iloc[0]
        self.assertEqual(int(low_conf["FP"]), 1)
        conflict = evidence[evidence["Evidence Balance"] == "conflict_negative"].iloc[0]
        self.assertEqual(int(conflict["FN"]), 1)

    def test_dynamic_topic_summaries_are_emitted_when_fields_exist(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, 1, 0, 0],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, 1, 0, 0],
                "topic_final": ["Unknown", "Unknown", "N7", "N4"],
                "topic_final_group": ["unknown", "unknown", "nonurban", "nonurban"],
                "dynamic_topic_id": ["DUR_0001", "DUR_0001", "DUR_0002", ""],
                "dynamic_topic_name_zh": ["brownfield", "brownfield", "transport", ""],
                "dynamic_topic_keywords": ["brownfield; renewal", "brownfield; renewal", "transport; model", ""],
                "dynamic_topic_size": [2, 2, 1, ""],
                "dynamic_topic_confidence": [0.8, 0.7, 0.4, ""],
                "dynamic_topic_source_pool": ["unknown_pool", "unknown_pool", "nonurban_review_pool", ""],
                "dynamic_to_fixed_topic_candidate": ["U2", "U2", "Nonurban_Other", ""],
                "dynamic_mapping_status": [
                    "mapped_to_fixed",
                    "mapped_to_fixed",
                    "candidate_new_nonurban_topic",
                    "",
                ],
                "dynamic_binary_candidate_label": ["1", "1", "0", ""],
                "dynamic_binary_candidate_confidence": [0.8, 0.7, 0.4, ""],
                "dynamic_binary_candidate_action": [
                    "supports_current_label",
                    "supports_current_label",
                    "supports_current_label",
                    "",
                ],
                "dynamic_binary_candidate_reason": ["a", "b", "c", ""],
                "dynamic_binary_review_priority": ["low", "low", "low", ""],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)

        quality = summarize_dynamic_topic_quality(aligned.merged, "demo").iloc[0]
        self.assertEqual(int(quality["Candidate Pool Count"]), 3)
        self.assertEqual(int(quality["Dynamic Topic Count"]), 2)
        self.assertAlmostEqual(float(quality["Unknown Dynamic Coverage"]), 1.0, places=6)

        distribution = summarize_dynamic_topic_distribution(aligned.merged, "demo")
        top = distribution[distribution["Dynamic Topic ID"] == "DUR_0001"].iloc[0]
        self.assertEqual(int(top["Count"]), 2)
        self.assertAlmostEqual(float(top["Predicted Positive Rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(top["Unknown Topic Rate"]), 1.0, places=6)

        crosswalk = summarize_dynamic_fixed_crosswalk(aligned.merged, "demo")
        self.assertTrue(
            ((crosswalk["Dynamic Topic ID"] == "DUR_0001") & (crosswalk["Topic Final"] == "Unknown")).any()
        )

        candidates = summarize_dynamic_topic_candidates(aligned.merged, "demo")
        self.assertEqual(set(candidates["Dynamic Topic ID"]), {"DUR_0002"})

        binary_recommendations = summarize_dynamic_binary_recommendations(aligned.merged, "demo")
        self.assertEqual(set(binary_recommendations["Candidate Action"]), {"supports_current_label"})
        self.assertEqual(int(binary_recommendations["Total"].sum()), 3)

    def test_validate_accuracy_bounds_rejects_percentage_overflow(self):
        with self.assertRaises(ValueError):
            validate_accuracy_bounds(
                pd.DataFrame(
                    [
                        {"File": "demo", "Metric": "Urban Renewal", "Accuracy": 6363.64},
                    ]
                ),
                context="unit_test",
            )

    def test_boundary_bucket_unknown_conflict_and_bootstrap_outputs(self):
        truth_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, 0, 1, 0],
            }
        )
        pred_df = pd.DataFrame(
            {
                "Article Title": ["A", "B", "C", "D"],
                Schema.IS_URBAN_RENEWAL: [1, "", "", 0],
                "topic_final": ["U9", "Unknown", "Unknown", "N4"],
                "decision_source": ["stage2_classifier", "unknown_review", "unknown_review", "rule_model_fusion"],
                "review_reason": ["", "rule_local_cross_group_both_weak", "local_unknown_rule_weak", ""],
                "boundary_bucket": [
                    "same_family_or_single_source",
                    "governance_policy_finance_boundary",
                    "social_impact_boundary",
                    "same_family_or_single_source",
                ],
                "family_conflict_pattern": ["U9_vs_U9", "U9_vs_N3", "U12_vs_N4", "N4_vs_N4"],
                "topic_family_rule": ["urban", "urban", "urban", "nonurban"],
                "topic_family_local": ["urban", "nonurban", "unknown", "nonurban"],
                "llm_family_hint": ["", "1", "1", ""],
            }
        )
        aligned = align_truth_pred(truth_df, pred_df, strict=True)

        boundary = summarize_boundary_bucket_metrics(aligned.merged, "demo")
        self.assertIn("governance_policy_finance_boundary", set(boundary["Boundary Bucket"]))

        unknown_conflict = summarize_unknown_conflict_analysis(aligned.merged, "demo")
        self.assertTrue(
            (unknown_conflict["Boundary Bucket"] == "governance_policy_finance_boundary").any()
        )

        bootstrap = summarize_bootstrap_ci(aligned.merged, "demo", bootstrap_samples=80)
        self.assertEqual(set(bootstrap["Metric"]), {"Accuracy", "F1"})
        self.assertEqual(int(bootstrap.iloc[0]["Bootstrap Samples"]), 80)

    def test_mcnemar_compares_top_two_files(self):
        frame_a = pd.DataFrame(
            {
                Schema.TITLE: ["A", "B", "C", "D"],
                f"{Schema.IS_URBAN_RENEWAL}_truth": [1, 0, 1, 0],
                f"{Schema.IS_URBAN_RENEWAL}_pred": [1, 0, 1, 1],
            }
        )
        frame_b = pd.DataFrame(
            {
                Schema.TITLE: ["A", "B", "C", "D"],
                f"{Schema.IS_URBAN_RENEWAL}_truth": [1, 0, 1, 0],
                f"{Schema.IS_URBAN_RENEWAL}_pred": [1, 1, 0, 0],
            }
        )
        frame_c = pd.DataFrame(
            {
                Schema.TITLE: ["A", "B", "C", "D"],
                f"{Schema.IS_URBAN_RENEWAL}_truth": [1, 0, 1, 0],
                f"{Schema.IS_URBAN_RENEWAL}_pred": [0, 1, 0, 1],
            }
        )
        mcnemar = summarize_mcnemar({"file_a": frame_a, "file_b": frame_b, "file_c": frame_c})
        self.assertEqual(len(mcnemar), 1)
        self.assertEqual(str(mcnemar.iloc[0]["Metric"]), "Urban Renewal")
        self.assertGreaterEqual(float(mcnemar.iloc[0]["P Value"]), 0.0)
        self.assertLessEqual(float(mcnemar.iloc[0]["P Value"]), 1.0)

    def test_mcnemar_handles_aligned_frames_without_unsuffixed_title(self):
        # `align_truth_pred` merges on `_key` and produces suffixed title columns,
        # so the merged frame does not carry the unsuffixed `Schema.TITLE`.
        frame_a = pd.DataFrame(
            {
                "_key": ["k1", "k2", "k3"],
                f"{Schema.IS_URBAN_RENEWAL}_truth": [1, 0, 1],
                f"{Schema.IS_URBAN_RENEWAL}_pred": [1, 0, 0],
            }
        )
        frame_b = pd.DataFrame(
            {
                "_key": ["k1", "k2", "k3"],
                f"{Schema.IS_URBAN_RENEWAL}_truth": [1, 0, 1],
                f"{Schema.IS_URBAN_RENEWAL}_pred": [1, 1, 1],
            }
        )
        mcnemar = summarize_mcnemar({"file_a": frame_a, "file_b": frame_b})
        self.assertEqual(len(mcnemar), 1)
        self.assertEqual(str(mcnemar.iloc[0]["Metric"]), "Urban Renewal")
        self.assertGreaterEqual(float(mcnemar.iloc[0]["P Value"]), 0.0)
        self.assertLessEqual(float(mcnemar.iloc[0]["P Value"]), 1.0)


if __name__ == "__main__":
    unittest.main()
