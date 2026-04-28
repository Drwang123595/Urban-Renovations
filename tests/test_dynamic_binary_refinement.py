import pandas as pd

from src.config import Schema
from src.urban.dynamic_binary_refinement import DynamicBinaryRefinementConfig, DynamicBinaryRefiner


def test_dynamic_binary_refiner_resolves_unknown_rows_into_binary_label_and_topic():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Brownfield redevelopment and adaptive reuse",
                Schema.ABSTRACT: "Study of brownfield redevelopment projects and urban revitalization.",
                Schema.IS_URBAN_RENEWAL: "",
                "final_label": "",
                "urban_flag": "",
                "topic_final": "Unknown",
                "topic_final_group": "unknown",
                "taxonomy_coverage_status": "unknown",
                "dynamic_topic_id": "DUR_0001",
                "dynamic_topic_size": 40,
                "dynamic_topic_confidence": 0.86,
                "dynamic_mapping_status": "mapped_to_fixed",
                "dynamic_to_fixed_topic_candidate": "U9",
                "dynamic_binary_candidate_label": "1",
                "dynamic_binary_candidate_action": "possible_false_negative_cluster",
            }
        ]
    )

    refined = DynamicBinaryRefiner(DynamicBinaryRefinementConfig()).refine(frame, mutate_final_fields=True)

    assert refined.loc[0, "dynamic_binary_override_applied"] == 1
    assert str(refined.loc[0, Schema.IS_URBAN_RENEWAL]).strip() == "1"
    assert str(refined.loc[0, "final_label"]).strip() == "1"
    assert str(refined.loc[0, "topic_final"]).strip() == "U9"
    assert str(refined.loc[0, "taxonomy_coverage_status"]).strip() in {"binary_resolved", "covered", "open_set"}


def test_dynamic_binary_refiner_requires_core_anchor_for_positive_overrides():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Traffic model for commuting demand",
                Schema.ABSTRACT: "The study builds a transport simulation model for commuting demand.",
                "final_label": "",
                "urban_flag": "",
                "topic_final": "Unknown",
                "taxonomy_coverage_status": "unknown",
                "dynamic_topic_id": "DUR_0002",
                "dynamic_topic_size": 50,
                "dynamic_topic_confidence": 0.82,
                "dynamic_mapping_status": "candidate_new_urban_topic",
                "dynamic_to_fixed_topic_candidate": "Urban_Renewal_Other",
                "dynamic_binary_candidate_label": "1",
                "dynamic_binary_candidate_action": "needs_review",
            }
        ]
    )

    refined = DynamicBinaryRefiner(DynamicBinaryRefinementConfig()).refine(frame, mutate_final_fields=True)

    assert str(refined.loc[0, "dynamic_binary_override_applied"]).strip() in {"", "0"}
    assert str(refined.loc[0, "final_label"]).strip() == ""
    assert str(refined.loc[0, "topic_final"]).strip() == "Unknown"


def test_dynamic_binary_refiner_can_flip_existing_labels_when_enabled_and_review_triggered():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban regeneration of old districts",
                Schema.ABSTRACT: "This paper studies urban regeneration and redevelopment of old districts.",
                Schema.IS_URBAN_RENEWAL: "0",
                "final_label": "0",
                "urban_flag": "0",
                "topic_final": "N4",
                "taxonomy_coverage_status": "covered",
                "review_flag_raw": 1,
                "uncertain_nonurban_guard_action": "review",
                "urban_probability_score": 0.49,
                "binary_decision_threshold": 0.5,
                "dynamic_topic_id": "DUR_0003",
                "dynamic_topic_size": 60,
                "dynamic_topic_confidence": 0.88,
                "dynamic_mapping_status": "mapped_to_fixed",
                "dynamic_to_fixed_topic_candidate": "U1",
                "dynamic_binary_candidate_label": "1",
                "dynamic_binary_candidate_action": "possible_false_negative_cluster",
            }
        ]
    )

    config = DynamicBinaryRefinementConfig(
        unknown_only=False,
        allow_flip_existing=True,
        require_review_flag_for_flip=True,
        near_threshold_margin=0.2,
    )
    refined = DynamicBinaryRefiner(config).refine(frame, mutate_final_fields=True)

    assert refined.loc[0, "dynamic_binary_override_applied"] == 1
    assert str(refined.loc[0, "final_label"]).strip() == "1"
    assert str(refined.loc[0, "topic_final"]).strip() == "U1"


def test_dynamic_binary_refiner_does_not_flip_positive_to_negative_by_default():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal in historic districts",
                Schema.ABSTRACT: "This paper studies urban renewal policy and neighborhood revitalization.",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "U7",
                "taxonomy_coverage_status": "covered",
                "review_flag_raw": 1,
                "urban_probability_score": 0.51,
                "binary_decision_threshold": 0.5,
                "dynamic_topic_id": "DUR_0004",
                "dynamic_topic_size": 60,
                "dynamic_topic_confidence": 0.90,
                "dynamic_mapping_status": "mapped_to_fixed",
                "dynamic_to_fixed_topic_candidate": "N2",
                "dynamic_binary_candidate_label": "0",
                "dynamic_binary_candidate_action": "possible_false_positive_cluster",
            }
        ]
    )

    config = DynamicBinaryRefinementConfig(
        unknown_only=False,
        allow_flip_existing=True,
        require_review_flag_for_flip=True,
        near_threshold_margin=0.2,
    )
    refined = DynamicBinaryRefiner(config).refine(frame, mutate_final_fields=True)

    assert str(refined.loc[0, "dynamic_binary_override_applied"]).strip() in {"", "0"}
    assert str(refined.loc[0, "final_label"]).strip() == "1"
    assert str(refined.loc[0, "topic_final"]).strip() == "U7"


def test_dynamic_binary_refiner_does_not_flip_unknown_positive_to_negative():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban governance and neighborhood displacement",
                Schema.ABSTRACT: "This paper studies city policy and displacement in urban communities.",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "Unknown",
                "topic_final_group": "unknown",
                "taxonomy_coverage_status": "unknown",
                "review_flag_raw": 1,
                "urban_probability_score": 0.70,
                "binary_decision_threshold": 0.45,
                "dynamic_topic_id": "DUR_9001",
                "dynamic_topic_size": 120,
                "dynamic_topic_confidence": 0.90,
                "dynamic_mapping_status": "mapped_to_fixed",
                "dynamic_to_fixed_topic_candidate": "N1",
                "dynamic_binary_candidate_label": "0",
                "dynamic_binary_candidate_action": "possible_false_positive_cluster",
            }
        ]
    )

    refined = DynamicBinaryRefiner(DynamicBinaryRefinementConfig()).refine(frame, mutate_final_fields=True)

    assert str(refined.loc[0, "dynamic_binary_override_applied"]).strip() in {"", "0"}
    assert str(refined.loc[0, "final_label"]).strip() == "1"
    assert str(refined.loc[0, "topic_final"]).strip() == "Unknown"
