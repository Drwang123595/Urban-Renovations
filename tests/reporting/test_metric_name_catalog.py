import pandas as pd

from src.reporting.metric_name_catalog import (
    display_name_for_field,
    metric_dictionary_frame,
    rename_columns_for_display,
)


def test_metric_name_catalog_covers_core_display_fields():
    dictionary = metric_dictionary_frame()
    source_fields = set(dictionary["source_field"])

    expected = {
        "final_label",
        "urban_flag",
        "topic_final",
        "taxonomy_coverage_status",
        "decision_explanation",
        "dynamic_topic_id",
        "dynamic_mapping_status",
        "dynamic_binary_candidate_label",
        "binary_policy_action",
        "llm_adjudication_required",
        "llm_used",
    }

    assert expected.issubset(source_fields)
    assert display_name_for_field("final_label") == "最终二分类标签(final_label)"
    assert display_name_for_field("taxonomy_coverage_status") == "主题覆盖状态(taxonomy_coverage_status)"
    assert "used_for_final_binary" in dictionary.columns


def test_rename_columns_for_display_does_not_mutate_values_or_source_frame():
    frame = pd.DataFrame(
        {
            "final_label": ["1", "0"],
            "topic_final": ["U1", "N1"],
            "llm_used": [0, 0],
            "unmapped_internal_field": ["a", "b"],
        }
    )

    renamed = rename_columns_for_display(frame)

    assert frame.columns.tolist() == [
        "final_label",
        "topic_final",
        "llm_used",
        "unmapped_internal_field",
    ]
    assert renamed["最终二分类标签(final_label)"].tolist() == ["1", "0"]
    assert renamed["固定主题标签(topic_final)"].tolist() == ["U1", "N1"]
    assert renamed["LLM裁决实际使用(llm_used)"].tolist() == [0, 0]
    assert renamed["unmapped_internal_field"].tolist() == ["a", "b"]
