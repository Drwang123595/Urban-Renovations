import pandas as pd

from src.urban.pipeline.diagnostics import build_urban_diagnostics, build_urban_diagnostics_frame


def test_build_urban_diagnostics_summarizes_boundary_risks():
    frame = pd.DataFrame(
        [
            {
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "U9",
                "topic_final_group": "urban",
                "dynamic_binary_override_applied": 1,
                "binary_topic_consistency_flag": 0,
                "review_flag": 0,
            },
            {
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "N3",
                "topic_final_group": "nonurban",
                "dynamic_binary_override_applied": "",
                "binary_topic_consistency_flag": 1,
                "review_flag": 1,
                "review_reason": "binary_topic_inconsistency",
                "decision_source": "unknown_review",
            },
            {
                "final_label": "0",
                "urban_flag": "0",
                "topic_final": "Unknown",
                "topic_final_group": "unknown",
                "dynamic_binary_override_applied": 0,
                "binary_topic_consistency_flag": 0,
                "review_flag": 1,
                "review_reason": "unknown_review",
            },
        ]
    )

    diagnostics = build_urban_diagnostics(frame)

    assert diagnostics["total_rows"] == 3
    assert diagnostics["unknown_topic_count"] == 1
    assert diagnostics["binary_topic_conflict_count"] == 1
    assert diagnostics["dynamic_binary_override_count"] == 1
    assert diagnostics["high_risk_nonurban_positive_count"] == 1
    assert diagnostics["llm_adjudication_required_count"] == 2
    assert diagnostics["final_label_counts"] == {"1": 2, "0": 1}


def test_build_urban_diagnostics_frame_is_metric_value_table():
    frame = pd.DataFrame([{"final_label": "1", "topic_final": "U1", "topic_final_group": "urban"}])

    diagnostics_frame = build_urban_diagnostics_frame(frame)

    assert diagnostics_frame.columns.tolist() == ["metric", "value"]
    assert diagnostics_frame.loc[diagnostics_frame["metric"].eq("total_rows"), "value"].iloc[0] == 1
