import pandas as pd

from src.config import Schema
from src.urban.dynamic_topic_discovery import (
    DYNAMIC_BINARY_COLUMNS,
    DYNAMIC_TOPIC_COLUMNS,
    DynamicTopicConfig,
    DynamicTopicDiscovery,
    SOURCE_FULL_CORPUS_POOL,
    SOURCE_NONURBAN_REVIEW_POOL,
    SOURCE_UNKNOWN_POOL,
    STATUS_CANDIDATE_NEW_URBAN,
    STATUS_MAPPED_TO_FIXED,
    STATUS_NEEDS_REVIEW,
)


def _sample_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                Schema.TITLE: "Brownfield redevelopment and community renewal",
                Schema.ABSTRACT: "The paper studies brownfield regeneration, land reuse, and community renewal.",
                Schema.KEYWORDS_PLUS: "brownfield; redevelopment; renewal",
                "topic_final": "Unknown",
                "topic_final_group": "unknown",
                "taxonomy_coverage_status": "unknown",
                "final_label": "1",
                "urban_flag": "1",
                "review_flag_raw": 1,
                "review_reason_raw": "unknown_review",
                "llm_used": 0,
                "llm_attempted": 0,
            },
            {
                Schema.TITLE: "Brownfield regeneration and land value capture",
                Schema.ABSTRACT: "Urban brownfield renewal uses redevelopment finance and land value tools.",
                Schema.KEYWORDS_PLUS: "brownfield; finance; regeneration",
                "topic_final": "Unknown",
                "topic_final_group": "unknown",
                "taxonomy_coverage_status": "open_set",
                "final_label": "1",
                "urban_flag": "1",
                "review_flag_raw": 1,
                "review_reason_raw": "open_set",
                "llm_used": 0,
                "llm_attempted": 0,
            },
            {
                Schema.TITLE: "Traffic model for commuting demand",
                Schema.ABSTRACT: "The study builds a transport simulation model for commuting demand.",
                Schema.KEYWORDS_PLUS: "transport; model; commuting",
                "topic_final": "N7",
                "topic_final_group": "nonurban",
                "taxonomy_coverage_status": "binary_resolved",
                "final_label": "0",
                "urban_flag": "0",
                "review_flag_raw": 1,
                "review_reason_raw": "uncertain_nonurban_review",
                "llm_used": 0,
                "llm_attempted": 0,
            },
            {
                Schema.TITLE: "Housing market price volatility",
                Schema.ABSTRACT: "The paper studies housing markets without renewal or redevelopment.",
                Schema.KEYWORDS_PLUS: "housing market",
                "topic_final": "N4",
                "topic_final_group": "nonurban",
                "taxonomy_coverage_status": "covered",
                "final_label": "0",
                "urban_flag": "0",
                "review_flag_raw": 0,
                "review_reason_raw": "",
                "llm_used": 0,
                "llm_attempted": 0,
            },
        ]
    )


def test_dynamic_topic_discovery_appends_evidence_without_overwriting_core_contract():
    frame = _sample_prediction_frame()
    discovery = DynamicTopicDiscovery(
        DynamicTopicConfig(min_topic_size=1, max_topics=3, prefer_sklearn=False, mapping_min_score=0.05)
    )

    enriched = discovery.enrich(frame)

    assert set(DYNAMIC_TOPIC_COLUMNS).issubset(enriched.columns)
    assert set(DYNAMIC_BINARY_COLUMNS).issubset(enriched.columns)
    assert enriched["topic_final"].tolist() == frame["topic_final"].tolist()
    assert enriched["urban_flag"].tolist() == frame["urban_flag"].tolist()
    assert enriched["final_label"].tolist() == frame["final_label"].tolist()
    assert int(pd.to_numeric(enriched["llm_used"], errors="coerce").fillna(0).sum()) == 0
    assert int(pd.to_numeric(enriched["llm_attempted"], errors="coerce").fillna(0).sum()) == 0

    candidate_rows = enriched.iloc[:3]
    assert candidate_rows["dynamic_topic_id"].fillna("").astype(str).str.startswith("DUR_").all()
    assert not candidate_rows["dynamic_topic_keywords"].fillna("").astype(str).str.contains("unknown_review").any()
    assert enriched.loc[0, "dynamic_topic_source_pool"] == SOURCE_UNKNOWN_POOL
    assert enriched.loc[2, "dynamic_topic_source_pool"] == SOURCE_NONURBAN_REVIEW_POOL
    assert enriched.loc[3, "dynamic_topic_id"] == ""
    assert enriched.loc[0, "dynamic_binary_candidate_action"] in {"supports_current_label", "needs_review"}
    assert enriched.loc[0, "dynamic_binary_review_priority"] in {"low", "medium"}
    assert enriched.loc[0, "dynamic_mapping_status"] in {
        STATUS_MAPPED_TO_FIXED,
        STATUS_CANDIDATE_NEW_URBAN,
        STATUS_NEEDS_REVIEW,
    }


def test_dynamic_topic_discovery_can_include_full_corpus_background():
    discovery = DynamicTopicDiscovery(
        DynamicTopicConfig(min_topic_size=1, max_topics=4, prefer_sklearn=False, include_full_corpus=True)
    )

    enriched = discovery.enrich(_sample_prediction_frame())

    assert enriched.loc[3, "dynamic_topic_source_pool"] == SOURCE_FULL_CORPUS_POOL
    assert str(enriched.loc[3, "dynamic_topic_id"]).startswith("DUR_")
