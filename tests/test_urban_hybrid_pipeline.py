import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Schema
from src.urban_bertopic_service import BERTopicSignal
from src.urban_family_gate import FamilyGateDecision
from src.urban_hybrid_classifier import UrbanHybridClassifier
from src.urban_metadata import UrbanMetadataRecord, build_keywords
from src.urban_rule_filter import (
    METADATA_ROUTE_HARD_NEGATIVE,
    METADATA_ROUTE_UNCERTAIN,
    MetadataRouteResult,
    MetadataRuleFilter,
)
from src.urban_topic_classifier import TopicPrediction, UrbanTopicClassifier


def test_build_keywords_merges_author_and_keywords_plus():
    merged = build_keywords("Urban Renewal; Gentrification", "Gentrification; Brownfield")
    assert merged == "urban renewal; gentrification; brownfield"


def test_metadata_rule_filter_does_not_reject_metadata_only_negative():
    record = UrbanMetadataRecord(
        title="General city research",
        abstract="General urban research abstract.",
        keywords_plus="mechanical properties; recycled concrete",
        wos_categories="Mechanics",
        research_areas="Engineering",
    )
    result = MetadataRuleFilter().evaluate(record)
    assert result.route == METADATA_ROUTE_UNCERTAIN
    assert result.stage1_decision == "pass"
    assert "mechanics" in result.matched_negative_domains


def test_metadata_rule_filter_excludes_math_term_misuse_to_n8():
    record = UrbanMetadataRecord(
        title="Urban renewal moves in dimer models",
        abstract="This paper studies bipartite graph tiling methods and combinatorics.",
    )
    result = MetadataRuleFilter().evaluate(record)
    assert result.route == METADATA_ROUTE_HARD_NEGATIVE
    assert result.stage1_decision == "exclude"
    assert result.reason == "math_term_misuse"
    assert result.topic_rule == "N8"


def test_metadata_rule_filter_excludes_rural_nonurban_to_n9():
    record = UrbanMetadataRecord(
        title="Governing property-led rural regeneration in Ireland",
        abstract="This article examines the rural renewal scheme and rural development effects.",
    )
    result = MetadataRuleFilter().evaluate(record)
    assert result.route == METADATA_ROUTE_HARD_NEGATIVE
    assert result.reason == "rural_nonurban"
    assert result.topic_rule == "N9"


def test_metadata_rule_filter_marks_greenfield_risk_and_n2_candidate():
    record = UrbanMetadataRecord(
        title="New town expansion and housing growth at the urban fringe",
        abstract="This paper examines greenfield suburban growth and sprawl.",
    )
    result = MetadataRuleFilter().evaluate(record)
    assert result.route == METADATA_ROUTE_UNCERTAIN
    assert result.stage1_decision == "pass"
    assert "greenfield_expansion_risk" in result.stage1_risk_tags
    assert result.topic_rule == "N2"


def test_metadata_rule_filter_detects_explicit_renewal_other_object_governance_to_n3():
    record = UrbanMetadataRecord(
        title="Urban regeneration discourse in city governance networks",
        abstract=(
            "This paper studies local governance, institutional discourse and policy narratives in cities. "
            "Urban regeneration is discussed as a general background context rather than a concrete project or intervention."
        ),
    )
    result = MetadataRuleFilter().evaluate(record)
    assert result.route == METADATA_ROUTE_UNCERTAIN
    assert result.topic_rule == "N3"
    assert result.rule_high_confidence is True
    assert "explicit_renewal_wording_but_other_object" in result.stage1_risk_tags


def test_metadata_rule_filter_detects_explicit_renewal_other_object_method_to_n8():
    record = UrbanMetadataRecord(
        title="A machine learning framework for urban renewal policy text mining",
        abstract=(
            "This article proposes a deep learning and remote sensing evaluation framework. "
            "Urban renewal is only a broad application context and no specific project object or intervention is studied."
        ),
    )
    result = MetadataRuleFilter().evaluate(record)
    assert result.route == METADATA_ROUTE_UNCERTAIN
    assert result.topic_rule == "N8"
    assert result.rule_high_confidence is True


def test_topic_classifier_supports_title_abstract_only_prediction_without_training(tmp_path):
    classifier = UrbanTopicClassifier(
        train_dir=tmp_path,
        master_metadata_path=tmp_path / "missing.xlsx",
    )
    urban_record = UrbanMetadataRecord(
        title="Brownfield redevelopment and regeneration in an old industrial district",
        abstract="Studies redevelopment governance and regeneration financing in an existing industrial site.",
    )
    urban_pred = classifier.predict(urban_record)
    assert urban_pred.topic_label == "U5"
    assert urban_pred.topic_group == "urban"

    nonurban_record = UrbanMetadataRecord(
        title="Optimization algorithm for transport accessibility prediction",
        abstract="This paper proposes a neural network framework for mobility classification.",
    )
    nonurban_pred = classifier.predict(nonurban_record)
    assert nonurban_pred.topic_label in {"N7", "N8"}
    assert nonurban_pred.topic_group == "nonurban"


def test_topic_classifier_ignores_metadata_fields_in_prediction(tmp_path):
    classifier = UrbanTopicClassifier(
        train_dir=tmp_path,
        master_metadata_path=tmp_path / "missing.xlsx",
    )
    common = {
        "title": "Adaptive reuse strategies in historic districts",
        "abstract": "This paper evaluates adaptive reuse in a historic district and heritage-led regeneration.",
    }
    record_a = UrbanMetadataRecord(
        **common,
        keywords_plus="remote sensing; gis",
        wos_categories="Engineering, Civil",
        research_areas="Remote Sensing",
    )
    record_b = UrbanMetadataRecord(
        **common,
        keywords_plus="mechanical properties",
        wos_categories="Mechanics",
        research_areas="Materials Science",
    )
    pred_a = classifier.predict(record_a)
    pred_b = classifier.predict(record_b)
    assert pred_a.topic_label == pred_b.topic_label
    assert pred_a.topic_group == pred_b.topic_group
    assert pred_a.binary_probability == pred_b.binary_probability


def test_topic_classifier_short_text_stays_low_confidence(tmp_path):
    classifier = UrbanTopicClassifier(
        train_dir=tmp_path,
        master_metadata_path=tmp_path / "missing.xlsx",
    )
    record = UrbanMetadataRecord(title="Urban renewal", abstract="")
    pred = classifier.predict(record)
    assert pred.confidence <= 0.45


def test_topic_classifier_does_not_auto_treat_plain_gentrification_as_u12(tmp_path):
    classifier = UrbanTopicClassifier(
        train_dir=tmp_path,
        master_metadata_path=tmp_path / "missing.xlsx",
    )
    record = UrbanMetadataRecord(
        title="The new gentrifiers and neighborhood change",
        abstract="This paper studies social class change, identity and housing markets in changing neighborhoods.",
    )
    pred = classifier.predict(record)
    assert pred.topic_label != "U12"


def test_topic_classifier_keeps_redevelopment_finance_positive(tmp_path):
    classifier = UrbanTopicClassifier(
        train_dir=tmp_path,
        master_metadata_path=tmp_path / "missing.xlsx",
    )
    record = UrbanMetadataRecord(
        title="Tax increment financing and PPP in urban redevelopment",
        abstract="This paper analyzes TIF, PPP, redevelopment finance and land value capture in urban renewal projects.",
    )
    pred = classifier.predict(record)
    assert pred.topic_label == "U10"
    assert pred.topic_group == "urban"


class _NoCallLLMStrategy:
    def process(self, *args, **kwargs):
        raise AssertionError("LLM review should not be called")


class _LLMStrategy:
    def process(self, *args, **kwargs):
        return {
            Schema.IS_URBAN_RENEWAL: "1",
            "urban_parse_reason": "single_digit_line",
        }


class _NegativeLLMStrategy:
    def process(self, *args, **kwargs):
        return {
            Schema.IS_URBAN_RENEWAL: "0",
            "urban_parse_reason": "single_digit_line",
        }


def _prediction(
    topic_label: str,
    topic_group: str,
    *,
    topic_name: str = "topic",
    confidence: float,
    margin: float,
    matched_terms=None,
    binary_score: float = 0.0,
    binary_probability: float = 0.5,
    top_candidates=None,
):
    return TopicPrediction(
        topic_label=topic_label,
        topic_group=topic_group,
        topic_name=topic_name,
        confidence=confidence,
        matched_terms=matched_terms or [],
        binary_score=binary_score,
        binary_probability=binary_probability,
        margin=margin,
        top_candidates=top_candidates or [],
    )


class _HighConfidenceUrbanClassifier:
    def predict(self, _record):
        return _prediction(
            "U9",
            "urban",
            topic_name="renewal governance institutions participation",
            confidence=0.91,
            margin=2.4,
            matched_terms=["abstract:governance"],
            binary_score=1.7,
            binary_probability=0.83,
            top_candidates=["U9:8.40", "U10:6.00", "N3:3.20"],
        )


class _UnknownClassifier:
    def predict(self, _record):
        return _prediction(
            "Unknown",
            "unknown",
            topic_name="Unknown",
            confidence=0.34,
            margin=0.2,
            matched_terms=[],
            binary_score=0.0,
            binary_probability=0.51,
            top_candidates=["U9:2.20", "N3:2.10", "U15:1.90"],
        )


class _WeakConflictClassifier:
    def predict(self, _record):
        return _prediction(
            "N3",
            "nonurban",
            topic_name="general urban governance",
            confidence=0.48,
            margin=0.8,
            matched_terms=["abstract:governance"],
            binary_score=-0.1,
            binary_probability=0.49,
            top_candidates=["N3:4.20", "U9:3.80", "N1:2.70"],
        )


class _StrongUrbanConflictClassifier:
    def predict(self, _record):
        return _prediction(
            "U9",
            "urban",
            topic_name="renewal governance institutions participation",
            confidence=0.91,
            margin=4.2,
            matched_terms=["abstract:urban regeneration", "abstract:participation"],
            binary_score=1.6,
            binary_probability=0.82,
            top_candidates=["U9:8.40", "N3:5.20", "U10:4.10"],
        )


class _VeryStrongUrbanConflictClassifier:
    def predict(self, _record):
        return _prediction(
            "U9",
            "urban",
            topic_name="renewal governance institutions participation",
            confidence=0.92,
            margin=4.2,
            matched_terms=["abstract:urban regeneration", "abstract:participation"],
            binary_score=1.8,
            binary_probability=0.87,
            top_candidates=["U9:8.90", "N3:4.70", "U10:3.10"],
        )


class _HousingMarketClassifier:
    def predict(self, _record):
        return _prediction(
            "N4",
            "nonurban",
            topic_name="general housing market and real estate",
            confidence=0.61,
            margin=0.24,
            matched_terms=["abstract:housing market"],
            binary_score=0.2,
            binary_probability=0.54,
            top_candidates=["N4:4.40", "U12:4.10", "N1:3.60"],
        )


class _UrbanCoreN1Classifier:
    def predict(self, _record):
        return _prediction(
            "N1",
            "nonurban",
            topic_name="general urbanization and expansion",
            confidence=0.56,
            margin=0.35,
            matched_terms=["abstract:urban development"],
            binary_score=0.1,
            binary_probability=0.53,
            top_candidates=["N1:4.30", "U4:4.00", "N4:3.20"],
        )


class _StrongNonurbanUnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["N3", "U9", "U12"],
            matched_candidate_terms=["abstract:governance", "abstract:policy"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["abstract:governance", "abstract:policy"],
            stage1_risk_tags=["explicit_renewal_wording_but_other_object"],
            stage1_conflict_flag=1,
            topic_rule="N3",
            topic_rule_group="nonurban",
            topic_rule_name="general urban governance",
            topic_rule_score=6.4,
            topic_rule_margin=2.8,
            topic_rule_top3=["N3:6.40", "U9:5.10", "U12:3.20"],
            topic_rule_matches=["abstract:governance", "abstract:policy"],
            rule_high_confidence=True,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


class _WeakVeryLowNonurbanUnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["N1", "N3"],
            matched_candidate_terms=["abstract:suburban growth"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["abstract:suburban growth"],
            stage1_risk_tags=[],
            stage1_conflict_flag=0,
            topic_rule="N1",
            topic_rule_group="nonurban",
            topic_rule_name="general urbanization and expansion",
            topic_rule_score=1.0,
            topic_rule_margin=0.0,
            topic_rule_top3=["N1:1.00", "N3:0.80"],
            topic_rule_matches=["abstract:suburban growth"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_low_confidence",
        )


class _UrbanCoreRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U4", "N1", "N4"],
            matched_candidate_terms=["title:urban redevelopment", "abstract:downtown"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["title:urban redevelopment", "abstract:downtown"],
            stage1_risk_tags=[],
            stage1_conflict_flag=1,
            topic_rule="U4",
            topic_rule_group="urban",
            topic_rule_name="old city inner city and downtown regeneration",
            topic_rule_score=4.2,
            topic_rule_margin=0.4,
            topic_rule_top3=["U4:4.20", "N1:4.00", "N4:3.10"],
            topic_rule_matches=["title:urban redevelopment", "abstract:downtown"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


class _U1UnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U1", "N1", "N3"],
            matched_candidate_terms=[
                "title:regeneration",
                "abstract:regeneration",
                "anchor:urban regeneration",
                "anchor:regeneration",
            ],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=[
                "title:regeneration",
                "abstract:regeneration",
                "anchor:urban regeneration",
                "anchor:regeneration",
            ],
            stage1_risk_tags=[],
            stage1_conflict_flag=0,
            topic_rule="U1",
            topic_rule_group="urban",
            topic_rule_name="urban renewal and regeneration projects",
            topic_rule_score=3.85,
            topic_rule_margin=0.0,
            topic_rule_top3=["U1:3.85", "N1:3.50", "N3:2.60"],
            topic_rule_matches=[
                "title:regeneration",
                "abstract:regeneration",
                "anchor:urban regeneration",
                "anchor:regeneration",
            ],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_low_confidence",
        )


class _U4UnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U4", "N1", "N7"],
            matched_candidate_terms=[
                "title:redevelopment",
                "abstract:redevelopment",
                "anchor:urban redevelopment",
                "anchor:redevelopment",
            ],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=[
                "title:redevelopment",
                "abstract:redevelopment",
                "anchor:urban redevelopment",
                "anchor:redevelopment",
            ],
            stage1_risk_tags=[],
            stage1_conflict_flag=0,
            topic_rule="U4",
            topic_rule_group="urban",
            topic_rule_name="old city inner city and downtown regeneration",
            topic_rule_score=3.85,
            topic_rule_margin=1.0,
            topic_rule_top3=["U4:3.85", "N1:3.10", "N7:2.40"],
            topic_rule_matches=[
                "title:redevelopment",
                "abstract:redevelopment",
                "anchor:urban redevelopment",
                "anchor:redevelopment",
            ],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_low_confidence",
        )


class _GentrificationHousingRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U12", "N4", "N1"],
            matched_candidate_terms=["title:gentrification", "abstract:neighborhood change"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["title:gentrification", "abstract:neighborhood change"],
            stage1_risk_tags=[],
            stage1_conflict_flag=1,
            topic_rule="U12",
            topic_rule_group="urban",
            topic_rule_name="gentrification exclusion and neighborhood change",
            topic_rule_score=5.3,
            topic_rule_margin=1.2,
            topic_rule_top3=["U12:5.30", "N4:4.80", "N1:3.60"],
            topic_rule_matches=["title:gentrification", "abstract:neighborhood change"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


class _U12UnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U12", "N4", "N7"],
            matched_candidate_terms=[
                "title:gentrification",
                "abstract:gentrification",
                "anchor:gentrification",
            ],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=[
                "title:gentrification",
                "abstract:gentrification",
                "anchor:gentrification",
            ],
            stage1_risk_tags=[],
            stage1_conflict_flag=0,
            topic_rule="U12",
            topic_rule_group="urban",
            topic_rule_name="gentrification exclusion and neighborhood change",
            topic_rule_score=5.10,
            topic_rule_margin=0.75,
            topic_rule_top3=["U12:5.10", "N4:4.20", "N7:3.10"],
            topic_rule_matches=[
                "title:gentrification",
                "abstract:gentrification",
                "anchor:gentrification",
            ],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_low_confidence",
        )


class _WeakUrbanUnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U15", "U9", "N3"],
            matched_candidate_terms=["abstract:health impact", "abstract:regeneration"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["abstract:health impact", "abstract:regeneration"],
            stage1_risk_tags=[],
            stage1_conflict_flag=0,
            topic_rule="U15",
            topic_rule_group="urban",
            topic_rule_name="renewal comprehensive impacts",
            topic_rule_score=3.8,
            topic_rule_margin=1.7,
            topic_rule_top3=["U15:3.80", "U9:2.40", "N3:2.10"],
            topic_rule_matches=["abstract:health impact", "abstract:regeneration"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_low_confidence",
        )


class _VeryWeakU12UnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U12", "N4", "N1"],
            matched_candidate_terms=["title:gentrification", "abstract:neighborhood change"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["title:gentrification", "abstract:neighborhood change"],
            stage1_risk_tags=[],
            stage1_conflict_flag=1,
            topic_rule="U12",
            topic_rule_group="urban",
            topic_rule_name="gentrification exclusion and neighborhood change",
            topic_rule_score=1.25,
            topic_rule_margin=0.25,
            topic_rule_top3=["U12:1.25", "N4:1.10", "N1:0.90"],
            topic_rule_matches=["title:gentrification", "abstract:neighborhood change"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


class _VeryWeakUrbanNoBundleRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U15", "N3", "N5"],
            matched_candidate_terms=["abstract:community impact"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["abstract:community impact"],
            stage1_risk_tags=[],
            stage1_conflict_flag=1,
            topic_rule="U15",
            topic_rule_group="urban",
            topic_rule_name="renewal comprehensive impacts",
            topic_rule_score=1.25,
            topic_rule_margin=0.25,
            topic_rule_top3=["U15:1.25", "N3:1.10", "N5:0.95"],
            topic_rule_matches=["abstract:community impact"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


class _WeakNonurbanUnknownRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["N3", "N1", "U9"],
            matched_candidate_terms=["abstract:policy", "abstract:governance"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["abstract:policy", "abstract:governance"],
            stage1_risk_tags=[],
            stage1_conflict_flag=0,
            topic_rule="N3",
            topic_rule_group="nonurban",
            topic_rule_name="general urban governance",
            topic_rule_score=3.7,
            topic_rule_margin=1.8,
            topic_rule_top3=["N3:3.70", "N1:2.90", "U9:2.40"],
            topic_rule_matches=["abstract:policy", "abstract:governance"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_low_confidence",
        )


class _WeakRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["U9", "N3", "U15"],
            matched_candidate_terms=["abstract:governance"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["abstract:governance"],
            stage1_risk_tags=[],
            stage1_conflict_flag=1,
            topic_rule="U9",
            topic_rule_group="urban",
            topic_rule_name="renewal governance institutions participation",
            topic_rule_score=4.1,
            topic_rule_margin=0.6,
            topic_rule_top3=["U9:4.10", "N3:3.90", "U15:2.50"],
            topic_rule_matches=["abstract:governance"],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


def _signal(
    *,
    available=True,
    status="ready",
    topic_id=1,
    topic_name="cluster",
    topic_probability=0.81,
    is_outlier=False,
    topic_count=48,
    topic_pos_rate=0.9,
    mapped_label="U9",
    mapped_group="urban",
    mapped_name="renewal governance institutions participation",
    label_purity=0.84,
    mapped_label_share=0.78,
    top_terms="urban renewal, governance, participation",
):
    return BERTopicSignal(
        available=available,
        status=status,
        topic_id=topic_id,
        topic_name=topic_name,
        topic_probability=topic_probability,
        is_outlier=is_outlier,
        topic_count=topic_count,
        topic_pos_rate=topic_pos_rate,
        mapped_label=mapped_label,
        mapped_group=mapped_group,
        mapped_name=mapped_name,
        label_purity=label_purity,
        mapped_label_share=mapped_label_share,
        top_terms=top_terms,
        reason="",
    )


class _FakeBERTopicService:
    def __init__(self, signal):
        self.signal = signal

    def predict(self, _record):
        return self.signal


class _NullBERTopicService(_FakeBERTopicService):
    def __init__(self):
        super().__init__(_signal(available=False, status="disabled", mapped_label="", mapped_group="", mapped_name=""))


class _StubFamilyGate:
    def __init__(self, *, final_family: str, confidence: float = 0.78, probability_urban: float = 0.82):
        self.enabled = True
        self.final_family = final_family
        self.confidence = confidence
        self.probability_urban = probability_urban

    def describe_conflict(self, *, rule_label: str, local_label: str, rule_family: str, local_family: str):
        return "governance_policy_finance_boundary", f"{rule_label}_vs_{local_label}"

    def predict(self, *, route_result, topic_prediction, **kwargs):
        return FamilyGateDecision(
            rule_family=route_result.topic_rule_group,
            local_family=topic_prediction.topic_group,
            final_family=self.final_family,
            confidence=self.confidence,
            probability_urban=self.probability_urban,
            decision_source="family_gate_stub",
            boundary_bucket="governance_policy_finance_boundary",
            family_conflict_pattern=f"{route_result.topic_rule}_vs_{topic_prediction.topic_label}",
            features={},
        )


class _AnchorGuardNeutralFamilyGate:
    def __init__(self, *, final_family: str = "nonurban", confidence: float = 0.92, probability_urban: float = 0.18):
        self.enabled = True
        self.final_family = final_family
        self.confidence = confidence
        self.probability_urban = probability_urban

    def describe_conflict(self, *, rule_label: str, local_label: str, rule_family: str, local_family: str):
        return "nonurban_rule_nonurban_local", f"{rule_label}_vs_{local_label}"

    def predict(self, *, route_result, topic_prediction, **kwargs):
        return FamilyGateDecision(
            rule_family=route_result.topic_rule_group,
            local_family=topic_prediction.topic_group,
            final_family=self.final_family,
            confidence=self.confidence,
            probability_urban=self.probability_urban,
            decision_source="family_gate_anchor_guard_neutral",
            boundary_bucket="nonurban_rule_nonurban_local",
            family_conflict_pattern=f"{route_result.topic_rule}_vs_{topic_prediction.topic_label}",
            features={},
        )


class _AnchorGuardNonurbanWithUrbanAltClassifier:
    def predict(self, _record):
        return _prediction(
            "N3",
            "nonurban",
            topic_name="general urban governance",
            confidence=0.67,
            margin=1.8,
            matched_terms=["abstract:policy", "abstract:governance"],
            binary_score=0.65,
            binary_probability=0.59,
            top_candidates=["N3:6.20", "U9:4.70", "N1:2.90"],
        )


class _AnchorGuardNonurbanNoSupportClassifier:
    def predict(self, _record):
        return _prediction(
            "N3",
            "nonurban",
            topic_name="general urban governance",
            confidence=0.63,
            margin=1.5,
            matched_terms=["abstract:policy", "abstract:governance"],
            binary_score=-0.2,
            binary_probability=0.21,
            top_candidates=["N3:6.00", "N1:4.10", "N4:3.30"],
        )


class _AnchorGuardFallbackUrbanCandidateClassifier:
    def predict(self, _record):
        return _prediction(
            "N3",
            "nonurban",
            topic_name="general urban governance",
            confidence=0.66,
            margin=1.2,
            matched_terms=["abstract:policy", "abstract:governance"],
            binary_score=0.18,
            binary_probability=0.34,
            top_candidates=["N3:6.40", "U9:5.20", "U10:3.90", "N1:3.20"],
        )


class _N8TechnicalRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["N8", "N1", "N3"],
            matched_candidate_terms=["abstract:model", "abstract:optimization"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="rule_topic_n8",
            stage1_hit_signals=["abstract:model", "abstract:optimization"],
            stage1_risk_tags=["generic_technical_risk"],
            stage1_conflict_flag=0,
            topic_rule="N8",
            topic_rule_group="nonurban",
            topic_rule_name="pure methods algorithms and modeling",
            topic_rule_score=5.8,
            topic_rule_margin=2.1,
            topic_rule_top3=["N8:5.80", "N1:2.60", "N3:2.20"],
            topic_rule_matches=["abstract:model", "abstract:optimization"],
            rule_high_confidence=True,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


class _N8RenewalRiskRuleFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["N8", "U9", "N3"],
            matched_candidate_terms=["abstract:urban renewal", "abstract:model"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="rule_topic_n8",
            stage1_hit_signals=["abstract:urban renewal", "abstract:model"],
            stage1_risk_tags=["explicit_renewal_wording_but_other_object"],
            stage1_conflict_flag=1,
            topic_rule="N8",
            topic_rule_group="nonurban",
            topic_rule_name="pure methods algorithms and modeling",
            topic_rule_score=6.2,
            topic_rule_margin=2.6,
            topic_rule_top3=["N8:6.20", "U9:4.90", "N3:3.10"],
            topic_rule_matches=["abstract:urban renewal", "abstract:model"],
            rule_high_confidence=True,
            review_flag_rule=1,
            review_reason_rule="rule_group_conflict",
        )


class _N8LocalAgreeClassifier:
    def predict(self, _record):
        return _prediction(
            "N8",
            "nonurban",
            topic_name="pure methods algorithms and modeling",
            confidence=0.63,
            margin=1.4,
            matched_terms=["abstract:model"],
            binary_score=0.1,
            binary_probability=0.41,
            top_candidates=["N8:6.10", "N3:2.90", "N1:2.20"],
        )


class _N8UrbanConflictClassifier:
    def predict(self, _record):
        return _prediction(
            "U9",
            "urban",
            topic_name="renewal governance institutions participation",
            confidence=0.40,
            margin=1.4,
            matched_terms=["abstract:urban renewal", "abstract:governance"],
            binary_score=1.1,
            binary_probability=0.61,
            top_candidates=["U9:6.40", "N8:5.80", "N3:2.40"],
        )


class _UnknownRuleReviewFlagFilter:
    def evaluate(self, _record):
        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason="uncertain_pass",
            candidate_topic_buckets=["N3", "N1"],
            matched_candidate_terms=["abstract:policy"],
            matched_negative_domains=[],
            matched_negative_keywords=[],
            matched_related_domains=["urban studies"],
            stage1_decision="pass",
            stage1_reason_tag="uncertain_pass",
            stage1_hit_signals=["abstract:policy"],
            stage1_risk_tags=[],
            stage1_conflict_flag=0,
            topic_rule="Unknown",
            topic_rule_group="unknown",
            topic_rule_name="Unknown",
            topic_rule_score=0.5,
            topic_rule_margin=0.0,
            topic_rule_top3=["Unknown:0.50", "N3:0.40", "N1:0.35"],
            topic_rule_matches=[],
            rule_high_confidence=False,
            review_flag_rule=1,
            review_reason_rule="rule_unknown",
        )


def _assert_explainability_contract(result):
    for column in [
        "decision_explanation",
        "primary_positive_evidence",
        "primary_negative_evidence",
        "evidence_balance",
        "decision_rule_stack",
        "binary_decision_evidence",
        "urban_probability_score",
        "binary_decision_source",
        "taxonomy_coverage_status",
    ]:
        assert str(result.get(column, "")).strip()
    explanation = result["decision_explanation"]
    assert "score=" in explanation
    assert "source=" in explanation
    assert "topic=" in explanation
    assert "review=" in explanation


def _assert_final_label_follows_binary_score(result):
    expected = (
        "1"
        if float(result["urban_probability_score"]) >= float(result["binary_decision_threshold"])
        else "0"
    )
    assert result["final_label"] == expected
    assert result["urban_flag"] == expected
    assert result[Schema.IS_URBAN_RENEWAL] == expected


def test_hybrid_classifier_short_circuits_stage1_hard_negative(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _HighConfidenceUrbanClassifier(),
    )
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    result = classifier.classify(
        "Urban renewal moves in dimer models",
        "This paper studies bipartite graph tiling methods and combinatorics.",
    )
    assert result["final_label"] == "0"
    assert result["decision_source"] == "stage1_rule"
    assert result["topic_label"] == "N8"
    assert result["topic_final"] == "N8"
    _assert_explainability_contract(result)


def test_hybrid_classifier_uses_rule_model_fusion_when_rule_and_local_agree(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _HighConfidenceUrbanClassifier(),
    )
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    result = classifier.classify(
        "Urban renewal governance and participation in old districts",
        "This paper studies governance, stakeholders and participation in urban renewal implementation.",
    )
    assert result["final_label"] == "1"
    assert result["decision_source"] == "rule_model_fusion"
    assert result["topic_final"] == "U9"
    assert result["topic_label"] == "U9"
    _assert_explainability_contract(result)


def test_hybrid_classifier_prefers_rule_when_local_unknown(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    result = classifier.classify(
        "Tax increment financing for urban renewal projects",
        "This article studies redevelopment finance, PPP, and land value capture in urban renewal.",
    )
    assert result["final_label"] == "1"
    assert result["decision_source"] == "rule_model_fusion"
    assert result["topic_final"] == "U10"


def test_hybrid_classifier_returns_unknown_for_cross_group_weak_conflict(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _WeakConflictClassifier(),
    )
    classifier = UrbanHybridClassifier(_LLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _WeakRuleFilter()
    result = classifier.classify(
        "Governance and participation in contested districts",
        "This paper examines governance networks and local institutions in contested districts.",
    )
    assert result["decision_source"] == "unknown_review"
    assert result["final_label"] in {"0", "1"}
    assert result["urban_flag"] == result["final_label"]
    assert result[Schema.IS_URBAN_RENEWAL] == result["final_label"]
    assert result["topic_final"] == "Unknown"
    assert result["review_flag_raw"] == 1
    assert result["review_flag"] == 1
    assert result["binary_decision_source"] == "binary_confidence_resolution"
    assert result["binary_topic_consistency_flag"] == 1
    assert result["taxonomy_coverage_status"] == "unknown"
    assert "topic=Unknown/unknown" in result["decision_explanation"]
    assert "topic_final=Unknown" in result["primary_negative_evidence"]
    _assert_explainability_contract(result)
    assert result["binary_audit_resolution_action"] == "positive_binary_conflict_audit"
    assert result["llm_attempted"] == 1
    assert result["llm_used"] == 0
    assert result["llm_family_hint"] == "1"


def test_hybrid_final_binary_label_is_not_overwritten_by_unknown_topic():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    base = {
        "urban_probability_score": 0.72,
        "binary_decision_threshold": 0.45,
        "binary_decision_source": "binary_confidence_resolution",
        "binary_decision_evidence": "score_above_threshold",
        "taxonomy_coverage_status": "unknown",
        "topic_rule": "",
        "topic_local_label": "Unknown",
        "family_probability_urban": 0.8,
        "topic_binary_probability": 0.7,
        "llm_family_hint": "",
        "stage1_risk_tags": "",
    }

    result = classifier._build_final_result(
        base,
        final_topic="Unknown",
        decision_source="unknown_review",
        decision_reason="binary_positive_topic_unknown",
        confidence=0.72,
        review_flag=1,
        review_reason="binary_topic_inconsistency",
        binary_label="1",
    )

    assert result["final_label"] == "1"
    assert result["urban_flag"] == "1"
    assert result[Schema.IS_URBAN_RENEWAL] == "1"
    assert result["topic_final"] == "Unknown"
    assert result["binary_topic_consistency_flag"] == 1
    assert result["review_flag"] == 1
    assert result["binary_decision_source"] == "binary_confidence_resolution"
    assert result["binary_topic_consistency_flag"] == 1
    assert result["taxonomy_coverage_status"] == "unknown"
    assert "topic=Unknown/unknown" in result["decision_explanation"]
    assert "topic_final=Unknown" in result["primary_negative_evidence"]
    _assert_explainability_contract(result)
    assert result["binary_audit_resolution_action"] == "positive_binary_conflict_audit"


def test_hybrid_classifier_can_disable_llm_for_unknown_review(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _WeakConflictClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _LLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
    )
    classifier.rule_filter = _WeakRuleFilter()
    result = classifier.classify(
        "Governance and participation in contested districts",
        "This paper examines governance networks and local institutions in contested districts.",
    )
    assert result["decision_source"] == "unknown_review"
    assert result["final_label"] in {"0", "1"}
    assert result["urban_flag"] == result["final_label"]
    assert result["binary_decision_source"] == "binary_confidence_resolution"
    assert result["llm_attempted"] == 0
    assert result["llm_family_hint"] == ""


def test_hybrid_classifier_keeps_bertopic_as_auxiliary_hint_only(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _HighConfidenceUrbanClassifier(),
    )
    bertopic_service = _FakeBERTopicService(
        _signal(
            mapped_label="N3",
            mapped_group="nonurban",
            mapped_name="general urban governance",
            topic_probability=0.83,
            topic_count=58,
            label_purity=0.87,
            mapped_label_share=0.82,
        )
    )
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=bertopic_service)
    result = classifier.classify(
        "Urban renewal governance and participation in old districts",
        "This paper studies governance, stakeholders and participation in urban renewal implementation.",
    )
    assert result["decision_source"] != "bertopic_primary"
    assert result["topic_final"] == "U9"
    assert result["bertopic_hint_label"] == "N3"
    assert result["bertopic_hint_conflict_flag"] == 1


def test_hybrid_classifier_resolves_unknown_when_rule_urban_and_llm_family_align(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(_LLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _WeakUrbanUnknownRuleFilter()
    result = classifier.classify(
        "Health impacts in regeneration districts",
        "This paper studies health, well-being and quality of life in regeneration areas.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U15"
    assert result["final_label"] == "1"
    assert result["llm_family_hint"] == "1"
    _assert_explainability_contract(result)


def test_hybrid_classifier_resolves_unknown_when_rule_nonurban_and_llm_family_align(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(_NegativeLLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _WeakNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Local governance narratives in city policy",
        "This paper studies policy discourse and city governance in a general urban context.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "N3"
    _assert_final_label_follows_binary_score(result)
    assert result["binary_decision_source"] == "binary_confidence_resolution"
    assert float(result["urban_probability_score"]) >= float(result["binary_decision_threshold"])
    assert result["binary_recall_calibration_tier"] == "context_relevance_floor"


def test_hybrid_classifier_offline_recovers_u1_when_online_llm_hints_disabled(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    monkeypatch.setattr("src.config.Config.URBAN_HYBRID_ONLINE_LLM_HINTS_ENABLED", False)
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _U1UnknownRuleFilter()
    result = classifier.classify(
        "Urban regeneration strategy in old districts",
        "This paper studies regeneration and implementation in an existing urban district.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U1"
    assert result["final_label"] == "1"
    assert result["llm_attempted"] == 0
    assert result["unknown_recovery_path"] == "unknown_offline_curated_rule"
    assert "trigger=u1_anchor_bundle" in result["unknown_recovery_evidence"]


def test_hybrid_classifier_offline_recovers_u12_gentrification_without_llm(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
    )
    classifier.rule_filter = _U12UnknownRuleFilter()
    result = classifier.classify(
        "Gentrification and neighborhood transition",
        "This paper studies gentrification and displacement in an existing neighborhood.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U12"
    assert result["final_label"] == "1"
    assert result["unknown_recovery_path"] == "unknown_offline_curated_rule"
    assert "trigger=u12_gentrification_bundle" in result["unknown_recovery_evidence"]


def test_hybrid_classifier_offline_recovers_u4_redevelopment_without_llm(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
    )
    classifier.rule_filter = _U4UnknownRuleFilter()
    result = classifier.classify(
        "Urban redevelopment in inner-city districts",
        "This paper studies redevelopment and revitalization in existing downtown areas.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U4"
    assert result["final_label"] == "1"
    assert result["unknown_recovery_path"] == "unknown_offline_curated_rule"
    assert "trigger=u4_redevelopment_bundle" in result["unknown_recovery_evidence"]


def test_hybrid_classifier_offline_recovers_strong_local_u9_without_llm(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _VeryStrongUrbanConflictClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
    )
    classifier.rule_filter = _StrongNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Coalitions in urban regeneration",
        "This paper studies participation, governance and coalitions in urban regeneration programmes.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U9"
    assert result["final_label"] == "1"
    assert result["unknown_recovery_path"] == "unknown_offline_curated_local"
    assert "trigger=n3_to_u9_strong_local" in result["unknown_recovery_evidence"]


def test_hybrid_classifier_resolves_cross_group_unknown_with_dual_urban_support(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _WeakConflictClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _LLMStrategy(),
        bertopic_service=_FakeBERTopicService(
            _signal(
                mapped_label="U9",
                mapped_group="urban",
                mapped_name="renewal governance institutions participation",
                topic_probability=0.82,
                topic_count=56,
                label_purity=0.86,
                mapped_label_share=0.79,
            )
        ),
    )
    classifier.rule_filter = _WeakRuleFilter()
    result = classifier.classify(
        "Governance and participation in contested districts",
        "This paper examines governance networks and local institutions in contested districts.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U9"
    assert result["final_label"] == "1"


def test_hybrid_classifier_resolves_curated_nonurban_rule_to_strong_local_urban(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _StrongUrbanConflictClassifier(),
    )
    classifier = UrbanHybridClassifier(_LLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _StrongNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Coalitions in urban regeneration",
        "This paper studies participation, governance and coalitions in urban regeneration programmes.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U9"
    assert result["final_label"] == "1"


def test_hybrid_classifier_resolves_curated_u12_rule_over_housing_market_local(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _HousingMarketClassifier(),
    )
    classifier = UrbanHybridClassifier(_LLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _GentrificationHousingRuleFilter()
    result = classifier.classify(
        "Gentrification and neighborhood change",
        "This paper studies gentrification, neighborhood change and housing value shifts in an existing city.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U12"
    assert result["final_label"] == "1"


def test_hybrid_classifier_resolves_weak_nonurban_rule_when_local_unknown_and_llm_negative(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(_NegativeLLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _WeakNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Suburban housing and city growth",
        "This paper studies a general city context and suburban growth without renewal intervention.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "N3"
    assert result["final_label"] == "0"


def test_hybrid_classifier_resolves_very_weak_nonurban_rule_when_local_unknown_and_llm_negative(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(_NegativeLLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _WeakVeryLowNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Suburban growth and city expansion",
        "This paper studies suburban growth and general urban expansion without any renewal project.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "N1"
    assert result["final_label"] == "0"
    assert result["binary_decision_source"] == "binary_confidence_resolution"
    assert "weak_nonurban_hint" in result["binary_decision_evidence"]


def test_hybrid_classifier_resolves_curated_u4_rule_over_n1_local(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UrbanCoreN1Classifier(),
    )
    classifier = UrbanHybridClassifier(_LLMStrategy(), bertopic_service=_NullBERTopicService())
    classifier.rule_filter = _UrbanCoreRuleFilter()
    result = classifier.classify(
        "Urban redevelopment in downtown districts",
        "This paper studies urban redevelopment and downtown regeneration in an existing built-up area.",
    )
    assert result["decision_source"] == "unknown_hint_resolution"
    assert result["topic_final"] == "U4"
    assert result["final_label"] == "1"


def test_hybrid_classifier_family_gate_keeps_unknown_audit_fields_without_forcing_recovery(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _HousingMarketClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_StubFamilyGate(final_family="urban"),
    )
    classifier.rule_filter = _GentrificationHousingRuleFilter()
    result = classifier.classify(
        "Gentrification and neighborhood change",
        "This paper studies gentrification, neighborhood change and housing value shifts in an existing city.",
    )
    assert result["decision_source"] == "unknown_review"
    assert result["topic_final"] == "Unknown"
    assert result["topic_family_rule"] == "urban"
    assert result["topic_family_local"] == "nonurban"
    assert result["topic_family_final"] == "unknown"
    assert result["family_predicted_family"] == "urban"
    assert result["family_decision_source"] == "family_gate_stub"
    assert result["topic_within_family_label"] == "U12"
    assert result["unknown_recovery_path"] == "retained_unknown"
    assert result["boundary_bucket"] == "governance_policy_finance_boundary"
    assert result["family_conflict_pattern"] == "U12_vs_N4"


def test_hybrid_classifier_does_not_override_nonunknown_with_family_gate(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _HighConfidenceUrbanClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_StubFamilyGate(final_family="nonurban", confidence=0.95, probability_urban=0.04),
    )
    result = classifier.classify(
        "Urban renewal governance and participation in old districts",
        "This paper studies governance, stakeholders and participation in urban renewal implementation.",
    )
    assert result["decision_source"] == "rule_model_fusion"
    assert result["topic_final"] == "U9"
    assert result["topic_family_final"] == "urban"
    assert result["family_predicted_family"] == "nonurban"
    assert result["unknown_recovery_path"] == "not_triggered"


def test_hybrid_classifier_blocks_family_gate_recovery_when_llm_conflicts(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _LLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        family_gate=_StubFamilyGate(final_family="nonurban", confidence=0.95, probability_urban=0.04),
    )
    classifier.rule_filter = _VeryWeakUrbanNoBundleRuleFilter()
    result = classifier.classify(
        "Community impacts in changing districts",
        "This paper studies community impact under weak and ambiguous evidence.",
    )
    assert result["decision_source"] == "unknown_review"
    assert result["topic_final"] == "Unknown"
    assert result["family_predicted_family"] == "nonurban"
    assert result["unknown_recovery_path"] == "retained_unknown"


def test_anchor_guard_promotes_nonurban_with_core_anchor_and_support(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _AnchorGuardNonurbanWithUrbanAltClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_StubFamilyGate(final_family="urban", confidence=0.90, probability_urban=0.86),
    )
    classifier.rule_filter = _StrongNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Urban redevelopment governance coalitions in existing districts",
        "This paper examines policy coalitions and governance in urban redevelopment programmes.",
    )
    assert result["decision_source"] == "anchor_guard_promotion"
    assert result["topic_final"] == "U9"
    assert result["final_label"] == "1"
    assert result["anchor_guard_flag"] == 1
    assert result["anchor_guard_action"] == "promote"
    assert "urban redevelopment" in result["anchor_guard_hits"]
    _assert_explainability_contract(result)


def test_anchor_guard_sends_core_anchor_without_support_to_unknown_review(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _AnchorGuardNonurbanNoSupportClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _WeakNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Urban redevelopment discourse in policy narratives",
        "This article studies policy narratives and governance discourse without intervention evidence.",
    )
    assert result["decision_source"] == "anchor_guard_review"
    assert result["topic_final"] == "N3"
    assert result["final_label"] == "0"
    assert result["review_flag_raw"] == 1
    assert result["review_flag"] == 1
    assert result["anchor_guard_flag"] == 1
    assert result["anchor_guard_action"] == "review"
    assert result["binary_topic_consistency_flag"] == 0
    assert result["taxonomy_coverage_status"] == "covered"


def test_anchor_guard_can_promote_with_urban_within_family_fallback(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _AnchorGuardFallbackUrbanCandidateClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _WeakNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Urban redevelopment governance framework",
        "This paper studies urban redevelopment governance in existing districts.",
    )
    assert result["decision_source"] == "anchor_guard_promotion"
    assert result["topic_final"] == "U9"
    assert result["final_label"] == "1"
    assert float(result["urban_probability_score"]) >= float(result["binary_decision_threshold"])
    assert "decision_adjust=+0.16(anchor_guard_promotion)" in result["binary_decision_evidence"]
    assert "urban_within_candidate" in result["decision_reason"]


def test_uncertain_nonurban_gate_downgrades_n3_rule_local_agree_to_unknown(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _AnchorGuardNonurbanNoSupportClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _WeakNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "City governance and policy discourse",
        "This paper studies city governance and policy discourse in general urban contexts.",
    )
    assert result["decision_source"] == "uncertain_nonurban_review"
    assert result["topic_final"] == "N3"
    _assert_final_label_follows_binary_score(result)
    assert result["uncertain_nonurban_guard_flag"] == 1
    assert result["uncertain_nonurban_guard_action"] == "review"
    assert result["binary_topic_consistency_flag"] == 1
    assert result["taxonomy_coverage_status"] == "covered"


def test_uncertain_nonurban_gate_keeps_n8_technical_without_renewal_signal(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _N8LocalAgreeClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _N8TechnicalRuleFilter()
    result = classifier.classify(
        "Optimization model for urban simulation",
        "This paper develops an optimization model for urban systems simulation and benchmarking.",
    )
    assert result["topic_final"] == "N8"
    assert result["final_label"] == "0"
    assert result["decision_source"] == "rule_model_fusion"
    assert result["uncertain_nonurban_guard_flag"] == 1
    assert result["uncertain_nonurban_guard_action"] == "keep_0"


def test_uncertain_nonurban_gate_promotes_n8_with_renewal_and_strong_urban_local(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _N8UrbanConflictClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_StubFamilyGate(final_family="urban", confidence=0.90, probability_urban=0.86),
    )
    classifier.rule_filter = _N8RenewalRiskRuleFilter()
    result = classifier.classify(
        "Urban renewal governance evaluation model",
        "This paper evaluates urban renewal governance using a hybrid quantitative model.",
    )
    assert result["decision_source"] in {"uncertain_nonurban_promotion", "anchor_guard_promotion"}
    assert result["topic_final"] == "U9"
    assert result["final_label"] == "1"
    if result["decision_source"] == "uncertain_nonurban_promotion":
        assert result["uncertain_nonurban_guard_flag"] == 1
        assert result["uncertain_nonurban_guard_action"] == "promote"
    else:
        assert result["anchor_guard_flag"] == 1
        assert result["anchor_guard_action"] == "promote"


def test_anchor_guard_does_not_override_stage1_hard_negative_math():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    result = classifier.classify(
        "Urban renewal moves in dimer models",
        "This paper studies bipartite graph tiling methods and combinatorics.",
    )
    assert result["decision_source"] == "stage1_rule"
    assert result["topic_final"] == "N8"
    assert result["final_label"] == "0"
    assert result["urban_probability_score"] == 0.02
    assert result["binary_decision_source"] == "binary_hard_negative_override"
    assert result["evidence_balance"] == "hard_negative"
    assert "final=0" in result["decision_explanation"]
    assert result["anchor_guard_flag"] == 0
    _assert_explainability_contract(result)


def test_anchor_guard_does_not_override_stage1_hard_negative_rural():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    result = classifier.classify(
        "Rural regeneration and urban renewal policy in villages",
        "This paper studies rural renewal schemes and countryside development governance.",
    )
    assert result["decision_source"] == "stage1_rule"
    assert result["topic_final"] == "N9"
    assert result["final_label"] == "0"
    assert result["urban_probability_score"] == 0.02
    assert result["binary_decision_source"] == "binary_hard_negative_override"
    assert result["evidence_balance"] == "hard_negative"
    assert "final=0" in result["decision_explanation"]
    assert result["anchor_guard_flag"] == 0


def test_anchor_guard_does_not_trigger_on_urban_transformation_only(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _AnchorGuardNonurbanNoSupportClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _WeakNonurbanUnknownRuleFilter()
    result = classifier.classify(
        "Urban transformation discourse in policy networks",
        "This paper studies governance discourse and policy narratives in cities.",
    )
    assert result["topic_final"] == "N3"
    _assert_final_label_follows_binary_score(result)


def test_binary_recall_calibration_promotes_urban_context_floor():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    base = {
        "family_probability_urban": 0.02,
        "topic_binary_probability": 0.0,
        "topic_rule": "N3",
        "topic_local_label": "N3",
        "topic_within_family_label": "",
        "bertopic_hint_label": "",
        "llm_family_hint": "",
        "stage1_risk_tags": "",
    }
    label, confidence, review_flag, review_reason = classifier._apply_binary_decision(
        base,
        record=UrbanMetadataRecord(
            title="Urban governance in metropolitan housing policy",
            abstract="This paper studies policy coordination, planning and community infrastructure in cities.",
        ),
        route_result=MetadataRouteResult(route=METADATA_ROUTE_UNCERTAIN, reason="uncertain_pass"),
        final_topic="Unknown",
        decision_source="unknown_review",
        decision_reason="rule_unknown_local_unknown",
        confidence=0.2,
        review_flag=1,
        review_reason="unknown_review",
    )
    assert label == "1"
    assert confidence >= 0.45
    assert review_flag == 1
    assert "binary_low_confidence" in review_reason
    assert base["binary_recall_calibration_flag"] == 1
    assert base["binary_recall_calibration_tier"] == "context_relevance_floor"
    assert base["urban_probability_score"] >= base["binary_decision_threshold"]


def test_binary_recall_calibration_promotes_final_urban_topic_floor():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    base = {
        "family_probability_urban": 0.02,
        "topic_binary_probability": 0.0,
        "topic_rule": "N3",
        "topic_local_label": "N3",
        "topic_within_family_label": "",
        "bertopic_hint_label": "",
        "llm_family_hint": "",
        "stage1_risk_tags": "",
    }
    label, confidence, review_flag, review_reason = classifier._apply_binary_decision(
        base,
        record=UrbanMetadataRecord(
            title="Displacement and neighborhood change",
            abstract="This paper studies residents and housing pressure in the contemporary city.",
        ),
        route_result=MetadataRouteResult(route=METADATA_ROUTE_UNCERTAIN, reason="uncertain_pass"),
        final_topic="U12",
        decision_source="rule_model_fusion",
        decision_reason="rule_local_agree:U12",
        confidence=0.3,
        review_flag=0,
        review_reason="",
    )
    assert label == "1"
    assert confidence >= 0.45
    assert review_flag == 1
    assert "binary_low_confidence" in review_reason
    assert base["binary_recall_calibration_flag"] == 1
    assert base["binary_recall_calibration_tier"] == "final_topic_urban_floor"


def test_binary_recall_calibration_blocks_pure_n8_technical_risk():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    base = {
        "family_probability_urban": 0.02,
        "topic_binary_probability": 0.41,
        "topic_rule": "N8",
        "topic_local_label": "N8",
        "topic_within_family_label": "",
        "bertopic_hint_label": "",
        "llm_family_hint": "",
        "stage1_risk_tags": "generic_technical_risk",
    }
    label, confidence, review_flag, review_reason = classifier._apply_binary_decision(
        base,
        record=UrbanMetadataRecord(
            title="Optimization model for urban simulation",
            abstract="This paper develops an optimization model for urban systems simulation and benchmarking.",
        ),
        route_result=MetadataRouteResult(route=METADATA_ROUTE_UNCERTAIN, reason="uncertain_pass"),
        final_topic="N8",
        decision_source="rule_model_fusion",
        decision_reason="rule_local_agree:N8",
        confidence=0.3,
        review_flag=1,
        review_reason="rule_group_conflict",
    )
    assert label == "0"
    assert confidence > 0.45
    assert review_flag == 1
    assert base["binary_recall_calibration_flag"] == 0
    assert base["binary_recall_calibration_tier"] == "blocked"
    assert "generic_technical_n8_without_substantive_anchor" in base["binary_recall_calibration_reason"]


def test_open_set_promotes_unmapped_renewal_retrofit_topic(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _UnknownRuleReviewFlagFilter()
    result = classifier.classify(
        "Climate adaptation retrofit of existing districts",
        "This paper studies retrofit strategies for existing districts, older building stock, and urban fabric.",
    )
    assert result["final_label"] == "1"
    assert result["topic_final"] == "Urban_Renewal_Other"
    assert result["topic_final_group"] == "urban"
    assert result["open_set_flag"] == 1
    assert result["open_set_topic"] == "Urban_Renewal_Other"
    assert result["open_set_reason"] == "renewal_action_existing_urban_object"
    assert result["taxonomy_coverage_status"] == "open_set"
    assert "Urban_Renewal_Other" in result["primary_positive_evidence"]
    assert "open_set=renewal_action_existing_urban_object" in result["primary_positive_evidence"]
    assert result["evidence_balance"] in {"positive", "strong_positive", "low_confidence_positive"}


def test_open_set_promotes_unmapped_policy_project_with_renewal_anchor(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _UnknownRuleReviewFlagFilter()
    result = classifier.classify(
        "Regeneration strategy for waterfront adaptation",
        "This paper studies a regeneration strategy and implementation project for waterfront built environment adaptation.",
    )
    assert result["final_label"] == "1"
    assert result["topic_final"] == "Urban_Renewal_Other"
    assert result["open_set_reason"] in {
        "renewal_action_existing_urban_object",
        "policy_project_intervention_built_environment",
        "core_renewal_anchor",
    }
    assert result["taxonomy_coverage_status"] == "open_set"


def test_open_set_does_not_absorb_general_city_governance(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _UnknownClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _UnknownRuleReviewFlagFilter()
    result = classifier.classify(
        "General city governance narratives",
        "This paper studies city policy discourse and local institutions in general urban contexts.",
    )
    assert result["topic_final"] == "Unknown"
    assert result["open_set_flag"] == 0
    assert result["open_set_topic"] == ""
    assert result["taxonomy_coverage_status"] == "unknown"
    assert result["binary_audit_resolution_action"] == "positive_binary_conflict_audit"


def test_hybrid_classifier_preserves_review_rule_signal_for_nonunknown(monkeypatch):
    monkeypatch.setattr(
        "src.urban_hybrid_classifier.UrbanTopicClassifier",
        lambda: _AnchorGuardNonurbanNoSupportClassifier(),
    )
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=False,
        family_gate=_AnchorGuardNeutralFamilyGate(),
    )
    classifier.rule_filter = _UnknownRuleReviewFlagFilter()
    result = classifier.classify(
        "City governance policy narratives",
        "This paper studies policy and governance narratives in urban contexts.",
    )
    assert result["topic_final"] == "N3"
    _assert_final_label_follows_binary_score(result)
    assert result["review_flag_raw"] == 1
    assert result["review_flag"] == 1
    assert "rule_unknown" in result["review_reason_raw"]
    assert result["binary_recall_calibration_tier"] == "context_relevance_floor"
