from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .urban_metadata import UrbanMetadataRecord, normalize_phrase
from .urban_topic_taxonomy import (
    COMMON_EXISTING_URBAN_OBJECTS,
    UNKNOWN_TOPIC_GROUP,
    UNKNOWN_TOPIC_LABEL,
    UNKNOWN_TOPIC_NAME,
    score_all_topics,
    topic_group_for_label,
    topic_name_for_label,
)


METADATA_ROUTE_HARD_NEGATIVE = "hard_negative"
METADATA_ROUTE_CANDIDATE = "candidate_topic_bucket"
METADATA_ROUTE_UNCERTAIN = "uncertain"

STAGE1_DECISION_EXCLUDE = "exclude"
STAGE1_DECISION_PASS = "pass"

RULE_DECISION_REJECT = METADATA_ROUTE_HARD_NEGATIVE
RULE_DECISION_PASS = METADATA_ROUTE_UNCERTAIN
RULE_DECISION_REVIEW = METADATA_ROUTE_UNCERTAIN

HARD_NEGATIVE_DOMAINS = {
    "mechanics",
    "geochemistry & geophysics",
    "materials science",
    "physics, applied",
    "engineering, electrical & electronic",
}

RELATED_DOMAINS = {
    "urban studies",
    "regional & urban planning",
    "geography",
    "development studies",
    "public administration",
    "architecture",
    "sociology",
    "political science",
    "history",
    "anthropology",
    "economics",
    "government & law",
    "construction & building technology",
    "engineering, civil",
    "green & sustainable science & technology",
    "environmental studies",
    "social sciences, interdisciplinary",
    "public, environmental & occupational health",
    "transportation",
    "management",
    "area studies",
}

R0_URBAN_RENEWAL_TRIGGER = ("urban renewal",)
R0_MATH_TERMS = (
    "dimer",
    "bipartite graph",
    "teichmuller",
    "cluster algebra",
    "tiling",
    "combinatorics",
)
R4_RURAL_NONURBAN = (
    "rural regeneration",
    "rural renewal",
    "rural gentrification",
    "village revitalization",
    "agricultural regeneration",
    "rural development",
    "fishing community",
)
R5_GREENFIELD_EXPANSION = (
    "new town",
    "greenfield",
    "urban expansion",
    "sprawl",
    "urban fringe",
    "suburban growth",
)
R6_METHOD_ALGO = (
    "algorithm",
    "deep learning",
    "machine learning",
    "cnn",
    "optimization",
    "framework",
    "numerical simulation",
)
R6_MATERIALS_POLLUTION = (
    "recycled concrete",
    "compressive strength",
    "cement",
    "hydration",
    "soil contamination",
    "pollution",
    "thermal storage",
    "wastewater treatment",
)

M1_BACKGROUND_SUPPORT_ONLY = (
    "in the context of urban renewal",
    "support for urban renewal",
    "renewal area",
    "for future renewal planning",
    "plays a role in regeneration",
    "background context",
    "general background context",
    "broad application context",
    "application context",
    "general context",
)
M2_SOCIAL_HISTORY_MEDIA_RISK = (
    "discourse",
    "representation",
    "photography",
    "memory",
    "cinema",
    "era of urban renewal",
    "media coverage",
)

RISK_BACKGROUND_SUPPORT = "background_support_risk"
RISK_SOCIAL_HISTORY_MEDIA = "social_history_media_risk"
RISK_GREENFIELD_EXPANSION = "greenfield_expansion_risk"
RISK_GENERIC_TECHNICAL = "generic_technical_risk"
RISK_EXPLICIT_RENEWAL_OTHER_OBJECT = "explicit_renewal_wording_but_other_object"

R7_EXPLICIT_RENEWAL_WORDING = (
    "urban renewal",
    "urban regeneration",
    "urban redevelopment",
    "redevelopment",
    "regeneration",
    "renewal",
    "revitalization",
)
R7_GENERAL_GOVERNANCE = (
    "governance",
    "policy",
    "law",
    "program",
    "programme",
    "institution",
    "city governance",
    "local governance",
)
R7_GENERAL_METHOD = (
    "framework",
    "model",
    "algorithm",
    "machine learning",
    "deep learning",
    "remote sensing",
    "gis",
    "assessment",
    "evaluation",
    "design",
)
R7_STRONG_RENEWAL_MECHANISMS = (
    "tax increment financing",
    "tif",
    "ppp",
    "public private partnership",
    "reit",
    "land value capture",
    "compensation",
    "relocation",
    "resettlement",
    "demolition",
    "adaptive reuse",
    "slum upgrading",
    "urban village",
    "brownfield",
    "historic district",
    "historic quarter",
    "station area",
    "public space renewal",
    "street renewal",
)


@dataclass
class MetadataRouteResult:
    route: str
    reason: str
    candidate_topic_buckets: List[str] = field(default_factory=list)
    matched_candidate_terms: List[str] = field(default_factory=list)
    matched_negative_domains: List[str] = field(default_factory=list)
    matched_negative_keywords: List[str] = field(default_factory=list)
    matched_related_domains: List[str] = field(default_factory=list)
    stage1_decision: str = STAGE1_DECISION_PASS
    stage1_reason_tag: str = "uncertain_pass"
    stage1_hit_signals: List[str] = field(default_factory=list)
    stage1_risk_tags: List[str] = field(default_factory=list)
    stage1_conflict_flag: int = 0
    topic_rule: str = UNKNOWN_TOPIC_LABEL
    topic_rule_group: str = UNKNOWN_TOPIC_GROUP
    topic_rule_name: str = UNKNOWN_TOPIC_NAME
    topic_rule_score: float = 0.0
    topic_rule_margin: float = 0.0
    topic_rule_top3: List[str] = field(default_factory=list)
    topic_rule_matches: List[str] = field(default_factory=list)
    rule_high_confidence: bool = False
    review_flag_rule: int = 0
    review_reason_rule: str = ""

    @property
    def decision(self) -> str:
        return self.route

    @property
    def matched_positive_signals(self) -> List[str]:
        return self.matched_candidate_terms


class MetadataRuleFilter:
    def __init__(self):
        self.hard_negative_domains = HARD_NEGATIVE_DOMAINS
        self.related_domains = RELATED_DOMAINS

    def _match_phrases(self, text: str, phrases: Sequence[str]) -> List[str]:
        normalized_text = normalize_phrase(text).replace("-", " ")
        return [
            phrase
            for phrase in phrases
            if normalize_phrase(phrase).replace("-", " ") in normalized_text
        ]

    def _collect_metadata_audit(
        self,
        record: UrbanMetadataRecord,
    ) -> tuple[list[str], list[str], list[str]]:
        domain_tokens = set(record.domain_tokens)
        keyword_tokens = set(record.keywords_plus_tokens)
        matched_negative_domains = sorted(domain_tokens & self.hard_negative_domains)
        matched_negative_keywords = sorted(
            token
            for token in keyword_tokens
            if token in {
                "mechanical properties",
                "recycled concrete",
                "compressive strength",
                "pollution",
                "soil contamination",
            }
        )
        matched_related_domains = sorted(domain_tokens & self.related_domains)
        return matched_negative_domains, matched_negative_keywords, matched_related_domains

    def _detect_explicit_renewal_other_object(
        self,
        normalized_text: str,
        *,
        background_hits: Sequence[str],
        media_risk_hits: Sequence[str],
    ) -> tuple[str, str, List[str]]:
        renewal_hits = self._match_phrases(normalized_text, R7_EXPLICIT_RENEWAL_WORDING)
        if not renewal_hits:
            return "", "", []

        object_hits = self._match_phrases(normalized_text, COMMON_EXISTING_URBAN_OBJECTS)
        mechanism_hits = self._match_phrases(normalized_text, R7_STRONG_RENEWAL_MECHANISMS)
        if object_hits or mechanism_hits:
            return "", "", []
        if not background_hits and not media_risk_hits:
            return "", "", []

        method_hits = self._match_phrases(normalized_text, R7_GENERAL_METHOD)
        if method_hits:
            return "N8", "explicit_renewal_wording_but_other_object_method", list(background_hits) + renewal_hits + method_hits

        governance_hits = self._match_phrases(normalized_text, R7_GENERAL_GOVERNANCE)
        if governance_hits:
            return "N3", "explicit_renewal_wording_but_other_object_governance", list(background_hits) + list(media_risk_hits) + renewal_hits + governance_hits

        return "", "", []

    def _score_rule_topics(self, record: UrbanMetadataRecord) -> tuple[str, float, float, List[str], List[str], bool]:
        scored = score_all_topics(title=record.title, abstract=record.abstract)
        best = scored[0] if scored else {"label": UNKNOWN_TOPIC_LABEL, "score": 0.0, "matched_terms": [], "combo_hits": []}
        second_score = float(scored[1]["score"]) if len(scored) > 1 else 0.0
        best_label = str(best["label"])
        best_score = float(best["score"])
        margin = round(max(best_score - second_score, 0.0), 4)
        top3 = [
            f"{item['label']}:{float(item['score']):.2f}"
            for item in scored[:3]
        ]
        matched_terms = list(best.get("matched_terms", []))
        combo_hits = list(best.get("combo_hits", []))
        high_confidence = (
            (best_score >= 6.0 and margin >= 3.0)
            or (best_score >= 5.0 and bool(combo_hits))
        )
        if best_score < 4.0 or margin < 2.0:
            high_confidence = False
        if best_score <= 0.0:
            best_label = UNKNOWN_TOPIC_LABEL
        return best_label, round(best_score, 4), margin, top3, matched_terms, high_confidence

    def _stage1_conflict_flag(self, topic_rule_top3: List[str]) -> int:
        if len(topic_rule_top3) < 2:
            return 0
        top1_label = topic_rule_top3[0].split(":", 1)[0]
        top2_label = topic_rule_top3[1].split(":", 1)[0]
        return int(topic_group_for_label(top1_label) != topic_group_for_label(top2_label))

    def _review_rule_signal(
        self,
        topic_rule: str,
        topic_rule_score: float,
        topic_rule_margin: float,
        stage1_conflict_flag: int,
    ) -> tuple[int, str]:
        if topic_rule == UNKNOWN_TOPIC_LABEL:
            return 1, "rule_unknown"
        if stage1_conflict_flag == 1:
            return 1, "rule_group_conflict"
        if topic_rule_margin < 2.0:
            return 1, "rule_low_margin"
        if topic_rule_score < 4.0:
            return 1, "rule_low_score"
        return 0, ""

    def _route_reason(
        self,
        *,
        other_object_topic: str,
        other_object_reason: str,
        rule_high_confidence: bool,
        topic_rule: str,
        greenfield_hits: List[str],
        method_algo_hits: List[str],
        materials_hits: List[str],
    ) -> str:
        if other_object_topic:
            return other_object_reason
        if rule_high_confidence:
            return f"rule_topic_{topic_rule.lower()}"
        if greenfield_hits and topic_rule == "N2":
            return "greenfield_expansion_signal"
        if method_algo_hits and topic_rule == "N8":
            return "method_signal"
        if materials_hits and topic_rule == "N10":
            return "environment_signal"
        return "uncertain_pass"

    def evaluate(self, record: UrbanMetadataRecord) -> MetadataRouteResult:
        title_abstract_text = normalize_phrase(record.title_abstract_text()).replace("-", " ")
        matched_negative_domains, matched_negative_keywords, matched_related_domains = (
            self._collect_metadata_audit(record)
        )

        risk_tags: List[str] = []
        background_support_hits = self._match_phrases(title_abstract_text, M1_BACKGROUND_SUPPORT_ONLY)
        if background_support_hits:
            risk_tags.append(RISK_BACKGROUND_SUPPORT)
        social_history_hits = self._match_phrases(title_abstract_text, M2_SOCIAL_HISTORY_MEDIA_RISK)
        if social_history_hits:
            risk_tags.append(RISK_SOCIAL_HISTORY_MEDIA)
        other_object_topic, other_object_reason, other_object_hits = self._detect_explicit_renewal_other_object(
            title_abstract_text,
            background_hits=background_support_hits,
            media_risk_hits=social_history_hits,
        )
        if other_object_topic:
            risk_tags.append(RISK_EXPLICIT_RENEWAL_OTHER_OBJECT)

        r0_trigger = self._match_phrases(title_abstract_text, R0_URBAN_RENEWAL_TRIGGER)
        r0_math_terms = self._match_phrases(title_abstract_text, R0_MATH_TERMS)
        if r0_trigger and r0_math_terms:
            return MetadataRouteResult(
                route=METADATA_ROUTE_HARD_NEGATIVE,
                reason="math_term_misuse",
                candidate_topic_buckets=["N8"],
                matched_candidate_terms=r0_trigger + r0_math_terms,
                matched_negative_domains=matched_negative_domains,
                matched_negative_keywords=matched_negative_keywords,
                matched_related_domains=matched_related_domains,
                stage1_decision=STAGE1_DECISION_EXCLUDE,
                stage1_reason_tag="math_term_misuse",
                stage1_hit_signals=r0_trigger + r0_math_terms,
                stage1_risk_tags=[],
                stage1_conflict_flag=0,
                topic_rule="N8",
                topic_rule_group=topic_group_for_label("N8"),
                topic_rule_name=topic_name_for_label("N8"),
                topic_rule_score=9.0,
                topic_rule_margin=9.0,
                topic_rule_top3=["N8:9.00"],
                topic_rule_matches=r0_trigger + r0_math_terms,
                rule_high_confidence=True,
                review_flag_rule=0,
                review_reason_rule="",
            )

        rural_hits = self._match_phrases(title_abstract_text, R4_RURAL_NONURBAN)
        if rural_hits:
            return MetadataRouteResult(
                route=METADATA_ROUTE_HARD_NEGATIVE,
                reason="rural_nonurban",
                candidate_topic_buckets=["N9"],
                matched_candidate_terms=rural_hits,
                matched_negative_domains=matched_negative_domains,
                matched_negative_keywords=matched_negative_keywords,
                matched_related_domains=matched_related_domains,
                stage1_decision=STAGE1_DECISION_EXCLUDE,
                stage1_reason_tag="rural_nonurban",
                stage1_hit_signals=rural_hits,
                stage1_risk_tags=[],
                stage1_conflict_flag=0,
                topic_rule="N9",
                topic_rule_group=topic_group_for_label("N9"),
                topic_rule_name=topic_name_for_label("N9"),
                topic_rule_score=9.0,
                topic_rule_margin=9.0,
                topic_rule_top3=["N9:9.00"],
                topic_rule_matches=rural_hits,
                rule_high_confidence=True,
                review_flag_rule=0,
                review_reason_rule="",
            )

        greenfield_hits = self._match_phrases(title_abstract_text, R5_GREENFIELD_EXPANSION)
        method_algo_hits = self._match_phrases(title_abstract_text, R6_METHOD_ALGO)
        materials_hits = self._match_phrases(title_abstract_text, R6_MATERIALS_POLLUTION)
        if greenfield_hits:
            risk_tags.append(RISK_GREENFIELD_EXPANSION)
        if method_algo_hits or materials_hits:
            risk_tags.append(RISK_GENERIC_TECHNICAL)

        topic_rule, topic_rule_score, topic_rule_margin, topic_rule_top3, topic_rule_matches, rule_high_confidence = (
            self._score_rule_topics(record)
        )
        if other_object_topic:
            topic_rule = other_object_topic
            topic_rule_score = 6.4
            topic_rule_margin = 3.0
            topic_rule_top3 = [f"{other_object_topic}:6.40"] + [
                item for item in topic_rule_top3 if not item.startswith(f"{other_object_topic}:")
            ][:2]
            topic_rule_matches = list(dict.fromkeys(other_object_hits + topic_rule_matches))
            rule_high_confidence = True
        topic_rule_group = topic_group_for_label(topic_rule)
        topic_rule_name = topic_name_for_label(topic_rule)

        stage1_hit_signals = topic_rule_matches + greenfield_hits + method_algo_hits + materials_hits
        candidate_buckets = [item.split(":", 1)[0] for item in topic_rule_top3]
        stage1_conflict_flag = self._stage1_conflict_flag(topic_rule_top3)
        review_flag_rule, review_reason_rule = self._review_rule_signal(
            topic_rule,
            topic_rule_score,
            topic_rule_margin,
            stage1_conflict_flag,
        )
        reason = self._route_reason(
            other_object_topic=other_object_topic,
            other_object_reason=other_object_reason,
            rule_high_confidence=rule_high_confidence,
            topic_rule=topic_rule,
            greenfield_hits=greenfield_hits,
            method_algo_hits=method_algo_hits,
            materials_hits=materials_hits,
        )

        return MetadataRouteResult(
            route=METADATA_ROUTE_UNCERTAIN,
            reason=reason,
            candidate_topic_buckets=candidate_buckets,
            matched_candidate_terms=topic_rule_matches,
            matched_negative_domains=matched_negative_domains,
            matched_negative_keywords=matched_negative_keywords,
            matched_related_domains=matched_related_domains,
            stage1_decision=STAGE1_DECISION_PASS,
            stage1_reason_tag=reason,
            stage1_hit_signals=stage1_hit_signals,
            stage1_risk_tags=sorted(set(risk_tags)),
            stage1_conflict_flag=stage1_conflict_flag,
            topic_rule=topic_rule,
            topic_rule_group=topic_rule_group,
            topic_rule_name=topic_rule_name,
            topic_rule_score=topic_rule_score,
            topic_rule_margin=topic_rule_margin,
            topic_rule_top3=topic_rule_top3,
            topic_rule_matches=topic_rule_matches,
            rule_high_confidence=rule_high_confidence,
            review_flag_rule=review_flag_rule,
            review_reason_rule=review_reason_rule,
        )
