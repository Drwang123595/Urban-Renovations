"""Four-step evidence summaries for stable urban-renewal decisions."""

from __future__ import annotations

from .evidence import EvidenceBundle, FourStepEvidence


def build_four_step_evidence(evidence: EvidenceBundle) -> FourStepEvidence:
    main_subject = main_subject_evidence(evidence)
    return FourStepEvidence(
        core_object_evidence=core_object_summary(evidence),
        renewal_action_evidence=renewal_action_summary(evidence),
        main_subject_evidence=main_subject,
        risk_evidence=risk_summary(evidence),
        auxiliary_evidence=auxiliary_summary(evidence),
        positive_evidence=positive_summary(evidence),
        negative_evidence=negative_summary(evidence, main_subject),
        llm_semantic_evidence=llm_summary(evidence),
    )


def main_subject_evidence(evidence: EvidenceBundle) -> str:
    if not (evidence.rule.has_renewal_action and evidence.rule.has_existing_urban_object):
        return "missing_core_object_or_action"
    if evidence.llm.used:
        if evidence.llm.action_is_main_subject is True and evidence.llm.is_background_only is not True:
            return "llm_confirms_main_subject"
        if evidence.llm.action_is_main_subject is False or evidence.llm.is_background_only is True:
            return "llm_rejects_main_subject"

    article = evidence.article
    title = str(article.title or "").lower()
    normalized = str(article.normalized_text or "").lower()
    method_markers = (
        "method paper",
        "benchmark dataset",
        "benchmark data",
        "case study dataset",
        "test dataset",
        "uses old district",
        "uses urban renewal",
        "using urban renewal",
    )
    method_hits = {
        "algorithm",
        "framework",
        "model",
        "machine learning",
        "deep learning",
        "neural network",
        "graph neural network",
    }
    if any(marker in normalized for marker in method_markers) and any(hit in normalized for hit in method_hits):
        return "background_or_method_only"

    title_has_action = any(hit in title for hit in evidence.rule.renewal_action_hits)
    title_has_object = any(hit in title for hit in evidence.rule.existing_urban_object_hits)
    title_has_urban_renewal = any(
        anchor in title
        for anchor in (
            "urban renewal",
            "urban regeneration",
            "urban redevelopment",
            "adaptive reuse",
            "brownfield redevelopment",
            "slum upgrading",
        )
    )
    if title_has_urban_renewal or (title_has_action and title_has_object):
        return "core_object_and_action_in_title_or_abstract"

    abstract = str(article.abstract or "").lower()
    abstract_has_action = any(hit in abstract for hit in evidence.rule.renewal_action_hits)
    abstract_has_object = any(hit in abstract for hit in evidence.rule.existing_urban_object_hits)
    if abstract_has_action and abstract_has_object:
        return "core_object_and_action_in_title_or_abstract"
    return "uncertain_main_subject"


def core_object_summary(evidence: EvidenceBundle) -> str:
    if evidence.rule.existing_urban_object_hits:
        return ",".join(evidence.rule.existing_urban_object_hits[:5])
    if evidence.llm.used and evidence.llm.existing_urban_object:
        return f"llm={evidence.llm.existing_urban_object}"
    return "none"


def renewal_action_summary(evidence: EvidenceBundle) -> str:
    if evidence.rule.renewal_action_hits:
        return ",".join(evidence.rule.renewal_action_hits[:5])
    if evidence.llm.used and evidence.llm.renewal_action:
        return f"llm={evidence.llm.renewal_action}"
    return "none"


def risk_summary(evidence: EvidenceBundle) -> str:
    parts: list[str] = []
    if evidence.rule.hard_exclusion_reason:
        parts.append(f"hard_exclusion={evidence.rule.hard_exclusion_reason}")
    if evidence.rule.risk_hits:
        parts.append("risk=" + ",".join(evidence.rule.risk_hits))
    if evidence.llm.used and evidence.llm.exclusion_risk:
        parts.append(f"llm_risk={evidence.llm.exclusion_risk}")
    return "; ".join(dict.fromkeys(part for part in parts if part)) or "none"


def auxiliary_summary(evidence: EvidenceBundle) -> str:
    parts: list[str] = []
    if evidence.topic.topic_candidate:
        parts.append(f"topic={evidence.topic.topic_candidate}")
    if evidence.topic.topic_group:
        parts.append(f"topic_group={evidence.topic.topic_group}")
    if evidence.score:
        parts.append(f"binary_score={evidence.score:.4f}")
    if evidence.threshold:
        parts.append(f"binary_threshold={evidence.threshold:.4f}")
    if evidence.family.family_probability_urban:
        parts.append(f"family_probability={evidence.family.family_probability_urban:.4f}")
    if evidence.cluster.cluster_id:
        parts.append(f"cluster={evidence.cluster.cluster_id}")
    candidate_label = str(evidence.dynamic.get("candidate_label", "") or "")
    if candidate_label:
        parts.append(f"dynamic_candidate={candidate_label}")
    return "; ".join(dict.fromkeys(part for part in parts if part)) or "none"


def positive_summary(evidence: EvidenceBundle) -> str:
    parts: list[str] = []
    if evidence.rule.renewal_action_hits:
        parts.append("renewal_action=" + ",".join(evidence.rule.renewal_action_hits[:5]))
    if evidence.rule.existing_urban_object_hits:
        parts.append("existing_urban_object=" + ",".join(evidence.rule.existing_urban_object_hits[:5]))
    if evidence.rule.policy_project_hits:
        parts.append("policy_project_intervention=" + ",".join(evidence.rule.policy_project_hits[:5]))
    if evidence.topic.topic_group == "urban":
        parts.append(f"topic={evidence.topic.topic_candidate}")
    if evidence.family.family_probability_urban >= 0.60:
        parts.append(f"family_probability={evidence.family.family_probability_urban:.4f}")
    if evidence.llm.used and evidence.llm.label_hint == "1":
        parts.append(f"llm={evidence.llm.reason}")
    return "; ".join(dict.fromkeys(part for part in parts if part)) or "none"


def negative_summary(evidence: EvidenceBundle, main_subject: str) -> str:
    parts: list[str] = []
    if evidence.rule.hard_exclusion_reason:
        parts.append(f"hard_exclusion={evidence.rule.hard_exclusion_reason}")
    if evidence.rule.risk_hits:
        parts.append("risk=" + ",".join(evidence.rule.risk_hits))
    if main_subject != "core_object_and_action_in_title_or_abstract":
        parts.append(f"main_subject={main_subject}")
    if evidence.topic.topic_group in {"nonurban", "unknown"}:
        parts.append(f"topic_group={evidence.topic.topic_group}")
    if evidence.llm.used and evidence.llm.label_hint == "0":
        parts.append(f"llm={evidence.llm.reason}")
    return "; ".join(dict.fromkeys(part for part in parts if part)) or "none"


def llm_summary(evidence: EvidenceBundle) -> str:
    llm = evidence.llm
    if not llm.attempted:
        return ""
    parts = [
        f"used={int(llm.used)}",
        f"label={llm.label_hint}",
        f"confidence={llm.confidence:.4f}",
        f"object_is_existing_urban={llm.object_is_existing_urban}",
        f"renewal_action_present={llm.renewal_action_present}",
        f"action_is_main_subject={llm.action_is_main_subject}",
        f"background_only={llm.is_background_only}",
        f"risk={llm.exclusion_risk}",
        f"reason={llm.reason}",
    ]
    return "; ".join(parts)
