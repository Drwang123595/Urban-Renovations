"""Input normalization for stable urban-renewal strategy decisions."""

from __future__ import annotations

from typing import Any, Mapping

from ...runtime.config import Schema
from ..core.metadata import normalize_phrase
from .evidence import ArticleEvidenceInput


def build_article_input(row: Mapping[str, Any]) -> ArticleEvidenceInput:
    title = _text(row.get(Schema.TITLE, row.get("Article Title", "")))
    abstract = _text(row.get(Schema.ABSTRACT, row.get("Abstract", "")))
    author_keywords = _text(row.get(Schema.AUTHOR_KEYWORDS, ""))
    keywords_plus = _text(row.get(Schema.KEYWORDS_PLUS, ""))
    keywords = _text(row.get(Schema.KEYWORDS, ""))
    wos_categories = _text(row.get(Schema.WOS_CATEGORIES, ""))
    research_areas = _text(row.get(Schema.RESEARCH_AREAS, ""))
    normalized_text = normalize_phrase(
        " ".join(
            part
            for part in (
                title,
                abstract,
                author_keywords,
                keywords_plus,
                keywords,
                wos_categories,
                research_areas,
            )
            if part
        )
    ).replace("-", " ")
    return ArticleEvidenceInput(
        title=title,
        abstract=abstract,
        author_keywords=author_keywords,
        keywords_plus=keywords_plus,
        keywords=keywords,
        wos_categories=wos_categories,
        research_areas=research_areas,
        normalized_text=normalized_text,
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
