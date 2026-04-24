from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List

from ..runtime.config import Schema


_SPLIT_RE = re.compile(r"\s*;\s*")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def split_multi_value(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [item.strip() for item in _SPLIT_RE.split(text) if item.strip()]


def normalize_phrase(value: str) -> str:
    return _SPACE_RE.sub(" ", normalize_text(value).lower())


def unique_phrases(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = normalize_phrase(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def build_keywords(author_keywords: Any, keywords_plus: Any) -> str:
    merged = unique_phrases(
        split_multi_value(author_keywords) + split_multi_value(keywords_plus)
    )
    return "; ".join(merged)


def tokenize_text(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(normalize_phrase(text))
    if not tokens:
        return []
    bigrams = [f"{tokens[idx]}_{tokens[idx + 1]}" for idx in range(len(tokens) - 1)]
    return tokens + bigrams


@dataclass(frozen=True)
class UrbanMetadataRecord:
    title: str
    abstract: str
    author_keywords: str = ""
    keywords_plus: str = ""
    keywords: str = ""
    wos_categories: str = ""
    research_areas: str = ""

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "UrbanMetadataRecord":
        author_keywords = normalize_text(row.get(Schema.AUTHOR_KEYWORDS, ""))
        keywords_plus = normalize_text(row.get(Schema.KEYWORDS_PLUS, ""))
        keywords = normalize_text(row.get(Schema.KEYWORDS, "")) or build_keywords(
            author_keywords,
            keywords_plus,
        )
        return cls(
            title=normalize_text(row.get(Schema.TITLE, "")),
            abstract=normalize_text(row.get(Schema.ABSTRACT, "")),
            author_keywords=author_keywords,
            keywords_plus=keywords_plus,
            keywords=keywords,
            wos_categories=normalize_text(row.get(Schema.WOS_CATEGORIES, "")),
            research_areas=normalize_text(row.get(Schema.RESEARCH_AREAS, "")),
        )

    def to_output_dict(self) -> Dict[str, str]:
        return {
            Schema.AUTHOR_KEYWORDS: self.author_keywords,
            Schema.KEYWORDS_PLUS: self.keywords_plus,
            Schema.KEYWORDS: self.keywords,
            Schema.WOS_CATEGORIES: self.wos_categories,
            Schema.RESEARCH_AREAS: self.research_areas,
        }

    @property
    def keyword_tokens(self) -> List[str]:
        return unique_phrases(split_multi_value(self.keywords))

    @property
    def keywords_plus_tokens(self) -> List[str]:
        return unique_phrases(split_multi_value(self.keywords_plus))

    @property
    def domain_tokens(self) -> List[str]:
        return unique_phrases(
            split_multi_value(self.wos_categories) + split_multi_value(self.research_areas)
        )

    def title_abstract_text(self) -> str:
        parts = [self.title, self.abstract]
        return " [SEP] ".join(part for part in parts if part)

    def title_abstract_weighted_text(self) -> str:
        parts: List[str] = []
        parts.extend([self.title] * 2 if self.title else [])
        parts.extend([self.abstract] * 3 if self.abstract else [])
        return " ".join(parts)

    def combined_text(self) -> str:
        parts = [
            self.title,
            self.abstract,
            self.keywords,
            self.wos_categories,
            self.research_areas,
        ]
        return " [SEP] ".join(part for part in parts if part)

    def weighted_text(self) -> str:
        return self.title_abstract_weighted_text()
