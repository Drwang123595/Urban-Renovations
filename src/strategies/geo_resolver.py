import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCALE_LEVELS = {
    "1": "1. Global Scale",
    "2": "2. Multi-national / Continental Scale",
    "3": "3. National / Single-country Scale",
    "4": "4. Multi-provincial / Sub-national Regional Scale",
    "5": "5. Single-provincial / State Scale",
    "6": "6. Multi-city / Megaregion Scale",
    "7": "7. Single-city / Municipal Scale",
    "8": "8. District / County Scale",
    "9": "9. Micro / Neighborhood / Block Scale",
}


@dataclass(frozen=True)
class GeoEntity:
    geo_id: str
    canonical_name: str
    country_code: str = ""
    admin_level: str = ""
    feature_code: str = ""
    parent_geo_id: str = ""
    population: int = 0
    source: str = "builtin_overlay"
    mapped_scale_level: str = ""
    hierarchy_path: str = ""


@dataclass(frozen=True)
class GeoResolution:
    resolved_study_area: str = ""
    resolved_geo_id: str = ""
    area_hierarchy_path: str = ""
    mapped_spatial_scale_level: str = ""
    scale_decision_source: str = ""
    geo_resolution_status: str = "not_applicable"
    geo_resolution_reason: str = ""
    geo_resolution_confidence: float = 0.0
    geo_source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resolved_study_area": self.resolved_study_area,
            "resolved_geo_id": self.resolved_geo_id,
            "area_hierarchy_path": self.area_hierarchy_path,
            "mapped_spatial_scale_level": self.mapped_spatial_scale_level,
            "scale_decision_source": self.scale_decision_source,
            "geo_resolution_status": self.geo_resolution_status,
            "geo_resolution_reason": self.geo_resolution_reason,
            "geo_resolution_confidence": self.geo_resolution_confidence,
            "geo_source": self.geo_source,
        }


class GeoResolver:
    """Resolve extracted study areas to a local gazetteer-backed spatial scale."""

    _SPLIT_RE = re.compile(r"\s*(?:;|/|&|\band\b)\s*", re.IGNORECASE)
    _COUNTRY_SCALE = SCALE_LEVELS["3"]
    _PROVINCE_SCALE = SCALE_LEVELS["5"]
    _CITY_SCALE = SCALE_LEVELS["7"]
    _REFERENCE_CACHE: Dict[str, tuple[Dict[str, GeoEntity], Dict[str, List[str]]]] = {}

    def __init__(self, reference_dir: Optional[Path] = None):
        self.reference_dir = Path(reference_dir) if reference_dir else self._default_reference_dir()
        cache_key = str(self.reference_dir.resolve())
        cached = self._REFERENCE_CACHE.get(cache_key)
        if cached:
            self.entities, self.alias_to_ids = cached
            return
        self.entities: Dict[str, GeoEntity] = {}
        self.alias_to_ids: Dict[str, List[str]] = {}
        self._load_builtin_overlay()
        self._load_reference_files()
        self._REFERENCE_CACHE[cache_key] = (self.entities, self.alias_to_ids)

    @staticmethod
    def _default_reference_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "Data" / "reference" / "geography"

    @staticmethod
    def normalize(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\x00", "")
        text = "".join(ch for ch in text if ch >= " " or ch in "\t\n\r")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
        text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    @classmethod
    def compact(cls, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", cls.normalize(value))

    def _add_entity(self, entity: GeoEntity, aliases: Iterable[str]) -> None:
        self.entities[entity.geo_id] = entity
        for alias in {entity.canonical_name, *aliases}:
            normalized = self.normalize(alias).strip(" .;:,")
            compact = self.compact(alias)
            for key in {normalized, compact}:
                if not key:
                    continue
                bucket = self.alias_to_ids.setdefault(key, [])
                if entity.geo_id not in bucket:
                    bucket.append(entity.geo_id)

    def _load_builtin_overlay(self) -> None:
        entries = [
            GeoEntity("global", "Global", mapped_scale_level=SCALE_LEVELS["1"], source="builtin_macro"),
            GeoEntity("world", "World", mapped_scale_level=SCALE_LEVELS["1"], source="builtin_macro"),
            GeoEntity("europe", "Europe", mapped_scale_level=SCALE_LEVELS["2"], source="builtin_macro"),
            GeoEntity("eu", "European Union", mapped_scale_level=SCALE_LEVELS["2"], source="builtin_macro"),
            GeoEntity("global_south", "Global South", mapped_scale_level=SCALE_LEVELS["2"], source="builtin_overlay"),
            GeoEntity("cn", "China", country_code="CN", feature_code="PCLI", mapped_scale_level=SCALE_LEVELS["3"], source="builtin_country"),
            GeoEntity("in", "India", country_code="IN", feature_code="PCLI", mapped_scale_level=SCALE_LEVELS["3"], source="builtin_country"),
            GeoEntity("gb", "United Kingdom", country_code="GB", feature_code="PCLI", mapped_scale_level=SCALE_LEVELS["3"], source="builtin_country"),
            GeoEntity("us", "United States", country_code="US", feature_code="PCLI", mapped_scale_level=SCALE_LEVELS["3"], source="builtin_country"),
            GeoEntity("cn-gd", "Guangdong Province", country_code="CN", admin_level="ADM1", feature_code="ADM1", mapped_scale_level=SCALE_LEVELS["5"], source="builtin_admin"),
            GeoEntity("cn-shenzhen", "Shenzhen", country_code="CN", admin_level="PPL", feature_code="PPLA2", parent_geo_id="cn-gd", mapped_scale_level=SCALE_LEVELS["7"], hierarchy_path="China > Guangdong Province > Shenzhen", source="builtin_city"),
            GeoEntity("cn-beijing", "Beijing", country_code="CN", admin_level="PPL", feature_code="PPLC", mapped_scale_level=SCALE_LEVELS["7"], hierarchy_path="China > Beijing", source="builtin_city"),
            GeoEntity("cn-shanghai", "Shanghai", country_code="CN", admin_level="PPL", feature_code="PPLA", mapped_scale_level=SCALE_LEVELS["7"], hierarchy_path="China > Shanghai", source="builtin_city"),
            GeoEntity("hk", "Hong Kong", country_code="HK", admin_level="PPL", feature_code="PCLI", mapped_scale_level=SCALE_LEVELS["7"], hierarchy_path="Hong Kong", source="builtin_city"),
            GeoEntity("hk-sham-shui-po", "Sham Shui Po", country_code="HK", admin_level="ADM2", feature_code="ADM2", parent_geo_id="hk", mapped_scale_level=SCALE_LEVELS["8"], hierarchy_path="Hong Kong > Sham Shui Po", source="builtin_admin"),
            GeoEntity("cn-yrd", "Yangtze River Delta", country_code="CN", admin_level="REGION", feature_code="RGN", mapped_scale_level=SCALE_LEVELS["6"], hierarchy_path="China > Yangtze River Delta", source="project_overlay"),
            GeoEntity("cn-prd", "Pearl River Delta", country_code="CN", admin_level="REGION", feature_code="RGN", mapped_scale_level=SCALE_LEVELS["6"], hierarchy_path="China > Pearl River Delta", source="project_overlay"),
            GeoEntity("cn-gba", "Greater Bay Area", country_code="CN", admin_level="REGION", feature_code="RGN", mapped_scale_level=SCALE_LEVELS["6"], hierarchy_path="China > Guangdong-Hong Kong-Macao Greater Bay Area", source="project_overlay"),
            GeoEntity("es-barcelona", "Barcelona", country_code="ES", admin_level="PPL", feature_code="PPLA2", mapped_scale_level=SCALE_LEVELS["7"], source="builtin_city"),
            GeoEntity("pl-poznan", "Poznan", country_code="PL", admin_level="PPL", feature_code="PPLA", mapped_scale_level=SCALE_LEVELS["7"], source="builtin_city"),
        ]
        aliases = {
            "global": ["worldwide"],
            "world": ["worldwide"],
            "eu": ["EU", "E.U."],
            "cn": ["PRC", "Chinese"],
            "gb": ["UK", "U.K.", "Britain", "British", "England", "Scotland", "Wales"],
            "us": ["USA", "U.S.", "U.S.A.", "America", "American"],
            "cn-gd": ["Guangdong", "Guangdong Sheng"],
            "cn-shenzhen": ["Shenzhen City"],
            "cn-beijing": ["Beijing Municipality", "Beijing City"],
            "cn-shanghai": ["Shanghai Municipality", "Shanghai City"],
            "hk": ["Hong Kong SAR", "Hong Kong Special Administrative Region"],
            "hk-sham-shui-po": ["Sham Shui Po District", "Sham Shui Po in Hong Kong"],
            "cn-yrd": ["Yangtze River Delta region", "YRD"],
            "cn-prd": ["Pearl River Delta region", "PRD"],
            "cn-gba": ["Guangdong-Hong Kong-Macao Greater Bay Area", "GBA"],
            "pl-poznan": ["Poznań"],
        }
        for entity in entries:
            self._add_entity(entity, aliases.get(entity.geo_id, []))

    def _load_reference_files(self) -> None:
        entities_path = self.reference_dir / "geo_entities.csv"
        aliases_path = self.reference_dir / "geo_aliases.csv"
        if not entities_path.exists():
            return
        loaded: Dict[str, GeoEntity] = {}
        with entities_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                geo_id = str(row.get("geo_id") or "").strip()
                name = str(row.get("canonical_name") or "").strip()
                scale = str(row.get("mapped_scale_level") or "").strip()
                if not geo_id or not name or not scale:
                    continue
                loaded[geo_id] = GeoEntity(
                    geo_id=geo_id,
                    canonical_name=name,
                    country_code=str(row.get("country_code") or ""),
                    admin_level=str(row.get("admin_level") or ""),
                    feature_code=str(row.get("feature_code") or ""),
                    parent_geo_id=str(row.get("parent_geo_id") or ""),
                    population=int(float(row.get("population") or 0)),
                    source=str(row.get("source") or "reference"),
                    mapped_scale_level=scale,
                    hierarchy_path=str(row.get("hierarchy_path") or ""),
                )
        alias_rows: Dict[str, List[str]] = {}
        if aliases_path.exists():
            with aliases_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    geo_id = str(row.get("geo_id") or "").strip()
                    alias = str(row.get("alias") or "").strip()
                    if geo_id and alias:
                        alias_rows.setdefault(geo_id, []).append(alias)
        for geo_id, entity in loaded.items():
            self._add_entity(entity, alias_rows.get(geo_id, []))

    def resolve(self, area: str, llm_scale_level: Optional[str] = None) -> GeoResolution:
        cleaned = self._clean_area(area)
        if not cleaned:
            return GeoResolution(geo_resolution_status="not_applicable", geo_resolution_reason="empty_area")

        direct = self._resolve_single(cleaned)
        if direct:
            source = "mapping" if llm_scale_level == direct.mapped_scale_level else "mapping_override_llm"
            if not llm_scale_level:
                source = "mapping_inferred_scale"
            return self._resolution_from_entities([direct], "matched", source, 0.98)

        parts = self._split_area(cleaned)
        if len(parts) >= 2:
            matched = []
            for part in parts:
                entity = self._resolve_single(part)
                if entity:
                    matched.append(entity)
            if len(matched) == len(parts):
                scale = self._multi_entity_scale(matched)
                return self._resolution_from_entities(
                    matched,
                    "matched_multiple",
                    "mapping_override_llm" if llm_scale_level != scale else "mapping",
                    0.94,
                    scale_override=scale,
                )

        if llm_scale_level:
            return GeoResolution(
                resolved_study_area=cleaned,
                mapped_spatial_scale_level=llm_scale_level,
                scale_decision_source="llm_fallback_unresolved",
                geo_resolution_status="unresolved_or_ambiguous",
                geo_resolution_reason="area_not_found_in_local_gazetteer",
                geo_resolution_confidence=0.35,
            )

        return GeoResolution(
            resolved_study_area=cleaned,
            geo_resolution_status="unresolved_or_ambiguous",
            geo_resolution_reason="area_not_found_and_no_llm_scale",
            geo_resolution_confidence=0.0,
        )

    def _resolve_single(self, area: str) -> Optional[GeoEntity]:
        candidates = self._candidate_keys(area)
        for key in candidates:
            ids = self.alias_to_ids.get(key, [])
            if len(ids) == 1:
                return self.entities.get(ids[0])
            if len(ids) > 1:
                ranked = sorted(
                    (self.entities[item] for item in ids if item in self.entities),
                    key=self._entity_rank,
                    reverse=True,
                )
                return ranked[0] if ranked else None
        return None

    def _entity_rank(self, entity: GeoEntity) -> tuple[int, int, int]:
        source = entity.source.lower()
        feature = entity.feature_code.upper()
        admin = entity.admin_level.upper()
        if source.startswith("builtin") or source == "project_overlay":
            source_score = 1000
        elif source == "natural_earth":
            source_score = 900
        elif "country" in source or feature.startswith("PCL"):
            source_score = 850
        elif admin == "ADM1" or feature == "ADM1":
            source_score = 760
        elif admin == "PPL" or feature.startswith("PPL"):
            source_score = 700
        elif admin in {"ADM2", "ADM3"} or feature in {"ADM2", "ADM3"}:
            source_score = 650
        else:
            source_score = 500
        scale_score = 100 - self._scale_number(entity.mapped_scale_level)
        return (source_score, scale_score, entity.population)

    def _candidate_keys(self, area: str) -> List[str]:
        cleaned = self._clean_area(area)
        candidates = [cleaned]
        without_parentheses = re.sub(r"\([^)]*\)", "", cleaned).strip(" .;:,")
        candidates.append(without_parentheses)
        if "," in without_parentheses:
            first_fragment = without_parentheses.split(",", 1)[0].strip(" .;:,")
            candidates.append(first_fragment)
        extraction_patterns = [
            r"\bcity of\s+([^,;()]+)",
            r"\bmunicipality of\s+([^,;()]+)",
            r"\bprovince of\s+([^,;()]+)",
            r"\bdistrict of\s+([^,;()]+)",
            r"\bregion of\s+([^,;()]+)",
            r"\bUK city of\s+([^,;()]+)",
        ]
        for pattern in extraction_patterns:
            match = re.search(pattern, without_parentheses, flags=re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip(" .;:,"))
        for base in list(candidates):
            normalized_base = re.sub(
                r"\b(metropolitan area|municipal corporation|city|municipality|"
                r"province|state|district|county|region)\b",
                "",
                base,
                flags=re.IGNORECASE,
            ).strip(" .;:,")
            candidates.append(normalized_base)
            candidates.append(
                re.sub(
                    r"^(?:the|a|an|historic|postwar|rural|central|old|typical|"
                    r"selected|local|urban)\s+",
                    "",
                    normalized_base,
                    flags=re.IGNORECASE,
                ).strip(" .;:,")
            )
        keys: List[str] = []
        for candidate in candidates:
            normalized = self.normalize(candidate).strip(" .;:,")
            compact = self.compact(candidate)
            for key in (normalized, compact):
                if key and key not in keys:
                    keys.append(key)
        return keys

    def _split_area(self, area: str) -> List[str]:
        if re.search(r"\b(?:in|within|of)\b", area, flags=re.IGNORECASE):
            return []
        return [part.strip(" .;:,") for part in self._SPLIT_RE.split(area) if part.strip(" .;:,")]

    def _clean_area(self, area: str) -> str:
        text = re.sub(r"\([^)]*\bimplicit\b[^)]*\)", "", str(area), flags=re.IGNORECASE)
        text = re.sub(r"\bimplicit(?:ly)?\b", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" .;:,")

    def _resolution_from_entities(
        self,
        entities: List[GeoEntity],
        status: str,
        scale_source: str,
        confidence: float,
        scale_override: Optional[str] = None,
    ) -> GeoResolution:
        names = [entity.canonical_name for entity in entities]
        ids = [entity.geo_id for entity in entities]
        hierarchy = [entity.hierarchy_path or entity.canonical_name for entity in entities]
        scale = scale_override or entities[0].mapped_scale_level
        sources = sorted({entity.source for entity in entities if entity.source})
        return GeoResolution(
            resolved_study_area="; ".join(names),
            resolved_geo_id="; ".join(ids),
            area_hierarchy_path=" | ".join(hierarchy),
            mapped_spatial_scale_level=scale,
            scale_decision_source=scale_source,
            geo_resolution_status=status,
            geo_resolution_reason="gazetteer_match",
            geo_resolution_confidence=confidence,
            geo_source="; ".join(sources),
        )

    def _multi_entity_scale(self, entities: List[GeoEntity]) -> str:
        scale_numbers = [self._scale_number(entity.mapped_scale_level) for entity in entities]
        scale_numbers = [number for number in scale_numbers if number]
        if not scale_numbers:
            return ""
        unique = set(scale_numbers)
        if unique == {3}:
            return SCALE_LEVELS["2"]
        if unique == {5}:
            return SCALE_LEVELS["4"]
        if unique == {7}:
            return SCALE_LEVELS["6"]
        return SCALE_LEVELS[str(min(scale_numbers))]

    @staticmethod
    def _scale_number(scale_level: str) -> int:
        match = re.match(r"^([1-9])\.", str(scale_level).strip())
        return int(match.group(1)) if match else 99
