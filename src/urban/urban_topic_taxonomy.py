from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .urban_metadata import normalize_phrase


UNKNOWN_TOPIC_LABEL = "Unknown"
UNKNOWN_TOPIC_GROUP = "unknown"
UNKNOWN_TOPIC_NAME = "Unknown"
OPEN_SET_URBAN_LABEL = "Urban_Renewal_Other"
OPEN_SET_NONURBAN_LABEL = "Nonurban_Other"
OPEN_SET_URBAN_NAME = "urban renewal other"
OPEN_SET_NONURBAN_NAME = "nonurban other"

U_TOPIC_LABELS = tuple(f"U{idx}" for idx in range(1, 16))
N_TOPIC_LABELS = tuple(f"N{idx}" for idx in range(1, 11))
TOPIC_ORDER = U_TOPIC_LABELS + N_TOPIC_LABELS

COMMON_RENEWAL_ANCHORS = (
    "urban renewal",
    "urban regeneration",
    "urban redevelopment",
    "urban transformation",
    "urban revitalization",
    "city regeneration",
    "district regeneration",
    "area regeneration",
    "redevelopment",
    "regeneration",
    "renewal",
    "revitalization",
    "upgrading",
    "rehabilitation",
    "adaptive reuse",
    "brownfield redevelopment",
    "brownfield regeneration",
    "community renewal",
    "estate regeneration",
    "old city renewal",
    "old town renewal",
    "old district renewal",
    "urban village redevelopment",
    "slum upgrading",
    "public space renewal",
    "street renewal",
    "station area renewal",
)

# Core anchors are reserved for hard false-negative protection in the hybrid
# post-processing stage and intentionally exclude broad wording like
# "urban transformation".
CORE_RENEWAL_ANCHORS = (
    "urban renewal",
    "urban regeneration",
    "urban redevelopment",
    "urban revitalization",
    "brownfield redevelopment",
    "brownfield regeneration",
    "slum upgrading",
    "informal settlement upgrading",
    "adaptive reuse",
)

COMMON_EXISTING_URBAN_OBJECTS = (
    "old neighborhood",
    "old neighbourhood",
    "old community",
    "old district",
    "old town",
    "old city",
    "inner city",
    "inner-city",
    "city centre",
    "city center",
    "historic district",
    "historic quarter",
    "brownfield",
    "industrial site",
    "urban village",
    "housing estate",
    "public housing estate",
    "waterfront",
    "station area",
)

COMMON_RURAL_ANCHORS = (
    "rural",
    "agricultural",
    "farmland",
    "countryside",
    "fishing village",
    "fishing community",
    "village revitalization",
)

COMMON_METHOD_ANCHORS = (
    "algorithm",
    "framework",
    "model",
    "optimization",
    "simulation",
    "machine learning",
    "deep learning",
    "neural network",
    "cnn",
    "transformer",
)


def _topic(
    group: str,
    name: str,
    legacy_label: str,
    *,
    seeds: Sequence[str] = (),
    context_terms: Sequence[str] = (),
    context_anchors: Sequence[str] = (),
    combo_rules: Sequence[Sequence[str]] = (),
    exclude_terms: Sequence[str] = (),
    context_bonus: float = 1.5,
    requires_anchor: bool = False,
    anchor_terms: Sequence[str] = (),
    penalize_if_renewal_anchor: bool = False,
    missing_anchor_penalty: float = 1.75,
    renewal_anchor_penalty: float = 1.75,
) -> Dict[str, object]:
    return {
        "group": group,
        "name": name,
        "legacy_label": legacy_label,
        "seeds": list(seeds),
        "context_terms": list(context_terms),
        "context_anchors": list(context_anchors),
        "combo_rules": [tuple(rule) for rule in combo_rules],
        "exclude_terms": list(exclude_terms),
        "context_bonus": float(context_bonus),
        "requires_anchor": bool(requires_anchor),
        "anchor_terms": list(anchor_terms),
        "penalize_if_renewal_anchor": bool(penalize_if_renewal_anchor),
        "missing_anchor_penalty": float(missing_anchor_penalty),
        "renewal_anchor_penalty": float(renewal_anchor_penalty),
    }


TOPIC_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "U1": _topic(
        "urban",
        "old neighborhood renewal",
        "U2",
        seeds=(
            "old neighborhood renewal",
            "old neighbourhood renewal",
            "old community renewal",
            "housing estate renewal",
            "estate regeneration",
            "neighborhood rehabilitation",
            "community renewal",
            "old residential area renewal",
        ),
        context_terms=(
            "old neighborhood",
            "old neighbourhood",
            "old community",
            "housing estate",
            "public housing estate",
            "rehabilitation",
            "upgrading",
            "renewal",
            "regeneration",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS,
        combo_rules=(
            ("old neighborhood", "renewal"),
            ("old neighbourhood", "renewal"),
            ("old community", "upgrading"),
            ("housing estate", "regeneration"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
    ),
    "U2": _topic(
        "urban",
        "urban village renewal",
        "U2",
        seeds=(
            "urban village",
            "urban village redevelopment",
            "urban village renewal",
            "urban village regeneration",
            "village in the city",
            "chengzhongcun",
        ),
        context_terms=(
            "collective land",
            "migrant housing",
            "redevelopment",
            "renewal",
            "regeneration",
            "village transformation",
        ),
        context_anchors=("urban village", "village in the city", "chengzhongcun"),
        combo_rules=(
            ("urban village", "redevelopment"),
            ("urban village", "renewal"),
            ("urban village", "regeneration"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
    ),
    "U3": _topic(
        "urban",
        "slum or informal settlement upgrading",
        "U2",
        seeds=(
            "slum upgrading",
            "informal settlement upgrading",
            "squatter settlement upgrading",
            "favela upgrading",
            "informal settlement rehabilitation",
        ),
        context_terms=(
            "slum",
            "informal settlement",
            "squatter settlement",
            "favela",
            "upgrading",
            "rehabilitation",
            "basic services",
        ),
        context_anchors=("slum", "informal settlement", "squatter settlement", "favela"),
        combo_rules=(
            ("slum", "upgrading"),
            ("informal settlement", "upgrading"),
            ("favela", "upgrading"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
    ),
    "U4": _topic(
        "urban",
        "old city or inner-city regeneration",
        "U2",
        seeds=(
            "inner city regeneration",
            "inner-city regeneration",
            "old city renewal",
            "old town regeneration",
            "city centre regeneration",
            "city center regeneration",
            "downtown redevelopment",
            "inner-city redevelopment",
        ),
        context_terms=(
            "inner city",
            "inner-city",
            "old city",
            "old town",
            "city centre",
            "city center",
            "downtown",
            "redevelopment",
            "regeneration",
            "revitalization",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
        combo_rules=(
            ("inner city", "regeneration"),
            ("old city", "renewal"),
            ("old town", "regeneration"),
            ("downtown", "redevelopment"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
    ),
    "U5": _topic(
        "urban",
        "brownfield or industrial land redevelopment",
        "U2",
        seeds=(
            "brownfield redevelopment",
            "brownfield regeneration",
            "brownfield renewal",
            "industrial land redevelopment",
            "industrial site redevelopment",
            "post-industrial redevelopment",
        ),
        context_terms=(
            "brownfield",
            "industrial land",
            "industrial site",
            "remediation",
            "redevelopment",
            "regeneration",
        ),
        context_anchors=("brownfield", "industrial land", "industrial site", "post-industrial"),
        combo_rules=(
            ("brownfield", "redevelopment"),
            ("brownfield", "regeneration"),
            ("industrial site", "redevelopment"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
    ),
    "U6": _topic(
        "urban",
        "industrial heritage or historic building reuse",
        "U3",
        seeds=(
            "adaptive reuse",
            "historic building reuse",
            "industrial heritage reuse",
            "factory conversion",
            "warehouse conversion",
        ),
        context_terms=(
            "adaptive reuse",
            "industrial heritage",
            "historic building",
            "heritage building",
            "reuse",
            "conversion",
        ),
        context_anchors=("adaptive reuse", "industrial heritage", "historic building", "heritage building"),
        combo_rules=(
            ("adaptive reuse", "historic building"),
            ("industrial heritage", "reuse"),
            ("factory", "conversion"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
    ),
    "U7": _topic(
        "urban",
        "historic district conservation renewal",
        "U3",
        seeds=(
            "historic district revitalization",
            "historic quarter regeneration",
            "conservation-led regeneration",
            "heritage-led regeneration",
            "historic district renewal",
        ),
        context_terms=(
            "historic district",
            "historic quarter",
            "heritage conservation",
            "conservation",
            "revitalization",
            "regeneration",
        ),
        context_anchors=("historic district", "historic quarter", "heritage conservation"),
        combo_rules=(
            ("historic district", "revitalization"),
            ("historic quarter", "regeneration"),
            ("heritage conservation", "regeneration"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
    ),
    "U8": _topic(
        "urban",
        "public space or street renewal",
        "U3",
        seeds=(
            "public space renewal",
            "public realm renewal",
            "street renewal",
            "streetscape improvement",
            "plaza revitalization",
            "waterfront public space",
        ),
        context_terms=(
            "public space",
            "public realm",
            "street",
            "streetscape",
            "plaza",
            "waterfront",
            "revitalization",
            "renewal",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS + ("public space", "public realm", "street", "streetscape"),
        combo_rules=(
            ("public space", "renewal"),
            ("public realm", "renewal"),
            ("street", "renewal"),
            ("streetscape", "improvement"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + ("public space renewal", "public realm renewal", "street renewal"),
    ),
    "U9": _topic(
        "urban",
        "renewal governance institutions participation",
        "U1",
        seeds=(
            "renewal governance",
            "regeneration governance",
            "redevelopment governance",
            "collaborative governance",
            "stakeholder negotiation",
            "institutional arrangement",
            "renewal implementation",
            "governance mechanism",
            "implementation path",
            "delivery mechanism",
            "planning process",
            "participatory renewal",
        ),
        context_terms=(
            "governance",
            "institution",
            "participation",
            "stakeholder",
            "coalition",
            "implementation",
            "negotiation",
            "delivery",
            "co production",
            "coproduction",
            "co creation",
            "cocreation",
            "implementation path",
            "delivery mechanism",
            "governance mechanism",
            "planning process",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS,
        combo_rules=(
            ("urban renewal", "governance"),
            ("urban regeneration", "participation"),
            ("urban regeneration", "policy"),
            ("urban regeneration", "governance"),
            ("redevelopment", "stakeholder"),
            ("redevelopment", "policy"),
            ("renewal", "implementation"),
            ("regeneration", "delivery"),
            ("redevelopment", "institutional arrangement"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
        missing_anchor_penalty=2.1,
    ),
    "U10": _topic(
        "urban",
        "renewal finance and policy tools",
        "U1",
        seeds=(
            "tax increment financing",
            "public private partnership",
            "urban renewal reit",
            "redevelopment finance",
            "land value capture",
            "renewal financing",
            "redevelopment funding",
            "renewal subsidy",
            "redevelopment incentive",
            "financial instrument",
        ),
        context_terms=(
            "tif",
            "ppp",
            "reit",
            "finance",
            "financing",
            "policy tool",
            "subsidy",
            "incentive",
            "funding",
            "grant",
            "public private partnership",
            "land value capture",
            "policy instrument",
            "financial instrument",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS,
        combo_rules=(
            ("redevelopment", "financing"),
            ("renewal", "reit"),
            ("regeneration", "ppp"),
            ("urban renewal", "tax increment financing"),
            ("redevelopment", "land value capture"),
            ("regeneration", "subsidy"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS + (
            "tax increment financing",
            "tif",
            "ppp",
            "public private partnership",
            "reit",
            "land value capture",
            "redevelopment finance",
        ),
        missing_anchor_penalty=2.3,
    ),
    "U11": _topic(
        "urban",
        "relocation compensation resettlement eviction",
        "U1",
        seeds=(
            "resettlement compensation",
            "relocation compensation",
            "forced eviction",
            "demolition compensation",
            "resettlement",
            "relocation",
        ),
        context_terms=(
            "resettlement",
            "relocation",
            "compensation",
            "eviction",
            "demolition",
            "displacement",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
        combo_rules=(
            ("resettlement", "urban renewal"),
            ("demolition", "compensation"),
            ("relocation", "redevelopment"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
    ),
    "U12": _topic(
        "urban",
        "gentrification exclusion and neighborhood change",
        "U4",
        seeds=(
            "gentrification",
            "touristification",
            "renoviction",
            "new-build gentrification",
            "neighborhood change",
            "neighbourhood change",
        ),
        context_terms=(
            "displacement",
            "social exclusion",
            "exclusion",
            "social mix",
            "neighborhood change",
            "neighbourhood change",
            "place identity",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS + ("gentrification",),
        combo_rules=(
            ("gentrification", "regeneration"),
            ("gentrification", "redevelopment"),
            ("neighborhood change", "renewal"),
            ("displacement", "redevelopment"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
        missing_anchor_penalty=1.95,
    ),
    "U13": _topic(
        "urban",
        "renewal evaluation sensing and machine learning",
        "U5",
        seeds=(
            "urban renewal evaluation",
            "urban regeneration assessment",
            "renewal suitability",
            "renewal identification",
            "remote sensing urban renewal",
            "urban renewal machine learning",
        ),
        context_terms=(
            "evaluation",
            "assessment",
            "remote sensing",
            "gis",
            "machine learning",
            "deep learning",
            "change detection",
            "spatial analysis",
            "classification",
            "suitability",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
        combo_rules=(
            ("remote sensing", "urban renewal"),
            ("evaluation", "redevelopment"),
            ("machine learning", "urban regeneration"),
            ("gis", "urban renewal"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
    ),
    "U14": _topic(
        "urban",
        "tod and station area upgrading",
        "U2",
        seeds=(
            "transit oriented redevelopment",
            "station area renewal",
            "station area redevelopment",
            "tod renewal",
            "rail transit-led redevelopment",
        ),
        context_terms=(
            "tod",
            "transit oriented development",
            "station area",
            "metro station",
            "rail transit",
            "redevelopment",
            "renewal",
        ),
        context_anchors=("station area", "metro station", "tod", "transit oriented development", "rail transit"),
        combo_rules=(
            ("station area", "renewal"),
            ("tod", "redevelopment"),
            ("metro station", "regeneration"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + ("station area", "metro station", "tod", "transit oriented development", "rail transit"),
        missing_anchor_penalty=2.2,
    ),
    "U15": _topic(
        "urban",
        "renewal comprehensive impacts",
        "U4",
        seeds=(
            "quality of life after regeneration",
            "health impacts of urban renewal",
            "social impacts of redevelopment",
            "environmental impacts of regeneration",
            "public health and urban renewal",
            "well being after regeneration",
            "equity impacts of redevelopment",
            "renewal quality of life",
        ),
        context_terms=(
            "quality of life",
            "well-being",
            "well being",
            "health",
            "public health",
            "mental health",
            "equity",
            "social equity",
            "livability",
            "social impact",
            "social impacts",
            "environmental impact",
            "environmental impacts",
            "accessibility",
            "quality of life",
            "community perception",
            "resident perception",
            "wellness",
        ),
        context_anchors=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
        combo_rules=(
            ("quality of life", "urban renewal"),
            ("health", "urban regeneration"),
            ("livability", "redevelopment"),
            ("equity", "urban renewal"),
            ("public health", "urban renewal"),
            ("well being", "regeneration"),
            ("community perception", "redevelopment"),
        ),
        exclude_terms=COMMON_RURAL_ANCHORS,
        requires_anchor=True,
        anchor_terms=COMMON_RENEWAL_ANCHORS + COMMON_EXISTING_URBAN_OBJECTS,
        missing_anchor_penalty=2.2,
    ),
    "N1": _topic(
        "nonurban",
        "general urbanization and expansion",
        "N4",
        seeds=(
            "urbanization",
            "urban growth",
            "metropolitan growth",
            "city growth",
            "regional urbanization",
        ),
        context_terms=(
            "urbanization",
            "urban growth",
            "migration",
            "metropolitan",
            "expansion",
            "development",
        ),
        penalize_if_renewal_anchor=True,
        renewal_anchor_penalty=2.15,
    ),
    "N2": _topic(
        "nonurban",
        "new town and greenfield development",
        "N4",
        seeds=(
            "new town",
            "greenfield development",
            "greenfield",
            "suburban expansion",
            "urban fringe development",
        ),
        context_terms=(
            "new town",
            "greenfield",
            "suburban",
            "fringe",
            "expansion",
            "suburbanization",
        ),
        combo_rules=(
            ("new town", "development"),
            ("greenfield", "development"),
            ("suburban", "expansion"),
        ),
        penalize_if_renewal_anchor=True,
        renewal_anchor_penalty=2.4,
    ),
    "N3": _topic(
        "nonurban",
        "general urban governance",
        "N1",
        seeds=(
            "urban governance",
            "local governance",
            "city governance",
            "governance network",
            "city management",
        ),
        context_terms=(
            "governance",
            "policy",
            "institution",
            "management",
            "local government",
            "planning",
        ),
        penalize_if_renewal_anchor=True,
        renewal_anchor_penalty=2.45,
    ),
    "N4": _topic(
        "nonurban",
        "housing market and real estate",
        "N1",
        seeds=(
            "housing market",
            "real estate market",
            "mortgage foreclosure",
            "housing prices",
            "property value",
        ),
        context_terms=(
            "housing market",
            "real estate",
            "mortgage",
            "foreclosure",
            "rent",
            "property value",
            "hedonic",
        ),
        penalize_if_renewal_anchor=True,
        renewal_anchor_penalty=2.45,
    ),
    "N5": _topic(
        "nonurban",
        "general social problems poverty and crime",
        "N3",
        seeds=(
            "urban poverty",
            "crime",
            "homelessness",
            "violence",
            "social deprivation",
        ),
        context_terms=(
            "poverty",
            "crime",
            "policing",
            "inequality",
            "deprivation",
            "unemployment",
        ),
        penalize_if_renewal_anchor=True,
    ),
    "N6": _topic(
        "nonurban",
        "informal settlement social-spatial studies",
        "N3",
        seeds=(
            "informal settlement livelihood",
            "slum poverty",
            "informal housing vulnerability",
        ),
        context_terms=(
            "informal settlement",
            "slum",
            "squatter",
            "vulnerability",
            "livelihood",
            "social network",
        ),
        penalize_if_renewal_anchor=True,
    ),
    "N7": _topic(
        "nonurban",
        "transport mobility and accessibility",
        "N1",
        seeds=(
            "travel behavior",
            "transit accessibility",
            "commuting pattern",
            "public transport network",
            "mobility pattern",
        ),
        context_terms=(
            "transport",
            "transit",
            "mobility",
            "accessibility",
            "commuting",
            "walkability",
        ),
        penalize_if_renewal_anchor=True,
    ),
    "N8": _topic(
        "nonurban",
        "pure methods algorithms and modeling",
        "N2",
        seeds=(
            "algorithm",
            "neural network",
            "optimization model",
            "numerical simulation",
            "graph model",
            "bipartite graph",
        ),
        context_terms=COMMON_METHOD_ANCHORS + (
            "numerical",
            "performance",
            "prediction",
            "classification",
        ),
        penalize_if_renewal_anchor=True,
        renewal_anchor_penalty=2.35,
    ),
    "N9": _topic(
        "nonurban",
        "rural agricultural and countryside change",
        "N4",
        seeds=(
            "rural",
            "rural gentrification",
            "rural regeneration",
            "agricultural",
            "farmland",
            "countryside",
            "fishing community",
        ),
        context_terms=COMMON_RURAL_ANCHORS + (
            "village",
            "rural development",
        ),
        combo_rules=(
            ("rural", "gentrification"),
            ("agricultural", "development"),
            ("fishing community", "gentrification"),
        ),
    ),
    "N10": _topic(
        "nonurban",
        "ecological restoration and environmental treatment",
        "N2",
        seeds=(
            "ecological restoration",
            "soil remediation",
            "pollution treatment",
            "wastewater treatment",
            "habitat restoration",
            "soil contamination",
        ),
        context_terms=(
            "restoration",
            "remediation",
            "pollution",
            "contamination",
            "wastewater",
            "soil",
            "habitat",
            "ecology",
        ),
        penalize_if_renewal_anchor=True,
        renewal_anchor_penalty=2.25,
    ),
    OPEN_SET_URBAN_LABEL: _topic(
        "urban",
        OPEN_SET_URBAN_NAME,
        "",
    ),
    OPEN_SET_NONURBAN_LABEL: _topic(
        "nonurban",
        OPEN_SET_NONURBAN_NAME,
        "",
    ),
}

TOPIC_ENGLISH_NAMES = {
    label: str(definition["name"])
    for label, definition in TOPIC_DEFINITIONS.items()
}

TOPIC_CHINESE_NAMES = {
    "U1": "老旧住区更新",
    "U2": "城中村更新",
    "U3": "贫民窟或非正规住区升级",
    "U4": "旧城或内城再生",
    "U5": "棕地或工业用地再开发",
    "U6": "工业遗产或历史建筑再利用",
    "U7": "历史街区保护更新",
    "U8": "公共空间或街道更新",
    "U9": "更新治理、制度与参与",
    "U10": "更新融资与政策工具",
    "U11": "搬迁、补偿、安置与驱逐",
    "U12": "绅士化、排斥与社区变化",
    "U13": "更新评估、遥感与机器学习",
    "U14": "TOD与站点地区升级",
    "U15": "更新综合影响",
    "N1": "一般城市化与城市扩张",
    "N2": "新城与绿地开发",
    "N3": "一般城市治理",
    "N4": "住房市场与房地产",
    "N5": "一般社会问题、贫困与犯罪",
    "N6": "非正规住区社会空间研究",
    "N7": "交通、流动性与可达性",
    "N8": "纯方法、算法与建模",
    "N9": "农村、农业与乡村变化",
    "N10": "生态修复与环境治理",
    UNKNOWN_TOPIC_LABEL: "未知主题",
}

TOPIC_CHINESE_NAMES.update(
    {
        OPEN_SET_URBAN_LABEL: "Urban renewal other",
        OPEN_SET_NONURBAN_LABEL: "Nonurban other",
    }
)

LEGACY_TOPIC_NAMES = {
    "U1": "renewal governance policy finance",
    "U2": "renewal projects redevelopment regeneration",
    "U3": "heritage reuse public realm renewal",
    "U4": "renewal consequences social impacts",
    "U5": "renewal assessment design sensing optimization",
    "N1": "general urban studies governance development",
    "N2": "materials engineering pollution pure technical",
    "N3": "social cultural historical discourse media",
    "N4": "new town greenfield expansion rural nonurban",
}

LEGACY_TOPIC_GROUPS = {
    "U1": "urban",
    "U2": "urban",
    "U3": "urban",
    "U4": "urban",
    "U5": "urban",
    "N1": "nonurban",
    "N2": "nonurban",
    "N3": "nonurban",
    "N4": "nonurban",
}


def is_topic_label(label: str) -> bool:
    return label in TOPIC_DEFINITIONS


def topic_group_for_label(label: str) -> str:
    if label == UNKNOWN_TOPIC_LABEL:
        return UNKNOWN_TOPIC_GROUP
    return str(TOPIC_DEFINITIONS.get(label, {}).get("group", UNKNOWN_TOPIC_GROUP))


def topic_name_for_label(label: str) -> str:
    if label == UNKNOWN_TOPIC_LABEL:
        return UNKNOWN_TOPIC_NAME
    return str(TOPIC_DEFINITIONS.get(label, {}).get("name", UNKNOWN_TOPIC_NAME))


def topic_name_zh_for_label(label: str) -> str:
    if label == UNKNOWN_TOPIC_LABEL:
        return TOPIC_CHINESE_NAMES[UNKNOWN_TOPIC_LABEL]
    return str(TOPIC_CHINESE_NAMES.get(label, TOPIC_CHINESE_NAMES[UNKNOWN_TOPIC_LABEL]))


def legacy_topic_for_label(label: str) -> tuple[str, str, str]:
    if label == UNKNOWN_TOPIC_LABEL or not is_topic_label(label):
        return "", "", ""
    legacy_label = str(TOPIC_DEFINITIONS[label].get("legacy_label", "") or "")
    return (
        legacy_label,
        LEGACY_TOPIC_GROUPS.get(legacy_label, ""),
        LEGACY_TOPIC_NAMES.get(legacy_label, ""),
    )


def urban_flag_for_topic_label(label: str) -> str:
    if label == UNKNOWN_TOPIC_LABEL or not label:
        return ""
    if label.startswith("U"):
        return "1"
    if label.startswith("N"):
        return "0"
    return ""


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _match_weighted_terms(
    *,
    terms: Sequence[str],
    field_texts: Dict[str, str],
    base_score: float,
) -> tuple[float, List[str]]:
    score = 0.0
    matched: List[str] = []
    field_weights = {"title": 1.35, "abstract": 1.0}
    for term in terms:
        normalized = normalize_phrase(term)
        if not normalized:
            continue
        for field_name, field_text in field_texts.items():
            if normalized in field_text:
                score += base_score * field_weights.get(field_name, 1.0)
                matched.append(f"{field_name}:{normalized}")
    return score, _dedupe(matched)


def score_topic_definition(
    label: str,
    *,
    title: str,
    abstract: str,
) -> Dict[str, object]:
    definition = TOPIC_DEFINITIONS[label]
    field_texts = {
        "title": normalize_phrase(title).replace("-", " "),
        "abstract": normalize_phrase(abstract).replace("-", " "),
    }
    combined_text = " ".join(value for value in field_texts.values() if value)
    score = 0.0
    matched: List[str] = []

    seed_score, seed_matches = _match_weighted_terms(
        terms=definition.get("seeds", []),
        field_texts=field_texts,
        base_score=3.0,
    )
    score += seed_score
    matched.extend(seed_matches)

    context_anchors = [
        normalize_phrase(anchor).replace("-", " ")
        for anchor in definition.get("context_anchors", [])
        if normalize_phrase(anchor)
    ]
    anchor_hits = [anchor for anchor in context_anchors if anchor in combined_text]
    if definition.get("context_terms"):
        context_score, context_matches = _match_weighted_terms(
            terms=definition.get("context_terms", []),
            field_texts=field_texts,
            base_score=1.0,
        )
        if context_matches:
            if context_anchors:
                if anchor_hits:
                    score += context_score + float(definition.get("context_bonus", 1.5))
                    matched.extend(context_matches)
                    matched.extend(f"anchor:{anchor}" for anchor in anchor_hits[:3])
            else:
                score += context_score
                matched.extend(context_matches)

    combo_hits: List[str] = []
    for combo in definition.get("combo_rules", []):
        normalized_combo = [normalize_phrase(term).replace("-", " ") for term in combo if normalize_phrase(term)]
        if normalized_combo and all(term in combined_text for term in normalized_combo):
            score += 4.0
            combo_hits.append("+".join(normalized_combo))
    matched.extend(f"combo:{combo}" for combo in combo_hits)

    exclude_hits = []
    for term in definition.get("exclude_terms", []):
        normalized = normalize_phrase(term).replace("-", " ")
        if normalized and normalized in combined_text:
            score -= 3.0
            exclude_hits.append(normalized)
    matched.extend(f"exclude:{term}" for term in exclude_hits)

    if definition.get("requires_anchor"):
        required_anchors = [
            normalize_phrase(anchor).replace("-", " ")
            for anchor in definition.get("anchor_terms", [])
            if normalize_phrase(anchor)
        ]
        if required_anchors and not any(anchor in combined_text for anchor in required_anchors):
            score -= float(definition.get("missing_anchor_penalty", 1.75))
            matched.append("penalty:missing_anchor")

    if definition.get("penalize_if_renewal_anchor"):
        renewal_hits = [
            normalize_phrase(anchor).replace("-", " ")
            for anchor in COMMON_RENEWAL_ANCHORS
            if normalize_phrase(anchor) and normalize_phrase(anchor).replace("-", " ") in combined_text
        ]
        substantive_object_hits = [
            normalize_phrase(anchor).replace("-", " ")
            for anchor in COMMON_EXISTING_URBAN_OBJECTS
            if normalize_phrase(anchor) and normalize_phrase(anchor).replace("-", " ") in combined_text
        ]
        substantive_mechanism_hits = [
            normalize_phrase(anchor).replace("-", " ")
            for anchor in ("compensation", "relocation", "resettlement", "tif", "ppp", "reit", "land value capture")
            if normalize_phrase(anchor) and normalize_phrase(anchor).replace("-", " ") in combined_text
        ]
        if renewal_hits and (substantive_object_hits or substantive_mechanism_hits):
            score -= float(definition.get("renewal_anchor_penalty", 1.75))
            matched.append("penalty:renewal_anchor_present")

    return {
        "label": label,
        "group": str(definition["group"]),
        "name": str(definition["name"]),
        "score": round(score, 4),
        "matched_terms": _dedupe(matched),
        "combo_hits": combo_hits,
    }


def score_all_topics(*, title: str, abstract: str) -> List[Dict[str, object]]:
    scored = [score_topic_definition(label, title=title, abstract=abstract) for label in TOPIC_ORDER]
    order_index = {label: idx for idx, label in enumerate(TOPIC_ORDER)}
    scored.sort(key=lambda item: (-float(item["score"]), order_index.get(str(item["label"]), 999)))
    return scored
