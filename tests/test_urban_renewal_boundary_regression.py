import json
from pathlib import Path

from src.prompts import PromptGenerator


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "urban_renewal_boundary_cases.json"


def test_boundary_fixture_is_large_and_balanced():
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(cases) >= 50
    labels = {case["expected_label"] for case in cases}
    assert labels == {0, 1}


def test_boundary_fixture_covers_required_error_modes():
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    tags = {tag for case in cases for tag in case["tags"]}
    required_tags = {
        "densification",
        "heritage_conservation",
        "public_realm",
        "blight_demolition",
        "gentrification_effect",
        "finance_reit",
        "governance_policy",
        "general_sustainability",
        "uhi_climate",
        "social_history",
        "urbanism_overview",
        "remote_sensing_method",
        "optimization_method",
        "materials_tech",
    }
    assert required_tags.issubset(tags)


def test_prompt_family_uses_broad_academic_definition():
    required_phrases = [
        "task definition",
        "core decision rule",
        "important positive bias for this task",
        "label 0 only in the following cases",
        "boundary handling",
        "relatively broad but still academic sense",
        "gentrification in an existing urban built environment generally counts as 1 when it is a main research topic",
    ]

    for mode in ["zero", "one", "few"]:
        prompt = PromptGenerator(shot_mode=mode).get_single_system_prompt().lower()
        for phrase in required_phrases:
            assert phrase in prompt


def test_few_prompt_covers_updated_boundary_examples_and_removes_softening_language():
    prompt = PromptGenerator(shot_mode="few").get_single_system_prompt().lower()
    required_phrases = [
        "slum upgrading and community participation in nairobi informal settlements",
        "financing brownfield redevelopment through urban regeneration partnerships",
        "state-led gentrification and displacement under urban renewal",
        "gentrification in hong kong? epistemology vs. ontology",
        "visual and aesthetic markers of gentrification in tourist destinations",
        "predictive modeling of fire incidence for future urban renewal planning",
        "pollution risks in soil samples from an urban renewal area",
        "new town expansion and housing growth at the urban fringe",
    ]
    for phrase in required_phrases:
        assert phrase in prompt

    forbidden_phrases = [
        "lean 1",
        "central historical episode",
        "urban discourse context",
        "{title}",
        "{abstract}",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in prompt
