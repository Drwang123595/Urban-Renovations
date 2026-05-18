from pathlib import Path

from src.urban import urban_metadata as legacy_metadata
from src.urban import urban_training_contract as legacy_training_contract
from src.urban.core import metadata as core_metadata
from src.urban.core import training_contract as core_training_contract


def test_core_metadata_is_canonical_implementation_and_legacy_path_aliases_it():
    assert core_metadata.UrbanMetadataRecord.__module__ == "src.urban.core.metadata"
    assert legacy_metadata.UrbanMetadataRecord is core_metadata.UrbanMetadataRecord
    assert legacy_metadata.normalize_phrase is core_metadata.normalize_phrase
    assert core_metadata.build_keywords("Urban Renewal; Brownfield", "brownfield; Retrofit") == (
        "urban renewal; brownfield; retrofit"
    )


def test_core_training_contract_is_canonical_implementation_and_legacy_path_aliases_it():
    assert core_training_contract.allowed_training_workbooks.__module__ == (
        "src.urban.core.training_contract"
    )
    assert legacy_training_contract.allowed_training_workbooks is core_training_contract.allowed_training_workbooks
    assert legacy_training_contract.assert_training_source_contract is (
        core_training_contract.assert_training_source_contract
    )


def test_urban_internal_modules_import_core_contracts_instead_of_root_legacy_wrappers():
    urban_root = Path(__file__).resolve().parents[2] / "src" / "urban"
    canonical_files = [
        path
        for path in urban_root.rglob("*.py")
        if path.name != "__init__.py"
        and path.name not in {
            "urban_metadata.py",
            "urban_training_contract.py",
        }
    ]

    offenders = []
    for path in canonical_files:
        text = path.read_text(encoding="utf-8")
        if "from ..urban_metadata" in text or "from ..urban_training_contract" in text:
            offenders.append(str(path.relative_to(urban_root)))

    assert offenders == []
