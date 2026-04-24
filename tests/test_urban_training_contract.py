from pathlib import Path

from src.config import Config
from src.urban_topic_classifier import UrbanTopicClassifier
from src.urban_training_contract import (
    allowed_training_workbooks,
    assert_training_source_contract,
    is_stable_release_training_workbook,
)


def test_allowed_training_workbooks_exclude_stable_release_workbook():
    workbooks = allowed_training_workbooks(Config.TRAIN_DIR)
    assert workbooks
    assert any("test1-test7_merged" in path.name for path in workbooks)
    assert all(not is_stable_release_training_workbook(path) for path in workbooks)


def test_training_contract_rejects_stable_release_workbook():
    stable_path = Config.TRAIN_DIR / f"{Config.STABLE_RELEASE_DATASET_ID}.xlsx"
    try:
        assert_training_source_contract([stable_path])
    except ValueError as error:
        assert "stable_release workbook" in str(error)
    else:
        raise AssertionError("Expected stable training contract to reject the stable workbook.")


def test_topic_classifier_records_clean_training_sources():
    classifier = UrbanTopicClassifier()
    assert classifier.training_sources
    assert all(not is_stable_release_training_workbook(path) for path in classifier.training_sources)
    assert all(isinstance(path, Path) for path in classifier.training_sources)
