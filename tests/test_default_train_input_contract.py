from pathlib import Path

from src.config import Config
from src.urban_training_contract import is_stable_release_training_workbook


def test_config_default_train_input_file_resolves_existing_workbook():
    resolved = Config.default_train_input_file()
    assert resolved is not None
    assert Path(resolved).exists()
    assert str(resolved).lower().endswith(".xlsx")


def test_config_requires_default_train_input_file_and_excludes_stable_workbook():
    resolved = Config.require_default_train_input_file()
    assert Path(resolved).exists()
    assert not is_stable_release_training_workbook(Path(resolved))

