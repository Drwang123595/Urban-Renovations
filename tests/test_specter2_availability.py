import importlib

from src.urban.specter2.config import Specter2Config
from src.urban.specter2.encoder import Specter2Encoder, check_availability


def test_encoder_reports_unavailable_when_adapters_dependency_is_missing(monkeypatch):
    import src.urban.specter2.encoder as encoder_module

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "adapters":
            raise ImportError("No module named 'adapters'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(encoder_module.importlib, "import_module", fake_import_module)

    availability = check_availability()
    assert availability.status == "specter2_unavailable"
    assert "adapters" in availability.reason

    result = Specter2Encoder(Specter2Config()).encode(
        [{"title": "Brownfield redevelopment", "abstract": "Urban renewal in old industrial land."}]
    )
    assert result.status == "specter2_unavailable"
    assert result.embeddings.shape == (0, 0)
