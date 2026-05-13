import numpy as np

from src.urban.specter2.cache import EmbeddingCache, make_cache_key


def test_cache_key_is_stable_for_same_model_adapter_title_and_abstract():
    key_a = make_cache_key(
        model_name="allenai/specter2_base",
        adapter_name="allenai/specter2_classification",
        title="Brownfield redevelopment",
        abstract="Urban renewal in old industrial land.",
    )
    key_b = make_cache_key(
        model_name="allenai/specter2_base",
        adapter_name="allenai/specter2_classification",
        title="Brownfield redevelopment",
        abstract="Urban renewal in old industrial land.",
    )
    key_c = make_cache_key(
        model_name="allenai/specter2_base",
        adapter_name="allenai/specter2_classification",
        title="Brownfield redevelopment",
        abstract="Transit demand modeling.",
    )

    assert key_a == key_b
    assert key_a != key_c
    assert len(key_a) == 64


def test_embedding_cache_round_trips_numpy_vector(tmp_path):
    cache = EmbeddingCache(tmp_path)
    key = make_cache_key(
        model_name="allenai/specter2_base",
        adapter_name="allenai/specter2_classification",
        title="A",
        abstract="B",
    )
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    cache.store(key, vector)
    loaded = cache.load(key)

    np.testing.assert_allclose(loaded, vector)
