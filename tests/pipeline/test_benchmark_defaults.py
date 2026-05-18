from pathlib import Path

from scripts.evaluation import benchmark_api_vs_classifier as benchmark


def test_benchmark_defaults_are_project_relative():
    assert benchmark.DEFAULT_BENCHMARK_INPUT == (
        benchmark.PROJECT_ROOT
        / "output"
        / "spreadsheet"
        / "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx"
    )
    assert benchmark.DEFAULT_BENCHMARK_OUTPUT_DIR == benchmark.PROJECT_ROOT / "output" / "spreadsheet"
    assert benchmark.default_benchmark_session_root("demo") == (
        benchmark.PROJECT_ROOT / "tmp" / "benchmark_sessions" / "demo"
    )

    for path in (
        benchmark.DEFAULT_BENCHMARK_INPUT,
        benchmark.DEFAULT_BENCHMARK_OUTPUT_DIR,
        benchmark.default_benchmark_session_root("demo"),
    ):
        assert Path(path).is_absolute()
        assert str(path).startswith(str(benchmark.PROJECT_ROOT))
