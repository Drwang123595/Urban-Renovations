from pathlib import Path


FORBIDDEN_ROOT_COMPAT_FILES = {
    "_compat.py",
    "config.py",
    "data_processor.py",
    "evaluation_core.py",
    "llm_client.py",
    "memory.py",
    "merged_output.py",
    "project_paths.py",
    "prompt_manifest.py",
    "prompt_strategy_manager.py",
    "prompt_strategy_registry.py",
    "prompts.py",
    "review_experiment_report.py",
    "review_workbook_analysis.py",
    "task_router.py",
    "urban_bertopic_service.py",
    "urban_binary_policy_v2.py",
    "urban_family_gate.py",
    "urban_hybrid_classifier.py",
    "urban_metadata.py",
    "urban_rule_filter.py",
    "urban_topic_classifier.py",
    "urban_topic_taxonomy.py",
    "urban_training_contract.py",
}


def test_src_root_does_not_contain_legacy_compatibility_wrappers():
    src_root = Path(__file__).resolve().parents[2] / "src"
    existing = sorted(name for name in FORBIDDEN_ROOT_COMPAT_FILES if (src_root / name).exists())

    assert existing == []
