from pathlib import Path


def test_scripts_root_contains_only_package_marker_and_directories():
    scripts_root = Path(__file__).resolve().parents[2] / "scripts"
    root_py_files = sorted(path.name for path in scripts_root.glob("*.py"))

    assert root_py_files == ["__init__.py"]


def test_scripts_do_not_use_compat_loader():
    scripts_root = Path(__file__).resolve().parents[2] / "scripts"
    offenders = []
    for path in scripts_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "scripts._compat" in text or "load_script_module" in text:
            offenders.append(str(path.relative_to(scripts_root)))

    assert offenders == []


def test_analysis_root_contains_only_package_marker_and_subdirectories():
    analysis_root = Path(__file__).resolve().parents[2] / "scripts" / "analysis"
    root_py_files = sorted(path.name for path in analysis_root.glob("*.py"))

    assert root_py_files == ["__init__.py"]
