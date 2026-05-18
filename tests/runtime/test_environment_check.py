from scripts.dev import check_environment


def test_environment_check_runs_lock_pip_and_import_checks(monkeypatch):
    calls = []

    def fake_run(command, *, cwd=None):
        calls.append(tuple(command))
        return check_environment.CheckResult(command=tuple(command), returncode=0, output="ok")

    monkeypatch.setattr(check_environment, "run_command", fake_run)

    results = check_environment.run_environment_checks(
        python_executable="python",
        required_modules=("pandas", "openpyxl"),
    )

    assert [result.returncode for result in results] == [0, 0, 0]
    assert calls[0] == ("uv", "lock", "--check")
    assert calls[1] == ("python", "-m", "pip", "check")
    assert calls[2][:2] == ("python", "-c")
    assert calls[2][2].startswith("import importlib.util; missing = []")
    assert "pandas" in calls[2][2]
    assert "openpyxl" in calls[2][2]
