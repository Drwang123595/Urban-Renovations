import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runtime.config import Config
from src.runtime.llm_client import DeepSeekClient, LLMQuotaExceededError
from src.runtime.resume import ResumeCheckpoint, summary_path_for_checkpoint


class _FakeRateLimitError(Exception):
    def __init__(self, message, *, status_code=429, body=None, response_text=""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(text=response_text, headers={})


def test_llm_client_raises_quota_exceeded_for_daily_limit(monkeypatch, capsys):
    monkeypatch.setattr("src.runtime.llm_client.RateLimitError", _FakeRateLimitError)
    monkeypatch.setattr("src.runtime.llm_client.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(Config, "DEBUG_SENSITIVE_LOGGING", False)

    def fake_create(**_kwargs):
        raise _FakeRateLimitError(
            "Error code: 429 - daily usage limit exceeded sk-test-secret",
            body={
                "code": "USAGE_LIMIT_EXCEEDED",
                "message": 'error: code=429 reason="DAILY_LIMIT_EXCEEDED" message="daily usage limit exceeded"',
            },
        )

    client = DeepSeekClient(api_key="sk-test-secret", base_url="https://sub2.de5.net/v1", model="demo")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    with pytest.raises(LLMQuotaExceededError) as exc_info:
        client.chat_completion([{"role": "user", "content": "Reply OK"}], max_retries=3)

    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "USAGE_LIMIT_EXCEEDED"
    output = capsys.readouterr().out
    assert "DAILY_LIMIT_EXCEEDED" in output
    assert "sk-test-secret" not in output


def test_llm_client_treats_429_too_many_requests_as_quota_without_retry(monkeypatch):
    monkeypatch.setattr("src.runtime.llm_client.RateLimitError", _FakeRateLimitError)
    monkeypatch.setattr("src.runtime.llm_client.time.sleep", lambda _seconds: None)
    attempts = 0

    def fake_create(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise _FakeRateLimitError("429 Too Many Requests", response_text="Too Many Requests")

    client = DeepSeekClient(api_key="sk-test-secret", base_url="https://sub2.de5.net/v1", model="demo")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    with pytest.raises(LLMQuotaExceededError):
        client.chat_completion([{"role": "user", "content": "Reply OK"}], max_retries=3)

    assert attempts == 1


def test_resume_checkpoint_skips_only_completed_and_keys_by_row_task_run_and_input(tmp_path):
    path = tmp_path / "predictions.xlsx.checkpoint.jsonl"
    checkpoint = ResumeCheckpoint(path)
    key_a = checkpoint.key(row_index=1, task_type="urban_renewal", run_id="run-a", input_fingerprint="fp")
    key_b = checkpoint.key(row_index=1, task_type="spatial", run_id="run-a", input_fingerprint="fp")
    key_c = checkpoint.key(row_index=1, task_type="urban_renewal", run_id="run-b", input_fingerprint="fp")

    checkpoint.append_completed(key_a, row={"Article Title": "Duplicate", "final_label": "1"})
    checkpoint.append_record(key_b, status="quota_exhausted", row={"Article Title": "Duplicate"})
    checkpoint.append_completed(key_c, row={"Article Title": "Duplicate", "final_label": "0"})

    loaded = ResumeCheckpoint(path)

    assert loaded.completed_row(key_a)["final_label"] == "1"
    assert loaded.completed_row(key_b) is None
    assert loaded.completed_row(key_c)["final_label"] == "0"
    assert key_a != key_b
    assert key_a != key_c

    summary = json.loads(summary_path_for_checkpoint(path).read_text(encoding="utf-8"))
    assert summary["completed"] == 2
    assert summary["quota_exhausted"] == 1
