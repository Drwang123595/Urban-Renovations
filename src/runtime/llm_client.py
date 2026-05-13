import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    from openai import OpenAI, APIError, RateLimitError
except ImportError:
    OpenAI = None

    class APIError(Exception):
        pass

    class RateLimitError(Exception):
        pass

from .config import Config


class EmptyLLMResponseError(RuntimeError):
    """Raised when a provider returns success without usable text."""


class DeepSeekClient:
    """
    Generic LLM Client wrapper compatible with OpenAI SDK.
    Despite the name (legacy), it supports any OpenAI-compatible provider.
    """
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or Config.API_KEY
        self.base_url = base_url or Config.BASE_URL
        self.model = model or Config.MODEL_NAME
        self.api_mode, sdk_base_url = self._resolve_api_mode_and_base_url(self.base_url)
        
        if not self.api_key:
            print("Warning: API Key is not set. API calls will fail.")

        self.client = None
        if OpenAI is not None:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=sdk_base_url,
                timeout=Config.TIMEOUT
            )

    def _resolve_api_mode_and_base_url(self, base_url: str) -> Tuple[str, str]:
        normalized = (base_url or "").rstrip("/")
        if normalized.lower().endswith("/responses"):
            return "responses", normalized.rsplit("/", 1)[0]
        return "chat_completions", base_url

    def _mask_secret(self, value: str) -> str:
        if not value:
            return ""
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}...{value[-4:]}"

    def _sanitize_diagnostic_text(self, value: Any, limit: int = 240) -> str:
        text = self._shorten(value, limit=limit)
        sensitive_values = [
            self.api_key,
            os.environ.get("HTTP_PROXY", ""),
            os.environ.get("HTTPS_PROXY", ""),
            os.environ.get("ALL_PROXY", ""),
            os.environ.get("NO_PROXY", ""),
        ]
        for item in sensitive_values:
            if item:
                text = text.replace(item, "[REDACTED]")

        text = re.sub(r"\bsk-[A-Za-z0-9\-_]+\b", "[REDACTED]", text)
        text = re.sub(r"\bhf_[A-Za-z0-9]+\b", "[REDACTED]", text)
        text = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)(\S+)", r"\1[REDACTED]", text)
        text = re.sub(r"(?i)(https?://)([^/\s:@]+):([^@\s]+)@", r"\1[REDACTED]:[REDACTED]@", text)
        return text

    def _shorten(self, value: Any, limit: int = 500) -> str:
        if value is None:
            return ""
        text = str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    def _extract_error_payload(self, error: APIError) -> str:
        body = getattr(error, "body", None)
        if body:
            return self._shorten(body)
        response = getattr(error, "response", None)
        if response is not None:
            try:
                return self._shorten(response.text)
            except Exception:
                return ""
        return ""

    def _print_api_error_diagnostics(self, error: APIError, attempt: int, max_retries: int):
        status_code = getattr(error, "status_code", None)
        request_id = getattr(error, "request_id", None)
        response = getattr(error, "response", None)
        response_headers = {}
        if response is not None:
            try:
                response_headers = dict(getattr(response, "headers", {}) or {})
            except Exception:
                response_headers = {}
        diagnostic_headers = {}
        for key in ["x-request-id", "request-id", "cf-ray", "server"]:
            if key in response_headers:
                diagnostic_headers[key] = response_headers[key]
        print(
            "API Error Diagnostic | "
            f"attempt={attempt+1}/{max_retries} | "
            f"status={status_code} | "
            f"model={self.model} | "
            f"base_url={self.base_url} | "
            f"timeout={Config.TIMEOUT} | "
            f"request_id={request_id or diagnostic_headers.get('x-request-id') or diagnostic_headers.get('request-id') or ''}"
        )
        if Config.DEBUG_SENSITIVE_LOGGING and diagnostic_headers:
            print(f"API Error Headers | {diagnostic_headers}")
        payload = self._extract_error_payload(error)
        if Config.DEBUG_SENSITIVE_LOGGING and payload:
            print(f"API Error Payload | {self._sanitize_diagnostic_text(payload)}")
        kind = "API_ERROR"
        if status_code == 401:
            kind = "UNAUTHORIZED_401"
        elif status_code == 403:
            kind = "FORBIDDEN_403"
        elif status_code == 404:
            kind = "NOT_FOUND_404"
        elif status_code == 429:
            kind = "RATE_LIMIT_429"
        print(f"API Error Kind | {kind}")

    def _messages_to_responses_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        instructions = []
        input_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role in {"system", "developer"}:
                instructions.append(content)
            elif role in {"user", "assistant"}:
                input_messages.append({"role": role, "content": content})
            else:
                input_messages.append({"role": "user", "content": f"{role}: {content}"})

        payload: Dict[str, Any] = {
            "input": input_messages if input_messages else "",
        }
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        return payload

    def _extract_responses_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        parts = []
        for item in getattr(response, "output", []) or []:
            content_items = getattr(item, "content", None)
            if isinstance(item, dict):
                content_items = item.get("content", content_items)
            for content_item in content_items or []:
                text = getattr(content_item, "text", None)
                if isinstance(content_item, dict):
                    text = content_item.get("text", text)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    def _describe_empty_response(self, response: Any) -> str:
        fields = []
        for name in ["id", "status", "finish_reason", "incomplete_details"]:
            value = getattr(response, name, None)
            if isinstance(response, dict):
                value = response.get(name, value)
            if value:
                fields.append(f"{name}={self._shorten(value, limit=120)}")
        output = getattr(response, "output", None)
        if isinstance(response, dict):
            output = response.get("output", output)
        if output is not None:
            try:
                fields.append(f"output_items={len(output)}")
            except TypeError:
                fields.append("output_items=unknown")
        return "; ".join(fields) or "no diagnostic fields"

    def _responses_completion(self, messages: List[Dict[str, str]], temperature: float):
        payload = self._messages_to_responses_payload(messages)
        response = self.client.responses.create(
            model=self.model,
            temperature=temperature,
            max_output_tokens=Config.MAX_TOKENS,
            stream=False,
            **payload,
        )
        text = self._extract_responses_text(response).strip()
        if not text:
            raise EmptyLLMResponseError(self._describe_empty_response(response))
        return text

    def _chat_completions_completion(self, messages: List[Dict[str, str]], temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=Config.MAX_TOKENS,
            stream=False
        )
        text = response.choices[0].message.content
        text = "" if text is None else str(text).strip()
        if not text:
            raise EmptyLLMResponseError("chat_completions returned empty message content")
        return text

    def _chat_completions_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_retries: int,
    ) -> Optional[str]:
        if not hasattr(self.client, "chat"):
            return None
        for attempt in range(max_retries):
            try:
                print(
                    "LLM Fallback | "
                    f"mode=chat_completions | attempt={attempt+1}/{max_retries} | "
                    f"model={self.model} | base_url={self.base_url}"
                )
                return self._chat_completions_completion(messages, temperature)
            except EmptyLLMResponseError as e:
                print(
                    "Empty LLM Fallback Response | "
                    f"attempt={attempt+1}/{max_retries} | "
                    f"diagnostic={self._sanitize_diagnostic_text(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
            except RateLimitError as e:
                print(f"Rate Limit Hit in fallback (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5 * (attempt + 1))
            except APIError as e:
                self._print_api_error_diagnostics(e, attempt, max_retries)
                print(
                    f"API Error in fallback (Attempt {attempt+1}/{max_retries}): "
                    f"{self._sanitize_diagnostic_text(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
            except Exception as e:
                print(
                    f"Unexpected Error in fallback (Attempt {attempt+1}/{max_retries}): "
                    f"{self._sanitize_diagnostic_text(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        return None

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_retries: int = 3) -> Optional[str]:
        """
        Call LLM API for chat completion using OpenAI SDK.
        """
        if self.client is None:
            raise RuntimeError(
                "openai package is not installed. Install the project runtime first "
                "with `python -m pip install -e .[dev]` in Python 3.13."
            )
        for attempt in range(max_retries):
            try:
                if self.api_mode == "responses":
                    return self._responses_completion(messages, temperature)

                return self._chat_completions_completion(messages, temperature)
                
            except RateLimitError as e:
                print(f"Rate Limit Hit (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5 * (attempt + 1)) # Aggressive backoff

            except EmptyLLMResponseError as e:
                print(
                    "Empty LLM Response | "
                    f"attempt={attempt+1}/{max_retries} | "
                    f"mode={self.api_mode} | model={self.model} | "
                    f"diagnostic={self._sanitize_diagnostic_text(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                elif self.api_mode == "responses":
                    fallback = self._chat_completions_fallback(
                        messages,
                        temperature,
                        max_retries=max(1, min(2, max_retries)),
                    )
                    if fallback:
                        return fallback
                    return None
                else:
                    return None
                
            except APIError as e:
                self._print_api_error_diagnostics(e, attempt, max_retries)
                print(
                    f"API Error (Attempt {attempt+1}/{max_retries}): "
                    f"{self._sanitize_diagnostic_text(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None
                    
            except Exception as e:
                print(
                    f"Unexpected Error (Attempt {attempt+1}/{max_retries}): "
                    f"{self._sanitize_diagnostic_text(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None
                    
        return None
