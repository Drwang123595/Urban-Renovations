import os
import re
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError
from .config import Config

class DeepSeekClient:
    """
    Generic LLM Client wrapper compatible with OpenAI SDK.
    Despite the name (legacy), it supports any OpenAI-compatible provider.
    """
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or Config.API_KEY
        self.base_url = base_url or Config.BASE_URL
        self.model = model or Config.MODEL_NAME
        
        if not self.api_key:
            print("Warning: API Key is not set. API calls will fail.")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=Config.TIMEOUT
        )

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
            
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_retries: int = 3) -> Optional[str]:
        """
        Call LLM API for chat completion using OpenAI SDK.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=Config.MAX_TOKENS,
                    stream=False
                )
                
                content = response.choices[0].message.content
                return content
                
            except RateLimitError as e:
                print(f"Rate Limit Hit (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5 * (attempt + 1)) # Aggressive backoff
                
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
