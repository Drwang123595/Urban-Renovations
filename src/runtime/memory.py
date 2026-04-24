import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import Config

class ConversationMemory:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        session_path: Optional[Union[str, Path]] = None,
        skip_index: bool = False,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the memory with an optional system prompt and session ID.
        If session_path is provided, it overrides the default UUID-based path.
        If session_id is provided, tries to load existing history.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[Dict[str, str]] = []
        self.created_at = time.time()
        self.warning_triggered = False
        self.skip_index = skip_index
        self.last_event = ""
        self.error_code: Optional[str] = None
        self.audit_metadata: Dict[str, Any] = {}
        self._initial_system_prompt = system_prompt or ""
        
        if session_path:
            self.session_path = Path(session_path)
            # Ensure parent dir exists
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.session_path = None
        
        # Try load if session_id existed (and we are using default path) or if explicit path exists
        if self._session_file_path.exists():
            self.load()
            if not self.messages and system_prompt:
                self.add_system_message(system_prompt)
        elif system_prompt:
            self.add_system_message(system_prompt)

        if audit_metadata:
            self.update_audit_metadata(audit_metadata)
            
    @property
    def _session_file_path(self) -> Path:
        if self.session_path:
            return self.session_path
        return Config.SESSIONS_DIR / f"{self.session_id}.json"

    def update_audit_metadata(self, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        payload = dict(metadata or {})
        payload.update(kwargs)
        for key, value in payload.items():
            if value is None:
                continue
            self.audit_metadata[str(key)] = self._sanitize_audit_value(value)

    def set_last_event(self, event: Optional[str]):
        self.last_event = str(event or "").strip()

    def set_error_code(self, error_code: Optional[str]):
        value = str(error_code or "").strip()
        self.error_code = value or None

    def add_system_message(self, content: str):
        self._add_message("system", content)

    def add_user_message(self, content: str):
        self._add_message("user", content)

    def add_assistant_message(self, content: str):
        self._add_message("assistant", content)

    def _add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.check_token_limit()
        # Removed auto-save on every message to improve I/O performance.
        # Call save() explicitly when needed.

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def clear(self):
        self.messages = []
        self.save()

    def save(self):
        """Persist conversation to JSON file."""
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "last_event": self.last_event,
            "error_code": self.error_code,
            "message_count": len(self.messages),
            "audit_metadata": self.audit_metadata,
            "messages": self._serialized_messages(),
        }
        with open(self._session_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # Update index only if not skipped
        if not self.skip_index:
            self._update_index()

    def load(self):
        """Load conversation from JSON file."""
        if not self._session_file_path.exists():
            return
            
        with open(self._session_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.session_id = data.get("session_id", self.session_id)
            self.created_at = data.get("created_at", self.created_at)
            self.messages = data.get("messages", [])
            self.last_event = str(data.get("last_event", self.last_event) or "")
            self.error_code = data.get("error_code") or self.error_code
            loaded_metadata = data.get("audit_metadata") or {}
            if isinstance(loaded_metadata, dict):
                self.update_audit_metadata(loaded_metadata)

    def _update_index(self):
        """Update the global index.json file."""
        index_file = Config.INDEX_FILE
        entries = []
        
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except:
                entries = []
                
        # Remove existing entry for this session
        entries = [e for e in entries if e["session_id"] != self.session_id]
                
        # Create summary
        title = "New Conversation"
        for msg in self.messages:
            if msg["role"] == "user":
                title = msg["content"][:50] + "..."
                break
        if title == "New Conversation" and self.audit_metadata:
            task_type = str(self.audit_metadata.get("task_type", "") or "").strip()
            strategy_name = str(self.audit_metadata.get("strategy_name", "") or "").strip()
            if task_type or strategy_name:
                title = " / ".join(part for part in (task_type, strategy_name) if part)
                
        entries.append({
            "session_id": self.session_id,
            "title": title,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "message_count": len(self.messages),
            "last_event": self.last_event,
            "error_code": self.error_code,
        })
        
        # Sort by updated_at desc
        entries.sort(key=lambda x: x["updated_at"], reverse=True)
        
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def check_token_limit(self):
        """Estimate token usage and warn if approaching limit."""
        if self.warning_triggered:
            return

        # Rough estimation: 1 token ~= 4 chars (English) or 1 char (Chinese)
        # We use a conservative estimate: len(content)
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        estimated_tokens = total_chars  # Conservative for mixed content
        
        limit = Config.MAX_CONTEXT_TOKENS
        threshold = limit * Config.TOKEN_WARNING_THRESHOLD
        
        if estimated_tokens > threshold:
            print(f"\n[WARNING] Session {self.session_id}: Context length ({estimated_tokens} chars) is approaching the limit ({limit})!")
            self.warning_triggered = True

    def is_context_full(self) -> bool:
        """Check if context is nearly full (for strategy decision making)."""
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        estimated_tokens = total_chars
        limit = Config.MAX_CONTEXT_TOKENS
        threshold = limit * Config.TOKEN_WARNING_THRESHOLD
        return estimated_tokens > threshold

    def _serialized_messages(self) -> List[Dict[str, str]]:
        if not Config.PERSIST_FULL_SESSIONS:
            return []

        serialized = []
        for item in self.messages:
            serialized.append(
                {
                    "role": str(item.get("role", "") or ""),
                    "content": self._sanitize_text(
                        item.get("content", ""),
                        max_chars=Config.SESSION_MESSAGE_MAX_CHARS,
                    ),
                }
            )
        return serialized

    def _sanitize_audit_value(self, value: Any) -> Any:
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, Path):
            value = str(value)
        elif not isinstance(value, str):
            try:
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                value = str(value)
        return self._sanitize_text(value, max_chars=Config.AUDIT_FIELD_MAX_CHARS)

    def _sanitize_text(self, value: Any, *, max_chars: int) -> str:
        text = str(value or "")
        sensitive_values = [
            Config.API_KEY,
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

        if len(text) > max_chars:
            text = text[:max_chars] + "...(truncated)"
        return text
