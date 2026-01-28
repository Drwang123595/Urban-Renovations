import json
import uuid
import time
from typing import List, Dict, Optional
from pathlib import Path
from .config import Config

class ConversationMemory:
    def __init__(self, system_prompt: Optional[str] = None, session_id: Optional[str] = None):
        """
        Initialize the memory with an optional system prompt and session ID.
        If session_id is provided, tries to load existing history.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[Dict[str, str]] = []
        self.created_at = time.time()
        self.warning_triggered = False
        
        # Try load if session_id existed, otherwise init
        if session_id and self._session_file_path.exists():
            self.load()
        elif system_prompt:
            self.add_system_message(system_prompt)
            
    @property
    def _session_file_path(self) -> Path:
        return Config.SESSIONS_DIR / f"{self.session_id}.json"

    def add_system_message(self, content: str):
        self._add_message("system", content)

    def add_user_message(self, content: str):
        self._add_message("user", content)

    def add_assistant_message(self, content: str):
        self._add_message("assistant", content)

    def _add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.check_token_limit()
        self.save()

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
            "messages": self.messages
        }
        with open(self._session_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # Update index (optional, can be optimized to not run every save)
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
                
        entries.append({
            "session_id": self.session_id,
            "title": title,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "message_count": len(self.messages)
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
