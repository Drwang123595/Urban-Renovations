from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..llm_client import DeepSeekClient
from ..prompts import PromptGenerator
from ..memory import ConversationMemory

class ExtractionStrategy(ABC):
    def __init__(self, client: DeepSeekClient, prompt_gen: PromptGenerator):
        self.client = client
        self.prompt_gen = prompt_gen
        self.memory: Optional[ConversationMemory] = None

    @abstractmethod
    def process(self, title: str, abstract: str) -> Dict[str, Any]:
        """
        Process a single paper and return extracted data.
        """
        pass

    def _get_or_create_memory(self, system_prompt: str) -> ConversationMemory:
        """
        Get existing memory or create new one if full/missing.
        Handles session rotation for long contexts.
        """
        if self.memory and self.memory.is_context_full():
            # Memory full, force save (already done by add_message) and reset
            print(f"\n[INFO] Context full (Session {self.memory.session_id}). Starting new session.")
            self.memory = None

        if self.memory is None:
            # Create new memory
            self.memory = ConversationMemory(system_prompt=system_prompt)
            print(f"\n[INFO] Created new session: {self.memory.session_id}")
            
        return self.memory

    def parse_tab_output(self, text: str) -> Dict[str, Any]:
        """Helper to parse TAB-separated output (4 fields)."""
        line = ""
        for raw in text.splitlines():
            if raw.strip():
                line = raw.strip()
                break
        if not line:
            return {}
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 4:
            return {}
        return {
            "是否属于城市更新研究": parts[0],
            "空间研究/非空间研究": parts[1],
            "空间等级": parts[2],
            "具体空间描述": parts[3],
        }

    def parse_single_output(self, text: str) -> str:
        """Helper to parse single line output."""
        for raw in text.splitlines():
            if raw.strip():
                return raw.strip()
        return ""

    def parse_two_field_output(self, text: str) -> Dict[str, Any]:
        """Helper to parse TAB-separated output (2 fields)."""
        line = self.parse_single_output(text)
        if not line:
            return {}
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 2:
            return {}
        return {"空间等级": parts[0], "具体空间描述": parts[1]}
