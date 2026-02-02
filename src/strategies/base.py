from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
import re
from ..llm_client import DeepSeekClient
from ..prompts import PromptGenerator
from ..memory import ConversationMemory

class ExtractionStrategy(ABC):
    def __init__(self, client: DeepSeekClient, prompt_gen: PromptGenerator):
        self.client = client
        self.prompt_gen = prompt_gen
        self.memory: Optional[ConversationMemory] = None

    @abstractmethod
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Process a single paper and return extracted data.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            session_path: Optional path to save conversation history (for semantic naming)
        """
        pass

    def _get_or_create_memory(self, system_prompt: str, session_path: Optional[Union[str, Path]] = None) -> ConversationMemory:
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
            # If session_path is provided, we skip global index update to prevent concurrency issues
            skip_index = True if session_path else False
            self.memory = ConversationMemory(
                system_prompt=system_prompt, 
                session_path=session_path,
                skip_index=skip_index
            )
            if not session_path:
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
        """
        Helper to parse single line output.
        Enhanced to extract specific tokens (1, 0, 待确定) if present,
        cleaning up potential noise like 'Step 1: 1' or '1 (Yes)'.
        """
        raw_text = text.strip()
        if not raw_text:
            return ""
            
        # Prioritize matching exact expected tokens
        # 1. Check for '待确定'
        if "待确定" in raw_text:
            return "待确定"
            
        # 2. Check for single digit 1 or 0 isolated or at start/end
        # Regex looks for 1 or 0 that is NOT surrounded by other digits (e.g. 10, 2024)
        # It allows surrounding text like "Output: 1" or "1."
        match = re.search(r'(?<!\d)(1|0)(?!\d)', raw_text)
        if match:
            return match.group(1)
            
        # Fallback: return the first non-empty line as is
        for raw in text.splitlines():
            if raw.strip():
                return raw.strip()
        return ""

    def parse_two_field_output(self, text: str) -> Dict[str, Any]:
        """Helper to parse TAB-separated output (2 fields)."""
        # For multi-field, we still take the first valid line
        line = ""
        for raw in text.splitlines():
            if raw.strip():
                line = raw.strip()
                break
                
        if not line:
            return {}
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 2:
            return {}
        return {"空间等级": parts[0], "具体空间描述": parts[1]}
