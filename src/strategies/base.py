from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
import re
from ..prompting.generator import PromptGenerator
from ..runtime.llm_client import DeepSeekClient
from ..runtime.memory import ConversationMemory

class ExtractionStrategy(ABC):
    def __init__(self, client: DeepSeekClient, prompt_gen: PromptGenerator):
        self.client = client
        self.prompt_gen = prompt_gen
        # Removed self.memory to ensure statelessness in concurrent execution

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

    def _create_memory(
        self,
        system_prompt: str,
        session_path: Optional[Union[str, Path]] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMemory:
        """
        Create a new isolated memory instance for a session.
        If session_path is provided, we skip global index update to prevent concurrency issues.
        """
        skip_index = True if session_path else False
        memory_audit = {"strategy_name": self.__class__.__name__}
        if audit_metadata:
            memory_audit.update(audit_metadata)
        memory = ConversationMemory(
            system_prompt=system_prompt, 
            session_path=session_path,
            skip_index=skip_index,
            audit_metadata=memory_audit,
        )
        if not session_path:
            # Only print for new auto-generated sessions
            print(f"\n[INFO] Created new session: {memory.session_id}")
            
        return memory

    def parse_tab_output(self, text: str) -> Dict[str, Any]:
        """Helper to parse TAB-separated output (4 fields). Enforces strict 1/0."""
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
            
        # Enforce 1/0 on first two fields
        field1 = self.parse_single_output(parts[0])
        field2 = self.parse_single_output(parts[1])
        
        return {
            "是否属于城市更新研究": field1,
            "空间研究/非空间研究": field2,
            "空间等级": parts[2],
            "具体空间描述": parts[3],
        }

    def parse_single_output(self, text: str) -> str:
        """
        Helper to parse single line output.
        Strictly returns '1' or '0'. Defaults to '0' if unclear.
        """
        raw_text = text.strip()
        if not raw_text:
            return "0"
            
        # 1. Pre-process to remove common labels that might contain digits
        # e.g. "Step 1: 0", "Field 1: 0" -> ": 0"
        clean_text = re.sub(r'(Step|Field|Phase)\s*\d+', '', raw_text, flags=re.IGNORECASE)
        
        # 2. Check for explicit 1 or 0
        match = re.search(r'(?<!\d)(1|0)(?!\d)', clean_text)
        if match:
            return match.group(1)
            
        # 2. If no clear 1/0 found, check for "Yes"/"No" keywords as fallback
        if re.search(r'\b(yes|true)\b', raw_text, re.IGNORECASE):
            return "1"
        
        # Default to 0 for any other case (including "Unsure", "待确定", etc.)
        return "0"

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
