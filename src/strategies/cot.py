from typing import Dict, Any, Optional, Union
import re
from pathlib import Path
from .base import ExtractionStrategy
from ..memory import ConversationMemory

class CoTStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        # CoT can use a fresh memory per paper (like Single) or long context.
        # Here we stick to independent processing for cleaner experimental isolation.
        
        skip_index = True if session_path else False
        
        memory = ConversationMemory(
            self.prompt_gen.get_cot_system_prompt(),
            session_path=session_path,
            skip_index=skip_index
        )
        
        prompt = self.prompt_gen.get_single_prompt(title, abstract)
        memory.add_user_message(prompt)
        
        resp = self.client.chat_completion(memory.get_messages())
        if not resp:
            return {}
            
        memory.add_assistant_message(resp)
        
        # Parse output: extract the last line or the line after </thinking>
        # The prompt asks for <thinking>...</thinking> then the result line.
        
        # Try to find the result line
        # Strategy: Look for the last non-empty line that looks like fields
        lines = resp.strip().splitlines()
        final_line = ""
        
        # Iterate backwards to find the data line
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith("<thinking>") or line.startswith("</thinking>"):
                continue
            # Basic heuristic: contains tabs or looks like our format
            if "\t" in line:
                final_line = line
                break
                
        if not final_line:
            # Fallback: just try parsing the whole text with base parser
            return self.parse_tab_output(resp)
            
        return self.parse_tab_output(final_line)
