from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy
from ..memory import ConversationMemory

class ReflectionStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        # Independent memory per paper
        skip_index = True if session_path else False
        
        memory = ConversationMemory(
            self.prompt_gen.get_reflection_system_prompt(),
            session_path=session_path,
            skip_index=skip_index
        )
        
        # Round 1: Initial Answer
        prompt = self.prompt_gen.get_single_prompt(title, abstract)
        memory.add_user_message(prompt)
        
        resp1 = self.client.chat_completion(memory.get_messages())
        if not resp1:
            return {}
        memory.add_assistant_message(resp1)
        
        # Round 2: Critique & Correction
        critique_prompt = self.prompt_gen.get_reflection_critique_prompt()
        memory.add_user_message(critique_prompt)
        
        resp2 = self.client.chat_completion(memory.get_messages())
        if not resp2:
            # If round 2 fails, return round 1
            return self.parse_tab_output(resp1)
            
        memory.add_assistant_message(resp2)
        
        # The final answer should be in resp2
        return self.parse_tab_output(resp2)
