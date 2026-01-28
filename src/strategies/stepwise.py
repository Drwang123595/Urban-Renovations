from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy
from ..memory import ConversationMemory

class StepwiseStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        # Initialize memory with system prompt
        skip_index = True if session_path else False
        
        memory = ConversationMemory(
            self.prompt_gen.get_step_system_prompt(),
            session_path=session_path,
            skip_index=skip_index
        )
        
        results: Dict[str, Any] = {}

        # Step 1: Include context (Title + Abstract)
        prompt1 = self.prompt_gen.get_step_prompt(1, title, abstract, include_context=True)
        memory.add_user_message(prompt1)
        
        resp1 = self.client.chat_completion(memory.get_messages())
        if not resp1:
            return {}
            
        memory.add_assistant_message(resp1)
        results["是否属于城市更新研究"] = self.parse_single_output(resp1)

        # Step 2: Simplified prompt (Context already in memory)
        prompt2 = self.prompt_gen.get_step_prompt(2, title, abstract, include_context=False)
        memory.add_user_message(prompt2)
        
        resp2 = self.client.chat_completion(memory.get_messages())
        if not resp2:
            return results
            
        memory.add_assistant_message(resp2)
        results["空间研究/非空间研究"] = self.parse_single_output(resp2)

        # Step 3: Simplified prompt
        prompt3 = self.prompt_gen.get_step_prompt(3, title, abstract, include_context=False)
        memory.add_user_message(prompt3)
        
        resp3 = self.client.chat_completion(memory.get_messages())
        if not resp3:
            return results
            
        memory.add_assistant_message(resp3)
        results.update(self.parse_two_field_output(resp3))
        
        return results
