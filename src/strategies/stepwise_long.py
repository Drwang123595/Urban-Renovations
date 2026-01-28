from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy

class StepwiseLongContextStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        # Reuse memory or create new one if full
        # Only set system prompt when creating new memory
        # Note: LongContext strategy typically reuses memory across calls. 
        # If session_path is provided (for a specific paper), it might conflict with the idea of "long context across papers".
        # However, if the user wants "long context" but also specific session files per paper, it's contradictory.
        # Assuming for "batch processing" with parallel execution, we treat it as isolated per paper to avoid race conditions,
        # OR we accept that session_path might point to a shared file (bad for parallelism).
        
        # For this refactor, we prioritize the "per paper isolation" implied by the parallel architecture.
        # If the user insists on long context across papers in parallel mode, they need a different architecture (single worker).
        # Here we treat it as isolated per paper if session_path is given.
        
        memory = self._get_or_create_memory(
            self.prompt_gen.get_step_system_prompt(),
            session_path=session_path
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
