from typing import Dict, Any
from .base import ExtractionStrategy

class StepwiseLongContextStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str) -> Dict[str, Any]:
        # Reuse memory or create new one if full
        # Only set system prompt when creating new memory
        memory = self._get_or_create_memory(self.prompt_gen.get_step_system_prompt())
        results: Dict[str, Any] = {}

        # Step 1
        prompt1 = self.prompt_gen.get_step_prompt(1, title, abstract)
        memory.add_user_message(prompt1)
        
        resp1 = self.client.chat_completion(memory.get_messages())
        if not resp1:
            return {}
            
        memory.add_assistant_message(resp1)
        results["是否属于城市更新研究"] = self.parse_single_output(resp1)

        # Step 2
        prompt2 = self.prompt_gen.get_step_prompt(2, title, abstract)
        memory.add_user_message(prompt2)
        
        resp2 = self.client.chat_completion(memory.get_messages())
        if not resp2:
            return results
            
        memory.add_assistant_message(resp2)
        results["空间研究/非空间研究"] = self.parse_single_output(resp2)

        # Step 3
        prompt3 = self.prompt_gen.get_step_prompt(3, title, abstract)
        memory.add_user_message(prompt3)
        
        resp3 = self.client.chat_completion(memory.get_messages())
        if not resp3:
            return results
            
        memory.add_assistant_message(resp3)
        results.update(self.parse_two_field_output(resp3))
        
        return results
