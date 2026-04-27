from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy

class SingleTurnStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        # Note: For single turn strategy, we might want to create a new session per paper 
        # or reuse one. Here we create a new one per process call for isolation.

        memory = self._create_isolated_memory(
            self.prompt_gen.get_single_system_prompt(),
            session_path=session_path,
        )
        
        prompt = self.prompt_gen.get_single_prompt(title, abstract)
        memory.add_user_message(prompt)
        
        resp = self.client.chat_completion(memory.get_messages())
        if not resp:
            return {}

        memory.add_assistant_message(resp)
        urban_val = self.parse_single_output(resp)
        return {"是否属于城市更新研究": urban_val}
