from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy
from ..memory import ConversationMemory

class SingleTurnStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        # Note: For single turn strategy, we might want to create a new session per paper 
        # or reuse one. Here we create a new one per process call for isolation.
        
        # Determine if we should skip index updates
        skip_index = True if session_path else False
        
        memory = ConversationMemory(
            self.prompt_gen.get_single_system_prompt(),
            session_path=session_path,
            skip_index=skip_index
        )
        
        prompt = self.prompt_gen.get_single_prompt(title, abstract)
        memory.add_user_message(prompt)
        
        resp = self.client.chat_completion(memory.get_messages())
        if not resp:
            return {}
            
        memory.add_assistant_message(resp)  # Save response to history
        return self.parse_tab_output(resp)
