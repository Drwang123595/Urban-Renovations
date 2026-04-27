from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy

class StepwiseStrategy(ExtractionStrategy):
    def process(self, title: str, abstract: str, session_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Stepwise strategy for Urban Renewal classification.
        Only Step 1 is used (Urban Renewal = 1 or 0).
        Spatial attributes are extracted separately using the 'spatial' strategy.
        """
        memory = self._create_isolated_memory(
            self.prompt_gen.get_step_system_prompt(),
            session_path=session_path,
        )

        results: Dict[str, Any] = {}

        prompt1 = self.prompt_gen.get_step_prompt(1, title, abstract, include_context=True)
        memory.add_user_message(prompt1)

        resp1 = self.client.chat_completion(memory.get_messages())
        if not resp1:
            return {}

        memory.add_assistant_message(resp1)
        results["是否属于城市更新研究"] = self.parse_single_output(resp1)

        return results
