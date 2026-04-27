import re
from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy
from ..prompting.generator import PromptGenerator
from ..runtime.llm_client import DeepSeekClient
from ..runtime.memory import ConversationMemory

class StepwiseLongContextStrategy(ExtractionStrategy):
    def __init__(
        self,
        client: DeepSeekClient,
        prompt_gen: PromptGenerator,
        max_samples_per_window: int = 50,
    ):
        super().__init__(client, prompt_gen)
        self.memory: Optional[ConversationMemory] = None
        self.max_samples_per_window = max(1, max_samples_per_window)
        self.samples_in_window = 0

    def _get_or_create_memory(
        self,
        system_prompt: str,
        session_path: Optional[Union[str, Path]] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMemory:
        """
        Get existing memory (for long context) or create new one (for isolated/first run).
        """
        if session_path:
            return self._create_memory(system_prompt, session_path, audit_metadata=audit_metadata)

        if self.memory is None:
            self.memory = self._create_memory(system_prompt, audit_metadata=audit_metadata)
            self.samples_in_window = 0
        elif audit_metadata:
            self.memory.update_audit_metadata(audit_metadata)

        if self.samples_in_window >= self.max_samples_per_window or self.memory.is_context_full():
            print(
                "[INFO] Resetting context for StepwiseLongContextStrategy "
                f"(samples={self.samples_in_window}, max_window={self.max_samples_per_window})."
            )
            self.memory.save()
            self.memory = self._create_memory(system_prompt, audit_metadata=audit_metadata)
            self.samples_in_window = 0

        return self.memory

    def _parse_single_output_with_reason(self, text: str) -> tuple[str, str]:
        raw_text = text.strip()
        if not raw_text:
            return "0", "empty_text"

        explicit_patterns = [
            r'(?:最终答案|最终结论|答案|结论)\s*[:：是为]?\s*([01])\b',
            r'"?(?:是否属于城市更新研究|is_urban_renewal)"?\s*[:=]\s*"?([01])"?',
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                return match.group(1), "explicit_answer_pattern"

        clean_text = re.sub(r'(Step|Field|Phase)\s*\d+', '', raw_text, flags=re.IGNORECASE)
        line_match = re.search(r'(?m)^\s*([01])\s*$', clean_text)
        if line_match:
            return line_match.group(1), "single_digit_line"

        match = re.search(r'(?<!\d)(1|0)(?!\d)', clean_text)
        if match:
            return match.group(1), "fallback_first_digit"

        if re.search(r'\b(yes|true)\b', raw_text, re.IGNORECASE):
            return "1", "fallback_boolean_yes"

        return "0", "no_label_detected"

    def _safe_save(self, memory: ConversationMemory, scene: str):
        try:
            memory.set_last_event(scene)
            memory.set_error_code("empty_response" if "empty_response" in scene else None)
            memory.save()
        except Exception as error:
            print(f"[WARN] Failed to persist urban session in {scene}: {error}")

    def process(
        self,
        title: str,
        abstract: str,
        session_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auxiliary_context: Optional[Dict[str, Any]] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Stepwise Long Context strategy for Urban Renewal classification.
        Only Step 1 is used (Urban Renewal = 1 or 0).
        Spatial attributes are extracted separately using the 'spatial' strategy.
        """
        memory = self._get_or_create_memory(
            self.prompt_gen.get_step_system_prompt(),
            session_path=session_path,
            audit_metadata=audit_metadata,
        )
        results: Dict[str, Any] = {}

        prompt1 = self.prompt_gen.get_step_prompt(
            1,
            title,
            abstract,
            include_context=True,
            metadata=metadata,
            auxiliary_context=auxiliary_context,
        )
        memory.add_user_message(prompt1)

        resp1 = self.client.chat_completion(memory.get_messages())
        if not resp1:
            self._safe_save(memory, "urban_empty_response")
            if not session_path:
                self.samples_in_window += 1
            return {
                "是否属于城市更新研究": "0",
                "urban_parse_reason": "empty_response",
            }

        memory.add_assistant_message(resp1)
        parsed_label, parse_reason = self._parse_single_output_with_reason(resp1)
        self._safe_save(memory, "urban_sample_completed")
        results["是否属于城市更新研究"] = parsed_label
        results["urban_parse_reason"] = parse_reason

        if not session_path:
            self.samples_in_window += 1

        return results
