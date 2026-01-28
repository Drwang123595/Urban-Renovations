import yaml
from pathlib import Path
from typing import List, Dict

class PromptGenerator:
    def __init__(self, shot_mode: str = "zero"):
        """
        Initialize PromptGenerator.
        
        Args:
            shot_mode (str): 'zero', 'one', or 'few'.
        """
        self.shot_mode = shot_mode
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """Load all YAML templates from src/templates directory."""
        templates = {}
        template_dir = Path(__file__).parent / "templates"
        
        # Load each yaml file
        for yaml_file in template_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if content:
                        templates.update(content)
            except Exception as e:
                print(f"Error loading template {yaml_file}: {e}")
                
        return templates

    def get_system_prompt(self) -> str:
        return self.get_single_system_prompt()

    def get_single_system_prompt(self) -> str:
        base_prompt = self.templates.get("single_system", "")
        
        if self.shot_mode == "zero":
            return base_prompt
            
        elif self.shot_mode == "one":
            # Assuming 'examples' key exists in single.yaml structure or loaded flat
            # Based on my yaml design, it might be nested under 'examples' key in the file
            # But yaml.safe_load returns a dict. Let's assume flat keys for simplicity or specific structure.
            # In single.yaml, I put 'examples' as a top level key.
            examples = self.templates.get("examples", {})
            return base_prompt + examples.get("one_shot", "")
            
        elif self.shot_mode == "few":
            examples = self.templates.get("examples", {})
            return base_prompt + examples.get("few_shot", "")
            
        return base_prompt

    def get_step_system_prompt(self) -> str:
        base_prompt = self.templates.get("step_system", "")
        # For stepwise, examples are also in the yaml (I put them in stepwise.yaml)
        # Note: yaml.safe_load merges all keys. If 'examples' key is used in multiple files, 
        # the last one loaded overwrites. This is a potential issue.
        # FIX: I should structure the templates dict better or namespace them in yaml.
        # For now, let's assume I need to load stepwise examples specifically.
        # In stepwise.yaml, I used the same 'examples' key.
        
        # To fix this conflict without changing yaml structure too much:
        # I will re-read stepwise.yaml specifically or rely on unique keys.
        # Better approach: In the yaml files, prefix keys like 'step_examples'.
        
        # However, since I already wrote the YAMLs with generic 'examples' key,
        # let's modify the YAMLs to be distinct or handle it here.
        # Actually, let's re-read the specific file to be safe, or just use the current dict 
        # and hope the merge order was lucky? No, that's bad.
        
        # Let's update the YAML files to have unique example keys first.
        # But wait, I can just load them into namespaced dicts in _load_templates.
        return base_prompt + self._get_step_examples()

    def _get_step_examples(self) -> str:
        # Helper to get stepwise examples safely
        # Since 'examples' key might be overwritten, let's look for 'step_examples' if I rename it,
        # OR just reload stepwise.yaml here for safety (a bit inefficient but safe).
        # OR better: I will modify the YAML files in the next step to have unique keys.
        # For now, assuming I will rename them to 'step_examples' in stepwise.yaml
        examples = self.templates.get("step_examples", {})
        if not examples: 
             # Fallback if I haven't renamed yet
             examples = self.templates.get("examples", {})
             
        if self.shot_mode == "one":
            return examples.get("one_shot", "")
        elif self.shot_mode == "few":
            return examples.get("few_shot", "")
        return ""

    def get_cot_system_prompt(self) -> str:
        """Chain of Thought System Prompt."""
        base = self.get_single_system_prompt()
        instruction = self.templates.get("cot_instruction", "")
        
        # Replace output instruction block
        # Using string replacement as before
        prompt = base.replace("OUTPUT REQUIREMENTS (Very important):", "OUTPUT REQUIREMENTS (OVERRIDDEN BELOW):")
        prompt = prompt.replace("- Output EXACTLY one line with 4 fields in order", "")
        prompt = prompt.replace("- Fields must be separated by TABs", "")
        prompt = prompt.replace("- Do NOT output any explanations", "")
        
        return prompt + "\n" + instruction

    def get_reflection_system_prompt(self) -> str:
        """System prompt for Reflection Strategy (Round 1)."""
        return self.get_single_system_prompt()
        
    def get_reflection_critique_prompt(self) -> str:
        """Prompt for the reflection step (Round 2)."""
        return self.templates.get("reflection_critique", "")

    def get_round_prompt(self, round_num: int, title: str, abstract: str) -> str:
        return self.get_single_prompt(title, abstract)

    def get_single_prompt(self, title: str, abstract: str) -> str:
        return f"[TITLE] {title}\n[ABSTRACT] {abstract}"

    def get_step_prompt(self, step_num: int, title: str, abstract: str) -> str:
        base = f"[TITLE] {title}\n[ABSTRACT] {abstract}\n"
        if step_num == 1:
            return base + "Step 1: Urban renewal study? Output only 1 or 0."
        if step_num == 2:
            return base + "Step 2: Spatial study? Output only 1 or 0."
        return base + "Step 3: Output spatial level and spatial description, TAB-separated."
