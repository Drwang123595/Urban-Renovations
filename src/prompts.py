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
        """Load all YAML templates from src/templates directory into separate namespaces."""
        templates = {}
        template_dir = Path(__file__).parent / "templates"
        
        # Load each yaml file
        for yaml_file in template_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if content:
                        # Use filename (without extension) as key for namespace isolation
                        templates[yaml_file.stem] = content
            except Exception as e:
                print(f"Error loading template {yaml_file}: {e}")
                
        return templates

    def get_system_prompt(self) -> str:
        return self.get_single_system_prompt()

    def get_single_system_prompt(self) -> str:
        # Load from 'single' namespace
        single_tmpl = self.templates.get("single", {})
        base_prompt = single_tmpl.get("single_system", "")
        
        if self.shot_mode == "zero":
            return base_prompt
            
        elif self.shot_mode == "one":
            examples = single_tmpl.get("examples", {})
            return base_prompt + examples.get("one_shot", "")
            
        elif self.shot_mode == "few":
            examples = single_tmpl.get("examples", {})
            return base_prompt + examples.get("few_shot", "")
            
        return base_prompt

    def get_step_system_prompt(self) -> str:
        # Load from 'stepwise' namespace
        step_tmpl = self.templates.get("stepwise", {})
        base_prompt = step_tmpl.get("step_system", "")
        
        # Get examples from the same namespace
        examples = step_tmpl.get("examples", {})
        
        if self.shot_mode == "one":
            return base_prompt + examples.get("one_shot", "")
        elif self.shot_mode == "few":
            return base_prompt + examples.get("few_shot", "")
            
        return base_prompt

    def get_cot_system_prompt(self) -> str:
        """Chain of Thought System Prompt."""
        # CoT reuses single's system prompt but overrides output format
        # It also has its own 'cot_instruction' in cot.yaml
        
        base = self.get_single_system_prompt()
        cot_tmpl = self.templates.get("cot", {})
        instruction = cot_tmpl.get("cot_instruction", "")
        
        # Replace output instruction block
        # Using string replacement as before
        prompt = base.replace("OUTPUT REQUIREMENTS (Very important):", "OUTPUT REQUIREMENTS (OVERRIDDEN BELOW):")
        # Handle the new "OUTPUT REQUIREMENTS" string if updated in YAML
        prompt = prompt.replace("OUTPUT REQUIREMENTS:", "OUTPUT REQUIREMENTS (OVERRIDDEN BELOW):") 
        
        prompt = prompt.replace("- Output EXACTLY one line with 4 fields in order", "")
        prompt = prompt.replace("- Output EXACTLY one line with 4 fields (TAB-separated).", "") # Updated YAML string
        prompt = prompt.replace("- Fields must be separated by TABs", "")
        prompt = prompt.replace("- Do NOT output any explanations", "")
        prompt = prompt.replace("- No headers, no explanations.", "") # Updated YAML string
        
        return prompt + "\n" + instruction

    def get_reflection_system_prompt(self) -> str:
        """System prompt for Reflection Strategy (Round 1)."""
        return self.get_single_system_prompt()
        
    def get_reflection_critique_prompt(self) -> str:
        """Prompt for the reflection step (Round 2)."""
        refl_tmpl = self.templates.get("reflection", {})
        return refl_tmpl.get("reflection_critique", "")

    def get_round_prompt(self, round_num: int, title: str, abstract: str) -> str:
        return self.get_single_prompt(title, abstract)

    def get_single_prompt(self, title: str, abstract: str) -> str:
        return f"[TITLE] {title}\n[ABSTRACT] {abstract}"

    def get_step_prompt(self, step_num: int, title: str, abstract: str, include_context: bool = True) -> str:
        """
        Generate prompt for a specific step.
        
        Args:
            step_num: 1, 2, or 3
            title: Paper title
            abstract: Paper abstract
            include_context: Whether to include Title and Abstract in the prompt.
                             For Step 1, should usually be True.
                             For Step 2/3, if in same session, can be False to save tokens.
        """
        if include_context:
            base = f"[TITLE] {title}\n[ABSTRACT] {abstract}\n"
        else:
            base = ""
            
        if step_num == 1:
            return base + "Step 1: Urban renewal study? Output only 1 or 0."
        if step_num == 2:
            return base + "Step 2: Spatial study? Output only 1 or 0."
        return base + "Step 3: Output spatial level and spatial description, TAB-separated."
