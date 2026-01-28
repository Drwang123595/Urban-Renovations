import pandas as pd
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from .config import Config
from .llm_client import DeepSeekClient
from .prompts import PromptGenerator
from .strategies import StrategyRegistry

class DataProcessor:
    def __init__(self, shot_mode: str = "zero", strategy: str = "single"):
        self.config = Config()
        self.client = DeepSeekClient()
        self.prompt_gen = PromptGenerator(shot_mode=shot_mode)
        self.strategy_name = strategy
        
        strategy_cls = StrategyRegistry.get_strategy(strategy)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {StrategyRegistry.list_strategies()}")
            
        self.strategy = strategy_cls(self.client, self.prompt_gen)
        
    def run_batch(self, input_file: str = None, output_file: str = None, limit: int = None):
        """
        Run the extraction on the Excel file.
        """
        input_path = Path(input_file) if input_file else self.config.INPUT_FILE
        task_name = input_path.stem  # e.g., "test1" from "test1.xlsx"

        # Define task directory structure: Data/{task_name}/
        task_dir = self.config.DATA_DIR / task_name
        output_dir = task_dir / "output"
        labels_dir = task_dir / "labels"
        result_dir = task_dir / "Result"
        
        # Ensure directories exist
        for d in [output_dir, labels_dir, result_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # --- Auto-Archive Labels ---
        # Copy input file to labels directory if not present
        label_file_path = labels_dir / input_path.name
        if not label_file_path.exists():
            print(f"Archiving input file to labels: {label_file_path}")
            shutil.copy2(input_path, label_file_path)
        else:
            print(f"Labels file already exists at: {label_file_path}")

        if not output_file:
            # Structure: Data/{task_name}/output/{strategy}_{shot}_{timestamp}.xlsx
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.strategy_name}_{self.prompt_gen.shot_mode}_{timestamp}.xlsx"
            output_file = output_dir / filename
        else:
            output_file = Path(output_file)
            
        print(f"Reading from {input_path}")
        df = pd.read_excel(input_path, engine="openpyxl")
        if "Article Title" not in df.columns or "Abstract" not in df.columns:
            df = pd.read_excel(input_path, engine="openpyxl", header=None)
            df = df.dropna(axis=1, how="all")
            col_names = []
            if df.shape[1] >= 1:
                col_names.append("Article Title")
            if df.shape[1] >= 2:
                col_names.append("Abstract")
            label_cols = ["是否属于城市更新研究", "空间研究/非空间研究", "空间等级", "具体空间描述"]
            remaining = df.shape[1] - len(col_names)
            col_names.extend(label_cols[:max(0, min(remaining, len(label_cols)))])
            if remaining > len(label_cols):
                col_names.extend([f"extra_{i+1}" for i in range(remaining - len(label_cols))])
            df.columns = col_names
        
        if limit:
            df = df.head(limit)
            
        # Ensure output directory exists (in case user provided custom path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results_list = []
        
        # Define known label columns to exclude from output
        # Add any other columns you want to hide from the result file
        exclude_cols = [
            "Label_UrbanRenewal", 
            "Label_Spatial", 
            "Label_Level", 
            "Label_Desc",
            "是否属于城市更新研究",  # Possible existing label
            "空间研究/非空间研究",
            "空间等级",
            "具体空间描述",
            "是否属于城市更新研究(人工)", # Common manual label name
        ]
        
        print(f"Processing {len(df)} papers with mode='{self.prompt_gen.shot_mode}'...")
        print(f"Task Workspace: {task_dir}")
        print(f"Output will be saved to: {output_file}")
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            title = str(row.get("Article Title", "") or "")
            abstract = str(row.get("Abstract", "") or "")
            
            if not title and not abstract:
                continue
                
            extracted = self.strategy.process(title, abstract)
            
            # Update row data
            # Clean row: remove label columns before adding new results
            res_row = row.to_dict()
            for col in exclude_cols:
                if col in res_row:
                    del res_row[col]
            
            res_row.update(extracted)
            results_list.append(res_row)
            
            # Save periodically (every 10 or last)
            if (index + 1) % 10 == 0 or (index + 1) == len(df):
                temp_df = pd.DataFrame(results_list)
                temp_df.to_excel(output_file, index=False, engine="openpyxl")
                
        print(f"Done. Saved to {output_file}")
