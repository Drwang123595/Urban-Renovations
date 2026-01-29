import pandas as pd
import time
import shutil
import re
from pathlib import Path
from typing import Dict, Any, List, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import Config
from .llm_client import DeepSeekClient
from .prompts import PromptGenerator
from .strategies import StrategyRegistry

class DataProcessor:
    def __init__(self, shot_mode: str = "zero", strategies: Union[str, List[str]] = "single"):
        self.config = Config()
        self.client = DeepSeekClient()
        self.prompt_gen = PromptGenerator(shot_mode=shot_mode)
        
        if isinstance(strategies, str):
            self.strategy_names = [strategies]
        else:
            self.strategy_names = strategies
            
        self.strategies = {}
        available_strategies = StrategyRegistry.list_strategies()
        
        for name in self.strategy_names:
            strategy_cls = StrategyRegistry.get_strategy(name)
            if not strategy_cls:
                raise ValueError(f"Unknown strategy: {name}. Available: {available_strategies}")
            # Instantiate each strategy separately
            self.strategies[name] = strategy_cls(self.client, self.prompt_gen)
        
        # Split strategies into Parallel and Serial groups
        self.serial_strategies = [name for name in self.strategy_names if name == "stepwise_long"]
        self.parallel_strategies = [name for name in self.strategy_names if name != "stepwise_long"]
        
    def run_batch(self, input_file: str = None, output_file: str = None, limit: int = None):
        """
        Run the extraction on the Excel file using Hybrid Scheduler (Serial + Parallel).
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

        # Prepare output files and result containers for each strategy
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_files = {}
        results_lists = {name: [] for name in self.strategy_names}
        
        if output_file:
            base_output = Path(output_file)
            if len(self.strategy_names) == 1:
                output_files[self.strategy_names[0]] = base_output
            else:
                for name in self.strategy_names:
                    new_name = f"{base_output.stem}_{name}{base_output.suffix}"
                    output_files[name] = base_output.parent / new_name
        else:
            for name in self.strategy_names:
                filename = f"{name}_{self.prompt_gen.shot_mode}_{timestamp}.xlsx"
                output_files[name] = output_dir / filename
            
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
            
        # Ensure output directory exists
        for f in output_files.values():
            f.parent.mkdir(parents=True, exist_ok=True)
        
        # Define known label columns to exclude from output
        exclude_cols = [
            "Label_UrbanRenewal", 
            "Label_Spatial", 
            "Label_Level", 
            "Label_Desc",
            "是否属于城市更新研究", 
            "空间研究/非空间研究",
            "空间等级",
            "具体空间描述",
            "是否属于城市更新研究(人工)", 
        ]
        
        print(f"Processing {len(df)} papers with mode='{self.prompt_gen.shot_mode}'...")
        print(f"Parallel Strategies: {self.parallel_strategies}")
        print(f"Serial Strategies: {self.serial_strategies}")
        print(f"Task Workspace: {task_dir}")
        for name, path in output_files.items():
            print(f"Output for {name}: {path}")
            
        # Parallel Execution Logic
        # Max workers for parallel strategies
        max_workers = self.config.MAX_WORKERS
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            title = str(row.get("Article Title", "") or "")
            abstract = str(row.get("Abstract", "") or "")
            
            if not title and not abstract:
                continue
                
            # Clean row: remove label columns before adding new results
            base_row = row.to_dict()
            for col in exclude_cols:
                if col in base_row:
                    del base_row[col]

            # Clean title for filename usage
            clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            clean_title = clean_title[:50]
            paper_id = f"{index+1:03d}_{clean_title}"

            # ---------------------------------------------------------
            # Hybrid Execution Block
            # ---------------------------------------------------------
            
            # 1. Submit Parallel Tasks (ThreadPool)
            futures = {}
            if self.parallel_strategies:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for name in self.parallel_strategies:
                        strategy_obj = self.strategies[name]
                        # Use ISOLATED session path for parallel execution
                        session_path = self.config.SESSIONS_DIR / task_name / paper_id / f"{name}.json"
                        
                        future = executor.submit(strategy_obj.process, title, abstract, session_path)
                        futures[future] = name
                    
                    # Wait for parallel tasks to complete and collect results
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            extracted = future.result()
                        except Exception as e:
                            extracted = {"Error": str(e)}
                        
                        res_row = base_row.copy()
                        res_row.update(extracted)
                        results_lists[name].append(res_row)

            # 2. Execute Serial Tasks (Main Thread)
            # CRITICAL: Do NOT pass session_path (or pass None) to reuse the shared memory object
            # This ensures Long Context memory is maintained across papers.
            for name in self.serial_strategies:
                strategy_obj = self.strategies[name]
                try:
                    # Pass None for session_path to trigger memory reuse in base.py
                    extracted = strategy_obj.process(title, abstract, session_path=None)
                except Exception as e:
                    extracted = {"Error": str(e)}
                
                res_row = base_row.copy()
                res_row.update(extracted)
                results_lists[name].append(res_row)
            
            # ---------------------------------------------------------
            
            # Save periodically
            if (index + 1) % 10 == 0 or (index + 1) == len(df):
                for name, res_list in results_lists.items():
                    if res_list:
                        temp_df = pd.DataFrame(res_list)
                        temp_df.to_excel(output_files[name], index=False, engine="openpyxl")
                
        print(f"Done. All results saved.")
