import pandas as pd
import time
import shutil
import re
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .merged_output import build_review_ready_merged_frame, load_task_input_frame
from ..prompting.generator import PromptGenerator
from ..runtime.config import Config, Schema
from ..runtime.llm_client import DeepSeekClient
from ..strategies import ExtractionStrategy, StrategyRegistry

class DataProcessor:
    def __init__(self, 
                 client: DeepSeekClient = None, 
                 prompt_gen: PromptGenerator = None,
                 shot_mode: str = "zero", 
                 strategies: Union[str, List[str]] = "single"):
        """
        Initialize DataProcessor with dependency injection.
        """
        self.config = Config()
        self.client = client or DeepSeekClient()
        self.prompt_gen = prompt_gen or PromptGenerator(shot_mode=shot_mode)
        
        if isinstance(strategies, str):
            self.strategy_names = [strategies]
        else:
            self.strategy_names = strategies
            
        self.strategies: Dict[str, ExtractionStrategy] = {}
        available_strategies = StrategyRegistry.list_strategies()
        
        for name in self.strategy_names:
            strategy_cls = StrategyRegistry.get_strategy(name)
            if not strategy_cls:
                raise ValueError(f"闈炴硶绛栫暐: {name}. 鍙€夌瓥鐣? {available_strategies}")
            self.strategies[name] = strategy_cls(self.client, self.prompt_gen)
        self._validate_prompt_routes()
        
        # Split strategies into Parallel and Serial groups
        self.serial_strategies = [name for name in self.strategy_names if name == "stepwise_long"]
        self.parallel_strategies = [name for name in self.strategy_names if name != "stepwise_long"]

    def _validate_prompt_routes(self):
        for name in self.strategy_names:
            if name == "cot":
                self.prompt_gen.get_cot_system_prompt()
            elif name == "spatial":
                self.prompt_gen.get_spatial_system_prompt()
            elif name == "stepwise" or name == "stepwise_long":
                self.prompt_gen.get_step_system_prompt()
            elif name == "reflection":
                self.prompt_gen.get_reflection_system_prompt()
            else:
                self.prompt_gen.get_single_system_prompt()

    def _raise_fatal_strategy_error(self, strategy_name: str, error: Exception):
        raise RuntimeError(
            f"绛栫暐鎵ц澶辫触骞跺凡缁堟: strategy={strategy_name}, error={type(error).__name__}: {error}"
        ) from error

    def _prepare_legacy_run_layout(self, input_path: Path, timestamp: str):
        task_name = input_path.stem
        task_dir = self.config.DATA_DIR / task_name
        run_dir = task_dir / "runs" / "research_matrix" / timestamp
        output_dir = run_dir / "predictions"
        labels_dir = task_dir / "input" / "labels"
        result_dir = run_dir / "reports"

        for directory in [output_dir, labels_dir, result_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        label_file_path = labels_dir / input_path.name
        if not label_file_path.exists():
            print(f"Archiving input file to labels: {label_file_path}")
            shutil.copy2(input_path, label_file_path)
        else:
            print(f"Labels file already exists at: {label_file_path}")

        return task_name, task_dir, output_dir

    def _build_legacy_output_files(self, output_file: str, output_dir: Path, timestamp: str) -> Dict[str, Path]:
        if output_file:
            base_output = Path(output_file)
            if len(self.strategy_names) == 1:
                return {self.strategy_names[0]: base_output}
            return {
                name: base_output.parent / f"{base_output.stem}_{name}{base_output.suffix}"
                for name in self.strategy_names
            }
        return {
            name: output_dir / f"{name}_{self.prompt_gen.shot_mode}_{timestamp}.xlsx"
            for name in self.strategy_names
        }

    def _legacy_header_names(self, width: int) -> List[str]:
        col_names = []
        if width >= 1:
            col_names.append("Article Title")
        if width >= 2:
            col_names.append("Abstract")
        label_cols = [
            Schema.IS_URBAN_RENEWAL,
            Schema.IS_SPATIAL,
            Schema.SPATIAL_LEVEL,
            Schema.SPATIAL_DESC,
        ]
        remaining = width - len(col_names)
        col_names.extend(label_cols[:max(0, min(remaining, len(label_cols)))])
        if remaining > len(label_cols):
            col_names.extend([f"extra_{i+1}" for i in range(remaining - len(label_cols))])
        return col_names

    def _load_legacy_input_frame(self, input_path: Path, limit: int = None) -> pd.DataFrame:
        df = pd.read_excel(input_path, engine="openpyxl")
        if "Article Title" not in df.columns or "Abstract" not in df.columns:
            df = pd.read_excel(input_path, engine="openpyxl", header=None)
            df = df.dropna(axis=1, how="all")
            df.columns = self._legacy_header_names(df.shape[1])
        return df.head(limit) if limit else df

    def _base_result_row(self, row, exclude_cols: List[str]) -> Dict:
        base_row = row.to_dict()
        for col in exclude_cols:
            if col in base_row:
                del base_row[col]
        return base_row

    def _paper_id(self, index: int, title: str) -> str:
        clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        return f"{index+1:03d}_{clean_title[:50]}"

    def _append_strategy_result(self, results_lists: Dict[str, List], name: str, base_row: Dict, extracted: Dict):
        res_row = base_row.copy()
        res_row.update(extracted)
        results_lists[name].append(res_row)

    def _run_parallel_strategies(
        self,
        executor: ThreadPoolExecutor,
        task_name: str,
        paper_id: str,
        title: str,
        abstract: str,
        base_row: Dict,
        results_lists: Dict[str, List],
    ):
        futures = {}
        for name in self.parallel_strategies:
            strategy_obj = self.strategies[name]
            session_path = self.config.SESSIONS_DIR / task_name / paper_id / f"{name}.json"
            future = executor.submit(strategy_obj.process, title, abstract, session_path)
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                extracted = future.result()
            except Exception as e:
                if isinstance(e, (FileNotFoundError, ValueError)):
                    self._raise_fatal_strategy_error(name, e)
                extracted = {"Error": str(e)}
            self._append_strategy_result(results_lists, name, base_row, extracted)

    def _run_serial_strategies(
        self,
        title: str,
        abstract: str,
        base_row: Dict,
        results_lists: Dict[str, List],
    ):
        for name in self.serial_strategies:
            strategy_obj = self.strategies[name]
            try:
                extracted = strategy_obj.process(title, abstract, session_path=None)
            except Exception as e:
                if isinstance(e, (FileNotFoundError, ValueError)):
                    self._raise_fatal_strategy_error(name, e)
                extracted = {"Error": str(e)}
            self._append_strategy_result(results_lists, name, base_row, extracted)

    def _save_legacy_results(self, output_files: Dict[str, Path], results_lists: Dict[str, List]):
        for name, res_list in results_lists.items():
            if res_list:
                temp_df = pd.DataFrame(res_list)
                temp_df.to_excel(output_files[name], index=False, engine="openpyxl")

    def run_batch(self, input_file: str = None, output_file: str = None, limit: int = None):
        """
        Run the extraction on the Excel file using Hybrid Scheduler (Serial + Parallel).
        """
        input_path = Path(input_file) if input_file else self.config.require_default_train_input_file()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        task_name, task_dir, output_dir = self._prepare_legacy_run_layout(input_path, timestamp)

        # Prepare output files and result containers for each strategy
        output_files = self._build_legacy_output_files(output_file, output_dir, timestamp)
        results_lists = {name: [] for name in self.strategy_names}
            
        print(f"Reading from {input_path}")
        df = self._load_legacy_input_frame(input_path, limit)
            
        # Ensure output directory exists
        for f in output_files.values():
            f.parent.mkdir(parents=True, exist_ok=True)
        
        # Define known label columns to exclude from output
        exclude_cols = [
            "Label_UrbanRenewal", 
            "Label_Spatial", 
            "Label_Level", 
            "Label_Desc",
            Schema.IS_URBAN_RENEWAL,
            Schema.IS_SPATIAL,
            Schema.SPATIAL_LEVEL,
            Schema.SPATIAL_DESC,
            f"{Schema.IS_URBAN_RENEWAL}(浜哄伐)", 
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
        
        # Instantiate executor OUTSIDE the loop to reuse threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row in tqdm(df.iterrows(), total=len(df)):
                title = str(row.get("Article Title", "") or "")
                abstract = str(row.get("Abstract", "") or "")
                
                if not title and not abstract:
                    continue
                    
                # Clean row: remove label columns before adding new results
                base_row = self._base_result_row(row, exclude_cols)
                paper_id = self._paper_id(index, title)

                # ---------------------------------------------------------
                # Hybrid Execution Block
                # ---------------------------------------------------------
                
                # 1. Submit Parallel Tasks (ThreadPool)
                if self.parallel_strategies:
                    self._run_parallel_strategies(
                        executor,
                        task_name,
                        paper_id,
                        title,
                        abstract,
                        base_row,
                        results_lists,
                    )

                # 2. Execute Serial Tasks (Main Thread)
                # CRITICAL: Do NOT pass session_path (or pass None) to reuse the shared memory object
                # This ensures Long Context memory is maintained across papers.
                self._run_serial_strategies(title, abstract, base_row, results_lists)
                
                # ---------------------------------------------------------
                
                # Save periodically
                if (index + 1) % 10 == 0 or (index + 1) == len(df):
                    self._save_legacy_results(output_files, results_lists)
                    
        print(f"Done. All results saved.")

        # Auto-merge for combined workflow (stepwise_long + spatial)
        self._auto_merge_results(output_files, results_lists, timestamp)

    def _auto_merge_results(self, output_files: Dict[str, Path], results_lists: Dict[str, List], timestamp: str):
        """
        Automatically merge results when both stepwise_long and spatial strategies are run together.
        Merge logic: Use stepwise_long's urban renewal judgment + spatial's spatial attributes.
        """
        has_stepwise = "stepwise_long" in self.strategy_names
        has_spatial = "spatial" in self.strategy_names

        if not (has_stepwise and has_spatial):
            return

        print("\n[INFO] Auto-merging stepwise_long + spatial results...")

        stepwise_file = output_files.get("stepwise_long")
        spatial_file = output_files.get("spatial")

        if not stepwise_file or not spatial_file:
            return

        if not stepwise_file.exists() or not spatial_file.exists():
            print("[WARN] Could not find output files for merge.")
            return

        try:
            df_stepwise = pd.read_excel(stepwise_file, engine="openpyxl")
            df_spatial = pd.read_excel(spatial_file, engine="openpyxl")

            if "Article Title" not in df_stepwise.columns or "Article Title" not in df_spatial.columns:
                print("[WARN] Missing 'Article Title' column for merge.")
                return

            key_col = "Article Title"
            df_stepwise["_key"] = df_stepwise[key_col].astype(str).str.strip().str.lower()
            df_spatial["_key"] = df_spatial[key_col].astype(str).str.strip().str.lower()

            spatial_columns = [
                "_key",
                Schema.IS_SPATIAL,
                Schema.SPATIAL_LEVEL,
                Schema.SPATIAL_DESC,
                "Reasoning",
                "Confidence",
            ]
            for column in [
                Schema.SPATIAL_VALIDATION_STATUS,
                Schema.SPATIAL_VALIDATION_REASON,
                Schema.SPATIAL_AREA_EVIDENCE,
            ]:
                if column in df_spatial.columns:
                    spatial_columns.append(column)

            merged = pd.merge(
                df_stepwise,
                df_spatial[spatial_columns],
                on="_key",
                suffixes=("", "_spatial"),
                how="left"
            )
            input_df = None
            for parent in stepwise_file.parents:
                input_df = load_task_input_frame(parent)
                if input_df is not None:
                    break
            merged = build_review_ready_merged_frame(merged, input_df=input_df)

            merge_output = stepwise_file.parent / f"merged_{timestamp}.xlsx"
            merged.to_excel(merge_output, index=False, engine="openpyxl")
            print(f"[INFO] Merged results saved to: {merge_output}")

        except Exception as e:
            print(f"[ERROR] Failed to merge results: {e}")
