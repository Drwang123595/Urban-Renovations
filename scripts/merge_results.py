import pandas as pd
import time
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import Config

def merge_results(task_name: str, strategies: List[str] = None):
    """
    Merge output files from different strategies into a single comparison Excel file.
    """
    config = Config()
    task_dir = config.DATA_DIR / task_name
    output_dir = task_dir / "output"
    result_dir = task_dir / "Result"
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    # Find all xlsx files in output_dir
    files = list(output_dir.glob("*.xlsx"))
    if not files:
        print("No output files found to merge.")
        return
        
    print(f"Found {len(files)} result files.")
    
    # Strategy mapping: try to guess strategy name from filename
    # Filename format: {strategy}_{shot}_{timestamp}.xlsx
    # We will load them all and merge on Index (assumed aligned) or Title
    
    merged_df = None
    
    # First, load the input file or labels file to get the base columns (Title, Abstract)
    # Actually, the first output file should contain them.
    
    base_columns = ["Article Title", "Abstract"]
    
    for file_path in files:
        print(f"Loading {file_path.name}...")
        df = pd.read_excel(file_path, engine="openpyxl")
        
        # Identify strategy name from filename
        # Simple heuristic: split by '_' and take first part
        # Better: user might have custom names. Let's use filename stem.
        strategy_name = file_path.stem.split('_')[0]
        shot_mode = file_path.stem.split('_')[1] if len(file_path.stem.split('_')) > 1 else "unknown"
        prefix = f"[{strategy_name.upper()}]"
        
        # Rename result columns
        rename_map = {
            "是否属于城市更新研究": f"{prefix} Urban Renewal",
            "空间研究/非空间研究": f"{prefix} Spatial Study",
            "空间等级": f"{prefix} Spatial Level",
            "具体空间描述": f"{prefix} Description"
        }
        
        # Select columns to merge
        # We keep base columns from the FIRST file only
        cols_to_keep = []
        if merged_df is None:
            # Initialize merged_df with base columns + first strategy results
            merged_df = df.rename(columns=rename_map)
        else:
            # For subsequent files, we only want the result columns
            # We assume rows are aligned by index. 
            # To be safe, we could merge on Article Title, but titles might have duplicates or slight diffs.
            # Index merge is safest if run_batch preserved order (it does).
            
            # Filter columns that are in our rename map
            result_cols = [c for c in df.columns if c in rename_map]
            subset = df[result_cols].rename(columns=rename_map)
            
            # Merge
            merged_df = pd.concat([merged_df, subset], axis=1)

    if merged_df is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = result_dir / f"merged_comparison_{timestamp}.xlsx"
        merged_df.to_excel(output_path, index=False)
        print(f"Successfully merged results to: {output_path}")
    else:
        print("Merge failed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge strategy results")
    parser.add_argument("task_name", help="Name of the task folder (e.g., test1)")
    args = parser.parse_args()
    
    merge_results(args.task_name)
