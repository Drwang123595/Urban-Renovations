import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import Config

def list_tasks():
    """List available tasks in Data directory (excluding 'train')."""
    tasks = []
    if not Config.DATA_DIR.exists():
        return tasks
        
    for p in Config.DATA_DIR.iterdir():
        if p.is_dir() and p.name != "train":
            tasks.append(p)
    return tasks

def select_from_list(items, prompt="Select item:"):
    if not items:
        print("No items found.")
        return None
        
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"{i+1}: {item.name}")
        
    while True:
        choice = input("\nEnter number: ").strip()
        if not choice:
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print("Invalid number.")
        except ValueError:
            print("Please enter a number.")

def evaluate_single_file(pred_file, truth_df):
    """Evaluate a single prediction file against the ground truth DataFrame."""
    try:
        df_pred = pd.read_excel(pred_file, engine="openpyxl")
    except Exception as e:
        print(f"Error reading {pred_file.name}: {e}")
        return None, None

    # Merge on Title
    join_col = "Article Title"
    if join_col not in df_pred.columns or join_col not in truth_df.columns:
        print(f"Error: '{join_col}' missing in {pred_file.name} or truth file.")
        return None, None

    df_pred["_key"] = df_pred[join_col].astype(str).str.strip().str.lower()
    # truth_df already has _key computed
    merged = pd.merge(truth_df, df_pred, on="_key", suffixes=("_truth", "_pred"), how="inner")
    
    if len(merged) == 0:
        print(f"Warning: No matching rows for {pred_file.name}")
        return None, None

    # Clean up redundant columns
    if "_key" in merged.columns:
        del merged["_key"]
    if "Article Title_pred" in merged.columns:
        del merged["Article Title_pred"]
    if "Abstract_pred" in merged.columns:
        del merged["Abstract_pred"]
    
    # Rename truth base columns back to clean names if needed
    if "Article Title_truth" in merged.columns:
        merged.rename(columns={"Article Title_truth": "Article Title"}, inplace=True)
    if "Abstract_truth" in merged.columns:
        merged.rename(columns={"Abstract_truth": "Abstract"}, inplace=True)

    mappings = [
        ("是否属于城市更新研究", "是否属于城市更新研究", "Urban Renewal"),
        ("空间研究/非空间研究", "空间研究/非空间研究", "Spatial Study"),
        ("空间等级", "空间等级", "Spatial Level"),
        ("具体空间描述", "具体空间描述", "Spatial Desc")
    ]

    metrics = []
    
    # Prepare Detail DF - Start with base columns
    base_cols = ["Article Title", "Abstract"]
    # Ensure these exist
    final_cols = [c for c in base_cols if c in merged.columns]
    
    # Helper for normalization
    def normalize(val):
        """Normalize values to integers 1/0 if possible, else return original string."""
        s = str(val).strip()
        if s.replace('.0', '') == '1': return 1
        if s.replace('.0', '') == '0': return 0
        return s # Return original string for '待确定' or other text

    # Calculate Metrics and Add Diff Columns
    for truth_col, pred_col, desc in mappings:
        t_col = f"{truth_col}_truth" if f"{truth_col}_truth" in merged.columns else truth_col
        p_col = f"{pred_col}_pred" if f"{pred_col}_pred" in merged.columns else pred_col

        if t_col not in merged.columns or p_col not in merged.columns:
            continue

        # Normalize columns for comparison
        t_norm = merged[t_col].apply(normalize)
        p_norm = merged[p_col].apply(normalize)
        
        # Determine if it's a numeric metric (Urban Renewal, Spatial Study)
        is_numeric = desc in ["Urban Renewal", "Spatial Study"]
        
        # STRICT BINARY MATCH LOGIC:
        # Diff = 1 if (Truth == Pred) AND (Pred is 0 or 1)
        # Diff = 0 otherwise (mismatch, or Pred is '待确定'/'Unknown')
        condition = (t_norm == p_norm) & (p_norm.isin([0, 1]))
        
        # Store Diff
        diff_col_name = f"Diff_{desc}"
        merged[diff_col_name] = np.where(condition, 1, 0)
        
        # Calculate Metrics based on strict diff
        correct = merged[diff_col_name].sum()
        total = len(merged)
        accuracy = correct / total * 100.0 if total > 0 else 0
        
        # Simple Confusion Matrix (Only for numeric-like fields where 1/0 matters)
        if is_numeric:
            # We map strictly: 1->1, 0->0, others->exclude or treat as 0?
            # Standard confusion matrix needs ground truth 0/1. 
            # If Truth is '待确定', we can't really say if it's TP/TN.
            # Here we follow the Diff logic: only count clear 0/1 cases.
            
            # Helper to map for matrix: 1->1, 0->0, others->-1
            def map_bin(v):
                if v == 1: return 1
                if v == 0: return 0
                return -1

            t_bin = t_norm.apply(map_bin)
            p_bin = p_norm.apply(map_bin)
            
            tp = ((t_bin == 1) & (p_bin == 1)).sum()
            tn = ((t_bin == 0) & (p_bin == 0)).sum()
            fp = ((t_bin == 0) & (p_bin == 1)).sum()
            fn = ((t_bin == 1) & (p_bin == 0)).sum()
        else:
            tp, tn, fp, fn = 0, 0, 0, 0
        
        metrics.append({
            "File": pred_file.stem,
            "Metric": desc,
            "Accuracy": accuracy,
            "Correct": correct,
            "Total": total,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        })
        
        # Add to column order: Truth, Pred, Diff
        final_cols.extend([t_col, p_col, diff_col_name])

    # Add any remaining columns that are not in final_cols yet
    # (e.g. Label_Level, Label_Desc etc if present in truth/pred)
    remaining = [c for c in merged.columns if c not in final_cols]
    final_cols.extend(remaining)
    
    detail_df = merged[final_cols]

    return pd.DataFrame(metrics), detail_df

def evaluate():
    print("\n=== Batch Evaluation Mode ===")
    Config.load_env()
    
    # 1. Select Task
    print("\nStep 1: Select Task")
    tasks = list_tasks()
    task_dir = select_from_list(tasks, prompt="Available Tasks:")
    if not task_dir:
        return

    output_dir = task_dir / "output"
    labels_dir = task_dir / "labels"
    result_dir = task_dir / "Result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Ground Truth (Once)
    if not labels_dir.exists():
        print("Labels directory not found.")
        return
        
    truth_files = list(labels_dir.glob("*.xlsx"))
    if not truth_files:
        print("No label files found.")
        return
    
    # If multiple label files, maybe ask? For now take first.
    truth_file = truth_files[0] 
    print(f"Using Ground Truth: {truth_file.name}")
    
    try:
        truth_df = pd.read_excel(truth_file, engine="openpyxl")
        # Prepare join key once
        if "Article Title" in truth_df.columns:
            truth_df["_key"] = truth_df["Article Title"].astype(str).str.strip().str.lower()
        else:
            print("Error: 'Article Title' column missing in truth file.")
            return
    except Exception as e:
        print(f"Error reading truth file: {e}")
        return

    # 3. Iterate all output files
    pred_files = list(output_dir.glob("*.xlsx"))
    if not pred_files:
        print("No output files found to evaluate.")
        return
        
    print(f"Found {len(pred_files)} prediction files. Processing...")

    for pred_file in pred_files:
        print(f"Evaluating {pred_file.name}...")
        metrics_df, detail_df = evaluate_single_file(pred_file, truth_df)
        
        if metrics_df is not None:
            # Generate Report Filename
            # naming: Eval_{Strategy}_{Shot}_{Timestamp}.xlsx
            # We can just prefix Eval_ to the original filename
            report_name = f"Eval_{pred_file.name}"
            report_path = result_dir / report_name
            
            with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
                # Sheet 1: Detail Comparison (reordered columns)
                detail_df.to_excel(writer, sheet_name="Detail Comparison", index=False)
                
                # Sheet 2: Metrics
                metrics_df.to_excel(writer, sheet_name="Quality Metrics", index=False)
                
            print(f"  -> Saved report to {report_path.name}")
            
    print("\nBatch evaluation complete.")

if __name__ == "__main__":
    evaluate()
