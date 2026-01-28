import sys
import shutil
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data_processor import DataProcessor

def create_mock_input():
    """Create a temporary input file with a special title."""
    data = {
        "Article Title": [
            'Urban Renewal: A "New" Era? Special Char Test!', 
            'Simple Title'
        ],
        "Abstract": [
            "This is a test abstract with special chars.",
            "This is a simple abstract."
        ]
    }
    df = pd.DataFrame(data)
    
    input_dir = Config.DATA_DIR / "verify_test"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = input_dir / "test_input.xlsx"
    df.to_excel(input_file, index=False)
    return input_file

def verify_results(task_name):
    """Check if output files and session logs exist."""
    task_dir = Config.DATA_DIR / task_name
    output_dir = task_dir / "output"
    sessions_dir = Config.SESSIONS_DIR / task_name
    
    print(f"\nVerifying results for task: {task_name}")
    
    # 1. Check Output Files
    print("\n[1] Checking Output Excel Files:")
    strategies = ["single", "cot"]
    all_outputs_exist = True
    for strat in strategies:
        files = list(output_dir.glob(f"*{strat}*.xlsx"))
        if files:
            print(f"  OK: Found output for {strat}: {[f.name for f in files]}")
        else:
            print(f"  FAIL: No output found for {strat}")
            all_outputs_exist = False
            
    # 2. Check Session Directories
    print("\n[2] Checking Session Directories:")
    if not sessions_dir.exists():
        print(f"  FAIL: Sessions directory missing: {sessions_dir}")
        return False
        
    paper_dirs = list(sessions_dir.glob("*"))
    print(f"  Found {len(paper_dirs)} paper session folders.")
    
    all_sessions_exist = True
    for p_dir in paper_dirs:
        print(f"  Checking folder: {p_dir.name}")
        # Verify sanitization
        if "?" in p_dir.name or ":" in p_dir.name or '"' in p_dir.name:
            print(f"    FAIL: Folder name contains forbidden characters!")
        else:
            print(f"    OK: Folder name sanitized.")
            
        # Check json files
        for strat in strategies:
            json_file = p_dir / f"{strat}.json"
            if json_file.exists():
                print(f"    OK: Found session log: {json_file.name}")
            else:
                print(f"    FAIL: Missing session log: {json_file.name}")
                all_sessions_exist = False

    return all_outputs_exist and all_sessions_exist

def main():
    print("Starting Verification Script...")
    
    # Setup
    Config.load_env()
    input_file = create_mock_input()
    print(f"Created mock input: {input_file}")
    
    # Run Processor
    # We use limit=1 to test just the first paper (the one with special chars)
    # We use 2 strategies: single and cot
    processor = DataProcessor(shot_mode="zero", strategies=["single", "cot"])
    
    print("\nRunning Batch Processing...")
    try:
        processor.run_batch(input_file=str(input_file), limit=1)
    except Exception as e:
        print(f"\nExecution Failed: {e}")
        return

    # Verify
    task_name = input_file.stem
    success = verify_results(task_name)
    
    if success:
        print("\n✅ VERIFICATION PASSED!")
    else:
        print("\n❌ VERIFICATION FAILED!")
        
    # Cleanup (Optional - comment out to inspect manually)
    # if input_file.parent.exists():
    #     shutil.rmtree(input_file.parent)

if __name__ == "__main__":
    main()
