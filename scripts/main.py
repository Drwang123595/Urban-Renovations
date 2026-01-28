import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data_processor import DataProcessor

def select_input_file():
    """Interactive file selection from Data/train directory."""
    train_dir = Config.TRAIN_DIR
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        return None
        
    files = list(train_dir.glob("*.xlsx"))
    
    if not files:
        print(f"No Excel files found in {train_dir}")
        return None
        
    print(f"\nFound {len(files)} Excel files in {train_dir}:")
    for i, f in enumerate(files):
        print(f"{i+1}: {f.name}")
        
    while True:
        choice = input("\nSelect input file number (or press Enter for default): ").strip()
        if not choice:
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return str(files[idx])
            print("Invalid number.")
        except ValueError:
            print("Please enter a number.")

def select_shot_mode():
    print("\nSelect shot mode:")
    print("1 = zero-shot")
    print("2 = one-shot")
    print("3 = few-shot")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "2":
        return "one"
    if choice == "3":
        return "few"
    return "zero"

def select_strategy():
    print("\nSelect strategy (support multiple selections, e.g. '1,3,4' or 'all'):")
    print("1 = single (一次性提问，每篇论文独立Session)")
    print("2 = stepwise (分三步提问，每篇论文独立Session)")
    print("3 = stepwise_long (分三步提问，跨论文保留长上下文)")
    print("4 = cot (思维链，要求先推理后输出)")
    print("5 = reflection (自我反思，输出前先自查修正)")
    
    choice_map = {
        "1": "single",
        "2": "stepwise",
        "3": "stepwise_long",
        "4": "cot",
        "5": "reflection"
    }
    
    while True:
        choice = input("Enter choice (e.g. 1,4 or all): ").strip().lower()
        if choice == "all":
            return list(choice_map.values())
            
        selections = []
        parts = choice.replace("，", ",").split(",")
        valid = True
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part in choice_map:
                selections.append(choice_map[part])
            elif part in choice_map.values():
                 selections.append(part)
            else:
                print(f"Invalid selection: {part}")
                valid = False
                break
        
        if valid and selections:
            # Remove duplicates while preserving order
            return list(dict.fromkeys(selections))
        
        if not valid or not selections:
            print("Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Urban Renovation Paper Extraction Experiment")
    parser.add_argument("--shot", choices=["zero", "one", "few"], default=None, help="Prompt shot mode")
    parser.add_argument("--strategy", help="Extraction strategy (comma separated or 'all')")
    parser.add_argument("--limit", type=int, help="Limit number of papers to process (for testing)")
    parser.add_argument("--input", help="Input Excel file path")
    parser.add_argument("--output", help="Output Excel file path")
    
    args = parser.parse_args()
    
    # Load env
    Config.load_env()
    
    if not Config.API_KEY:
        print("Error: DEEPSEEK_API_KEY not found in environment variables or .env file.")
        print("Please create a .env file in the project root with DEEPSEEK_API_KEY=your_key")
        return

    # Step 1: Select input file from Data/train
    if not args.input:
        selected_file = select_input_file()
        if selected_file:
            args.input = selected_file
            print(f"Selected input file: {args.input}")
        else:
            default_file = Config.INPUT_FILE
            if default_file and default_file.exists():
                args.input = str(default_file)
                print(f"Using default input file: {args.input}")
            else:
                print("No input file selected or default file not found.")
                return

    # Step 2: Select shot mode
    if not args.shot:
        args.shot = select_shot_mode()
    elif args.shot is None: 
         args.shot = select_shot_mode()

    # Step 3: Select strategy
    strategies = []
    if args.strategy is None:
        strategies = select_strategy()
    else:
        # Parse command line argument if provided (e.g. --strategy "single,cot")
        if args.strategy.lower() == "all":
             strategies = ["single", "stepwise", "stepwise_long", "cot", "reflection"]
        else:
             parts = args.strategy.replace("，", ",").split(",")
             strategies = [p.strip() for p in parts if p.strip()]

    print(f"\nStarting Experiment 1: Streaming Dialogue Extraction")
    print(f"Input: {args.input}")
    print(f"Mode: {args.shot}-shot, strategies={strategies}")
    
    processor = DataProcessor(shot_mode=args.shot, strategies=strategies)
    processor.run_batch(input_file=args.input, output_file=args.output, limit=args.limit)

if __name__ == "__main__":
    main()
