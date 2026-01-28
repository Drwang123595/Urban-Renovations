# Urban Renovation Paper Extraction Experiment Plan

## Project Structure
I will organize the code into a modular structure to facilitate iteration and prompt version switching, as requested.

```
Urban Renovation/
├── Data/
│   └── test1.xlsx             # Input data (contains Title, Abstract and target columns)
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration (API Key, Model, Paths)
│   ├── prompts.py             # Prompt management (Zero/One/Few shot logic)
│   ├── llm_client.py          # DeepSeek API interaction
│   └── data_processor.py      # Excel reading/writing and orchestration
├── main.py                    # Entry point to run the experiment
└── requirements.txt           # Dependencies
```

## Implementation Modules

### 1. Prompt Design (`src/prompts.py`)
This is the core component you requested. I will design a `PromptGenerator` class that supports switching between `zero-shot`, `one-shot`, and `few-shot`.
- **Zero-shot**: Only task description and input.
- **One-shot**: Adds one positive example.
- **Few-shot**: Adds multiple examples (positive and negative).
- **Structure**: The prompt will be split into a System Prompt (rules) and User Prompt (current paper).

### 2. LLM Client (`src/llm_client.py`)
- Encapsulate DeepSeek API calls.
- Support multi-turn dialogue (maintaining `messages` history).
- Handle JSON parsing for structured output.

### 3. Data Processor (`src/data_processor.py`)
- Read `Data/test1.xlsx`.
- Iterate through rows, extracting "Article Title" and "Abstract".
- For each paper, execute the 3-round dialogue with the LLM:
    - **Round 1**: Urban Renewal identification (1/0).
    - **Round 2**: Spatial/Non-spatial check (1/0).
    - **Round 3**: Spatial Level and Description extraction.
- Parse responses and fill the 4 target columns:
    - `是否属于城市更新研究`
    - `空间研究/非空间研究`
    - `空间等级`
    - `具体空间描述`
- Save results back to Excel.

### 4. Main Execution (`main.py`)
- Load config.
- Initialize `PromptGenerator` with the desired mode (e.g., `shot='one'`).
- Run the batch processing.

## Experiment Workflow
1.  **Setup**: Install necessary libraries (`pandas`, `openpyxl`, `requests`).
2.  **Code Implementation**: Create the file structure and implement the modules.
3.  **Execution**: Run `main.py` (defaulting to a safe test run or processing the whole file).
4.  **Verification**: Check if `Data/test1.xlsx` (or a copy) is correctly filled.

## Next Steps
Upon your confirmation, I will start creating the `src` directory and implementing the code.
