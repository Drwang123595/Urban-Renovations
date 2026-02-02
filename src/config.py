import os
from pathlib import Path
from typing import Optional

class Config:
    # Project Root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Data Paths
    DATA_DIR = PROJECT_ROOT / "Data"
    TRAIN_DIR = DATA_DIR / "train"  # New directory for original task files
    
    INPUT_FILE = TRAIN_DIR / "test1.xlsx" # Updated default
    
    # History Paths
    HISTORY_DIR = PROJECT_ROOT / "history"
    SESSIONS_DIR = HISTORY_DIR / "sessions"
    INDEX_FILE = HISTORY_DIR / "index.json"
    
    # API Settings (Default to DeepSeek if not set)
    # Priority: LLM_ > DEEPSEEK_
    API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    MODEL_NAME = os.environ.get("LLM_MODEL_NAME") or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    
    # Experiment Settings
    DEFAULT_SHOT_MODE = "zero"  # zero, one, few
    TEMPERATURE = 1.0
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 500))
    TIMEOUT = int(os.environ.get("TIMEOUT", 60))
    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 1)) # Default 1 for safety
    
    # Context Limits
    MAX_CONTEXT_TOKENS = 128000
    TOKEN_WARNING_THRESHOLD = 0.9  # Warn when 90% full

    @classmethod
    def load_env(cls, env_path: Optional[Path] = None):
        """Load environment variables from a .env file"""
        if env_path is None:
            env_path = cls.PROJECT_ROOT / ".env"
        
        if env_path.exists():
            print(f"Loading environment from {env_path}")
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    value = value.split("#", 1)[0].strip()
                    os.environ[key.strip()] = value
            
            # Update class attributes after loading env
            # Priority: LLM_ > DEEPSEEK_ > Default
            cls.API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", cls.API_KEY)
            cls.BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", cls.BASE_URL)
            cls.MODEL_NAME = os.environ.get("LLM_MODEL_NAME") or os.environ.get("DEEPSEEK_MODEL", cls.MODEL_NAME)
            
            # Update system settings
            cls.MAX_WORKERS = int(os.environ.get("MAX_WORKERS", cls.MAX_WORKERS))
            cls.MAX_TOKENS = int(os.environ.get("MAX_TOKENS", cls.MAX_TOKENS))
            cls.TIMEOUT = int(os.environ.get("TIMEOUT", cls.TIMEOUT))
            
        # Ensure directories exist
        cls.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
