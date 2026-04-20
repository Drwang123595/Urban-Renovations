import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

from .project_paths import DEFAULT_STABLE_DATASET_ID, dataset_paths, data_root


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

class Schema:
    # Excel Column Names (Input)
    TITLE = "Article Title"
    ABSTRACT = "Abstract"
    AUTHOR_KEYWORDS = "Author Keywords"
    KEYWORDS_PLUS = "Keywords Plus"
    KEYWORDS = "Keywords"
    WOS_CATEGORIES = "WoS Categories"
    RESEARCH_AREAS = "Research Areas"
    
    # Extraction Field Names (Internal & Output)
    IS_URBAN_RENEWAL = "是否属于城市更新研究"
    IS_SPATIAL = "空间研究/非空间研究"
    SPATIAL_LEVEL = "空间等级"
    SPATIAL_DESC = "具体空间描述"
    
    # All fields for consistent ordering
    FIELDS = [IS_URBAN_RENEWAL, IS_SPATIAL, SPATIAL_LEVEL, SPATIAL_DESC]

class Config:
    # Project Root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Data Paths
    DATA_DIR = data_root(PROJECT_ROOT)
    TRAIN_DIR = DATA_DIR / "train"
    EXPERIMENT_TRACKS = ("stable_release", "research_matrix", "legacy_archive")
    STABLE_RELEASE_DATASET_ID = DEFAULT_STABLE_DATASET_ID
    LEGACY_BASELINE_DATASET_ID = "test1-test7_merged"
    STABLE_RELEASE_TASK_DIR = DATA_DIR / STABLE_RELEASE_DATASET_ID
    LEGACY_BASELINE_TASK_DIR = DATA_DIR / LEGACY_BASELINE_DATASET_ID
    STABLE_RELEASE_INPUT_DIR = dataset_paths(STABLE_RELEASE_DATASET_ID, PROJECT_ROOT).input_dir
    STABLE_RELEASE_LABELS_DIR = dataset_paths(STABLE_RELEASE_DATASET_ID, PROJECT_ROOT).labels_dir
    STABLE_RELEASE_LEGACY_LABELS_DIR = dataset_paths(STABLE_RELEASE_DATASET_ID, PROJECT_ROOT).legacy_labels_dir
    STABLE_RELEASE_RUNS_DIR = dataset_paths(STABLE_RELEASE_DATASET_ID, PROJECT_ROOT).runs_dir
    STABLE_RELEASE_LABEL_FILE = (
        dataset_paths(STABLE_RELEASE_DATASET_ID, PROJECT_ROOT).label_file
    )
    STABLE_RELEASE_RESULT_DIR = STABLE_RELEASE_RUNS_DIR / "stable_release" / "20260417_unknown_recovery" / "reports"
    STABLE_RELEASE_OUTPUT_DIR = STABLE_RELEASE_RUNS_DIR / "stable_release" / "20260417_unknown_recovery" / "predictions"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = OUTPUT_DIR / "models"
    BERTOPIC_ARTIFACT_DIR = MODELS_DIR / "urban_bertopic_online_py313"
    URBAN_FAMILY_GATE_MODEL_PATH = MODELS_DIR / "urban_family_gate.joblib"
    URBAN_FAMILY_GATE_BOUNDARY_PACKAGE_PATH = OUTPUT_DIR / "doc" / "urban_family_gate_boundary_package.xlsx"
    PY313_VENV_PYTHON = PROJECT_ROOT / ".venv-bertopic313" / "Scripts" / "python.exe"
    
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
    PERSIST_FULL_SESSIONS = _env_flag("PERSIST_FULL_SESSIONS", False)
    AUDIT_FIELD_MAX_CHARS = int(os.environ.get("AUDIT_FIELD_MAX_CHARS", 240))
    SESSION_MESSAGE_MAX_CHARS = int(os.environ.get("SESSION_MESSAGE_MAX_CHARS", 1200))
    DEBUG_SENSITIVE_LOGGING = _env_flag("DEBUG_SENSITIVE_LOGGING", False)
    BERTOPIC_INTEGRITY_KEY = os.environ.get("BERTOPIC_INTEGRITY_KEY", "")
    BERTOPIC_PRIMARY_ENABLED = _env_flag("BERTOPIC_PRIMARY_ENABLED", True)
    BERTOPIC_PRIMARY_MIN_SUPPORT = int(os.environ.get("BERTOPIC_PRIMARY_MIN_SUPPORT", 35))
    BERTOPIC_PRIMARY_MIN_PURITY = float(os.environ.get("BERTOPIC_PRIMARY_MIN_PURITY", 0.80))
    BERTOPIC_PRIMARY_MIN_PROB = float(os.environ.get("BERTOPIC_PRIMARY_MIN_PROB", 0.50))
    BERTOPIC_PRIMARY_MIN_MAPPED_SHARE = float(
        os.environ.get("BERTOPIC_PRIMARY_MIN_MAPPED_SHARE", 0.70)
    )
    BERTOPIC_EMBEDDING_MODEL = os.environ.get("BERTOPIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    URBAN_HYBRID_LLM_ASSIST_ENABLED = _env_flag("URBAN_HYBRID_LLM_ASSIST_ENABLED", True)
    URBAN_HYBRID_ONLINE_LLM_HINTS_ENABLED = _env_flag("URBAN_HYBRID_ONLINE_LLM_HINTS_ENABLED", True)
    URBAN_FAMILY_GATE_ENABLED = _env_flag("URBAN_FAMILY_GATE_ENABLED", True)
    URBAN_FAMILY_GATE_THRESHOLD_URBAN = float(os.environ.get("URBAN_FAMILY_GATE_THRESHOLD_URBAN", 0.72))
    URBAN_FAMILY_GATE_THRESHOLD_NONURBAN = float(os.environ.get("URBAN_FAMILY_GATE_THRESHOLD_NONURBAN", 0.28))
    RECOMMENDED_PYTHON = "3.13"
    
    # Context Limits
    MAX_CONTEXT_TOKENS = 128000
    TOKEN_WARNING_THRESHOLD = 0.9  # Warn when 90% full

    @classmethod
    def load_env(cls, env_path: Optional[Path] = None):
        """Load environment variables from a .env file using python-dotenv"""
        if env_path is None:
            env_path = cls.PROJECT_ROOT / ".env"
            if not env_path.exists():
                env_path = cls.PROJECT_ROOT / "scripts" / ".env"
        
        if env_path.exists():
            print(f"Loading environment from {env_path}")
            load_dotenv(env_path)
            
            # Update class attributes after loading env
            # Priority: LLM_ > DEEPSEEK_ > Default
            cls.API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", cls.API_KEY)
            cls.BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", cls.BASE_URL)
            cls.MODEL_NAME = os.environ.get("LLM_MODEL_NAME") or os.environ.get("DEEPSEEK_MODEL", cls.MODEL_NAME)
            
            # Update system settings
            cls.MAX_WORKERS = int(os.environ.get("MAX_WORKERS", cls.MAX_WORKERS))
            cls.MAX_TOKENS = int(os.environ.get("MAX_TOKENS", cls.MAX_TOKENS))
            cls.TIMEOUT = int(os.environ.get("TIMEOUT", cls.TIMEOUT))
            cls.PERSIST_FULL_SESSIONS = _env_flag("PERSIST_FULL_SESSIONS", cls.PERSIST_FULL_SESSIONS)
            cls.AUDIT_FIELD_MAX_CHARS = int(os.environ.get("AUDIT_FIELD_MAX_CHARS", cls.AUDIT_FIELD_MAX_CHARS))
            cls.SESSION_MESSAGE_MAX_CHARS = int(
                os.environ.get("SESSION_MESSAGE_MAX_CHARS", cls.SESSION_MESSAGE_MAX_CHARS)
            )
            cls.DEBUG_SENSITIVE_LOGGING = _env_flag(
                "DEBUG_SENSITIVE_LOGGING",
                cls.DEBUG_SENSITIVE_LOGGING,
            )
            cls.BERTOPIC_INTEGRITY_KEY = os.environ.get(
                "BERTOPIC_INTEGRITY_KEY",
                cls.BERTOPIC_INTEGRITY_KEY,
            )
            cls.BERTOPIC_PRIMARY_ENABLED = _env_flag(
                "BERTOPIC_PRIMARY_ENABLED",
                cls.BERTOPIC_PRIMARY_ENABLED,
            )
            cls.BERTOPIC_PRIMARY_MIN_SUPPORT = int(
                os.environ.get("BERTOPIC_PRIMARY_MIN_SUPPORT", cls.BERTOPIC_PRIMARY_MIN_SUPPORT)
            )
            cls.BERTOPIC_PRIMARY_MIN_PURITY = float(
                os.environ.get("BERTOPIC_PRIMARY_MIN_PURITY", cls.BERTOPIC_PRIMARY_MIN_PURITY)
            )
            cls.BERTOPIC_PRIMARY_MIN_PROB = float(
                os.environ.get("BERTOPIC_PRIMARY_MIN_PROB", cls.BERTOPIC_PRIMARY_MIN_PROB)
            )
            cls.BERTOPIC_PRIMARY_MIN_MAPPED_SHARE = float(
                os.environ.get(
                    "BERTOPIC_PRIMARY_MIN_MAPPED_SHARE",
                    cls.BERTOPIC_PRIMARY_MIN_MAPPED_SHARE,
                )
            )
            cls.BERTOPIC_EMBEDDING_MODEL = os.environ.get(
                "BERTOPIC_EMBEDDING_MODEL",
                cls.BERTOPIC_EMBEDDING_MODEL,
            )
            cls.URBAN_HYBRID_LLM_ASSIST_ENABLED = _env_flag(
                "URBAN_HYBRID_LLM_ASSIST_ENABLED",
                cls.URBAN_HYBRID_LLM_ASSIST_ENABLED,
            )
            cls.URBAN_HYBRID_ONLINE_LLM_HINTS_ENABLED = _env_flag(
                "URBAN_HYBRID_ONLINE_LLM_HINTS_ENABLED",
                cls.URBAN_HYBRID_ONLINE_LLM_HINTS_ENABLED,
            )
            cls.URBAN_FAMILY_GATE_ENABLED = _env_flag(
                "URBAN_FAMILY_GATE_ENABLED",
                cls.URBAN_FAMILY_GATE_ENABLED,
            )
            cls.URBAN_FAMILY_GATE_THRESHOLD_URBAN = float(
                os.environ.get(
                    "URBAN_FAMILY_GATE_THRESHOLD_URBAN",
                    cls.URBAN_FAMILY_GATE_THRESHOLD_URBAN,
                )
            )
            cls.URBAN_FAMILY_GATE_THRESHOLD_NONURBAN = float(
                os.environ.get(
                    "URBAN_FAMILY_GATE_THRESHOLD_NONURBAN",
                    cls.URBAN_FAMILY_GATE_THRESHOLD_NONURBAN,
                )
            )
            
        # Ensure directories exist
        cls.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_runtime_environment(
        cls,
        *,
        require_py313: bool = False,
        warn_on_minor_drift: bool = False,
        required_modules: Optional[Iterable[str]] = None,
    ):
        expected_runtime = cls.RECOMMENDED_PYTHON
        current_runtime = f"{sys.version_info[0]}.{sys.version_info[1]}"
        recommended_entry = f"Use scripts/main_py313.py or {cls.PY313_VENV_PYTHON}."

        if sys.version_info[:2] >= (3, 14):
            raise RuntimeError(
                f"Python {current_runtime} is not supported by the project runtime. {recommended_entry}"
            )

        if require_py313 and sys.version_info[:2] != (3, 13):
            message = (
                f"Python {current_runtime} does not match the recommended runtime {expected_runtime}. "
                f"{recommended_entry}"
            )
            if warn_on_minor_drift and sys.version_info[:2] < (3, 14):
                print(f"[WARN] {message}")
            else:
                raise RuntimeError(message)

        modules = tuple(required_modules or ("openai", "pandas", "openpyxl"))
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(
                f"Missing required dependencies for runtime {current_runtime}: {', '.join(sorted(missing))}. "
                f"{recommended_entry}"
            )

        if "pandas" in modules:
            try:
                import pandas as pd  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to import pandas in runtime {current_runtime}. {recommended_entry}"
                ) from exc
            if not hasattr(pd, "DataFrame"):
                raise RuntimeError(
                    "Detected a broken pandas installation (imported module has no pandas.DataFrame). "
                    "Reinstall pandas in the active environment. "
                    f"{recommended_entry}"
                )
