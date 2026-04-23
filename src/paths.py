from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_SAVE_DIR = PROJECT_ROOT / "models"