from pathlib import Path

"""
Use this helper script to correctly place all models, plots etc.
Keep it clean.
"""
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_SAVE_DIR = PROJECT_ROOT / "models"