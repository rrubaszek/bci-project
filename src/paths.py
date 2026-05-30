from pathlib import Path

"""
Use this helper script to correctly place all models, plots etc.
Keep it clean.
"""
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
EMOTIV_RAW = DEFAULT_DATA_DIR / "emotiv" / "raw"
EMOTIV_CLEANED = DEFAULT_DATA_DIR / "emotiv" / "cleaned"

DEFAULT_BCI_DIR = DEFAULT_DATA_DIR / "bci_comp"

DEFAULT_OUT_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_SAVE_DIR = PROJECT_ROOT / "models"