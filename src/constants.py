from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"
MODELS_DIR = ROOT_DIR / "models"
RANDOM_STATE = 42
TARGET_COLUMN = "target"
