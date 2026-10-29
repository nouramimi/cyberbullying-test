import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = Path(os.getenv("RAW_CSV", DATA_DIR / "raw.csv"))
CLEAN_CSV = DATA_DIR / "cleaned.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MongoDB : "mongomock://" pour un test local sans serveur
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "cyberbullying_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "comments")

RANDOM_STATE = 42
TEST_SIZE = 0.2
