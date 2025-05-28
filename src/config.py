# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# === Load .env file if present ===
load_dotenv()

# === Base directory of the project ===
BASE_DIR = Path(os.getenv("PROJECT_BASE_DIR", Path.cwd()))

# === Standard Folder Paths ===
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR    = BASE_DIR / "data"
SRC_DIR     = BASE_DIR / "src"
OUTPUT_DIR  = FIGURES_DIR  # Alias if needed

# === Ensure directories exist ===
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)
