import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

data_dir_str = os.getenv("DATA_DIR")
if not data_dir_str:
    raise ValueError("❌ DATA_DIR not found in environment or .env file")

DATA_DIR = Path(data_dir_str)
NYC_API = os.getenv("NYC_API")
SOCRATA_API = os.getenv("SOCRATA_API")
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")
SOCRATA_APP_SECRET_TOKEN = os.getenv("SOCRATA_APP_SECRET_TOKEN")
