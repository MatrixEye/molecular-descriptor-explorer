from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"

META_FILE = DATA_PATH / "metadata.csv"
FMO_FILE = DATA_PATH / "global_FMO.csv"
FDA_FILE = DATA_PATH / "global_FDA.csv"
METADATA_FP_FILE = DATA_PATH / "metadata_with_fp.pkl"
