import sys
from pathlib import Path

# make ``argus`` importable from src/ and the experiments package importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
