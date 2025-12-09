import sys
from pathlib import Path

# Ensure project root is on sys.path for 'src' imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
