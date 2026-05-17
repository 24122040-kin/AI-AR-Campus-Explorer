"""
main.py — Quick entry point
  python main.py serve
  python main.py index data/images/
  python main.py demo "Đi từ nhà đến chợ lúc 8 giờ"
  python main.py status
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.cli import app

if __name__ == "__main__":
    app()
