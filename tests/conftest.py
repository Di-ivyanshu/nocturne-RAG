"""Make the project root importable so `from src import ...` works under pytest."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
