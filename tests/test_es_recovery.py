import importlib.util
from pathlib import Path


def test_es_recovery_importable():
    p = Path("experiments/es_recovery.py")
    spec = importlib.util.spec_from_file_location("es_recovery", p)
    assert spec is not None
