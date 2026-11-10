import importlib.util
from pathlib import Path


def test_ternary_baseline_importable():
    p = Path("experiments/ternary_baseline.py")
    spec = importlib.util.spec_from_file_location("ternary_baseline", p)
    assert spec is not None
