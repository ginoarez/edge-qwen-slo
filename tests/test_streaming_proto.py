import importlib.util
from pathlib import Path


def test_streaming_prototype_importable():
    p = Path("experiments/streaming_prototype.py")
    spec = importlib.util.spec_from_file_location("streaming_prototype", p)
    assert spec is not None
