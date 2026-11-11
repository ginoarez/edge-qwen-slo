import json
import tempfile
from pathlib import Path

from recommender.auto_tune import pick_best


def test_pick_best_prefers_slo_pass():
    data = [
        {"model": "a", "toks_median": 90, "slo_pass": False},
        {"model": "b", "toks_median": 110, "slo_pass": True},
    ]
    best = pick_best(data)
    assert best["model"] == "b"
