from optimizers.kv_cache import estimate_kv_bytes, recommend_kv_mode
from optimizers.prompt_compression import compress_text


def test_kv_cache_estimate():
    b = estimate_kv_bytes(1024, 8, 2048, "q4")
    assert b > 0
    assert recommend_kv_mode(4, 1024) == "kivi_int2"


def test_prompt_compression():
    t = "abcdefghijklmnopqrstuvwxyz"
    out = compress_text(t, 0.5)
    assert len(out) <= len(t)
