from recommender.recommend import recommend


def test_recommend_4gb_quality():
    r = recommend(4.0, "quality")
    assert r is not None
    assert r["model"] in {"qwen3-4b", "lfm2.5-8b-a1b", "qwen3-1.7b", "deepseek-v4-pro-qwen3.5-4b"}


def test_recommend_2gb_speed():
    r = recommend(2.0, "speed")
    assert r is not None
    assert r["min_vram_gb"] <= 2.0
