from hardware.memory_planner import plan_memory, recommend_mode


def test_plan_memory():
    p = plan_memory(8.0)
    assert p["trunk_gb"] + p["cache_gb"] <= 8.0


def test_recommend_mode():
    assert recommend_mode(8) == "laptop"
    assert recommend_mode(128) == "server"
