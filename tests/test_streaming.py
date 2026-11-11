from streaming.offload import ExpertLRUCache


def test_lru_cache_basic():
    c = ExpertLRUCache(100)
    assert c.put("e1", b"x", 40) is True
    assert c.get("e1") is not None
    assert c.stats()["entries"] == 1


def test_lru_cache_eviction():
    c = ExpertLRUCache(50)
    c.put("e1", b"x", 40)
    c.put("e2", b"y", 40)  # evicts e1
    assert c.get("e1") is None
    assert c.get("e2") is not None
