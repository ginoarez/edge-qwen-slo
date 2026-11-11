"""Conceptual MoE expert streaming cache (inspired by kimi-k3-in-c).

This module sketches how to keep a large MoE model in <4 GB by streaming
routed experts from disk and caching only the hot ones in RAM.
"""

from collections import OrderedDict


class ExpertLRUCache:
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.cache = OrderedDict()
        self.used = 0

    def get(self, expert_id):
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
            return self.cache[expert_id]
        return None

    def put(self, expert_id, data, size_bytes):
        if size_bytes > self.capacity:
            return False
        while self.used + size_bytes > self.capacity and self.cache:
            _, (old_size, _) = self.cache.popitem(last=False)
            self.used -= old_size
        self.cache[expert_id] = (size_bytes, data)
        self.used += size_bytes
        return True

    def stats(self):
        return {"entries": len(self.cache), "used_bytes": self.used, "capacity_bytes": self.capacity}
