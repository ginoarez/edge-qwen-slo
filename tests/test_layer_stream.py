from streaming.layer_stream import LayerStreamer


def test_layer_stream_plan():
    ls = LayerStreamer(num_layers=32, layer_size_bytes=100 * 1024 * 1024, vram_budget_bytes=500 * 1024 * 1024)
    p = ls.plan()
    assert p["resident_layers"] >= 1
    assert p["peak_vram_bytes"] <= ls.vram_budget
