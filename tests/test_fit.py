from hardware.fit import will_it_fit


def test_will_it_fit():
    r = will_it_fit("qwen3-4b", 8.0)
    assert "fits" in r
    assert r["peak_vram_gb"] > 0
