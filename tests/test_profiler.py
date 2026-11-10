from hardware.profiler import get_profile


def test_profile_has_keys():
    p = get_profile()
    assert "vram_gb" in p
    assert "cpu_cores" in p
    assert "ram_gb" in p
