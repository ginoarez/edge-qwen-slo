from benchmark import summarize


def test_slo_pass_when_median_above_100():
    results = [
        {"toks": 110.0, "ttft_s": 0.2},
        {"toks": 105.0, "ttft_s": 0.25},
        {"toks": 99.0, "ttft_s": 0.3},
    ]
    s = summarize("m", 3, results)
    assert s["slo_pass"] is True
    assert s["toks_median"] == 105.0


def test_slo_fail_when_median_below_100():
    results = [{"toks": 80.0, "ttft_s": 0.4} for _ in range(3)]
    s = summarize("m", 3, results)
    assert s["slo_pass"] is False
