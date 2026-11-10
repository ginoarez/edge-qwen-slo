from benchmark_matrix import summarize_runs


def test_summarize_runs_slo_pass():
    runs = [{"toks": 120.0, "ttft_s": 0.2}, {"toks": 110.0, "ttft_s": 0.25}, {"toks": 105.0, "ttft_s": 0.3}]
    s = summarize_runs(runs)
    assert s["slo_pass"] is True
    assert s["toks_median"] == 110.0


def test_summarize_runs_slo_fail():
    runs = [{"toks": 80.0, "ttft_s": 0.4} for _ in range(3)]
    s = summarize_runs(runs)
    assert s["slo_pass"] is False
