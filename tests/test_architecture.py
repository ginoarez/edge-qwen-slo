from recommender.architecture import select_architecture


def test_select_architecture():
    assert select_architecture(2, 4096, "quality") == "ssm"
    assert select_architecture(8, 4096, "quality") == "transformer"
