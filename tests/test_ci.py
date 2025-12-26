def test_imports():
    from fake_news_detector.dataset import FakeNewsDataModule, FakeNewsDataset
    from fake_news_detector.model import FakeNewsModel

    FakeNewsDataModule, FakeNewsDataset, FakeNewsModel
    assert True, "Problems with imports!"
