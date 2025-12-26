import pandas as pd
import pytest
from transformers import AutoTokenizer

from fake_news_detector.dataset import FakeNewsDataModule, FakeNewsDataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


@pytest.fixture
def sample_dataframe():
    data = {
        "title": ["Fake News 1", "Real News 1", "Fake News 2"],
        "text": ["Some fake text", "Some real text", "More fake text"],
        "label": [1, 0, 1],
    }
    return pd.DataFrame(data)


def test_dataset(tokenizer, sample_dataframe):
    dataset = FakeNewsDataset(sample_dataframe, tokenizer, max_length=128)
    assert len(dataset) == 3

    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["labels"] == 1
    assert sample["input_ids"].shape[0] == 128
    assert sample["attention_mask"].shape[0] == 128


@pytest.fixture
def tmp_data_dir(tmp_path):
    # Create fake csv files
    d = tmp_path / "raw"
    d.mkdir()

    fake_df = pd.DataFrame({"title": ["Fake1"] * 50, "text": ["Text1"] * 50})
    true_df = pd.DataFrame({"title": ["True1"] * 50, "text": ["Text1"] * 50})

    fake_df.to_csv(d / "Fake.csv", index=False)
    true_df.to_csv(d / "True.csv", index=False)

    return d


def test_datamodule(tmp_data_dir):
    dm = FakeNewsDataModule(
        data_dir=str(tmp_data_dir),
        fake_news_filename="Fake.csv",
        true_news_filename="True.csv",
        batch_size=10,
        max_length=32,
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Check that loaders are created
    assert val_loader is not None
    assert test_loader is not None

    # Check if we get data
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (10, 32)
    assert batch["labels"].shape == (10,)

    # Check split sizes approximately (70/15/15)
    # Total 100 samples
    # Train ~ 70
    assert len(dm.train_dataset) == 70
    assert len(dm.val_dataset) == 15
    # Test gets the rest, 15
    assert len(dm.test_dataset) == 15
