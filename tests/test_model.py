import pytest
import torch

from fake_news_detector.model import FakeNewsModel


def test_model_prediction():
    try:
        model = FakeNewsModel(model_name="distilbert-base-uncased")
    except OSError:
        pytest.skip("Model not found and no internet connection")

    batch_size = 2
    seq_len = 16

    # Vocabulary size of distilbert-base-uncased is 30522
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    # Forward pass
    logits = model(input_ids, attention_mask)

    # Output should be (batch_size, 1) as per code
    assert logits.shape == (
        batch_size,
        1,
    ), f"Output shape mismatch. Expected {(batch_size, 1)}, got {logits.shape}"
