import subprocess
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from fake_news_detector.model import FakeNewsModel


def pull_dvc_model(model_path: str):
    """Try to pull model file via DVC if it doesn't exist but .dvc file does."""
    path = Path(model_path)
    dvc_file = path.parent / (path.name + ".dvc")
    if not path.exists() and dvc_file.exists():
        print(
            f"Model file {model_path} not found, but {dvc_file} exists. Attempting DVC pull..."
        )
        try:
            subprocess.run(["dvc", "pull", str(dvc_file)], check=True)
            print("DVC pull successful.")
        except Exception as e:
            print(f"Warning: DVC pull failed: {e}")


@hydra.main(version_base=None, config_path="../conf", config_name="cfg")
def infer(cfg: DictConfig):
    """
    Run inference on a single text or file using a trained model checkpoint.

    The script expects 'model_path' and 'text' (or 'input_file') in the config
    or passed as arguments.

    Example:
        python fake_news_detector/infer.py model_path=models/best_model.ckpt text="Some fake news text"
    """

    # Check if model path is provided
    if not hasattr(cfg, "model_path") or not cfg.model_path:
        print("Error: 'model_path' is required.")
        return

    # Check if input text is provided
    if not hasattr(cfg, "text") and not hasattr(cfg, "input_file"):
        print("Error: Provide 'text' or 'input_file' argument.")
        return

    print(f"Loading model from {cfg.model_path}...")
    # Try DVC pull for model
    pull_dvc_model(cfg.model_path)

    try:
        model = FakeNewsModel.load_from_checkpoint(cfg.model_path)
    except FileNotFoundError:
        print(f"Error: Model file {cfg.model_path} not found.")
        return

    model.eval()
    model.freeze()

    tokenizer = AutoTokenizer.from_pretrained(cfg.module.name)

    # Prepare input text
    text = cfg.text if hasattr(cfg, "text") else Path(cfg.input_file).read_text()

    print("Tokenizing...")
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=cfg.data.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    print("Running inference...")
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prob = torch.sigmoid(logits).squeeze().item()

    prediction = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob

    print("-" * 30)
    print(f"Text: {text[:100]}...")
    print(f"Prediction: {prediction}")
    print(f"Probability (Fake): {prob:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    infer()
