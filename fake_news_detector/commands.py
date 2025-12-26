import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Union

import fire
import kagglehub
import torch

from fake_news_detector.model import FakeNewsModel

DEFAULT_DATA_DIR_STR = "data/raw"


def train(*args, **kwargs):
    """
    Run training pipeline.
    Passes all arguments to hydra-based train script.
    Usage: python commands.py train training.max_epochs=5
    """
    cmd = [sys.executable, "fake_news_detector/train.py"]

    overrides = []
    for k, v in kwargs.items():
        overrides.append(f"{k}={v}")

    overrides.extend(args)

    subprocess.run(cmd + overrides)


def infer(*args, **kwargs):
    """
    Run inference pipeline.
    Usage: python commands.py infer model_path=... text=...
    """
    cmd = [sys.executable, "fake_news_detector/infer.py"]
    overrides = []
    for k, v in kwargs.items():
        overrides.append(f"{k}={v}")
    overrides.extend(args)

    subprocess.run(cmd + overrides)


def download_data(output_dir: str = DEFAULT_DATA_DIR_STR) -> None:
    """
    Downloads the Fake and Real News Dataset from Kaggle and moves it to the output directory.

    Args:
        output_dir (str): Directory where the data will be saved. Defaults to 'data/raw'.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data to {output_path}...")

    # Load the latest version
    # kagglehub downloads to a cache directory by default
    try:
        path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        print(f"Data downloaded to cache at: {path}")

        # Move files from cache to our data directory
        src_path = Path(path)
        for file_name in ["Fake.csv", "True.csv"]:
            src_file = src_path / file_name
            dst_file = output_path / file_name
            if src_file.exists():
                shutil.copy(src_file, dst_file)
                print(f"Copied {file_name} to {dst_file}")
            else:
                print(f"Warning: {file_name} not found in {src_path}")

        print("Data download complete.")

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def ensure_data_exists(required_files: List[Union[str, Path]]) -> bool:
    """
    Checks if required data files exist.

    Args:
        required_files (List[Union[str, Path]]): List of file paths to check.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    missing_files = []
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        print(f"Missing required data files: {missing_files}")
        return False
    return True


def export_onnx(model_path: str, output_path: str = "model.onnx"):
    """
    Export PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the trained .ckpt model checkpoint.
        output_path (str): Path where the ONNX model will be saved. Defaults to "model.onnx".
    """
    print(f"Loading model from {model_path}...")
    try:
        model = FakeNewsModel.load_from_checkpoint(model_path)
        model.eval()

        # Create dummy input for tracing
        dummy_input = torch.randint(
            0, 1000, (1, 128)
        )  # Batch size 1, sequence length 128
        dummy_mask = torch.ones((1, 128))

        print(f"Exporting to ONNX at {output_path}...")
        model.to_onnx(
            output_path,
            (dummy_input, dummy_mask),
            export_params=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
        )
        print("Export successful!")

    except Exception as e:
        print(f"Failed to export model: {e}")
        raise


def export_tensorrt(onnx_path: str, output_path: str = "model.plan"):
    """
    Export ONNX model to TensorRT engine using trtexec.
    Note: This requires NVIDIA GPU and TensorRT installed on the system.

    Args:
        onnx_path (str): Path to the ONNX model file.
        output_path (str): Path where the TensorRT engine will be saved.
    """
    print(f"Exporting ONNX model from {onnx_path} to TensorRT...")

    # Check if trtexec is available
    if not shutil.which("trtexec"):
        print("Error: 'trtexec' not found in PATH. Please install TensorRT.")
        print("Skipping TensorRT export (requires NVIDIA GPU environment).")
        return

    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        "--fp16",
    ]

    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"TensorRT engine saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT export failed: {e}")


if __name__ == "__main__":
    fire.Fire()
