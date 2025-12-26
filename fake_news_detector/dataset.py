from pathlib import Path
from typing import Dict, List, Optional

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class FakeNewsDataset(Dataset):
    """
    A dataset for Fake News Classification task.
    Reads data from DataFrame, tokenizes text using HuggingFace tokenizer
    and returns input tensors for the model.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'title', 'text' and 'label' columns.
        tokenizer: HuggingFace tokenizer instance.
        max_length (int): Maximum sequence length for tokenization. Defaults to 512.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing:
                - input_ids (torch.Tensor): Token indices.
                - attention_mask (torch.Tensor): Attention mask.
                - labels (torch.Tensor): Class label (0 or 1).
        """
        row = self.data.iloc[idx]
        text = str(row["title"]) + " " + str(row["text"])
        label = row["label"]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class FakeNewsDataModule(L.LightningDataModule):
    """
    LightningDataModule for Fake News Detection.
    Handles data loading, splitting, and DataLoader creation.

    Args:
        data_dir (str): Path to the directory containing raw data files.
        fake_news_filename (str): Filename for fake news CSV.
        true_news_filename (str): Filename for true news CSV.
        model_name (str): Name of the pre-trained model (for tokenizer).
        batch_size (int): Batch size for DataLoaders.
        max_length (int): Maximum sequence length for tokenization.
        num_workers (int): Number of workers for DataLoaders.
        train_val_test_split (List[float]): Ratios for train/validation/test split.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        fake_news_filename: str = "Fake.csv",
        true_news_filename: str = "True.csv",
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        train_val_test_split: Optional[List[float]] = None,
        pin_memory: bool = True,
        subset_fraction: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        if train_val_test_split is None:
            train_val_test_split = [0.7, 0.15, 0.15]
        self.train_val_test_split = train_val_test_split
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_dir = Path(data_dir)

    def setup(self, stage: Optional[str] = None):
        """
        Prepares data for training and validation.
        Reads CSV files, merges them, adds labels, shuffles, and splits into train/val/test sets.

        Args:
            stage (Optional[str]): Stage of the training (fit, validate, test, predict).
        """
        # Read and merge data
        fake_df = pd.read_csv(self.data_dir / self.hparams.fake_news_filename)
        true_df = pd.read_csv(self.data_dir / self.hparams.true_news_filename)

        # 1 for Fake, 0 for Real
        fake_df["label"] = 1
        true_df["label"] = 0

        # Combine datasets
        df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
        # Shuffle
        df = df.sample(frac=1, random_state=self.hparams.seed).reset_index(drop=True)

        # Subset if requested
        if self.hparams.subset_fraction < 1.0:
            subset_size = int(len(df) * self.hparams.subset_fraction)
            df = df.iloc[:subset_size]

        # Split
        total_len = len(df)
        splits = self.train_val_test_split
        train_len = int(total_len * splits[0])
        val_len = int(total_len * splits[1])
        # The rest goes to test to ensure total matches (avoid rounding errors)

        train_df = df.iloc[:train_len]
        val_df = df.iloc[train_len : train_len + val_len]
        test_df = df.iloc[train_len + val_len :]

        self.train_dataset = FakeNewsDataset(
            train_df, self.tokenizer, self.hparams.max_length
        )
        self.val_dataset = FakeNewsDataset(
            val_df, self.tokenizer, self.hparams.max_length
        )
        self.test_dataset = FakeNewsDataset(
            test_df, self.tokenizer, self.hparams.max_length
        )

    def train_dataloader(self) -> DataLoader:
        """Returns DataLoader for training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns DataLoader for validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns DataLoader for test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=self.hparams.pin_memory,
        )
