from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from transformers import AutoModel


class FakeNewsModel(L.LightningModule):
    """
    LightningModule for Fake News Detection using DistilBERT.

    This model uses a pre-trained DistilBERT as a backbone and adds a linear
    classification head on top. It tracks various metrics like Accuracy, F1,
    Precision, Recall, and ROC-AUC during training and validation.

    Args:
        model_name (str): Name of the pre-trained model (e.g. "distilbert-base-uncased").
        lr (float): Learning rate for the optimizer.
        dropout (float): Dropout rate for the classifier head.
        weight_decay (float): Weight decay for the optimizer.
        scheduler_patience (int): Patience for ReduceLROnPlateau scheduler.
        scheduler_factor (float): Factor for ReduceLROnPlateau scheduler.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        lr: float = 2e-5,
        dropout: float = 0.2,
        weight_decay: float = 0.01,
        scheduler_patience: int = 2,
        scheduler_factor: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classifier head
        # DistilBERT hidden size is 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.bert.config.hidden_size, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_auroc = AUROC(task="binary")

        # Test metrics
        self.test_acc = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of token indices.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            torch.Tensor: Logits from the classifier (unnormalized scores).
        """
        # DistilBERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step logic. Calculates loss and logs metrics.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Calculated loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().unsqueeze(1)

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        preds = torch.sigmoid(logits)
        self.train_acc(preds, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step logic. Calculates metrics on validation set.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Index of the batch.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().unsqueeze(1)

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)

        # Update metrics
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_auroc(preds, labels)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)
        self.log("val_auroc", self.val_auroc)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step logic. Calculates metrics on test set.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Index of the batch.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().unsqueeze(1)

        logits = self(input_ids, attention_mask)
        preds = torch.sigmoid(logits)

        self.test_acc(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "frequency": 1,
            },
        }
