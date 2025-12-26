import subprocess
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from fake_news_detector.commands import download_data, ensure_data_exists
from fake_news_detector.dataset import FakeNewsDataModule
from fake_news_detector.make_plots import generate_plots
from fake_news_detector.model import FakeNewsModel


def get_git_commit_id():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def pull_dvc_data():
    try:
        print("Attempting to pull data via DVC...")
        subprocess.run(["dvc", "pull"], check=True)
        print("DVC pull successful.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("DVC pull failed or DVC not installed/configured.")
        return False


@hydra.main(version_base=None, config_path="../conf", config_name="cfg")
def train(cfg: DictConfig):
    # Check data existence
    data_dir = Path(cfg.data.data_dir)
    required_files = [
        data_dir / cfg.data.fake_news_filename,
        data_dir / cfg.data.true_news_filename,
    ]

    if not ensure_data_exists(required_files):
        print("Data not found.")
        # Try DVC first
        if not pull_dvc_data() or not ensure_data_exists(required_files):
            print("Attempting to download from source...")
            try:
                download_data(output_dir=cfg.data.data_dir)
            except Exception as e:
                print(f"Failed to download data: {e}")
                exit(1)

    # Set seed for reproducibility
    L.seed_everything(cfg.trainer.seed)

    # Init DataModule
    data_module = FakeNewsDataModule(
        data_dir=cfg.data.data_dir,
        fake_news_filename=cfg.data.fake_news_filename,
        true_news_filename=cfg.data.true_news_filename,
        model_name=cfg.module.name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
        train_val_test_split=cfg.data.train_val_test_split,
        pin_memory=cfg.data.pin_memory,
        subset_fraction=cfg.data.get("subset_fraction", 1.0),
        seed=cfg.trainer.seed,
    )

    # Init Model
    model = FakeNewsModel(
        model_name=cfg.module.name,
        lr=cfg.module.lr,
        dropout=cfg.module.dropout,
        weight_decay=cfg.module.weight_decay,
        scheduler_patience=cfg.trainer.scheduler.patience,
        scheduler_factor=cfg.trainer.scheduler.factor,
        freeze_backbone=cfg.module.get("freeze_backbone", False),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(**cfg.callbacks.model_checkpoint),
        EarlyStopping(**cfg.callbacks.early_stopping),
        RichModelSummary(**cfg.callbacks.rich_model_summary),
        LearningRateMonitor(**cfg.callbacks.learning_rate_monitor),
        # DeviceStatsMonitor() # Optional: useful for GPU monitoring
    ]

    # Logger
    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
    )

    # Log hyperparameters
    logger.log_hyperparams(cfg)
    logger.log_hyperparams({"git_commit_id": get_git_commit_id()})

    # Init Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

    # Log model to MLflow (for serving)
    import mlflow

    # Get best model path from checkpoint callback
    checkpoint_callback = trainer.checkpoint_callback

    if checkpoint_callback:
        best_model_path = checkpoint_callback.best_model_path

        if best_model_path:
            print(f"Logging model from {best_model_path} to MLflow...")
            # Load best model
            best_model = FakeNewsModel.load_from_checkpoint(best_model_path)

            # Log model to MLflow
            with mlflow.start_run(run_id=logger.run_id):
                mlflow.pytorch.log_model(
                    best_model, "model", registered_model_name="fake_news_distilbert"
                )
                print("Model logged to MLflow as 'model'")

    # Generate plots
    try:
        generate_plots()
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")


if __name__ == "__main__":
    train()
