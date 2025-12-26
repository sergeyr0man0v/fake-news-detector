from pathlib import Path

import fire
import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def generate_plots(output_dir: str = "plots"):
    """
    Generates training plots from MLflow metrics and saves them to the plots directory.

    Args:
        output_dir (str): Directory where plots will be saved. Defaults to 'plots'.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots in {output_path}...")

    try:
        # Connect to MLflow
        client = mlflow.tracking.MlflowClient(tracking_uri="file:./mlruns")
        experiment = client.get_experiment_by_name("fake_news_distilbert")

        if not experiment:
            print("Experiment 'fake_news_distilbert' not found.")
            return

        # Get the latest run
        runs = client.search_runs(
            experiment.experiment_id, order_by=["start_time DESC"], max_results=1
        )

        if not runs:
            print("No runs found.")
            return

        run_id = runs[0].info.run_id
        metrics_dir = Path("mlruns") / experiment.experiment_id / run_id / "metrics"

        if not metrics_dir.exists():
            print(f"Metrics directory not found at {metrics_dir}")
            return

        # Plot 1: Validation F1 Score
        if (metrics_dir / "val_f1").exists():
            val_f1 = pd.read_csv(
                metrics_dir / "val_f1",
                sep=" ",
                header=None,
                names=["time", "val", "step"],
            )
            plt.figure(figsize=(10, 6))
            plt.plot(
                val_f1["step"], val_f1["val"], marker="o", linestyle="-", color="b"
            )
            plt.title("Validation F1 Score over Steps")
            plt.xlabel("Step")
            plt.ylabel("F1 Score")
            plt.grid(True)
            plt.savefig(output_path / "val_f1.png")
            plt.close()
            print(f"Saved {output_path / 'val_f1.png'}")

        # Plot 2: Validation Loss
        if (metrics_dir / "val_loss").exists():
            val_loss = pd.read_csv(
                metrics_dir / "val_loss",
                sep=" ",
                header=None,
                names=["time", "val", "step"],
            )
            plt.figure(figsize=(10, 6))
            plt.plot(
                val_loss["step"], val_loss["val"], marker="o", linestyle="-", color="r"
            )
            plt.title("Validation Loss over Steps")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(output_path / "val_loss.png")
            plt.close()
            print(f"Saved {output_path / 'val_loss.png'}")

        # Plot 3: Training Loss (Epoch)
        if (metrics_dir / "train_loss_epoch").exists():
            train_loss = pd.read_csv(
                metrics_dir / "train_loss_epoch",
                sep=" ",
                header=None,
                names=["time", "val", "step"],
            )
            plt.figure(figsize=(10, 6))
            plt.plot(
                train_loss["step"],
                train_loss["val"],
                marker="s",
                linestyle="--",
                color="g",
            )
            plt.title("Training Loss per Epoch")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(output_path / "train_loss.png")
            plt.close()
            print(f"Saved {output_path / 'train_loss.png'}")

        print("Plot generation complete.")

    except Exception as e:
        print(f"Error generating plots: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(generate_plots)
