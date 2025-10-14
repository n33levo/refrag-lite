"""Logging utilities with MLflow and CSV fallback."""

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with rich formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Setup root logger with file and console handlers.

    Args:
        log_file: Optional file to write logs to
        level: Logging level

    Example:
        >>> setup_logging("logs/train.log")
    """
    handlers = [
        RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
    ]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
    )


class ExperimentLogger:
    """Experiment logger with MLflow + CSV fallback.

    Example:
        >>> logger = ExperimentLogger("my_experiment", use_mlflow=True)
        >>> with logger.start_run("run_1"):
        ...     logger.log_params({"lr": 0.001})
        ...     logger.log_metrics({"loss": 0.5}, step=0)
    """

    def __init__(
        self,
        experiment_name: str,
        use_mlflow: bool = True,
        csv_path: Optional[str] = None,
        tracking_uri: str = "file:./mlruns",
    ):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of experiment
            use_mlflow: Whether to use MLflow
            csv_path: Path to CSV file for fallback logging
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.csv_path = csv_path or f"reports/runs/{experiment_name}.csv"
        self.logger = get_logger(__name__)

        # Setup MLflow if available
        self.mlflow = None
        if use_mlflow:
            try:
                import mlflow

                self.mlflow = mlflow
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self.logger.info(f"MLflow tracking enabled: {tracking_uri}")
            except ImportError:
                self.logger.warning("MLflow not available, using CSV fallback")
                self.use_mlflow = False

        # Setup CSV fallback
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._csv_rows = []

    def start_run(self, run_name: Optional[str] = None):
        """Start a new run (context manager).

        Args:
            run_name: Optional run name

        Returns:
            Context manager
        """
        if self.use_mlflow and self.mlflow:
            return self.mlflow.start_run(run_name=run_name)
        else:
            return _DummyContextManager()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.

        Args:
            params: Dictionary of parameters

        Example:
            >>> logger.log_params({"lr": 0.001, "batch_size": 32})
        """
        if self.use_mlflow and self.mlflow:
            self.mlflow.log_params(params)

        # Also save to CSV
        self._csv_rows.append({"type": "param", **params})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number

        Example:
            >>> logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
        """
        if self.use_mlflow and self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)

        # Also save to CSV
        row = {"type": "metric", "step": step, **metrics}
        self._csv_rows.append(row)

    def log_artifact(self, path: str) -> None:
        """Log artifact file.

        Args:
            path: Path to artifact

        Example:
            >>> logger.log_artifact("model.pt")
        """
        if self.use_mlflow and self.mlflow:
            self.mlflow.log_artifact(path)

    def save_csv(self) -> None:
        """Save accumulated logs to CSV file."""
        if not self._csv_rows:
            return

        # Get all unique keys
        keys = set()
        for row in self._csv_rows:
            keys.update(row.keys())
        keys = sorted(keys)

        # Write CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._csv_rows)

        self.logger.info(f"Saved logs to {self.csv_path}")

    def __del__(self):
        """Save CSV on cleanup."""
        try:
            self.save_csv()
        except Exception:
            pass


class _DummyContextManager:
    """Dummy context manager for when MLflow is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
