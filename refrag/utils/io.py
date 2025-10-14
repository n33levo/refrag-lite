"""I/O utilities for loading/saving files and setting seeds."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with configuration

    Example:
        >>> config = load_yaml("configs/default.yaml")
        >>> print(config["seed"])
        42
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        path: Output path

    Example:
        >>> save_yaml({"seed": 42}, "config.yaml")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON object per line).

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries

    Example:
        >>> samples = load_jsonl("data/train.jsonl")
        >>> print(len(samples))
        1000
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str | Path) -> None:
    """Save list of dictionaries to JSONL file.

    Args:
        data: List of dictionaries
        path: Output path

    Example:
        >>> save_jsonl([{"id": 1}, {"id": 2}], "out.jsonl")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary

    Example:
        >>> data = load_json("config.json")
    """
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Output path
        indent: JSON indentation level

    Example:
        >>> save_json({"key": "value"}, "out.json")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
        deterministic: Use deterministic algorithms (slower but reproducible)

    Example:
        >>> set_seed(42)
        >>> # Now all random operations are reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for full determinism
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch 2.0+ deterministic mode
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Some ops don't support deterministic mode


def get_device(device: str = "auto") -> torch.device:
    """Get torch device.

    Args:
        device: Device string ("auto", "cuda", "mps", "cpu", "cuda:0", etc.)

    Returns:
        torch.device object

    Example:
        >>> device = get_device("auto")
        >>> print(device)
        cuda:0
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def save_checkpoint(
    state: Dict[str, Any],
    path: str | Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint.

    Args:
        state: Dictionary with model state, optimizer, etc.
        path: Output path
        is_best: If True, also save as "best.pt"

    Example:
        >>> save_checkpoint({
        ...     "model": model.state_dict(),
        ...     "optimizer": optimizer.state_dict(),
        ...     "epoch": 5,
        ... }, "checkpoint.pt")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

    if is_best:
        best_path = Path(path).parent / "best.pt"
        torch.save(state, best_path)


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        path: Path to checkpoint
        map_location: Device to load to

    Returns:
        Dictionary with checkpoint data

    Example:
        >>> checkpoint = load_checkpoint("checkpoint.pt", "cuda")
        >>> model.load_state_dict(checkpoint["model"])
    """
    return torch.load(path, map_location=map_location)
