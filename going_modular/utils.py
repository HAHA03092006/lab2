"""
Contains utility functions: seeding, device selection, metrics,
timing, and saving/loading PyTorch models.

Phase 4 extensions: deterministic seeding, torch.device return,
TensorBoard writer, atomic JSON I/O, artifacts resume/checkpoint,
safe weights_only loading, and environment info.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# History key schemas
# ---------------------------------------------------------------------------

HISTORY_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc")
HISTORY_KEYS_SIX = ("train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time")

ARTIFACT_REQUIRED = ("best_model.pth", "history.json", "config.json")


# ---------------------------------------------------------------------------
# Seed and device
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets random seeds for reproducibility.

    Seeds Python's random, NumPy, and PyTorch (CPU and CUDA when available).
    When ``deterministic`` is True, sets PyTorch/CUDA backend flags for
    deterministic behavior where supported.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Returns a ``torch.device`` for the requested or best available backend.

    Args:
        prefer: Optional backend name ("cuda", "mps", "cpu"). If None, selects
            cuda, then mps, then cpu. Raises ValueError if the requested
            backend is not available.

    Raises:
        ValueError: If ``prefer`` is invalid or the requested device is not
            available.
    """
    if prefer is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    prefer = prefer.lower()
    valid = ("cuda", "mps", "cpu")
    if prefer not in valid:
        raise ValueError(
            f"Invalid device preference '{prefer}'. Choose one of {valid}."
        )

    if prefer == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available on this system.")
    if prefer == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but not available on this system.")

    return torch.device(prefer)


def get_env_info(device: torch.device) -> Dict[str, str]:
    """Return device and library version info for config/logging."""
    info: Dict[str, str] = {
        "device": str(device),
        "torch": torch.__version__,
    }
    try:
        import torchvision  # type: ignore

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = "not installed"
    try:
        import numpy as np  # type: ignore

        info["numpy"] = np.__version__
    except ImportError:
        info["numpy"] = "not installed"
    return info


# ---------------------------------------------------------------------------
# Legacy utilities (kept for backward compatibility)
# ---------------------------------------------------------------------------

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates accuracy between true and predicted labels."""
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_pred)


def print_train_time(start: float, end: float, device=None) -> float:
    """Prints and returns the elapsed training time in seconds."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def save_model(
    model: torch.nn.Module, target_dir: str, model_name: str
) -> Path:
    """Saves a PyTorch model's state_dict to a target directory.

    Legacy signature kept for backward compatibility.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        raise ValueError("model_name should end with '.pt' or '.pth'")
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    return model_save_path


def load_model(
    model: torch.nn.Module, model_path: str, device: str = "cpu"
) -> torch.nn.Module:
    """Loads a saved state_dict into an existing model instance.

    Uses ``weights_only=True`` for safe loading. Does not fall back to unsafe
    loading if the payload is incompatible.
    """
    state_dict = torch.load(f=model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)


def save_training_metadata(
    history: Dict[str, List[float]],
    artifact_dir: str,
    config: Dict[str, object],
) -> Tuple[Path, Path]:
    """Persist training curves and their configuration beside a model checkpoint.

    Legacy function kept for backward compatibility. Validates the four
    legacy history keys.
    """
    missing_keys = set(HISTORY_KEYS) - history.keys()
    if missing_keys:
        raise ValueError(f"Training history is missing keys: {sorted(missing_keys)}")
    if not history["train_loss"]:
        raise ValueError("Training history is empty; refusing to save incomplete results.")
    if any(len(history[key]) != len(history["train_loss"]) for key in HISTORY_KEYS):
        raise ValueError("All training history series must have the same number of epochs.")

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    history_path = artifact_path / "history.json"
    config_path = artifact_path / "config.json"

    _atomic_write_json(history_path, history)
    _atomic_write_json(config_path, config)
    return history_path, config_path


def load_training_artifacts(
    model: torch.nn.Module,
    artifact_dir: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Dict[str, object]]:
    """Load a model checkpoint, its training curves, and saved run configuration.

    Legacy function kept for backward compatibility.
    """
    artifact_path = Path(artifact_dir)
    checkpoint_path = artifact_path / "best_model.pth"
    history_path = artifact_path / "history.json"
    config_path = artifact_path / "config.json"
    required_paths = (checkpoint_path, history_path, config_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Training artifacts are incomplete. Missing: "
            + ", ".join(missing_paths)
            + ". Run the notebook with RUN_MODE='train' first."
        )

    loaded_model = load_model(model, str(checkpoint_path), device=device)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing_keys = set(HISTORY_KEYS) - history.keys()
    if missing_keys:
        raise ValueError(f"Saved history is missing keys: {sorted(missing_keys)}")
    if not history["train_loss"]:
        raise ValueError(f"Saved training history is empty: {history_path}")
    if any(len(history[key]) != len(history["train_loss"]) for key in HISTORY_KEYS):
        raise ValueError(f"Saved training history has inconsistent lengths: {history_path}")
    return loaded_model, history, config


# ---------------------------------------------------------------------------
# Phase 4: atomic JSON helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to a temp file then atomically replace the target.

    If writing fails, the original target file is left intact and the temp
    file is removed. The error is re-raised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file with UTF-8 encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Phase 4: config / history / metrics
# ---------------------------------------------------------------------------

def save_config(artifact_dir: str, config: Dict[str, object]) -> Path:
    """Atomically save run configuration to ``config.json``.

    Validates that ``config`` is a non-empty dict before writing.
    """
    if not isinstance(config, dict) or not config:
        raise ValueError("config must be a non-empty dict.")
    artifact_path = Path(artifact_dir)
    config_path = artifact_path / "config.json"
    _atomic_write_json(config_path, config)
    return config_path


def save_history(artifact_dir: str, history: Dict[str, List[float]]) -> Path:
    """Atomically save training history to ``history.json``.

    Validates the six required history keys: train_loss, val_loss,
    train_acc, val_acc, lr, epoch_time. All series must have equal length
    and at least one epoch.
    """
    missing_keys = set(HISTORY_KEYS_SIX) - history.keys()
    if missing_keys:
        raise ValueError(
            f"History is missing keys: {sorted(missing_keys)}. "
            f"Required: {list(HISTORY_KEYS_SIX)}."
        )
    if not history["train_loss"]:
        raise ValueError("History is empty; refusing to save incomplete results.")
    n = len(history["train_loss"])
    for key in HISTORY_KEYS_SIX:
        if len(history[key]) != n:
            raise ValueError(
                f"All history series must have the same length; "
                f"'{key}' has {len(history[key])}, expected {n}."
            )

    artifact_path = Path(artifact_dir)
    history_path = artifact_path / "history.json"
    _atomic_write_json(history_path, history)
    return history_path


def load_experiment(artifact_dir: str) -> Tuple[Dict[str, List[float]], Dict[str, object]]:
    """Load ``history.json`` and ``config.json`` from an artifact directory.

    Raises ``FileNotFoundError`` if either file is missing and
    ``ValueError`` if the payload is invalid.
    """
    artifact_path = Path(artifact_dir)
    history_path = artifact_path / "history.json"
    config_path = artifact_path / "config.json"

    if not history_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {history_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {config_path}")

    history = _read_json(history_path)
    config = _read_json(config_path)

    missing_keys = set(HISTORY_KEYS_SIX) - history.keys()
    if missing_keys:
        raise ValueError(f"Saved history is missing keys: {sorted(missing_keys)}")

    return history, config


def save_test_metrics(artifact_dir: str, metrics: Dict[str, object]) -> Path:
    """Atomically save test metrics to ``test_metrics.json``.

    Refuses to write empty metrics (no placeholder/eval-fake data).
    """
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty dict; will not write empty metrics.")
    artifact_path = Path(artifact_dir)
    metrics_path = artifact_path / "test_metrics.json"
    _atomic_write_json(metrics_path, metrics)
    return metrics_path


def load_test_metrics(artifact_dir: str) -> Dict[str, object]:
    """Load test metrics from ``test_metrics.json``.

    Raises ``FileNotFoundError`` if the file is missing.
    """
    artifact_path = Path(artifact_dir)
    metrics_path = artifact_path / "test_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {metrics_path}")
    return _read_json(metrics_path)


# ---------------------------------------------------------------------------
# Phase 4: artifacts existence
# ---------------------------------------------------------------------------

def artifacts_exist(artifact_dir: str) -> bool:
    """Return True if all required training artifacts exist.

    Required: ``best_model.pth``, ``history.json``, ``config.json``.
    ``test_metrics.json`` is optional.
    """
    artifact_path = Path(artifact_dir)
    return all((artifact_path / name).is_file() for name in ARTIFACT_REQUIRED)


# ---------------------------------------------------------------------------
# Phase 4: TensorBoard writer
# ---------------------------------------------------------------------------

def create_writer(experiment_name: str, log_root: str = "runs") -> SummaryWriter:
    """Create a TensorBoard SummaryWriter at ``runs/<experiment_name>``.

    The caller is responsible for closing the writer (e.g. via try/finally).
    """
    log_dir = Path(log_root) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


# ---------------------------------------------------------------------------
# Phase 4: checkpoint resume
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    phase_epochs_completed: int,
    best_score: float,
    best_epoch: int,
    history: Dict[str, List[float]],
    phase_config: Dict[str, object],
) -> Path:
    """Save a resume checkpoint containing model, optimizer, epoch, score and history.

    Only state dicts and primitive/JSON-compatible data are stored — no
    raw model, optimizer, or callable objects. Saved with ``torch.save`` so it
    can be loaded back with ``weights_only=True``.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "phase_epochs_completed": phase_epochs_completed,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "history": history,
        "phase_config": phase_config,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_phase_config: Dict[str, object],
) -> Tuple[int, float, int, Dict[str, List[float]]]:
    """Load a resume checkpoint and verify phase config compatibility.

    Returns ``(phase_epochs_completed, best_score, best_epoch, history)``.

    Raises ``FileNotFoundError`` if the checkpoint is missing and
    ``ValueError`` if the phase config does not match.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    payload = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)

    saved_phase_config = payload.get("phase_config", {})
    if saved_phase_config != expected_phase_config:
        raise ValueError(
            "Phase config mismatch: checkpoint was created with a different "
            "run configuration. Resume refused."
        )

    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    return (
        payload["phase_epochs_completed"],
        payload["best_score"],
        payload["best_epoch"],
        payload["history"],
    )
