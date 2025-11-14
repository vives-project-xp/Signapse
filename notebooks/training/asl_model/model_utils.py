import os
from pathlib import Path
from typing import cast
import numpy as np
import torch
import torch.nn as nn

from data_utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR / "models"
MODEL_FILE = MODEL_DIR / "asl_alphabet_model.pth"
PACKAGE_MODEL_DIR = (
    THIS_DIR.parents[1]
    / "package"
    / "smart_gestures"
    / "alphabet"
    / "asl_model"
    / "models"
)
PACKAGE_MODEL_FILE = PACKAGE_MODEL_DIR / "asl_alphabet_model.pth"


def create_model(num_classes: int, in_dim: int) -> nn.Module:
    """
    Create a simple feedforward neural network model for hand landmark classification.

    Args:
        num_classes: number of output classes.
        in_dim: number of input features.
    """
    model = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def save_model(model: nn.Module, path: str = "asl_alphabet_model.pth") -> None:
    """
    Save the model state dictionary to a file in the MODEL_DIR.
    """
    # Ensure local model dir exists and save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = MODEL_DIR / path
    torch.save(model.state_dict(), local_path)

    # Also save into the package models folder (if present / create it)
    try:
        if not os.path.exists(PACKAGE_MODEL_DIR):
            os.makedirs(PACKAGE_MODEL_DIR, exist_ok=True)
        package_path = PACKAGE_MODEL_DIR / path
        torch.save(model.state_dict(), package_path)
    except Exception:
        # Do not fail if package path cannot be written; just continue
        pass


def load_model(
    path: str = "asl_alphabet_model.pth",
    model: nn.Module | None = None,
    num_classes: int | None = None,
    in_dim: int | None = None,
) -> nn.Module:
    """
    Load the model state dictionary from a file.

    Args:
        path: path to the saved state dict. If relative, resolved under `MODEL_DIR`.
        model: an existing `nn.Module` to load the state dict into. If provided,
            the function will load directly into this model instance.
        num_classes: when `model` is not provided, the number of classes to
            create a fresh model with.
        in_dim: when `model` is not provided, the input dimension for the model.

    Returns:
        The model with loaded weights on `DEVICE`.
    """
    # Resolve path relative to MODEL_DIR when not absolute
    resolved_path = path if os.path.isabs(path) else os.path.join(MODEL_DIR, path)

    # If not found in the local models folder, fallback to package models folder
    if not os.path.exists(resolved_path):
        pkg_candidate = str(PACKAGE_MODEL_DIR / path)
        if os.path.exists(pkg_candidate):
            resolved_path = pkg_candidate

    if model is None:
        if num_classes is None or in_dim is None:
            raise ValueError(
                "Either provide an existing `model` or both `num_classes` and `in_dim`."
            )
        model = create_model(int(num_classes), int(in_dim))

    state = torch.load(resolved_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE)
