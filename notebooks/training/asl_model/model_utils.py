"""
This module provides utility functions for creating, saving, and loading a PyTorch model
for ASL (American Sign Language) alphabet recognition. It defines the model architecture
and handles file paths for storing and retrieving model weights.
"""
import os
from pathlib import Path
from typing import cast
import numpy as np
import torch
import torch.nn as nn

from data_utils import *

# Define device for PyTorch operations (CUDA if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define base directories for model files
THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR / "models"
MODEL_FILE = MODEL_DIR / "asl_alphabet_model.pth"
# Define directory for storing the model within a package structure
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
    Creates a sequential neural network model.

    The model consists of three linear layers with ReLU activations and Dropout
    for regularization. It is designed for a classification task.

    Args:
        num_classes (int): The number of output classes for the final layer.
        in_dim (int): The dimensionality of the input features.

    Returns:
        nn.Module: The created PyTorch model, moved to the configured DEVICE.
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
    Saves the model's state dictionary to a file.

    This function saves the model weights to two locations: a local 'models'
    directory and a corresponding directory within a 'package' structure,
    facilitating both development/training and packaging.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (str, optional): The filename for the saved model.
                              Defaults to "asl_alphabet_model.pth".
    """
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
        pass


def load_model(
    path: str = "asl_alphabet_model.pth",
    model: nn.Module | None = None,
    num_classes: int | None = None,
    in_dim: int | None = None,
) -> nn.Module:
    """
    Loads a model's state dictionary from a file.

    It first checks for the model in the local 'models' directory, then falls
    back to the 'package' directory. If a model instance is not provided, it
    creates a new one using the provided `num_classes` and `in_dim`.

    Args:
        path (str, optional): The filename of the model to load.
                              Defaults to "asl_alphabet_model.pth".
        model (nn.Module | None, optional): An existing model instance to load
                                            the weights into. If None, a new
                                            model is created. Defaults to None.
        num_classes (int | None, optional): The number of output classes.
                                            Required if `model` is None.
        in_dim (int | None, optional): The input feature dimension.
                                       Required if `model` is None.

    Raises:
        ValueError: If `model` is not provided and either `num_classes` or
                    `in_dim` is missing.

    Returns:
        nn.Module: The loaded PyTorch model, moved to the configured DEVICE.
    """
    resolved_path = path if os.path.isabs(path) else os.path.join(MODEL_DIR, path)
    
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
