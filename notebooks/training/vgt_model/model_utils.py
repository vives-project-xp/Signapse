"""
This module provides utility functions for managing a PyTorch neural network model
for Vision-based Gesture Toolkit (VGT) alphabet sign language recognition.

It includes functions to create, save, and load the model, as well as defining
the device (CPU or GPU) and paths for model storage.
"""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from data_utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR / "models"
MODEL_FILE = MODEL_DIR / "vgt_alphabet_model.pth"
PACKAGE_MODEL_DIR = (
    THIS_DIR.parents[1]
    / "package"
    / "smart_gestures"
    / "alphabet"
    / "vgt_model"
    / "models"
)
MODEL_FILE_PACKAGE = PACKAGE_MODEL_DIR / "vgt_alphabet_model.pth"


def create_model(num_classes: int, in_dim: int) -> nn.Module:
    """
    Creates a sequential neural network model.

    The model consists of several linear layers with batch normalization,
    ReLU activation, and dropout. It is designed for classification tasks.

    Args:
        num_classes (int): The number of output classes for the final layer.
        in_dim (int): The number of input features for the first layer.

    Returns:
        nn.Module: The created PyTorch model, moved to the configured DEVICE.
    """
    model = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)


def save_model(model: nn.Module, path: str = 'vgt_alphabet_model.pth'):
    """
    Saves the model's state dictionary to a file.

    This function saves the model to two locations: the local `models` directory
    for training and a corresponding directory within the installable package
    to bundle the model with the application.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (str, optional): The filename for the saved model.
                              Defaults to 'vgt_alphabet_model.pth'.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = MODEL_DIR / path
    torch.save(model.state_dict(), local_path)

    # Also save to the package model directory
    # try:
    #     if not os.path.exists(PACKAGE_MODEL_DIR):
    #         os.makedirs(PACKAGE_MODEL_DIR, exist_ok=True)
    #     package_path = PACKAGE_MODEL_DIR / path
    #     torch.save(model.state_dict(), package_path)
    # except Exception:
    #     pass


def load_model(
        path: str = "vgt_alphabet_model.pth",
        model: nn.Module | None = None,
        num_classes: int | None = None,
        in_dim: int | None = None,
) -> nn.Module:
    """
    Loads a model's state dictionary from a file.

    It can either load the state into an existing model instance or create a
    new model if `num_classes` and `in_dim` are provided. The function
    searches for the model file in the local `models` directory first, then
    falls back to the package directory.

    Args:
        path (str, optional): The filename of the model to load.
                              Defaults to "vgt_alphabet_model.pth".
        model (nn.Module | None, optional): An existing model instance to load
                                            the state into. Defaults to None.
        num_classes (int | None, optional): The number of output classes.
                                            Required if `model` is None.
        in_dim (int | None, optional): The number of input features.
                                       Required if `model` is None.

    Returns:
        nn.Module: The loaded PyTorch model in evaluation mode, moved to the
                   configured DEVICE.

    Raises:
        ValueError: If `model` is None and either `num_classes` or `in_dim`
                    is not provided.
    """
    resolved_path = path if os.path.isabs(path) else os.path.join(MODEL_DIR, path)

    if not os.path.exists(resolved_path):
        # Try loading from package directory
        pkg_candidate = str(PACKAGE_MODEL_DIR / path)
        if os.path.exists(pkg_candidate):
            resolved_path = pkg_candidate

    if model is None:
        if num_classes is None or in_dim is None:
            raise ValueError(
                "num_classes and in_dim must be provided if model is None")
        model = create_model(int(num_classes), int(in_dim))

    state = torch.load(resolved_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE)
