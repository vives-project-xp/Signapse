import os
from pathlib import Path
from typing import cast
import numpy as np
import torch
import torch.nn as nn

from smart_gestures.alphabet.const import NUM_POINTS, IN_DIM
from .data_utils import get_classes, normalize_landmarks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR / "models"
MODEL_FILE = MODEL_DIR / "hand_gesture_model.pth"


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
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)


def save_model(model: nn.Module, path: str = 'hand_gesture_model.pth') -> None:
    """
    Save the model state dictionary to a file in the MODEL_DIR.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, path))


def load_model(model: nn.Module, path: str = 'hand_gesture_model.pth') -> nn.Module:
    """
    Load the model state dictionary from a file.
    """
    model = create_model(cast(int, model.num_classes), cast(int, model.in_dim))
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model
