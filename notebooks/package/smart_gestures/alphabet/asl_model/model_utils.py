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


class ASLModel():
    classes: list[str]

    def __init__(self, classes: list[str] = get_classes(), in_dim: int = IN_DIM):
        self.classes = classes
        self.in_dim = in_dim

        self.model = create_model(num_classes=len(
            self.classes), in_dim=self.in_dim)
        if MODEL_FILE.exists():
            self.model.load_state_dict(
                torch.load(MODEL_FILE, map_location=DEVICE))
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
        self.model.eval()

    def _prepare_input(self, landmarks: list[dict[str, float]]) -> np.ndarray:
        arr = normalize_landmarks(
            landmarks, root_idx=0, scale_method="wrist_to_middle")
        return arr.reshape(-1)

    def predict(self, landmarks: list[dict[str, float]]) -> str:
        if len(landmarks) != NUM_POINTS:
            raise ValueError(
                f"Expected {NUM_POINTS} landmarks, got {len(landmarks)}"
            )

        # Belangrijk: pas hier dezelfde preprocessing toe als bij training indien nodig
        x = torch.from_numpy(self._prepare_input(landmarks).reshape(
            1, IN_DIM)).to(DEVICE)  # type: ignore

        with torch.no_grad():
            logits = self.model(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_name = self.classes[pred_idx]
        return pred_name
