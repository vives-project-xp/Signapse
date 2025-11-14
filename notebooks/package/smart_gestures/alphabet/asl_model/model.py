import json
import os
from pathlib import Path
from typing import cast, Callable

import numpy as np
import torch
import torch.nn as nn

# Import package level constants
try:
    # Relative import when used as a package
    from smart_gestures.alphabet.const import NUM_POINTS, IN_DIM
except ImportError:
    # Otherwise, use absolute import
    from ..const import NUM_POINTS, IN_DIM

# print (f"NUM_POINTS: {NUM_POINTS}, IN_DIM: {IN_DIM}")

# Define the constant
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
MODEL_DIR = THIS_DIR / "models"

CLASSES_FILE = DATA_DIR / "classes.json"
MODEL_FILE = MODEL_DIR / "asl_alphabet_model.pth"

# Check if directories exist while loading the modules
if not CLASSES_FILE.exists():
    raise FileNotFoundError(f"Classes file not found at {CLASSES_FILE}")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")

# Normalize input function


def normalize_landmarks(lm: list[dict[str, float]], root_idx=0, scale_method="wrist_to_middle") -> np.ndarray:
    """
    Normalize the input landmarks.

    Args:
        lm: list of landmarks as dictionaries with 'x', 'y', 'z' keys.
        root_idx: index of the root landmark
        scale_method: method to scale the landmarks.

    Returns:
        normalized landmarks as a numpy array of shape (NUM_POINTS, 3).
    """
    # Convert list of dicts to numpy array
    arr = np.array([[p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)]
                    for p in lm], dtype=np.float32)

    # Check if the number of landmarks matches the expected number
    if arr.shape[0] != NUM_POINTS:
        raise ValueError(
            f"Expected {NUM_POINTS} landmarks per sample, got {arr.shape[0]}")
    
    # Translate so that the root landmark is at the origin
    root = arr[root_idx].copy()
    arr = arr - root

    # Scale the landmarks
    if scale_method == "wrist_to_middle":
        scale = np.linalg.norm(arr[9]) # Distance from wrist (0) to middle finger MCP (9)
    else:
        dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scale = dists.max() # Max distance between any two landmarks

    # Avoid division by zero
    if scale <= 1e-6:
        scale = 1.0

    # Normalize the landmarks
    arr = arr / scale
    # Return the normalized landmarks
    return arr 

# Create model function
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

# Get classes function
def get_classes() -> list[str]:
    """
    Load the classes from the CLASSES_FILE.

    Returns:
        list of classes.
    """
    with open(CLASSES_FILE, 'r') as f:
        classes = json.load(f)
    return classes

# Class to encapsulate the ASL model
class ASLModel():
    """
    A higher-level class to encapsulate the ASL alphabet model and its utilities.
    The class handles model loading, input preparation, and prediction.
    """
    def __init__(self):
        """
        Initialize the ASLModel by loading the model and classes.
        """
        self.classes = get_classes()
        self.in_dim = IN_DIM
        self.model = create_model(num_classes=len(
            self.classes), in_dim=self.in_dim)
        self.model.load_state_dict(
            torch.load(MODEL_FILE, map_location=DEVICE))
        self.model.eval()

    def prepare_input(self, lm: list[dict[str, float]]) -> torch.Tensor:
        """
        Prepare the input landmarks for prediction.

        Args:
            lm: list of landmarks as dictionaries with 'x', 'y', 'z' keys.
            
        Returns:
            input tensor of shape (1, in_dim).
        """
        arr = normalize_landmarks(
            lm, root_idx=0, scale_method="wrist_to_middle"
        )
        return arr.reshape(-1)

    def predict(self, lm: list[dict[str, float]]) -> str:
        """
        Predict the ASL letter from the input landmarks.

        Args:
            lm: list of landmarks as dictionaries with 'x', 'y', 'z' keys.
            
        Returns:
            predicted letter as a string.
        """
        if len(lm) != NUM_POINTS:
            raise ValueError(
                f"Expected {NUM_POINTS} landmarks, got {len(lm)}"
            )
        input_tensor = self.prepare_input(lm)
        input_tensor = torch.from_numpy(input_tensor.reshape(1, self.in_dim)).to(DEVICE)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_idx = int(torch.argmax(output, dim=1).item())
            predicted_class = self.classes[predicted_idx]
        return predicted_class 
