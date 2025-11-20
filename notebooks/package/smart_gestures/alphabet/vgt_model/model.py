import json
import os
from pathlib import Path
from typing import cast, Callable

import numpy as np
import torch
from torch import nn

# Constants
IN_DIM = 63
NUM_POINTS = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
MODEL_DIR = THIS_DIR / "models"

CLASSES_FILE = DATA_DIR / "classes.json"
MODEL_FILE = MODEL_DIR / "vgt_alphabet_model.pth"

if not CLASSES_FILE.exists():
    raise FileNotFoundError(f"Classes file not found: {CLASSES_FILE}")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

def normalize_landmarks(lm: list[dict[str, float]], root_index: int = 0,scale_methode="wrist_to_middle") -> np.ndarray:
    """
    Normalize a landmark list.
    Args:
        lm (list[dict[str, float]]): List of landmarks with 'x', 'y', 'z' keys.
        root_index (int, optional): Index of the root landmark. Defaults to 0.
        scale_methode (str, optional): Method to calculate scale. Defaults to "wrist_to_middle".
    Returns:
        np.ndarray: Normalized landmark array.
    """

    arr = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in lm], dtype=np.float32)

    if arr.shape[0] != NUM_POINTS:
        raise ValueError(f"Expected {NUM_POINTS} landmarks, got {arr.shape[0]}")
    
    root = arr[root_index].copy()
    arr = arr - root

    if scale_methode == "wrist_to_middle":
        # Madiapipe middle finger MCP is index 9
        scale = np.linalg.norm(arr[9])
    
    else: 
        dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scale = dists.max()
    
    if scale <= 1e-6:
        scale = 1.0
    
    arr = arr / scale
    return arr

def create_model(num_classes: int, in_dim: int = IN_DIM) -> nn.Module:
    """
    Create a simple feedforward neural network model.
    Args:
        num_classes (int): Number of output classes.
        in_dim (int, optional): Input dimension. Defaults to IN_DIM.
    Returns:
        nn.Module: The created model.
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

def get_classes() -> list[str]:
    """
    Load the class names from the classes.json file.
    Returns:
        list[str]: List of class names.
    """
    with open(CLASSES_FILE, 'r') as f:
        classes = json.load(f)
    return classes

class VGTModel():
    """
    A high level class for the VGT alphabet recognition model.
    This class handles loading the model, preparing input data, and making predictions.
    """

    def __init__(self):
        """
        Initialize the VGTModel.
        """
        self.classes = get_classes()
        self.in_dim = IN_DIM
        self.model = create_model(num_classes=len(self.classes), in_dim=self.in_dim)
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        self.model.eval()
    
    def prepare_input(self, landmarks: list[dict[str, float]]) -> np.ndarray:
        """
        Prepare input landmarks for the model.
        Args:
            landmarks (list[dict[str, float]]): List of landmarks.
        Returns:
            np.ndarray: Prepared input array.
        """
        arr = normalize_landmarks(landmarks, root_index=0, scale_methode="wrist_to_middle")
        return arr.reshape(-1)
    
    def predict(self, landmarks: list[dict[str, float]]) -> str:
        """
        Predict the class for the given landmarks.
        Args:
            landmarks (list[dict[str, float]]): List of landmarks.
        Returns:
            str: Predicted class name.
        """
        
        if len(landmarks) != NUM_POINTS:
            raise ValueError(f"Expected {NUM_POINTS} landmarks, got {len(landmarks)}")
        
        x_input = self.prepare_input(landmarks)
        x_tensor = torch.from_numpy(x_input.reshape(1, self.in_dim)).to(DEVICE)

        with torch.no_grad():
            logits = self.model(x_tensor)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_name = self.classes[pred_idx]
        return pred_name

