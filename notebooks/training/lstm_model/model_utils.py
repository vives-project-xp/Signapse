"""
This module provides utility functions for creating, saving, and loading a PyTorch model
for sequence-based sign language recognition (LSTM).
"""
import os
from pathlib import Path
from typing import cast
import numpy as np
import torch
import torch.nn as nn

# Assuming data_utils is available for context
from data_utils import * # --- Configuration ---
# Define device for PyTorch operations (CUDA if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define base directories for model files
THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR / "models"
MODEL_FILE = MODEL_DIR / "lstm_model.pth" 

# Path to the 'package/smart_gestures/gestures/lstm_model/models' directory.
# Assuming this model_utils.py is two levels deep from 'training' 
# (e.g., SMARTGLASSES/notebooks/training/lstm_model/model_utils.py)

# THIS_DIR.parents[4] should lead up to SMARTGLASSES/
# Then navigate down to: notebooks/package/smart_gestures/gestures/lstm_model/
PACKAGE_ROOT = THIS_DIR.parents[4] 

PACKAGE_MODEL_DIR = (
    PACKAGE_ROOT
    / "notebooks"
    / "package"
    / "smart_gestures"
    / "gestures"
    / "lstm_model"
    / "models"
)
PACKAGE_MODEL_FILE = PACKAGE_MODEL_DIR / "lstm_model.pth"

# --- LSTM Model Definition (Same as before) ---

class GestureLSTM(nn.Module):
    """
    A PyTorch LSTM model designed for sequence classification of sign language gestures.
    """
    def __init__(self, in_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super(GestureLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=in_dim,         
            hidden_size=hidden_size,   
            num_layers=num_layers,     
            batch_first=True,          
            dropout=dropout,           
            bidirectional=False        
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.dropout(last_time_step_out)
        out = self.fc(out)
        return out


def create_model(
    num_classes: int, 
    in_dim: int, 
    hidden_size: int = 128, 
    num_layers: int = 2
) -> nn.Module:
    """
    Creates an LSTM model instance.
    """
    model = GestureLSTM(
        in_dim=in_dim, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_classes=num_classes
    )
    return model.to(DEVICE)

# --- Save and Load Utilities (Adjusted paths, logic unchanged) ---

def save_model(model: nn.Module, path: str = "lstm_model.pth") -> None:
    """
    Saves the model's state dictionary to local and package directories.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = MODEL_DIR / path
    torch.save(model.state_dict(), local_path)

    try:
        if not os.path.exists(PACKAGE_MODEL_DIR):
            os.makedirs(PACKAGE_MODEL_DIR, exist_ok=True)
        package_path = PACKAGE_MODEL_DIR / path
        torch.save(model.state_dict(), package_path)
    except Exception as e:
        # Note: Added 'as e' for proper exception handling if needed for debugging
        print(f"Warning: Could not save model to package directory: {e}")
        pass


def load_model(
    path: str = "lstm_model.pth",
    model: nn.Module | None = None,
    num_classes: int | None = None,
    in_dim: int | None = None,
    hidden_size: int | None = 128, 
    num_layers: int | None = 2,
) -> nn.Module:
    """
    Loads a model's state dictionary from a file.
    """
    resolved_path = path if os.path.isabs(path) else os.path.join(MODEL_DIR, path)
    
    if not os.path.exists(resolved_path):
        # Fallback to the new package path
        pkg_candidate = str(PACKAGE_MODEL_DIR / path)
        if os.path.exists(pkg_candidate):
            resolved_path = pkg_candidate

    if model is None:
        if num_classes is None or in_dim is None:
            raise ValueError(
                "Either provide an existing `model` or both `num_classes` and `in_dim`."
            )
        model = create_model(
            num_classes=int(num_classes), 
            in_dim=int(in_dim),
            hidden_size=int(hidden_size) if hidden_size is not None else 128,
            num_layers=int(num_layers) if num_layers is not None else 2
        )

    state = torch.load(resolved_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE)