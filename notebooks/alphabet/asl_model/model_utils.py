import torch.nn as nn
import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(THIS_DIR, "models")

def create_model(num_classes: int, in_dim: int):
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

def save_model(model, path='hand_gesture_model.pth'):
    """
    Save the model state dictionary to a file in the MODEL_DIR.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, path))

def load_model(model, path='hand_gesture_model.pth'):
    """
    Load the model state dictionary from a file.
    """
    model = create_model(model.num_classes, model.in_dim)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model