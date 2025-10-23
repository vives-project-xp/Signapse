import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "dataset", "images")
HAND_LANDMARKS_JSON = os.path.join(DATA_DIR, "hand_landmarks.json")

# Dataset class for hand landmarks
class LandmarksDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, classes: List[str]):
        # Accept either flattened vectors (num_samples, 63) or sequences (num_samples, 21, 3)
        if not ((X.ndim == 2 and X.shape[1] == 63) or (X.ndim == 3 and X.shape[1:] == (21, 3))):
            raise AssertionError("Expected X to have shape (num_samples, 63) or (num_samples, 21, 3) for 21 landmarks with (x,y,z) coords")
        assert X.shape[0] == y.shape[0], "Number of samples in X and y must match"
        self.X = X
        self.y = y
        self.classes = classes

    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int):
        # return float tensor; keep shape as either (63,) or (21,3)
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def normalize_landmarks(lm, root_idx=0, scale_method="wrist_to_middle"):
    """
    Normalize a single sample of landmarks.
    - translate so the root (wrist) is at the origin
    - scale by a hand-size measure
    Returns an array of shape (21,3)
    """
    arr = np.array([[p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)] for p in lm], dtype=np.float32)
    if arr.shape[0] != 21:
        raise ValueError(f"Expected 21 landmarks per sample, got {arr.shape[0]}")
    root = arr[root_idx].copy()
    arr = arr - root

    if scale_method == "wrist_to_middle":
        # Mediapipe middle-finger MCP is index 9
        scale = np.linalg.norm(arr[9])
    else:
        dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scale = dists.max()

    if scale <= 1e-6:
        scale = 1.0
    arr = arr / scale
    return arr

def load_dataset_normalized(json_file: str, as_sequence: bool = True, scale_method: str = "wrist_to_middle") -> LandmarksDataset:
    """
    Load dataset and normalize landmarks per-sample.
    - as_sequence=True -> X shape (n,21,3)
    - as_sequence=False -> X shape (n,63) flattened after normalization
    """
    data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df['class'] = df['class'].astype('category')

    X_list = [normalize_landmarks(lm, root_idx=0, scale_method=scale_method) for lm in df['landmarks']]
    y = np.array(df['class'].cat.codes, dtype=np.int64)
    classes = df['class'].cat.categories.tolist()

    if as_sequence:
        X = np.stack(X_list).astype(np.float32)  # (n,21,3)
    else:
        X = np.stack(X_list).reshape(len(df), -1).astype(np.float32)  # (n,63)

    return LandmarksDataset(X, y, classes)

# Get classes
def get_classes() -> List[str]:
    return pd.read_json(HAND_LANDMARKS_JSON, lines=True)['class'].astype('category').cat.categories.tolist()

def split_dataset(dataset: LandmarksDataset, val_ratio: float = 0.2, random_seed: int = 42):
    """
    Split dataset into training and validation sets with reproducible results.
    A fixed random seed is used for shuffling before splitting.
    A ratio of 80/20 is used for training/validation split by default.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    return train_dataset, val_dataset

def get_loaders(train_dataset: LandmarksDataset, val_dataset: LandmarksDataset, batch_size: int = 32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader