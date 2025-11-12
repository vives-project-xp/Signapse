import json
import os
from pathlib import Path
from typing import cast, Callable
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

THIS_DIR = Path(__file__).parent
CLASSES_FILE = THIS_DIR / "data" / "classes.json"

if not CLASSES_FILE.exists():
    raise FileNotFoundError(f"Classes file not found: {CLASSES_FILE}")


class LandmarksDataset(Dataset[tuple[np.ndarray, int]]):
    def __init__(self, X: np.ndarray, y: np.ndarray, classes: list[str], preprocess: Callable[[np.ndarray], np.ndarray] | None = None):
        self.X = X
        self.y = y
        self.classes = classes
        self.preprocess = preprocess

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        x = self.X[idx]
        y = self.y[idx]
        if self.preprocess:
            x = self.preprocess(x)
        return x, y


def preprocess(lm: str) -> np.ndarray:
    return np.array(
        [[point['x'], point['y'], point['z']] for point in eval(lm)], dtype=np.float32)


def load_and_preprocess_dataset(csv_file: str) -> LandmarksDataset:
    """
    Load landmark data from CSV and preprocess it into a PyTorch Dataset.

    The CSV file should contain 'landmarks' and 'class' columns where:
    - 'landmarks' contains string representations of landmark coordinate lists
    - 'class' contains the label/category for each sample
    """
    df = pd.read_csv(csv_file)
    df['class'] = df['class'].astype('category')
    X = np.array(df['landmarks'].apply(preprocess).tolist())  # type: ignore
    y = np.array(df['class'].cat.codes, dtype=np.int64)
    classes = df['class'].cat.categories.tolist()
    return LandmarksDataset(X, y, classes, preprocess=None)

# Get classes


def get_classes() -> list[str]:
    return json.load(open(CLASSES_FILE, 'r', encoding='utf-8'))


def normalize_landmarks(lm, root_idx=0, scale_method="wrist_to_middle") -> np.ndarray:
    """
    Normalize a single sample of landmarks.
    - translate so the root (wrist) is at the origin
    - scale by a hand-size measure
    Returns an array of shape (21,3)
    """
    arr = np.array([[p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)]
                   for p in lm], dtype=np.float32)
    if arr.shape[0] != 21:
        raise ValueError(
            f"Expected 21 landmarks per sample, got {arr.shape[0]}")
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

# Split dataset into training and validation sets 80/20


def split_dataset(dataset: LandmarksDataset, val_ratio: float = 0.2, random_seed: int = 42) -> tuple[LandmarksDataset, LandmarksDataset]:
    """
    Split dataset into training and validation sets with reproducible results.
    A fixed random seed is used for shuffling before splitting.
    A ratio of 80/20 is used for training/validation split by default.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = cast(
        tuple[LandmarksDataset, LandmarksDataset],
        random_split(dataset, [train_size, val_size], generator=generator)
    )

    return train_dataset, val_dataset

# Create DataLoaders for training and validation sets


def get_loaders(train_dataset: LandmarksDataset, val_dataset: LandmarksDataset, batch_size: int = 32):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
