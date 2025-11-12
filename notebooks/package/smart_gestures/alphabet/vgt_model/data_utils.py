from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import random
import math

THIS_DIR = Path(__file__).parent
CLASSES_FILE = THIS_DIR / "data" / "classes.json"

if not CLASSES_FILE.exists():
    raise FileNotFoundError(f"Classes file not found: {CLASSES_FILE}")

# Dataset class for hand landmarks


class LandmarksDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: list[str],
        augment: bool = False,
        augment_prob: float = 0.5,
        noise_std: float = 0.01,
        rotate_deg: float = 10.0,
    ):
        # Accept either flattened vectors (num_samples, 63) or sequences (num_samples, 21, 3)
        if not ((X.ndim == 2 and X.shape[1] == 63) or (X.ndim == 3 and X.shape[1:] == (21, 3))):
            raise AssertionError(
                "Expected X to have shape (num_samples, 63) or (num_samples, 21, 3) for 21 landmarks with (x,y,z) coords")
        assert X.shape[0] == y.shape[0], "Number of samples in X and y must match"
        self.X = X
        self.y = y
        self.classes = classes
        self.is_flat = (X.ndim == 2)
        # Augmentation config
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.rotate_deg = rotate_deg

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # return float tensor; keep shape as either (63,) or (21,3)
        x_np = self.X[idx].copy()

        # Optional light augmentations for robustness
        if self.augment and random.random() < self.augment_prob:
            if self.is_flat:
                pts = x_np.reshape(21, 3)
            else:
                pts = x_np
            pts = apply_augmentations(
                pts, noise_std=self.noise_std, max_rotate_deg=self.rotate_deg)
            x_np = pts.reshape(-1) if self.is_flat else pts

        x = torch.from_numpy(x_np).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def _normalize_landmarks(lm, root_idx=0, scale_method="wrist_to_middle")-> np.ndarray:
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


def load_dataset_normalized(
    json_file: str,
    as_sequence: bool = True,
    scale_method: str = "wrist_to_middle",
    augment: bool = False,
    augment_prob: float = 0.5,
    noise_std: float = 0.01,
    rotate_deg: float = 10.0,
) -> LandmarksDataset:
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

    X_list = [_normalize_landmarks(
        lm, root_idx=0, scale_method=scale_method) for lm in df['landmarks']]
    y = np.array(df['class'].cat.codes, dtype=np.int64)
    classes = df['class'].cat.categories.tolist()

    if as_sequence:
        X = np.stack(X_list).astype(np.float32)  # (n,21,3)
    else:
        X = np.stack(X_list).reshape(len(df), -1).astype(np.float32)  # (n,63)

    return LandmarksDataset(
        X,
        y,
        classes,
        augment=augment,
        augment_prob=augment_prob,
        noise_std=noise_std,
        rotate_deg=rotate_deg,
    )

# Get classes


def get_classes() -> list[str]:
    return json.load(open(CLASSES_FILE, 'r', encoding='utf-8'))


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
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator)

    return train_dataset, val_dataset


def get_loaders(train_dataset: LandmarksDataset, val_dataset: LandmarksDataset, batch_size: int = 32):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ----- Augmentations for 3D hand landmarks -----
def random_rotation_matrix(max_deg: float = 10.0) -> np.ndarray:
    """Small random rotation around z-axis (camera facing) plus tiny tilt on x/y."""
    # Convert degrees to radians
    az = math.radians(random.uniform(-max_deg, max_deg))
    ax = math.radians(random.uniform(-max_deg * 0.2, max_deg * 0.2))
    ay = math.radians(random.uniform(-max_deg * 0.2, max_deg * 0.2))

    Rx = np.array(
        [[1, 0, 0], [0, math.cos(ax), -math.sin(ax)],
         [0, math.sin(ax), math.cos(ax)]],
        dtype=np.float32,
    )
    Ry = np.array(
        [[math.cos(ay), 0, math.sin(ay)], [0, 1, 0],
         [-math.sin(ay), 0, math.cos(ay)]],
        dtype=np.float32,
    )
    Rz = np.array(
        [[math.cos(az), -math.sin(az), 0],
         [math.sin(az), math.cos(az), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    return Rz @ Ry @ Rx


def apply_augmentations(points: np.ndarray, noise_std: float = 0.01, max_rotate_deg: float = 10.0) -> np.ndarray:
    """
    Apply light jitter and small rotation to normalized hand landmarks.
    points: (21, 3) numpy array
    """
    assert points.shape == (21, 3)
    # Small rotation
    R = random_rotation_matrix(max_rotate_deg)
    pts = (R @ points.T).T
    # Gaussian noise
    if noise_std > 0:
        pts = pts + np.random.normal(0.0, noise_std,
                                     size=pts.shape).astype(np.float32)
    return pts.astype(np.float32)
