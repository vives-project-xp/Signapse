"""
This module provides utility functions for handling and processing hand landmark data
for training machine learning models. It includes a custom PyTorch Dataset class for
hand landmarks, functions for data loading, normalization, augmentation, and splitting.

The main components are:
- LandmarksDataset: A PyTorch Dataset to handle landmark data.
- load_dataset_normalized: Loads and preprocesses landmark data from a JSON file.
- normalize_landmarks: Normalizes landmark coordinates.
- apply_augmentations: Applies random rotations and noise to landmark data.
- get_classes: Extracts the list of classes from the dataset file.
- split_dataset: Splits the dataset into training and validation sets.
- get_loaders: Creates PyTorch DataLoader instances for training and validation.
"""

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import random
import math

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
DATASET = DATA_DIR / "hand_landmarks.json"


class LandmarksDataset(Dataset):
    """
    A PyTorch Dataset for hand landmark data.

    This dataset stores hand landmark coordinates and their corresponding labels.
    It supports optional data augmentation, including noise and rotation, which
    can be applied on-the-fly during training.
    """

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
        """
        Initializes the LandmarksDataset.

        Args:
            X (np.ndarray): The input data, a numpy array of hand landmarks.
                            Shape can be (num_samples, 63) for flattened landmarks or
                            (num_samples, 21, 3) for 3D landmarks.
            y (np.ndarray): The corresponding labels for each sample.
            classes (list[str]): A list of class names.
            augment (bool, optional): Whether to apply augmentations. Defaults to False.
            augment_prob (float, optional): The probability of applying augmentations to a sample.
                                            Defaults to 0.5.
            noise_std (float, optional): The standard deviation of the noise to add for augmentation.
                                         Defaults to 0.01.
            rotate_deg (float, optional): The maximum degree of rotation to apply for augmentation.
                                          Defaults to 10.0.
        """
        if not (
            (X.ndim == 2 and X.shape[1] == 63)
            or (X.ndim == 3 and X.shape[1:] == (21, 3))
        ):
            raise AssertionError(
                "Expected X to have shape (num_samples, 63) or (num_samples, 21, 3) for 21 landmarks with (x,y,z) coords"
            )
        assert X.shape[0] == y.shape[0], "Number of samples in X and y must match"
        self.X = X
        self.y = y
        self.classes = classes
        self.is_flat = X.ndim == 2
        # Augmentation settings
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.rotate_deg = rotate_deg

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from the dataset at the given index.

        If augmentation is enabled, it may be applied based on the augmentation probability.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the landmark data as a torch.Tensor and the label
                   as a torch.Tensor.
        """
        x_np = self.X[idx].copy()
        x = torch.from_numpy(x_np).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def normalize_landmarks(lm, root_idx=0, scale_method="wrist_to_middle") -> np.ndarray:
    """
    Normalizes a set of 21 hand landmarks.

    This function centers the landmarks around a root point (typically the wrist)
    and scales them. The scaling can be based on the distance from the wrist to
    the middle finger MCP joint or the maximum distance between any two landmarks.

    Args:
        lm (list): A list of 21 landmark dictionaries, each with 'x', 'y', and 'z' keys.
        root_idx (int, optional): The index of the root landmark to center around.
                                  Defaults to 0 (wrist).
        scale_method (str, optional): The method for scaling. Can be "wrist_to_middle"
                                      or any other string to use the maximum inter-landmark
                                      distance. Defaults to "wrist_to_middle".

    Returns:
        np.ndarray: A numpy array of shape (21, 3) containing the normalized landmark coordinates.
    """
    arr = np.array(
        [[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in lm],
        dtype=np.float32,
    )
    if arr.shape[0] != 21:
        raise ValueError(f"Expected 21 landmarks per sample, got {arr.shape[0]}")
    root = arr[root_idx].copy()
    arr = arr - root

    if scale_method == "wrist_to_middle":
        scale = np.linalg.norm(arr[9])
    else:
        dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scale = dists.max()

    if scale <= 1e-6:
        scale = 1.0
    arr = arr / scale
    return arr


def load_dataset_normalized(
    json_file: str = str(DATASET),
    as_sequence: bool = True,
    scale_method: str = "wrist_to_middle",
    augment: bool = False,
    augment_prob: float = 0.5,
    noise_std: float = 0.01,
    rotate_deg: float = 10.0,
) -> LandmarksDataset:
    """
    Loads a landmark dataset from a JSON file and returns a normalized LandmarksDataset.

    Each line in the JSON file is expected to be a JSON object containing 'class' and
    'landmarks' keys.

    Args:
        json_file (str, optional): Path to the JSON file containing the dataset.
                                  If not provided, defaults to the module-level
                                  `DATASET` path.
        as_sequence (bool, optional): If True, returns landmarks as sequences of shape (21, 3).
                                      If False, returns flattened landmarks of shape (63,).
                                      Defaults to True.
        scale_method (str, optional): The normalization scaling method.
                                      See `normalize_landmarks` for details.
                                      Defaults to "wrist_to_middle".
        augment (bool, optional): Whether to enable augmentation in the returned dataset.
                                  Defaults to False.
        augment_prob (float, optional): Augmentation probability. Defaults to 0.5.
        noise_std (float, optional): Augmentation noise standard deviation. Defaults to 0.01.
        rotate_deg (float, optional): Augmentation maximum rotation in degrees. Defaults to 10.0.

    Returns:
        LandmarksDataset: An instance of the LandmarksDataset containing the loaded and
                          processed data.
    """
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df["class"] = df["class"].astype("category")

    X_list = [
        normalize_landmarks(lm, root_idx=0, scale_method=scale_method)
        for lm in df["landmarks"]
    ]
    y = np.array(df["class"].cat.codes, dtype=np.int64)
    classes = df["class"].cat.categories.tolist()

    if as_sequence:
        X = np.stack(X_list).astype(np.float32)
    else:
        X = np.stack(X_list).reshape(len(df), -1).astype(np.float32)

    return LandmarksDataset(
        X,
        y,
        classes,
        augment=augment,
        augment_prob=augment_prob,
        noise_std=noise_std,
        rotate_deg=rotate_deg,
    )


def get_classes() -> list[str]:
    """
    Extracts a sorted list of unique class names from the dataset JSON file.

    It reads the file specified by the global `DATASET` variable and parses it
    to find all unique values associated with the "class" key.

    Returns:
        list[str]: A sorted list of unique class names.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    classes_file = DATASET
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")

    with open(classes_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        if text[0] == "[":
            data = json.loads(text)
            if isinstance(data, list):
                classes = sorted(
                    {item.get("class") for item in data if "class" in item}
                )
                return [c for c in classes if c is not None]
    except json.JSONDecodeError:
        pass
    classes_set = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cls = obj.get("class")
        if cls is not None:
            classes_set.add(cls)

    return sorted(classes_set)


def split_dataset(
    dataset: LandmarksDataset, val_ratio: float = 0.2, random_seed: int = 42
):
    """
    Splits a dataset into training and validation sets.

    Args:
        dataset (LandmarksDataset): The dataset to split.
        val_ratio (float, optional): The proportion of the dataset to allocate to the
                                     validation set. Defaults to 0.2.
        random_seed (int, optional): The seed for the random number generator to ensure
                                     reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the training dataset and the validation dataset as
               PyTorch `Subset` objects.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset


def get_loaders(
    train_dataset: LandmarksDataset, val_dataset: LandmarksDataset, batch_size: int = 32
):
    """
    Creates DataLoader instances for training and validation datasets.

    Args:
        train_dataset (LandmarksDataset): The training dataset.
        val_dataset (LandmarksDataset): The validation dataset.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.

    Returns:
        tuple: A tuple containing the training DataLoader and the validation DataLoader.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def random_rotation_matrix(max_deg: float = 10.0) -> np.ndarray:
    """
    Generates a random 3D rotation matrix.

    The rotation is primarily around the z-axis, with smaller random rotations
    around the x and y axes.

    Args:
        max_deg (float, optional): The maximum rotation angle in degrees for the primary
                                   (z-axis) rotation. Defaults to 10.0.

    Returns:
        np.ndarray: A 3x3 numpy array representing the rotation matrix.
    """
    az = math.radians(random.uniform(-max_deg, max_deg))
    ax = math.radians(random.uniform(-max_deg * 0.2, max_deg * 0.2))
    ay = math.radians(random.uniform(-max_deg * 0.2, max_deg * 0.2))

    Rx = np.array(
        [[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]],
        dtype=np.float32,
    )
    Ry = np.array(
        [[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]],
        dtype=np.float32,
    )
    Rz = np.array(
        [[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    return Rz @ Ry @ Rx


def apply_augmentations(
    points: np.ndarray, noise_std: float = 0.01, max_rotate_deg: float = 10.0
) -> np.ndarray:
    """
    Applies a set of augmentations to a landmark point cloud.

    This includes a random rotation and the addition of Gaussian noise.

    Args:
        points (np.ndarray): The input landmark points, expected shape (21, 3).
        noise_std (float, optional): The standard deviation of the Gaussian noise to add.
                                     If 0, no noise is added. Defaults to 0.01.
        max_rotate_deg (float, optional): The maximum rotation angle in degrees.
                                          Defaults to 10.0.

    Returns:
        np.ndarray: The augmented landmark points, shape (21, 3).
    """
    assert points.shape == (21, 3)
    R = random_rotation_matrix(max_rotate_deg)
    pts = (R @ points.T).T
    if noise_std > 0:
        pts = pts + np.random.normal(0.0, noise_std, size=pts.shape).astype(np.float32)
    return pts.astype(np.float32)
