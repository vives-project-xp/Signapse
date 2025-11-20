"""
This module provides utilities for loading, preprocessing, and handling landmark data 
for sign language recognition models. It includes a custom PyTorch Dataset for 
hand landmarks, functions for data normalization, and data loading utilities.
"""
import json
import os
from pathlib import Path
from typing import cast, Callable

import pandas as pd
import numpy as np
import ast
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split

THIS_DIR = Path(__file__).parent
CLASSES_FILE = THIS_DIR / "data" / "classes.json"
HAND_LANDMARKS_CSV = THIS_DIR / "data" / "hand_landmarks.csv"


class LandmarksDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    A PyTorch Dataset for hand landmark data.

    Args:
        X (np.ndarray): A numpy array of landmark data.
        y (np.ndarray): A numpy array of labels.
        classes (list[str]): A list of class names.
        preprocess (Callable[[np.ndarray], np.ndarray] | None, optional): 
            A function to preprocess the landmark data. Defaults to None.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: list[str],
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    ):

        self.X = X
        self.y = y
        self.classes = classes
        self.preprocess = preprocess

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[np.ndarray, int]: A tuple containing the preprocessed landmark 
                                    data as a tensor and the corresponding label.
        """
        x = self.X[idx]
        y = self.y[idx]
        if self.preprocess:
            x = self.preprocess(x)
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.tensor(int(y), dtype=torch.long)
        return x_tensor, y_tensor


def preprocess(lm: str) -> np.ndarray:
    """
    Parses a string of landmarks into a numpy array.

    The string can be in JSON format or a Python literal. It also handles
    a special case where the string represents a list repeated multiple times,
    e.g., '[...]*10'.

    Args:
        lm (str): The string of landmarks to parse.

    Returns:
        np.ndarray: A numpy array of the parsed landmarks.
    
    Raises:
        ValueError: If the string cannot be parsed.
    """
    parsed = None
    try:
        parsed = json.loads(lm)
    except Exception:
        try:
            parsed = ast.literal_eval(lm)
        except Exception:
            mult_match = re.match(r"^\s*(\[[\s\S]*\])\s*\*\s*(\d+)\s*$", lm)
            if mult_match:
                inner = mult_match.group(1)
                count = int(mult_match.group(2))
                try:
                    inner_list = ast.literal_eval(inner)
                    parsed = inner_list * count
                except Exception as exc2:
                    raise ValueError(
                        "Could not parse repeated landmarks list; inner part failed to parse"
                    ) from exc2
            else:
                raise ValueError(
                    "Could not parse landmarks string: expected JSON or Python literal"
                )

    return np.array(
        [[point["x"], point["y"], point["z"]] for point in parsed], dtype=np.float32
    )


def load_and_preprocess_dataset(csv_file: str) -> LandmarksDataset:
    """
    Loads a dataset from a CSV file and preprocesses it.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        LandmarksDataset: The preprocessed dataset.
    """
    df = pd.read_csv(csv_file)
    df["class"] = df["class"].astype("category")
    X = np.array(df["landmarks"].apply(preprocess).tolist())
    y = np.array(df["class"].cat.codes, dtype=np.int64)
    classes = df["class"].cat.categories.tolist()
    return LandmarksDataset(X, y, classes, preprocess=None)


def get_classes() -> list[str]:
    """
    Loads the class names from the classes.json file.

    Returns:
        list[str]: A list of class names.
    """
    return json.load(open(CLASSES_FILE, "r", encoding="utf-8"))


def normalize_landmarks(lm, root_idx=0, scale_method="wrist_to_middle") -> np.ndarray:
    """
    Normalizes a set of landmarks.

    This function centers the landmarks around a root point (usually the wrist)
    and scales them based on a specified method.

    Args:
        lm: The landmarks to normalize.
        root_idx (int, optional): The index of the root landmark. Defaults to 0.
        scale_method (str, optional): The method to use for scaling. 
            Can be 'wrist_to_middle' or another value which will use the maximum 
            distance between any two points. Defaults to "wrist_to_middle".

    Returns:
        np.ndarray: The normalized landmarks.
        
    Raises:
        ValueError: If the number of landmarks is not 21.
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


def split_dataset(
    dataset: LandmarksDataset, val_ratio: float = 0.2, random_seed: int = 42
) -> tuple[LandmarksDataset, LandmarksDataset]:
    """
    Splits a dataset into training and validation sets.

    Args:
        dataset (LandmarksDataset): The dataset to split.
        val_ratio (float, optional): The ratio of the dataset to use for 
            validation. Defaults to 0.2.
        random_seed (int, optional): The random seed for the split. Defaults to 42.

    Returns:
        tuple[LandmarksDataset, LandmarksDataset]: A tuple containing the training 
                                                    and validation datasets.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = cast(
        tuple[LandmarksDataset, LandmarksDataset],
        random_split(dataset, [train_size, val_size], generator=generator),
    )

    return train_dataset, val_dataset


def get_loaders(
    train_dataset: LandmarksDataset, val_dataset: LandmarksDataset, batch_size: int = 32
):
    """
    Creates DataLoader instances for the training and validation datasets.

    Args:
        train_dataset (LandmarksDataset): The training dataset.
        val_dataset (LandmarksDataset): The validation dataset.
        batch_size (int, optional): The batch size for the loaders. Defaults to 32.

    Returns:
        A tuple containing the training and validation DataLoader instances.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
