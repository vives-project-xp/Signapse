import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "images")
HAND_LANDMARKS_CSV = os.path.join(DATA_DIR, "hand_landmarks.csv")

class LandmarksDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, classes: list, preprocess=None):
        self.X = X
        self.y = y
        self.classes = classes
        self.preprocess = preprocess

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.preprocess:
            x = self.preprocess(x)
        return x, y
preprocess = lambda lm: np.array([[point['x'], point['y'], point['z']] for point in eval(lm)], dtype=np.float32)
    
def load_and_preprocess_dataset(csv_file: str) -> LandmarksDataset:
    """
    Load landmark data from CSV and preprocess it into a PyTorch Dataset.

    The CSV file should contain 'landmarks' and 'class' columns where:
    - 'landmarks' contains string representations of landmark coordinate lists
    - 'class' contains the label/category for each sample
    """
    df = pd.read_csv(csv_file)
    df['class'] = df['class'].astype('category')
    X = np.array(df['landmarks'].apply(preprocess).tolist())
    y = np.array(df['class'].cat.codes, dtype=np.int64)
    classes = df['class'].cat.categories.tolist()
    return LandmarksDataset(X, y, classes, preprocess=None)

# Get classes
def get_classes():
    return(pd.read_csv(HAND_LANDMARKS_CSV)['class'].astype('category').cat.categories.tolist())

# Split dataset into training and validation sets 80/20
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

# Create DataLoaders for training and validation sets
def get_loaders(train_dataset: LandmarksDataset, val_dataset: LandmarksDataset, batch_size: int = 32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader