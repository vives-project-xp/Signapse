import os
import json
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from typing import cast, List, Tuple, Dict

# Constant paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "dataset")
GESTURE_MAP_PATH = os.path.join(THIS_DIR, "gesture_map.json")
FEATURE_SIZE = 258  # 33 pose * 4 + 21 left hand * 3 + 21 right hand * 3


# Dataset class for LSTM model
class GestureDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        classes: List[str],
    ):
        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must match")
        self.X = X
        self.y = torch.from_numpy(y).long()
        self.classes = classes

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_data = self.X[idx]
        y_label = self.y[idx]
        return torch.from_numpy(x_data).float(), y_label


# Get the gesture map
def get_gesture_map() -> Dict[str, int]:
    with open(GESTURE_MAP_PATH, "r") as f:
        gesture_map: Dict[str, int] = json.load(f)
    return gesture_map

# Get the classes from the gesture map
def get_classes() -> List[str]:
    gesture_map = get_gesture_map()
    classes = [""] * len(gesture_map)
    for name, idx in gesture_map.items():
        classes[idx] = name
    return classes

# Function to load dataset and preprocess sequences
def load_and_preprocess_dataset(
    data_path: str = DATA_DIR,
    gesture_map_path: str = GESTURE_MAP_PATH,
    feature_size: int = FEATURE_SIZE,
) -> GestureDataset:
    try:
        gesture_map = get_gesture_map()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Gesture map file not found at {gesture_map_path}")

    actions = list(gesture_map.keys())
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    print("Loading data from:", data_path)

    for action in actions:
        action_dir = os.path.join(data_path, action)
        if not os.path.isdir(action_dir):
            print(
                f"Warning: Action directory '{action_dir}' does not exist, skipping.")
            continue
        seq_folders = sorted(
            [
                d
                for d in os.listdir(action_dir)
                if d.isdigit() and os.path.isdir(os.path.join(action_dir, d))
            ],
            key=int,
        )

        for seq_folder in seq_folders:
            seq_path = os.path.join(action_dir, seq_folder)
            print(f"Processing sequence folder: {seq_path}")

            keypoint_pattern = os.path.join(seq_path, "**", "keypoints_*.npy")
            keypoint_files = glob.glob(keypoint_pattern, recursive=True)

            # Sort by numeric index extracted from filename (keypoints_<index>.npy)
            def _extract_index(path: str) -> int:
                base = os.path.basename(path)
                try:
                    return int(base.split("_")[-1].split(".")[0])
                except Exception:
                    return -1

            keypoint_files = sorted(keypoint_files, key=_extract_index)

            window: List[np.ndarray] = []
            # Load each keypoints file for this sequence
            for kf in keypoint_files:
                kf_path = kf
                try:
                    res = np.load(kf_path)
                    if res.shape[0] != feature_size:
                        print(
                            f"Warning: Unexpected feature size in {kf}, expected {feature_size}, got {res.shape[0]}. Skipping frame."
                        )
                        continue
                    window.append(res)
                except Exception as e:
                    print(f"Error loading {kf}: {e}. Skipping frame.")

            # If no valid frames were found for this sequence, skip it
            if not window:
                print(
                    f"Warning: No valid frames found in sequence {seq_path}, skipping sequence."
                )
                continue

            sequences.append(np.array(window, dtype=np.float32))
            labels.append(gesture_map[action])

    if not sequences:
        print("No sequences loaded. Please check the data directory and gesture map.")
        return GestureDataset([], np.array([]), [])

    y = np.array(labels, dtype=np.int64)
    classes = [""] * len(gesture_map)
    for name, idx in gesture_map.items():
        classes[idx] = name

    print(f"Loading complete. Loaded {len(sequences)} sequences.")
    return GestureDataset(sequences, y, classes)


# Function to split dataset into train, val, test
def split_dataset(
    dataset: GestureDataset,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[GestureDataset, GestureDataset, GestureDataset]:
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )
    return train_set, val_set, test_set

# Collate function for DataLoader
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0]
                           for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels_tensor = torch.stack(labels)
    return padded_sequences, labels_tensor, lengths

# Function to get DataLoaders
def get_loaders(
    train_dataset: GestureDataset,
    val_dataset: GestureDataset,
    test_dataset: GestureDataset,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader

# Example usage fro the functions in the model
def example_usage():
    classes = get_classes()
    print("Classes:", classes)
    dataset = load_and_preprocess_dataset()
    print(f"Total sequences loaded: {len(dataset)}")
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}, Test size: {len(test_set)}")
    train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set)
    print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}, Test loader batches: {len(test_loader)}")
# Uncomment to run example usage
# example_usage()
