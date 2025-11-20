import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import cast, Callable, List, Tuple
# Note: torch.nn.utils.rnn.pad_sequence is not needed here since we are implementing padding manually.

# --- Configuration Adjustment ---
THIS_DIR = Path(__file__).parent

# Data is currently located at: SMARTGLASSES/notebooks/package/smart_gestures/gestures/lstm_model/dataset
# Calculate the default path relative to data_utils.py (which is in SMARTGLASSES/notebooks/training/lstm_model/)
DEFAULT_DATA_PATH = (THIS_DIR / "data")
print(DEFAULT_DATA_PATH)

# DATA_PATH uses the environment variable if available, otherwise it uses the corrected default path.
DATA_PATH = Path(os.getenv("LAKEFS_DATA_PATH")) if os.getenv("LAKEFS_DATA_PATH") else DEFAULT_DATA_PATH

# --- Global Configuration for Sequence Handling ---
# Increased to 40 frames to avoid truncation of the longest sequences
SEQUENCE_LENGTH = 40 


# --- Data Normalization Utility (Sequence-aware) ---

def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """
    Normalizes a sequence of keypoints (frames) for relative movement and scale 
    using the hand landmarks from the first frame as the anchor.
    
    The input sequence shape is (N_frames, Total_Coords).
    """
    if sequence.ndim != 2:
        raise ValueError("Sequence must be 2D: (N_frames, Total_Coords)")

    N_frames = sequence.shape[0]
    # Keypoint structure: [POSE (33*4=132), LH (21*3=63), RH (21*3=63)] -> Total 258
    pose_size = 33 * 4 
    
    # Extract only the hand keypoints (last 126 coordinates)
    hand_keypoints_3d = sequence[:, pose_size:].reshape(N_frames, -1, 3)
    
    if hand_keypoints_3d.shape[1] < 2:
         # No hands detected in the sequence, skip normalization
         return sequence 

    # --- 1. Centering (using the first hand's wrist in the first frame) ---
    # The first hand's wrist is landmark index 0 of the first hand (Left Hand, index 0).
    root_coords = hand_keypoints_3d[0, 0, :3]
    
    # Center all hand coordinates across all frames relative to the root_coords
    centered_hand_keypoints = hand_keypoints_3d - root_coords

    # --- 2. Scaling (using wrist-to-middle finger base distance) ---
    # Left hand middle finger base is at index 9 (0-indexed)
    # Scale is calculated from the first frame's relative distance
    wrist_to_middle_dist = np.linalg.norm(centered_hand_keypoints[0, 9])
    
    scale = wrist_to_middle_dist
    if scale <= 1e-6:
        scale = 1.0 

    normalized_hand_keypoints = centered_hand_keypoints / scale
    
    # Flatten the hands back to (N_frames, 126)
    normalized_hands_flattened = normalized_hand_keypoints.reshape(N_frames, -1)
    
    # Reconstruct the full sequence: [Original POSE, Normalized Hands]
    normalized_sequence = np.concatenate([sequence[:, :pose_size], normalized_hands_flattened], axis=1)

    return normalized_sequence


# --- PyTorch Dataset for Sequence Data ---

class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    A PyTorch Dataset for sign language sequence data, handling variable-length 
    keypoint sequences for LSTM training.
    """
    def __init__(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        classes: List[str],
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.X = X
        self.y = y
        self.classes = classes
        self.preprocess = preprocess

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sequence from the dataset.
        Returns: (N_frames, N_features) tensor and the label tensor.
        """
        x = self.X[idx]
        y = self.y[idx]
        
        if self.preprocess:
            x = self.preprocess(x)
            
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.tensor(int(y), dtype=torch.long)
        
        return x_tensor, y_tensor


# --- Data Loading and Collection Utilities ---

def load_sequences(data_path: Path) -> tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Collects all sequences (keypoints) and their corresponding labels from 
    the nested directory structure: DATA_PATH / 'gesture_name' / 'sequence_num' / 'timestamp' / 'keypoints_*.npy'
    """
    X_sequences = []
    y_labels = []
    
    # 1. Get unique action/gesture names (directories)
    # Use os.scandir for efficiency
    gesture_dirs = sorted([d.name for d in os.scandir(data_path) if d.is_dir()])
    classes = gesture_dirs
    
    print(f"Found {len(classes)} gestures: {classes}")

    for class_idx, gesture_name in enumerate(classes):
        gesture_path = data_path / gesture_name
        
        # 2. Iterate through all sequence numbers
        for sequence_num_dir in os.listdir(gesture_path):
            sequence_num_path = gesture_path / sequence_num_dir
            if not os.path.isdir(sequence_num_path) or not sequence_num_dir.isdigit():
                continue

            # 3. Iterate through all timestamp folders (just taking the first/only one)
            timestamp_dirs = sorted([d.name for d in os.scandir(sequence_num_path) if d.is_dir()])
            if not timestamp_dirs:
                 continue
            
            # Assuming one timestamp folder per sequence folder
            sequence_dir = sequence_num_path / timestamp_dirs[0]
            
            # 4. Load all keypoints files for this sequence
            keypoint_files = sorted([f for f in os.listdir(sequence_dir) if f.startswith("keypoints_") and f.endswith(".npy")])
            
            if len(keypoint_files) > 0:
                # Load all frames and stack them into a single sequence array
                sequence_frames = [np.load(sequence_dir / f) for f in keypoint_files]
                full_sequence = np.stack(sequence_frames) # Shape (N_frames, Total_Coords)
                
                X_sequences.append(full_sequence)
                y_labels.append(class_idx)

    return X_sequences, np.array(y_labels, dtype=np.int64), classes


def load_dataset_for_lstm() -> SequenceDataset:
    """
    Loads all sign language sequences from the configured DATA_PATH.
    """
    # The data_path is now retrieved from the module's global configuration
    X, y, classes = load_sequences(DATA_PATH)
    
    return SequenceDataset(X, y, classes, preprocess=normalize_keypoints)


def split_dataset(
    dataset: SequenceDataset, val_ratio: float = 0.2, random_seed: int = 42
) -> tuple[SequenceDataset, SequenceDataset]:
    """
    Splits a SequenceDataset into training and validation sets.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    if train_size + val_size != total_size:
        train_size = total_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    
    train_dataset, val_dataset = cast(
        tuple[SequenceDataset, SequenceDataset],
        random_split(dataset, [train_size, val_size], generator=generator),
    )

    return train_dataset, val_dataset


# --- Collate Function for Padding and Lengths ---

def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads the sequences in the batch to SEQUENCE_LENGTH (40 frames), truncates long sequences, 
    and returns a tensor of the original sequence lengths.
    
    Returns: (X_batch, y_batch, lengths_batch)
    """
    # 1. Separate inputs (X) and labels (y)
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    padded_sequences = []
    # Store the original length of the sequence before padding/truncation
    sequence_lengths = [] 
    
    for seq in sequences:
        N_frames, N_features = seq.shape
        
        if N_frames > SEQUENCE_LENGTH:
            # Truncate long sequences
            padded_seq = seq[:SEQUENCE_LENGTH, :]
            sequence_lengths.append(SEQUENCE_LENGTH)
        elif N_frames < SEQUENCE_LENGTH:
            # Pad short sequences with zeros
            padding = torch.zeros((SEQUENCE_LENGTH - N_frames, N_features), dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
            sequence_lengths.append(N_frames) # Use the original length
        else:
            # Sequence is already the correct length
            padded_seq = seq
            sequence_lengths.append(SEQUENCE_LENGTH)

        padded_sequences.append(padded_seq)

    # Convert sequence lengths to tensor
    # NOTE: The lengths must be sorted by decreasing length for nn.utils.rnn.pack_padded_sequence 
    # if you choose to implement that later. For now, we return them unsorted.
    lengths_batch = torch.tensor(sequence_lengths, dtype=torch.int64)
    X_batch = torch.stack(padded_sequences)
    y_batch = torch.stack(labels)
    
    # Return the lengths tensor for use in masking/packing later in the model
    return X_batch, y_batch, lengths_batch


def get_loaders(
    train_dataset: SequenceDataset, val_dataset: SequenceDataset, batch_size: int = 32
):
    """
    Creates DataLoader instances using a custom collate_fn to handle padding and return sequence lengths.
    """
    # Pass the custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=pad_collate
    )
    return train_loader, val_loader