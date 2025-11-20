import json 
import os
from pathlib import Path
from typing import cast, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Constants for model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_DIM = 258
HIDDEN_DIM = 128
NUM_LAYERS = 2
SEQUENCE_LENGTH = 40

# Paths
THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR.parent.parent.parent / "data" 
MODEL_DIR = THIS_DIR.parent.parent.parent / "models"

CLASSES_MAP_PATH = DATA_DIR / "class_map.json"
MODEL_FILE = MODEL_DIR / "lstm_model.pth"

def normalize_landmarks(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 2:
        raise ValueError("Input sequence must be a 2D array")
    
    N_frames = sequence.shape[0]
    pose_size = 33 * 4

    hand_keypoints_3d = sequence[:, pose_size:].reshape(N_frames, -1, 3)

    if hand_keypoints_3d.shape[1] < 2:
        return sequence
    
    root_coords = hand_keypoints_3d[0, 0, :3]
    centered_hand_keypoints = hand_keypoints_3d - root_coords

    wrist_to_middle_dist = np.linalg.norm(centered_hand_keypoints[0, 9])
    
    scale = wrist_to_middle_dist
    if scale < 1e-6:
        scale = 1.0

    normalize_landmarks = centered_hand_keypoints / scale
    normalize_landmarks_flat = normalize_landmarks.reshape(N_frames, -1)
    normalized_sequence = np.concatenate([sequence[:, :pose_size], normalize_landmarks_flat], axis=1)

    return normalized_sequence

class LSTMModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        sorted_lengths, sorted_idx = sorted_lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx]

        packed_output, _ = self.lstm(packed_output, batch_first=True)

        last_time_step = output[torch.arange(output.size(0)), ]
