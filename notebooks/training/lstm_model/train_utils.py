"""
This module provides utility functions for training and evaluating PyTorch models
for sign language recognition, specifically adapted for sequential (LSTM) models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# The DataLoader now returns three elements (inputs, labels, lengths)
from typing import Tuple, List 
from tqdm import tqdm

# Assuming these imports work and DEVICE is defined in model_utils
from data_utils import *
from model_utils import *


# NOTE: The sort_batch_by_length helper function is removed because
# the sorting logic is now handled internally by GestureLSTM.forward()


# Update DataLoader tuple type to reflect the 3 outputs (X, y, length)
def train_model(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
    epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Trains a PyTorch LSTM model for a specified number of epochs, handling padding via packing.

    The model expects input shape (Batch Size, Sequence Length, Features) and sequence lengths.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        # CRITICAL CHANGE 1: Unpack 3 items (inputs, labels, lengths)
        for inputs, labels, lengths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            
            # NOTE: Sorting is now handled inside model_utils.py

            # 2. Move data to device (Note: lengths is kept on CPU/moved by the model internally)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            # 3. Pass inputs AND sequence lengths to the model
            outputs = model(inputs, lengths) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0) 
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# ----------------------------------------------------------------------

# Update DataLoader tuple type to reflect the 3 outputs (X, y, length)
def evaluate_model(
    model: nn.Module, dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Evaluates a PyTorch LSTM model on a given dataset, handling padding via packing.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # CRITICAL CHANGE 1: Unpack 3 items (inputs, labels, lengths)
        for inputs, labels, lengths in dataloader:
            
            # NOTE: Sorting is now handled inside model_utils.py

            # 2. Move data to device
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # 3. Pass inputs AND sequence lengths to the model
            outputs = model(inputs, lengths) 
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy