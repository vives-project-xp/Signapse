"""
This module provides utility functions for training and evaluating PyTorch models
for sign language recognition, specifically adapted for sequential (LSTM) models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm

# Assuming these imports work and DEVICE is defined in model_utils
from data_utils import *
from model_utils import *


def train_model(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Trains a PyTorch LSTM model for a specified number of epochs.

    The model expects input shape (Batch Size, Sequence Length, Features).

    Args:
        model (nn.Module): The PyTorch model (LSTM) to train.
        dataloader (DataLoader): The DataLoader for the training data (sequences).
        epochs (int, optional): The number of epochs to train for. Defaults to 50.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        # Reset hidden state at the start of each epoch for stateful LSTMs
        # If the LSTM is stateless (common in sequence classification), this line is optional.
        # if hasattr(model, 'hidden'): model.hidden = model.init_hidden() 
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Inputs shape: (Batch Size, N_frames, N_features)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            # --- KEY CHANGE: Pass the sequence (inputs) directly to the model ---
            # The model will internally handle the sequence dimension (N_frames).
            outputs = model(inputs) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# ----------------------------------------------------------------------

def evaluate_model(
    model: nn.Module, dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]]
):
    """
    Evaluates a PyTorch LSTM model on a given dataset.

    Args:
        model (nn.Module): The PyTorch model (LSTM) to evaluate.
        dataloader (DataLoader): The DataLoader for the validation data (sequences).

    Returns:
        float: The accuracy of the model in percent.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Inputs shape: (Batch Size, N_frames, N_features)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # --- KEY CHANGE: Pass the sequence (inputs) directly to the model ---
            outputs = model(inputs) 
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy