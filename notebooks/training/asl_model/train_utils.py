"""
This module provides utility functions for training and evaluating PyTorch models
for sign language recognition. It includes functions for training a model and
evaluating its performance on a validation set.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm

from data_utils import *
from model_utils import *


def train_model(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Trains a PyTorch model for a specified number of epochs.

    This function iterates over the dataset for a given number of epochs,
    calculates the loss, and updates the model's weights.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): The DataLoader for the training data.
        epochs (int, optional): The number of epochs to train for. 
            Defaults to 50.
        lr (float, optional): The learning rate for the optimizer. 
            Defaults to 1e-3.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")


def evaluate_model(
    model: nn.Module, dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]]
):
    """
    Evaluates a PyTorch model on a given dataset.

    This function calculates the accuracy of the model on the provided data.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): The DataLoader for the validation data.

    Returns:
        float: The accuracy of the model in percent.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs.view(inputs.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
