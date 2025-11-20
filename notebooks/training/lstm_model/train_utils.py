"""
This module provides utility functions for training and evaluating PyTorch models
for sign language recognition, specifically adapted for sequential (LSTM) models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, List 
from tqdm import tqdm

# Assuming these imports work and DEVICE is defined in model_utils
from data_utils import *
from model_utils import *

def train_model(
    model: nn.Module,
    dataloader: DataLoader, 
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
):
    """
    Trains a PyTorch LSTM model for a specified number of epochs.
    Includes a Learning Rate Scheduler.
    """
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with L2 Regularization (Weight Decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # SCHEDULER: Reduces LR by a factor of 0.5 if loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        for inputs, labels, lengths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            # Pass inputs AND sequence lengths to the model
            outputs = model(inputs, lengths) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Optional: Gradient Clipping (Highly recommended for LSTMs)
            # Prevents "exploding gradients" which LSTMs are prone to
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_size = inputs.size(0) 
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        
        # Step the scheduler based on the loss
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
        
        # Update the learning rate if loss has plateaued
        scheduler.step(epoch_loss)

# ----------------------------------------------------------------------

def evaluate_model(
    model: nn.Module, dataloader: DataLoader
):
    """
    Evaluates a PyTorch LSTM model on a given dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs, lengths) 
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy