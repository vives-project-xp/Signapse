import torch.optim as optim
import torch.nn as nn
import torch
from typing import Optional, Tuple, Any
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_utils import *
from data_utils import *
from callbacks import *


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    scheduler_type: Optional[str] = None,
    scheduler_kwargs: Optional[dict[str, Any]] = None,
    early_stopping_kwargs: Optional[dict[str, Any]] = None,
    checkpoint_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_kwargs = scheduler_kwargs or {}
    early_stopping_kwargs = early_stopping_kwargs or {}
    checkpoint_kwargs = checkpoint_kwargs or {}

    scheduler = create_scheduler(optimizer, scheduler_type, **scheduler_kwargs)
    early_stopper = None
    checkpoint = None

    if early_stopping_kwargs:
        early_stopper = EarlyStopping(**early_stopping_kwargs)
    if checkpoint_kwargs:
        checkpoint = ModelCheckpoint(**checkpoint_kwargs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_epoch_loss = running_loss / \
            len(train_loader.dataset)

        val_loss, val_acc = None, None
        if val_loader is not None:
            val_loss, val_acc = evaluate_metrics(model, val_loader, criterion)

        if scheduler is not None:
            if scheduler_type == "plateau":
                metric = val_loss if val_loss is not None else train_epoch_loss
                scheduler.step(metric)
            else:
                scheduler.step()

        if early_stopper is not None:
            monitor_value = val_loss if (val_loss is not None and early_stopper.mode == "min") else (
                val_acc if (val_acc is not None and early_stopper.mode ==
                            "max") else train_epoch_loss
            )
            stop = early_stopper.step(monitor_value, model=model)
            if stop and verbose:
                print(f"Early stopping at epoch {epoch+1}")

        if checkpoint is not None:
            metric_value = val_loss if (checkpoint.mode == "min" and val_loss is not None) else (
                val_acc if (checkpoint.mode ==
                            "max" and val_acc is not None) else train_epoch_loss
            )
            wrote = checkpoint.save(
                epoch=epoch+1, model=model, optimizer=optimizer, metric_value=metric_value)
            if wrote and verbose:
                print(f"Saved checkpoint to {checkpoint.filepath}")

        if verbose:
            if val_loss is not None and val_acc is not None:
                print(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_epoch_loss:.4f}")
                
        if early_stopper is not None and early_stopper.should_stop:
            break


def evaluate_model(model: nn.Module, dataloader: DataLoader[dict[str, Any]]) -> float:
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
    return accuracy


def evaluate_metrics(model: nn.Module, dataloader: DataLoader[dict[str, Any]], criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = criterion or nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    if was_training:
        model.train()
    return avg_loss, acc
