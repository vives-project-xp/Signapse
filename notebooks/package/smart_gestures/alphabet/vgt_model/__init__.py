"""vgt_model package

This package contains code and models for the VGT-based gesture recognition.
"""

from .data_utils import (
    get_classes,
    get_loaders,
    normalize_landmarks,
)
from .model_utils import (
    VGTModel,
    create_model,
    load_model,
    DEVICE,
    MODEL_DIR,
)
from .train_utils import (
    train_model,
    evaluate_model,
)

__all__ = [
    "get_classes",
    "get_loaders",
    "normalize_landmarks",
    "VGTModel",
    "create_model",
    "load_model",
    "DEVICE",
    "MODEL_DIR",
    "train_model",
    "evaluate_model", 
]
