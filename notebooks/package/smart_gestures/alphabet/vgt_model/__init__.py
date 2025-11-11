"""vgt_model package

This package contains code and models for the VGT-based gesture recognition.
"""

from .data import (
    get_classes,
    get_loaders,
    _normalize_landmarks,
)
from .model import (
    VGTModel as VGT
)

__all__ = [
    "get_classes",
    "get_loaders",
    "_normalize_landmarks",
    "VGT"
]
