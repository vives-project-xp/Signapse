"""
VGT Model Subpackage

This package contains a high level interface for performing predictions using
a pre-trained Vlaams Gebaren Taal (VGT) alphabet model.

Available Components:
- VGTModel: A class that encapsulates the VGT alphabet model and its utilities.
- get_classes: Function to load the classes used by the model.
"""

# Import necessary components
from .model import VGTModel, get_classes

# Define the public API of the package
__all__ = [
    "VGTModel",
    "get_classes",
]
