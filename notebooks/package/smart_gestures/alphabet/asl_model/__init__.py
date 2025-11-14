"""
ASL Model Subpackage

This package contains a high level interface for performing predictions using
a pre-trained American Sign Language (ASL) alphabet model.

Available Components:
- ASLModel: A class that encapsulates the ASL alphabet model and its utilities.
- get_classes: Function to load the classes used by the model.
"""

# Import necessary components
from .model import ASLModel, get_classes

# Define the public API of the package
__all__ = [
    "ASLModel",
    "get_classes",
]
