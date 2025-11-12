# Smart Gestures

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python package for sign language alphabet recognition using PyTorch and MediaPipe hand tracking.

## Overview

Smart Gestures provides PyTorch-based models and utilities for training and deploying sign language alphabet recognition systems. The package includes:

- **ASL** (American Sign Language) alphabet recognition
- **VGT** (Vlaamse Gebarentaal / Flemish Sign Language) alphabet recognition
- **LSTM-based** gesture recognition (experimental)

## Features

- 🤖 **Pre-trained models** for ASL and VGT alphabets
- 📊 **Data loading utilities** with built-in augmentation support
- 🎯 **Training utilities** with callbacks (early stopping, model checkpoints, learning rate scheduling)
- 🔧 **Model utilities** for creating, loading, and evaluating models
- 📈 **Real-time inference** support with MediaPipe hand landmarks
- 🎨 **Data augmentation** (rotation, noise, scaling)

## Installation

### From PyPI (Recommended)

```bash
pip install smart-gestures
```

### From Source

```bash
pip install git+https://github.com/vives-project-xp/SmartGlasses.git#subdirectory=notebooks/package
```

### Development Installation

```bash
git clone https://github.com/vives-project-xp/SmartGlasses.git
cd SmartGlasses/notebooks/package
pip install -e .
```

### Requirements

- Python 3.12+
- PyTorch 2.9.0+
- MediaPipe 0.10.14+
- NumPy 2.3.4+
- Pandas 2.3.3+
- tqdm 4.67.1+

## Quick Start

### Import the Package

```python
from smart_gestures.alphabet import asl_model, vgt_model
```

### Get Available Classes

```python
# ASL alphabet classes
asl_classes = asl_model.get_classes()
print(f"ASL classes: {asl_classes}")

# VGT alphabet classes
vgt_classes = vgt_model.get_classes()
print(f"VGT classes: {vgt_classes}")
```

### Load a Pre-trained Model

```python
import torch
from smart_gestures.alphabet.asl_model import create_model, load_model, get_classes, DEVICE

# Get classes
classes = get_classes()
num_classes = len(classes)

# Create model architecture
model = create_model(num_classes=num_classes, in_dim=63)

# Load weights
model_path = "path/to/hand_gesture_model.pth"
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
```

### Make Predictions

```python
import numpy as np
import torch

# Prepare input: 21 landmarks with x, y, z coordinates
landmarks = np.random.rand(21, 3).astype(np.float32)  # Replace with actual landmarks
input_tensor = torch.from_numpy(landmarks.reshape(1, 63)).to(DEVICE)

# Predict
with torch.no_grad():
    logits = model(input_tensor)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    predicted_class = classes[pred_idx]
    
print(f"Prediction: {predicted_class}")
```

### Normalize Landmarks (VGT Model)

```python
from smart_gestures.alphabet.vgt_model import normalize_landmarks

# Raw landmarks from MediaPipe (list of dicts with x, y, z)
raw_landmarks = [{"x": 0.5, "y": 0.3, "z": 0.1}, ...]  # 21 landmarks

# Normalize (wrist-to-middle finger scaling)
normalized = normalize_landmarks(raw_landmarks, method="wrist_to_middle")
```

## Training a Model

### ASL Model Training

```python
from smart_gestures.alphabet.asl_model import (
    get_classes, 
    get_loaders,
    create_model,
    train_model,
    evaluate_model
)
from smart_gestures.alphabet.asl_model.data_utils import (
    load_and_preprocess_dataset,
    split_dataset,
    HAND_LANDMARKS_CSV
)
from smart_gestures.alphabet.asl_model.model_utils import save_model

# Load data
classes = get_classes()
dataset = load_and_preprocess_dataset(HAND_LANDMARKS_CSV)
train_dataset, val_dataset = split_dataset(dataset, val_ratio=0.2, random_seed=42)
train_loader, val_loader = get_loaders(train_dataset, val_dataset, batch_size=32)

# Create model
in_dim = 63  # 21 landmarks * 3 coordinates
num_classes = len(classes)
model = create_model(num_classes, in_dim)

# Train
train_model(model, train_loader, epochs=20, lr=1e-3)

# Evaluate
accuracy = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {accuracy:.2f}%")

# Save
save_model(model, path="my_model.pth")
```

### VGT Model Training (Advanced)

```python
from smart_gestures.alphabet.vgt_model import (
    create_model,
    train_model,
    evaluate_model
)
from smart_gestures.alphabet.vgt_model.data_utils import (
    load_dataset_normalized,
    split_dataset,
    get_loaders,
    get_classes,
    HAND_LANDMARKS_JSON
)
from smart_gestures.alphabet.vgt_model.model_utils import save_model

# Load normalized dataset with augmentation
dataset = load_dataset_normalized(
    HAND_LANDMARKS_JSON,
    as_sequence=False,
    scale_method="wrist_to_middle",
    augment=True,
    augment_prob=0.5,
    noise_std=0.02,
    rotate_deg=15
)

train_dataset, val_dataset = split_dataset(dataset, val_ratio=0.2, random_seed=42)
train_loader, val_loader = get_loaders(train_dataset, val_dataset, batch_size=32)

# Create model
classes = get_classes()
model = create_model(num_classes=len(classes), in_dim=63)

# Train with callbacks
train_model(
    model,
    train_loader,
    val_loader=val_loader,
    epochs=50,
    lr=1e-3,
    scheduler_type="plateau",
    scheduler_kwargs={"factor": 0.5, "patience": 5},
    early_stopping_kwargs={"patience": 10, "min_delta": 0.001},
    checkpoint_kwargs={"filepath": "checkpoints/best_model.pth"}
)

# Evaluate
accuracy = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {accuracy:.2f}%")
```

## Command-Line Training

Both ASL and VGT models include command-line training scripts:

### ASL Training

```bash
python -m smart_gestures.alphabet.asl_model.run_training \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.001 \
    --output models/hand_gesture_model.pth
```

### VGT Training

```bash
python -m smart_gestures.alphabet.vgt_model.run_training \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --augment \
    --augment_prob 0.5 \
    --scheduler plateau \
    --early_stopping \
    --output models/hand_gesture_model.pth
```

## Package Structure

```
smart_gestures/
├── __init__.py
├── alphabet/
│   ├── __init__.py
│   ├── const.py
│   ├── asl_model/
│   │   ├── __init__.py
│   │   ├── data_utils.py      # Data loading and preprocessing
│   │   ├── model_utils.py     # Model architecture and utilities
│   │   ├── train_utils.py     # Training and evaluation functions
│   │   ├── run_training.py    # CLI training script
│   │   ├── data/              # Dataset files
│   │   └── models/            # Saved model checkpoints
│   └── vgt_model/
│       ├── __init__.py
│       ├── data_utils.py      # Data loading with normalization
│       ├── model_utils.py     # Model architecture
│       ├── train_utils.py     # Training with callbacks
│       ├── callbacks.py       # Training callbacks
│       ├── run_training.py    # CLI training script
│       ├── test_camera.py     # Real-time testing utility
│       ├── data/              # Dataset files
│       ├── dataset/           # Raw dataset
│       └── models/            # Saved model checkpoints
└── gestures/
    └── lstm_model/            # LSTM-based gesture recognition (experimental)
```

## API Reference

### ASL Model

```python
from smart_gestures.alphabet import asl_model

# Data utilities
classes = asl_model.get_classes()
train_loader, val_loader = asl_model.get_loaders(train_dataset, val_dataset, batch_size=32)

# Model utilities
model = asl_model.create_model(num_classes=26, in_dim=63)
model = asl_model.load_model(path="model.pth", num_classes=26, in_dim=63)

# Training utilities
asl_model.train_model(model, train_loader, epochs=20, lr=1e-3)
accuracy = asl_model.evaluate_model(model, val_loader)

# Device
device = asl_model.DEVICE  # 'cuda' if available, else 'cpu'
```

### VGT Model

```python
from smart_gestures.alphabet import vgt_model

# Data utilities
classes = vgt_model.get_classes()
train_loader, val_loader = vgt_model.get_loaders(train_dataset, val_dataset, batch_size=32)
normalized = vgt_model.normalize_landmarks(landmarks, method="wrist_to_middle")

# Model utilities
model = vgt_model.create_model(num_classes=26, in_dim=63)
model = vgt_model.load_model(path="model.pth", num_classes=26, in_dim=63)

# Training utilities (with callbacks)
vgt_model.train_model(
    model, train_loader, val_loader=val_loader, 
    epochs=50, lr=1e-3,
    scheduler_type="plateau",
    early_stopping_kwargs={"patience": 10}
)
accuracy = vgt_model.evaluate_model(model, val_loader)

# Device and paths
device = vgt_model.DEVICE
model_dir = vgt_model.MODEL_DIR
```

## Data Format

### Input Format

Models expect hand landmarks in the following format:

```python
# 21 hand landmarks with x, y, z coordinates
landmarks = [
    {"x": 0.5, "y": 0.3, "z": 0.1},
    {"x": 0.6, "y": 0.4, "z": 0.2},
    # ... 21 landmarks total
]

# Or as numpy array: shape (21, 3)
landmarks_array = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # (21, 3)

# Flattened for model input: shape (1, 63)
model_input = landmarks_array.reshape(1, 63)
```

### Dataset Files

- **ASL**: CSV file at `smart_gestures/alphabet/asl_model/data/hand_landmarks.csv`
- **VGT**: JSON file at `smart_gestures/alphabet/vgt_model/data/hand_landmarks.json`

## Usage in Production

### FastAPI Integration Example

```python
from fastapi import APIRouter, HTTPException
import numpy as np
import torch
from smart_gestures.alphabet.asl_model import create_model, get_classes, DEVICE

classes = get_classes()
model = create_model(num_classes=len(classes), in_dim=63)
model.load_state_dict(torch.load("path/to/model.pth", map_location=DEVICE))
model.eval()

router = APIRouter()

@router.post("/predict")
async def predict(landmarks: list[dict]):
    """
    Predict sign language letter from hand landmarks.
    
    Args:
        landmarks: List of 21 hand landmarks with x, y, z coordinates
        
    Returns:
        Predicted letter
    """
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)
    
    if pts.shape != (21, 3):
        raise HTTPException(status_code=400, detail="Expected 21 landmarks")
    
    x = torch.from_numpy(pts.reshape(1, 63)).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        predicted_letter = classes[pred_idx]
    
    return {"prediction": predicted_letter}
```

### Flask Integration Example

```python
from flask import Flask, request, jsonify
import numpy as np
import torch
from smart_gestures.alphabet.vgt_model import create_model, get_classes, normalize_landmarks, DEVICE

app = Flask(__name__)

classes = get_classes()
model = create_model(num_classes=len(classes), in_dim=63)
model.load_state_dict(torch.load("path/to/model.pth", map_location=DEVICE))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    landmarks = request.json['landmarks']
    
    # Normalize landmarks
    normalized = normalize_landmarks(landmarks, method="wrist_to_middle")
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in normalized], dtype=np.float32)
    
    x = torch.from_numpy(pts.reshape(1, 63)).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        
    return jsonify({'prediction': classes[pred_idx]})
```

## Model Architecture

Both ASL and VGT models use a feedforward neural network architecture:

```
Input (63 features: 21 landmarks × 3 coordinates)
    ↓
Linear(63 → 128) + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + ReLU + Dropout(0.3)
    ↓
Linear(64 → num_classes)
    ↓
Output (class logits)
```

## Dataset Format

The package expects hand landmark data in the following formats:

### CSV Format (ASL)

```csv
class,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
A,0.5,0.3,0.1,0.6,0.4,0.2,...
B,0.4,0.2,0.0,0.5,0.3,0.1,...
```

### JSON Format (VGT)

```json
{
  "A": [
    [[x0,y0,z0], [x1,y1,z1], ..., [x20,y20,z20]],
    [[x0,y0,z0], [x1,y1,z1], ..., [x20,y20,z20]]
  ],
  "B": [...]
}
```

## Performance

| Model | Classes | Accuracy | Parameters |
|-------|---------|----------|------------|
| ASL   | 26      | ~95%     | ~10K       |
| VGT   | 26      | ~93%     | ~10K       |

*Note: Actual performance depends on dataset quality and size.*

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:

```python
train_loader, val_loader = get_loaders(train_dataset, val_dataset, batch_size=16)
```

### Model Not Learning

Try adjusting learning rate:

```python
train_model(model, train_loader, epochs=50, lr=1e-4)  # Lower LR
```

### Poor Accuracy

- Ensure landmarks are normalized correctly
- Add data augmentation during training
- Collect more training data
- Verify dataset labels are correct

## Citation

If you use this package in your research, please cite:

```bibtex
@software{smart_gestures2025,
  title = {Smart Gestures: Sign Language Alphabet Recognition},
  author = {Stijnen, Simon and Deleare, Lynn and Westerman, Olivier},
  year = {2025},
  organization = {VIVES University of Applied Sciences},
  url = {https://github.com/vives-project-xp/SmartGlasses}
}
```

## License

GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Authors

- **Simon Stijnen**
- **Lynn Deleare**
- **Olivier Westerman**

Maintained by **VIVES University of Applied Sciences - Project XP**

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Links

- **Documentation**: [GitHub Repository](https://github.com/vives-project-xp/SmartGlasses)
- **Issue Tracker**: [GitHub Issues](https://github.com/vives-project-xp/SmartGlasses/issues)
- **Source Code**: [GitHub](https://github.com/vives-project-xp/SmartGlasses/tree/main/notebooks/package)

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [PyTorch](https://pytorch.org/) for the deep learning framework
- VIVES University of Applied Sciences for supporting this project
