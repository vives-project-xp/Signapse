import json
import os
from pathlib import Path
from typing import cast, Callable

import numpy as np
import torch
from torch import nn

# Constants
IN_DIM = 63
NUM_POINTS = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
MODEL_DIR = THIS_DIR / "models"

CLASSES_FILE = DATA_DIR / "classes.json"
MODEL_FILE = MODEL_DIR / "vgt_alphabet_model.pth"
