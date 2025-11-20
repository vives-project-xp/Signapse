from enum import Enum
from typing import Final

from pydantic import BaseModel

__version__: Final = "0.1.0"


class FastAPITags(Enum):
    ROOT = "Default"
    WEBSOCKET = "WebSocket"
    ALPHABET = "Alphabet"
    ASL_MODEL = "ASL Model"
    VGT_MODEL = "VGT Model"
    KEYPOINTS = "Keypoints"


IN_DIM = 63  # 21 landmarks * (x,y,z)
NUM_POINTS = 21  # exact 21 punten


# ---- Data schema ----
class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0
