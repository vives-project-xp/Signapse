import os
import json
from typing import Dict

# Configurable paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "dataset")  # Directory where gesture data is stored, change this if needed

# Create a mapping from gesture names to integer labels
def create_gesture_map(data_path: str) -> Dict[str, int]:
    gestures = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    gesture_map = {gesture: idx for idx, gesture in enumerate(gestures)}
    return gesture_map

# Example usage to create gesture map
gesture_map = create_gesture_map(DATA_DIR)
print("Gesture Map:", gesture_map)

# Export gesture map to JSON
with open(os.path.join(THIS_DIR, "gesture_map.json"), "w") as f:
    json.dump(gesture_map, f)
print("Gesture map saved to gesture_map.json")
