from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2 as cv
from mediapipe.python.solutions import hands as mp_hands

from const import FastAPITags, Landmark
from schemas import KeypointsResponse

router = APIRouter(
    prefix="/keypoints",
    tags=[FastAPITags.KEYPOINTS],
)

# Initialize MediaPipe Hands once at module load for maximum performance
# This eliminates the overhead of creating a new instance for each request
_hands_detector = mp_hands.Hands(
    static_image_mode=True,  # Optimized for single images
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.5,  # Balanced detection threshold
    min_tracking_confidence=0.5,  # Balanced tracking threshold
)


@router.post("/")
async def extract_keypoints(
    image: UploadFile = File(...),
) -> KeypointsResponse:
    """
    Accept an image file (multipart/form-data), detect hand(s) using MediaPipe,
    and return the list of 21 landmarks as normalized (x, y, z) floats.

    Optimized for speed by using a persistent MediaPipe model instance
    with fixed parameters for consistent, fast performance.
    """
    # read bytes
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file")

    # convert to numpy image
    nparr = np.frombuffer(data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    # check if image decoding was successful
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported format")

    # convert BGR to RGB for mediapipe
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Use the persistent detector instance for faster processing
    results = _hands_detector.process(img_rgb)

    if not results or not results.multi_hand_landmarks:  # type: ignore
        # no hand detected
        raise HTTPException(status_code=404, detail="No hand detected")

    hand_landmarks = results.multi_hand_landmarks[0]  # type: ignore
    landmarks: List[Landmark] = []
    for lm in hand_landmarks.landmark:  # type: ignore
        # lm.x, lm.y, lm.z are normalized to [0,1] (z is relative)
        landmarks.append(Landmark(x=float(lm.x), y=float(lm.y), z=float(lm.z)))  # type: ignore

    return KeypointsResponse(landmarks=landmarks)


__all__ = [
    "router",
]
