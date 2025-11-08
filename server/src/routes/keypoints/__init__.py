from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2 as cv
from mediapipe.python.solutions import hands as mp_hands

from const import FastAPITags, Landmark, NUM_POINTS

router = APIRouter(
    prefix="/keypoints",
    tags=[FastAPITags.KEYPOINTS],
)
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

@router.post("/", response_model=List[Landmark])
async def extract_keypoints(
    image: UploadFile = File(...), 
    # static_image_mode: bool = True,
    # max_num_hands: int = 2,
    # min_detection_confidence: float = 0.5,
    # min_tracking_confidence: float = 0.5,
    ) -> List[Landmark]:
    """
    Accept an image file (multipart/form-data), detect a single hand using MediaPipe,
    and return the list of 21 landmarks as normalized (x, y, z) floats.

    Returns an empty list when no hand is detected.
    """
    # read bytes
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file")

    # convert to numpy image
    nparr = np.frombuffer(data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    # convert BGR to RGB for mediapipe
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Use static_image_mode=True because we're processing single uploaded images
    results = hands.process(img_rgb)

    if not results or not results.multi_hand_landmarks:  # type: ignore
        # no hand detected: return empty list (client can decide what to do)
        return []

    hand_landmarks = results.multi_hand_landmarks[0]  # type: ignore
    landmarks: List[Landmark] = []
    for lm in hand_landmarks.landmark:  # type: ignore
        # lm.x, lm.y, lm.z are normalized to [0,1] (z is relative)
        landmarks.append(Landmark(x=float(lm.x), y=float(
            lm.y), z=float(lm.z)))  # type: ignore

    # Sanity: enforce expected number of points (21)
    if len(landmarks) != NUM_POINTS:
        # still return whatever we have but note in logs could be added
        return landmarks

    return landmarks


__all__ = [
    "router",
]
