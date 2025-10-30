import os
from fastapi import APIRouter, HTTPException
import numpy as np
import torch

from const import IN_DIM, NUM_POINTS, FastAPITags

from schemas import PredictBody, PredictResponse

from schemas.classes import ClassesResponse
from smart_gestures.alphabet.asl_model import create_model, get_classes, DEVICE

classes = get_classes()

model = create_model(num_classes=len(classes), in_dim=IN_DIM)
model_path = os.path.join(
    os.path.dirname(__file__),
    "../../notebooks/alphabet/asl_model/models/hand_gesture_model.pth",
)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

router = APIRouter(
    prefix="/asl",
    tags=[FastAPITags.ASL_MODEL],
)


@router.get("/classes")
async def asl_model_classes() -> ClassesResponse:
    return ClassesResponse(classes=classes)


@router.post("/predict")
async def asl_model_predict(body: PredictBody) -> PredictResponse:
     # naar (21,3) -> (1,63)
    pts = np.array([[lm.x, lm.y, lm.z]
                   for lm in body.landmarks], dtype=np.float32)
    if pts.shape != (NUM_POINTS, 3):
        raise HTTPException(
            status_code=400, detail=f"Expected shape (21,3), got {pts.shape}"
        )

    # Belangrijk: pas hier dezelfde preprocessing toe als bij training indien nodig
    x = torch.from_numpy(pts.reshape(1, IN_DIM)).to(DEVICE)  # type: ignore

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_name = classes[pred_idx]

    return PredictResponse(prediction=pred_name)


__all__ = [
    "router",
]
