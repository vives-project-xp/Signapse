from fastapi import APIRouter, HTTPException

from smart_gestures.alphabet.asl_model import get_classes, ASLModel

from const import FastAPITags
from schemas import ClassesResponse, PredictBody, PredictResponse

classes = get_classes()

model = ASLModel()

router = APIRouter(
    prefix="/asl",
    tags=[FastAPITags.ASL_MODEL],
)


@router.get("/classes")
async def asl_model_classes() -> ClassesResponse:
    return ClassesResponse(classes=classes)


@router.post("/predict")
async def asl_model_predict(body: PredictBody) -> PredictResponse:
    landmarks_dicts = [landmark.model_dump() for landmark in body.landmarks]
    try:
        pred_name = model.predict(landmarks_dicts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(prediction=pred_name)


__all__ = [
    "router",
]
