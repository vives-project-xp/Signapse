from fastapi import APIRouter, HTTPException

from const import FastAPITags

from schemas import ClassesResponse, PredictBody, PredictResponse

from smart_gestures.alphabet.vgt_model import get_classes, VGTModel

classes = get_classes()

model = VGTModel()

router = APIRouter(
    prefix="/vgt",
    tags=[FastAPITags.VGT_MODEL],
)


@router.get("/classes")
async def vgt_model_classes() -> ClassesResponse:
    return ClassesResponse(classes=classes)


@router.post("/predict")
async def vgt_model_predict(body: PredictBody) -> PredictResponse:
    landmarks_dicts = [landmark.model_dump() for landmark in body.landmarks]
    try:
        pred_name = model.predict(landmarks_dicts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(prediction=pred_name)


__all__ = [
    "router",
]
