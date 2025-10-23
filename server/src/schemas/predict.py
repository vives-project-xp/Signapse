from pydantic import BaseModel, Field
from const import NUM_POINTS, Landmark


class PredictBody(BaseModel):
    landmarks: list[Landmark] = Field(
        min_length=NUM_POINTS,
        max_length=NUM_POINTS,
        description="List of landmarks representing hand keypoints",
    )


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class name", )
