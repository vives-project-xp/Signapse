from pydantic import BaseModel, Field
from const import NUM_POINTS, Landmark


class KeypointsResponse(BaseModel):
    landmarks: list[Landmark] = Field(
        min_length=NUM_POINTS,
        max_length=NUM_POINTS,
        description="List of detected hand keypoints",
    )
