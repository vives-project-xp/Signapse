from pydantic import BaseModel, Field


class ClassesResponse(BaseModel):
    classes: list[str] = Field(..., description="List of class names")
