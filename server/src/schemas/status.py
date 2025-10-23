from pydantic import BaseModel, Field
from const import __version__


class StatusResponse(BaseModel):
    version: str = Field(
        description="API version (semantic, e.g. 0.1.0)",
        frozen=True,
        init=True,
        pattern=r"^\d+\.\d+\.\d+$",
        examples=[__version__],
    )
