from typing import cast

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from const import FastAPITags, __version__
from schemas import StatusResponse

router = APIRouter(
    tags=[FastAPITags.ROOT],
)


@router.get("/")
async def root() -> StatusResponse:
    return cast(StatusResponse, RedirectResponse(url="/health"))


@router.get("/health")
async def health_check() -> StatusResponse:
    return StatusResponse(version=__version__)
