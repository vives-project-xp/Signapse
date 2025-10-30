from typing import cast
from schemas import StatusResponse
from const import __version__
from fastapi import APIRouter
from const import FastAPITags
from fastapi.responses import RedirectResponse

router = APIRouter(
    tags=[FastAPITags.ROOT],
)


@router.get("/")
async def root() -> StatusResponse:
    return cast(StatusResponse, RedirectResponse(url="/health"))


@router.get("/health")
async def health_check() -> StatusResponse:
    return StatusResponse(version=__version__)
