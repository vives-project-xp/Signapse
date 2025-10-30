from const import FastAPITags
from fastapi import APIRouter

from . import connection

router = APIRouter(
    tags=[FastAPITags.WEBSOCKET],
)

router.include_router(connection.router)

__all__ = [
    "router",
]
