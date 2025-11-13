from fastapi import APIRouter

from const import FastAPITags

from . import asl_model, vgt_model

router = APIRouter(
    prefix="/alphabet",
    tags=[FastAPITags.ALPHABET],
)

router.include_router(asl_model.router)
router.include_router(vgt_model.router)

__all__ = [
    "router",
]
