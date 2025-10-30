from fastapi import FastAPI

from const import __version__
from websocket.connection_manager import ConnectionManager

import routes

manager = ConnectionManager()

app = FastAPI(
    title="Smart Glasses Hand Gesture Recognition API",
    description="API for recognizing hand gestures using a pre-trained model.",
    version=__version__,
    debug=False,
)

app.include_router(routes.root.router)
app.include_router(routes.ws.router)
app.include_router(routes.alphabet.router)
