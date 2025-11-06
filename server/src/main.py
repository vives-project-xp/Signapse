from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from const import __version__
from websocket.connection_manager import ConnectionManager
import routes
import os
import logging

# Reduce TensorFlow/absl/glog verbosity to avoid noisy startup logs from
# MediaPipe / TensorFlow. These environment variables MUST be set before
# importing any module that may in turn import mediapipe or tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
# Optionally silence absl python logger too
logging.getLogger("absl").setLevel(logging.ERROR)


manager = ConnectionManager()

app = FastAPI(
    title="Smart Glasses Hand Gesture Recognition API",
    description="API for recognizing hand gestures using a pre-trained model.",
    version=__version__,
    debug=False,
)

# Allow CORS from all origins. This is useful for development and for clients
# served from different origins (web, mobile, etc.). If you need to restrict
# origins later, replace ["*"] with a list of allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.root.router)
app.include_router(routes.ws.router)
app.include_router(routes.alphabet.router)
app.include_router(routes.keypoints.router)
