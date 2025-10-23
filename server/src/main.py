import os
from typing import Dict, List, Union, cast

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

from smart_gestures.alphabet.asl_model import get_classes, create_model, DEVICE

from const import __version__, IN_DIM, NUM_POINTS
from websocket.connection_manager import ConnectionManager
from schemas import StatusResponse, ClassesResponse, PredictBody, PredictResponse

app = FastAPI(
    title="Smart Glasses Hand Gesture Recognition API",
    description="API for recognizing hand gestures using a pre-trained model.",
    version=__version__,
    debug=False,
)
manager = ConnectionManager()


_classes_raw: Union[Dict[str, str], List[str]] = get_classes()
if isinstance(_classes_raw, dict):
    # index->naam dict naar lijst; gesorteerd op index
    class_names: List[str] = [
        name
        for _, name in sorted(
            ((int(k), v)
             for k, v in cast(Dict[str, str], _classes_raw).items()),
            key=lambda kv: kv[0],
        )
    ]
else:
    class_names = list(_classes_raw)

# Create model and load weights
model = create_model(num_classes=len(class_names), in_dim=IN_DIM)
model_path = os.path.join(
    os.path.dirname(__file__),
    "../../notebooks/alphabet/asl_model/models/hand_gesture_model.pth",
)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()


@app.get("/")
async def root() -> StatusResponse:
    return StatusResponse(version=__version__)


@app.get("/class-names")
def get_class_names() -> ClassesResponse:
    return ClassesResponse(classes=class_names)


@app.post("/predict")
def predict(body: PredictBody) -> PredictResponse:
    # naar (21,3) -> (1,63)
    pts = np.array([[lm.x, lm.y, lm.z]
                   for lm in body.landmarks], dtype=np.float32)
    if pts.shape != (NUM_POINTS, 3):
        raise HTTPException(
            status_code=400, detail=f"Expected shape (21,3), got {pts.shape}"
        )

    # Belangrijk: pas hier dezelfde preprocessing toe als bij training indien nodig
    x = torch.from_numpy(pts.reshape(1, IN_DIM)).to(DEVICE)  # type: ignore

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_name = class_names[pred_idx]

    return PredictResponse(prediction=pred_name)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Get websocket ip and port as client_id
    if websocket.client is None:
        await websocket.close()
        return
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    await manager.connect(websocket, client_id)
    await manager.broadcast(f"Client {client_id} connected")
    try:
        while True:
            data = await websocket.receive_text()
            # simple broadcast protocol; you can replace with JSON or other formats
            await manager.broadcast(f"{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(f"Client {client_id} disconnected")


# Run the application with `python -m fastapi dev src/main.py`
