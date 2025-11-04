from typing import Any, cast
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from const import __version__
from websocket import connection_manager as manager
from websocket.message import BaseMessage

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Get websocket ip and port as client_id
    if websocket.client is None:
        await websocket.close()
        return
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    await manager.connect(websocket, client_id)
    try:
        while True:
            try:
                data = cast(BaseMessage[Any], BaseMessage.model_validate_json(await websocket.receive_text()))
            except Exception:
                continue  # Ignore invalid messages
            await manager.handle_message(client_id, data)
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(client_id)
