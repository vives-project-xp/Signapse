from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from websocket import connection_manager as manager

router = APIRouter()


@router.websocket("/ws")
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
            # simple broadcast protocol
            await manager.broadcast(f"{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(f"Client {client_id} disconnected")
