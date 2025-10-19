from fastapi import WebSocket


class ConnectionManager:
    """Manage active WebSocket connections keyed by client id."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)

    async def send_personal_message(self, message: str, client_id: str) -> None:
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_text(message)

    async def broadcast(self, message: str) -> None:
        for connection in list(self.active_connections.values()):
            try:
                await connection.send_text(message)
            except Exception:
                # ignore send errors; clients that error will be cleaned up on disconnect
                pass
