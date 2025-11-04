from typing import Any
from fastapi import WebSocket
from const import __version__
from .message import BaseMessage, StatusMessage, ClassesMessage, PredictMessage, Version


class ConnectionManager:
    """Manage active WebSocket connections keyed by client id."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)

    async def send_personal_message(self, client_id: str, message: BaseMessage[Any]) -> None:
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_text(message.model_dump_json())

    async def broadcast(self, message: BaseMessage[Any]) -> None:
        for connection in list(self.active_connections.values()):
            try:
                await connection.send_text(message.model_dump_json())
            except Exception:
                # ignore send errors; clients that error will be cleaned up on disconnect
                pass

    async def handle_message(self, client_id: str, message: BaseMessage[Any]) -> None:
        """Process an incoming message from a client. Placeholder for custom logic."""
        match message.type:
            case "status":
                # Handle status message
                await self.send_personal_message(client_id, StatusMessage(
                    data=Version(version=__version__)
                ))
            case "classes":
                # Handle classes message
                await self.send_personal_message(client_id, ClassesMessage(
                    data=["class1", "class2", "class3"]
                ))
            case "predict":
                # Handle predict message
                await self.send_personal_message(client_id, PredictMessage(
                    data=["prediction1", "prediction2"]
                ))
            case _:
                # Unknown message type
                pass
