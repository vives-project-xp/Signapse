from .connection_manager import ConnectionManager
from .message import BaseMessage, StatusMessage, ClassesMessage, PredictMessage

connection_manager = ConnectionManager()

__all__ = [
    "connection_manager",
    "BaseMessage",
    "StatusMessage",
    "ClassesMessage",
    "PredictMessage",
    "ConnectionManager",
]
