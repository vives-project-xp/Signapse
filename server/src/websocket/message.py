from typing import Literal, TypeVar, Generic
from abc import ABC
from pydantic import BaseModel

# Generic type
T_DATA = TypeVar("T_DATA")


class Version(BaseModel):
    version: str


class Receive():
    class BaseMessage(ABC, BaseModel):
        """
        Represents a WebSocket message with a type and associated data.
        """

        type: Literal["status", "classes", "predict"]
        data: dict[str, str] | None


class Send():
    class BaseMessage(ABC, BaseModel, Generic[T_DATA]):
        """
        Represents a WebSocket message with a type and associated data.
        """

        type: str
        data: T_DATA

    class StatusMessage(BaseMessage[Version]):
        """
        A message subclass specifically for status updates.
        """

        type: str = "status"

    class ClassesMessage(BaseMessage[list[str]]):
        """
        A message subclass specifically for classification results.
        """

        type: str = "classes"

    class PredictMessage(BaseMessage[list[str]]):
        """
        A message subclass specifically for prediction results.
        """

        type: str = "predict"
