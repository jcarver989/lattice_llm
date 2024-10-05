from pydantic import BaseModel
from typing import Literal


class Node(BaseModel):
    id: str
    source: str
    is_active: bool = False


class Edge(BaseModel):
    source_id: str
    destination_id: str


class TextContentBlock(BaseModel):
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[TextContentBlock]


class ExecuteResult(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
    messages: list[Message]
