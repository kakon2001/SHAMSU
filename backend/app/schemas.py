from typing import Literal, Optional

from pydantic import BaseModel


class FileNode(BaseModel):
    name: str
    path: str
    type: Literal["file", "dir"]
    children: Optional[list["FileNode"]] = None


class FileContent(BaseModel):
    path: str
    content: str


class SaveFileRequest(BaseModel):
    path: str
    content: str
