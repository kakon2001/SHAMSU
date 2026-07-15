from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..agent.tools import IGNORED_DIRS, resolve_in_workspace
from ..config import settings
from ..schemas import FileContent, FileNode, SaveFileRequest

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("", response_model=FileNode)
async def get_file_tree() -> FileNode:
    return _build_tree(settings.workdir_path, settings.workdir_path)


@router.get("/content", response_model=FileContent)
async def get_file_content(path: str = Query(...)) -> FileContent:
    target = _resolve_or_400(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"'{path}' is not a file")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read '{path}': {exc}")
    return FileContent(path=path, content=text)


@router.put("/content", response_model=FileContent)
async def save_file_content(body: SaveFileRequest) -> FileContent:
    """Direct save from the editor â€” a user-initiated action, so no approval gate."""
    target = _resolve_or_400(body.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")
    return FileContent(path=body.path, content=body.content)


def _resolve_or_400(path: str) -> Path:
    try:
        return resolve_in_workspace(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _build_tree(path: Path, root: Path) -> FileNode:
    rel = path.relative_to(root).as_posix() if path != root else "."
    if path.is_dir():
        children = [
            _build_tree(child, root)
            for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            if child.name not in IGNORED_DIRS
        ]
        return FileNode(name=(path.name if path != root else "."), path=rel, type="dir", children=children)
    return FileNode(name=path.name, path=rel, type="file")


@router.delete("/content")
async def delete_file_content(path: str = Query(...)) -> dict[str, bool]:
    target = _resolve_or_400(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"'{path}' does not exist")
    if target.is_dir():
        raise HTTPException(status_code=400, detail=f"'{path}' is a directory")
    try:
        target.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{path}': {exc}")
    return {"ok": True}
