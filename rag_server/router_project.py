from fastapi import APIRouter, Body, HTTPException
from typing import Any, Dict
from .project_manager import get_manager

router = APIRouter(prefix="/project", tags=["project"])

@router.post("/register")
def register_project(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    key = str(body.get("key", "")).strip()
    name = str(body.get("name", "")).strip()
    hint = str(body.get("project_path", "")).strip()
    
    manager = get_manager()
    
    # If key is provided, check existing or create with this key
    if key:
        existing = manager.get(key)
        if existing:
            return existing
            
    entry = manager.register(key=key, name=name, hint=hint)
    return entry

@router.get("/list")
def list_projects():
    return {"projects": get_manager().list_all()}

@router.get("/{key}")
def get_project(key: str):
    proj = get_manager().get(key)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj
