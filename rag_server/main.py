from fastapi import FastAPI
from .router_ingest import router as ingest_router
from .router_query import router as query_router
from .router_sync import router as sync_router
from .router_project import router as project_router
from .middleware import ProjectKeyMiddleware
from .settings import settings

app = FastAPI(title="RAG Server")
app.add_middleware(ProjectKeyMiddleware)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(sync_router)
app.include_router(project_router)


@app.on_event("startup")
def _startup():
    from embed_server.lifecycle import ensure_running  # noqa: PLC0415
    ensure_running(settings.embed_server_url)


@app.get("/healthz")
def health():
    from .embeddings import get_embedder_info  # noqa: PLC0415
    return {"status": "ok", "embed_server": get_embedder_info()}
