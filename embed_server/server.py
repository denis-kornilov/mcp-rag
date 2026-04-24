"""Entry point: python -m embed_server.server (or python embed_server/server.py)"""
import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("EMBED_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("EMBED_SERVER_PORT", "8001"))
    workers = int(os.environ.get("EMBED_SERVER_WORKERS", "1"))
    uvicorn.run(
        "embed_server.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
