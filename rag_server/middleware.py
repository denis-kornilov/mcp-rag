import json
from starlette.types import ASGIApp, Scope, Receive, Send
from starlette.responses import Response
from .project_manager import get_manager
from .project_context import set_project
from .error_reporter import ErrorReporter

error_reporter = ErrorReporter("rag_server.middleware")

# Paths exempt from project-key validation
_EXEMPT_PATHS = {"/healthz", "/project/register", "/project/list"}


def _json_error(status: int, error: str, message: str) -> Response:
    body = json.dumps({"error": error, "message": message}, ensure_ascii=False).encode("utf-8")
    return Response(
        content=body,
        status_code=status,
        media_type="application/json",
    )


class ProjectKeyMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Exempt endpoints — accessible without a project key
        if path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        key = headers.get(b"x-project-key", b"").decode("utf-8", errors="ignore").strip()

        if not key:
            resp = _json_error(
                401,
                "project_key_required",
                "X-Project-Key header is required. "
                "Run the MCP gateway from your project directory — it creates .mcp-rag automatically.",
            )
            await resp(scope, receive, send)
            return

        try:
            manager = get_manager()
            proj = manager.get(key)
        except Exception as exc:
            error_reporter.warn(stage="middleware_manager_lookup", message="project manager error", exc=exc)
            proj = None

        if proj is None:
            resp = _json_error(
                403,
                "project_not_found",
                f"Project key '{key[:8]}...' is not registered on this RAG server. "
                "Call POST /project/register with your key first, or restart the MCP gateway.",
            )
            await resp(scope, receive, send)
            return

        try:
            set_project(
                chroma_path=proj["chroma_path"],
                project_root=proj["project_root"],
                key=key,
            )
        except Exception as exc:
            error_reporter.warn(stage="middleware_set_project", message="failed to set project context", exc=exc)

        await self.app(scope, receive, send)
