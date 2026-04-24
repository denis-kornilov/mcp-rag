from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pydantic import AliasChoices
from pathlib import Path
import os
import re
from .error_reporter import ErrorReporter

# .env load priority (lowest → highest):
#   1. mcp-rag/.env        — project-wide defaults
#   2. rag_server/.env     — rag_server module overrides
#   3. CWD/.env            — runtime / deployment overrides
CWD_DIR  = Path.cwd().resolve()
PKG_DIR  = Path(__file__).resolve().parents[1]   # mcp-rag root
SELF_DIR = Path(__file__).resolve().parent        # rag_server/

ENV_PATH_PKG  = PKG_DIR  / ".env"
ENV_PATH_SELF = SELF_DIR / ".env"
ENV_PATH_CWD  = CWD_DIR  / ".env"

BASE_DIR = CWD_DIR

error_reporter = ErrorReporter("rag_server.settings")
try:
    from dotenv import load_dotenv  # type: ignore
    if ENV_PATH_PKG.exists():
        load_dotenv(str(ENV_PATH_PKG), override=False, encoding="utf-8")
    if ENV_PATH_SELF.exists():
        load_dotenv(str(ENV_PATH_SELF), override=True, encoding="utf-8")
    if ENV_PATH_CWD.exists() and ENV_PATH_CWD != ENV_PATH_SELF:
        load_dotenv(str(ENV_PATH_CWD), override=True, encoding="utf-8")
except Exception as exc:
    error_reporter.warn(
        stage="settings_load_dotenv",
        message="fallback to pydantic env_file",
        exc=exc,
    )



_WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_windows_abs_path(value: str | None) -> bool:
    return bool(value and _WIN_ABS_RE.match(value))


def _default_cache_root() -> Path:
    # Use PKG_DIR for model caches by default to keep them shared across projects
    # unless HF_HOME etc. are explicitly overridden in .env
    return PKG_DIR / ".cache"


def _normalize_path(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    # Keep Windows paths on Windows unchanged. On non-Windows, remap them into
    # a local cache root instead of creating literal "D:\..." directories.
    if _is_windows_abs_path(raw):
        if os.name == "nt":
            return raw
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        return str(_default_cache_root() / "windows" / drive / tail)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return str(path)


def _normalize_project_root(value: str | None) -> str:
    if not value:
        return str(BASE_DIR.resolve())
    raw = value.strip()
    if _is_windows_abs_path(raw) and os.name != "nt":
        return str(BASE_DIR.resolve())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return str(path)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ENV_PATH_PKG), str(ENV_PATH_SELF), str(ENV_PATH_CWD)),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    chroma_path: str = Field(
        default="./.mcp_rag/db",
        validation_alias=AliasChoices("CHROMA_PATH", "chroma_path"),
    )
    # embed_server — URL of the primary compute worker
    embed_server_url: str = Field(default="http://127.0.0.1:8001", validation_alias=AliasChoices("EMBED_SERVER_URL", "embed_server_url"))
    # Multiple workers (comma-separated). When set, overrides embed_server_url.
    embed_server_urls: str = Field(default="", validation_alias=AliasChoices("EMBED_SERVER_URLS", "embed_server_urls"))
    # Max chars to send per chunk (truncation before HTTP call)
    embed_max_chars: int = Field(default=4000, validation_alias=AliasChoices("EMBED_MAX_CHARS", "embed_max_chars"))
    # Workspace root for applying code plans
    project_root: str = Field(default=".", validation_alias=AliasChoices("PROJECT_ROOT", "project_root"))
    # Ingest settings
    auto_ingest_max_file_bytes: int = Field(default=800_000, validation_alias=AliasChoices("AUTO_INGEST_MAX_FILE_BYTES", "auto_ingest_max_file_bytes"))
    # Number of chunks sent to embed_server per HTTP call during ingest
    embed_batch_size: int = Field(default=16, validation_alias=AliasChoices("EMBED_BATCH_SIZE", "embed_batch_size"))
    auto_ingest_allowlist: str = Field(
        default="",
        validation_alias=AliasChoices("AUTO_INGEST_ALLOWLIST", "auto_ingest_allowlist"),
    )
    auto_ingest_extensions: str = Field(
        default=".py,.md,.txt,.json,.yaml,.yml,.toml,.ini,.js,.ts,.tsx,.css,.html",
        validation_alias=AliasChoices("AUTO_INGEST_EXTENSIONS", "auto_ingest_extensions"),
    )
    # Server base URL for local client/tools
    rag_server: str = Field(default="http://127.0.0.1:8000", validation_alias=AliasChoices("RAG_SERVER", "rag_server"))
    # Multi-project support
    project_key: str = Field(default="", validation_alias=AliasChoices("PROJECT_KEY", "project_key"))
    server_data_root: str = Field(default="./mcp_rag_projects", validation_alias=AliasChoices("SERVER_DATA_ROOT", "server_data_root"))
    # Hybrid search (BM25 + vector RRF)
    hybrid_search_enabled: bool = Field(default=False, validation_alias=AliasChoices("HYBRID_SEARCH_ENABLED", "hybrid_search_enabled"))
    # Cross-encoder reranker model
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias=AliasChoices("RERANKER_MODEL", "reranker_model"))
    # File system watcher (watchdog)
    fs_watcher_enabled: bool = Field(default=False, validation_alias=AliasChoices("FS_WATCHER_ENABLED", "fs_watcher_enabled"))
    # RAG backend: "local" (in-process) or "http" (remote rag_server via HTTP)
    rag_backend: str = Field(default="local", validation_alias=AliasChoices("RAG_BACKEND", "rag_backend"))
    # Embed backend name (mirrors EMBED_BACKEND used by embed_server)
    embed_backend: str = Field(default="bge-m3-onnx", validation_alias=AliasChoices("EMBED_BACKEND", "embed_backend"))

    def model_post_init(self, __context) -> None:
        self.chroma_path = _normalize_path(self.chroma_path) or str(CWD_DIR / ".mcp_rag" / "db")
        self.project_root = _normalize_project_root(self.project_root)
        # Resolve server_data_root relative to the mcp-rag package dir, not cwd.
        # This ensures the gateway (started from a project dir) always finds
        # the shared RAG data, regardless of which project is open.
        if not Path(self.server_data_root).is_absolute():
            self.server_data_root = str((PKG_DIR / self.server_data_root).resolve())

settings = Settings()
