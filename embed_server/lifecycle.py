"""Auto-start embed_server as a local subprocess when EMBED_SERVER_URL is localhost.

Called from rag_server startup and from mcp_rag_server gateway init.
If the URL points to a remote host — does nothing (assumes server is already running).
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger("embed_server.lifecycle")

_proc: subprocess.Popen | None = None
_started = False


def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _is_alive(url: str) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/healthz", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_running(embed_server_url: str = "http://127.0.0.1:8001") -> bool:
    """Start embed_server subprocess if URL is local and server is not running.

    Returns True if server is alive after the call, False on failure.
    """
    global _proc, _started

    if not _is_local(embed_server_url):
        # Remote server — caller is responsible for starting it
        return _is_alive(embed_server_url)

    if _is_alive(embed_server_url):
        logger.info("[lifecycle] embed_server already running at %s", embed_server_url)
        return True

    if _started:
        logger.warning("[lifecycle] embed_server was started but not responding")
        return False

    parsed = urlparse(embed_server_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8001)

    # Locate embed_server package relative to this file
    embed_dir = Path(__file__).resolve().parent
    pkg_root = str(embed_dir.parent)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", pkg_root)

    # Load embed_server/.env so HF_HOME and other vars are available in subprocess
    _dotenv_path = embed_dir / ".env"
    if _dotenv_path.exists():
        with open(_dotenv_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip()
                # Resolve relative paths against embed_dir
                if _k == "HF_HOME" and _v and not os.path.isabs(_v):
                    _v = str((embed_dir / _v.lstrip("./")).resolve())
                env.setdefault(_k, _v)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "embed_server.main:app",
        "--host", host,
        "--port", port,
        "--log-level", "warning",
    ]

    logger.info("[lifecycle] starting embed_server pid=? host=%s port=%s", host, port)
    _proc = subprocess.Popen(
        cmd,
        cwd=pkg_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _started = True
    atexit.register(_shutdown)

    # Wait up to 30s for the server to become healthy
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if _proc.poll() is not None:
            logger.error("[lifecycle] embed_server exited early (code=%s)", _proc.returncode)
            return False
        if _is_alive(embed_server_url):
            logger.info("[lifecycle] embed_server ready pid=%d", _proc.pid)
            return True
        time.sleep(0.5)

    logger.error("[lifecycle] embed_server did not become healthy within 30s")
    return False


def _shutdown() -> None:
    global _proc
    if _proc is None or _proc.poll() is not None:
        return
    logger.info("[lifecycle] stopping embed_server pid=%d", _proc.pid)
    try:
        _proc.send_signal(signal.SIGINT)
        _proc.wait(timeout=5)
    except Exception:
        try:
            _proc.terminate()
            _proc.wait(timeout=3)
        except Exception:
            _proc.kill()
    _proc = None
