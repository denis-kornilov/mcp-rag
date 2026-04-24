from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from rag_server.error_reporter import ErrorReporter


def _is_local_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}



@dataclass
class _ServerTarget:
    host: str
    port: int
    health_url: str


class RAGServerLifecycle:
    """Manage local RAG server process for MCP gateway lifecycle."""

    def __init__(self, server_url: str, workdir: Path) -> None:
        parsed = urlparse(server_url)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 80)
        health_url = f"{parsed.scheme or 'http'}://{host}:{port}/healthz"
        self._target = _ServerTarget(host=host, port=port, health_url=health_url)
        self._workdir = workdir
        self._proc: subprocess.Popen[str] | None = None
        self._started_by_gateway = False
        self._logger = logging.getLogger("rag_server.lifecycle")
        self._error = ErrorReporter("rag_server.lifecycle")
        self._handlers_installed = False

    def ensure_running(self, timeout_s: float = 20.0) -> None:
        if self._is_healthy():
            self._logger.info(
                "RAG lifecycle: upstream already running host=%s port=%s",
                self._target.host,
                self._target.port,
            )
            return
        if not _is_local_host(self._target.host):
            raise RuntimeError(
                f"RAG server is unreachable at {self._target.health_url}, "
                f"and auto-start is disabled for non-local host '{self._target.host}'"
            )

        self._start_local_server()
        self._wait_until_healthy(timeout_s=timeout_s)

    def shutdown(self) -> None:
        proc = self._proc
        if proc is None or not self._started_by_gateway:
            return
        if proc.poll() is not None:
            return

        self._logger.info("RAG lifecycle: stopping local RAG server pid=%s", proc.pid)
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=4.0)
        except Exception as exc:
            self._error.warn(
                stage="rag_lifecycle_stop_sigint",
                message=f"graceful stop failed pid={proc.pid}; trying terminate",
                exc=exc,
            )
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception as exc2:
                self._error.warn(
                    stage="rag_lifecycle_stop_terminate",
                    message=f"terminate failed pid={proc.pid}; trying kill",
                    exc=exc2,
                )
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception as exc3:
                    self._error.error(
                        stage="rag_lifecycle_stop_kill",
                        message=f"kill failed pid={proc.pid}",
                        exc=exc3,
                    )
                    raise
        finally:
            self._proc = None
            self._started_by_gateway = False

    def _is_healthy(self) -> bool:
        try:
            resp = requests.get(self._target.health_url, timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def _start_local_server(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "rag_server.main:app",
            "--host",
            self._target.host,
            "--port",
            str(self._target.port),
            "--log-level", "warning",
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(self._workdir))
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._workdir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._started_by_gateway = True
        self._logger.info(
            "RAG lifecycle: started local RAG server pid=%s host=%s port=%s",
            self._proc.pid,
            self._target.host,
            self._target.port,
        )
        self._install_cleanup_hooks()

    def _wait_until_healthy(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._is_healthy():
                self._logger.info(
                    "RAG lifecycle: healthy host=%s port=%s",
                    self._target.host,
                    self._target.port,
                )
                return
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"RAG server process exited early with code {proc.returncode}")
            time.sleep(0.25)
        raise RuntimeError(f"RAG server health check timeout after {timeout_s:.1f}s at {self._target.health_url}")

    def _install_cleanup_hooks(self) -> None:
        if self._handlers_installed:
            return
        self._handlers_installed = True
        atexit.register(self.shutdown)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
                def _handler(signum, frame, _prev=previous):
                    self.shutdown()
                    if _prev in (signal.SIG_DFL, signal.SIG_IGN, signal.default_int_handler):
                        return
                    if callable(_prev):
                        _prev(signum, frame)
                signal.signal(sig, _handler)
            except Exception as exc:
                self._error.warn(
                    stage="rag_lifecycle_install_signal_hook",
                    message=f"failed to install handler for signal={sig}",
                    exc=exc,
                )
