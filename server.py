import argparse
import os
import signal
import subprocess
import sys
import time
import platform


def kill_tree_windows(pid: int):
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Run uvicorn with reliable Ctrl+C handling (Windows friendly)")
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", default=os.getenv("PORT", "8000"))
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--reload-dir", default=os.getenv("RELOAD_DIR", "app"), help="Directory to watch for reload changes")
    ap.add_argument("--reload-exclude", action="append", default=["serve.py", "run_server.bat"], help="Patterns to exclude from reload watcher (can repeat)")
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "uvicorn", "rag_server.main:app", "--host", str(args.host), "--port", str(args.port)]
    if args.reload:
        cmd.append("--reload")
        if args.reload_dir:
            cmd.extend(["--reload-dir", args.reload_dir])
        for pat in (args.reload_exclude or []):
            cmd.extend(["--reload-exclude", pat])

    creationflags = 0
    if platform.system() == "Windows":
        # Create a new process group so we can send CTRL_BREAK_EVENT
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen(cmd, creationflags=creationflags)

    def _graceful_shutdown():
        if platform.system() == "Windows":
            # Always try CTRL_BREAK to the whole group
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except Exception:
                pass
            time.sleep(1.0)
            if proc.poll() is None:
                # Fallback to terminate, then force kill tree
                try:
                    proc.terminate()
                except Exception:
                    pass
                time.sleep(0.5)
            if proc.poll() is None:
                kill_tree_windows(proc.pid)
        else:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    # Forward SIGINT/SIGTERM to child reliably
    def _sig_handler(signum, frame):
        _graceful_shutdown()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        # Some platforms may not allow setting handlers
        pass

    try:
        return_code = proc.wait()
        sys.exit(return_code)
    except KeyboardInterrupt:
        _graceful_shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
