import argparse
import pathlib
import sys
import time
from datetime import datetime, UTC

from rag_server.settings import settings
from rag_server.ingest_ops import _get_default_exts, plan_ingest_work, sync_project


DEBUG_LOG_PATH = pathlib.Path(__file__).resolve().parents[1] / "debug.log"


class IngestProgressReporter:
    def __init__(self, debug_log_path: pathlib.Path) -> None:
        self._debug_log_path = debug_log_path
        self._tty = sys.stdout.isatty()
        self._rendered_lines = 0
        self._planned_files = 0
        self._planned_chunks = 0
        self._phase = "init"
        self._phase_started_at = time.monotonic()
        self._phase_current = 0
        self._phase_total = 0
        self._current_file = ""
        self._last_chunks_in_file = 0

    def __call__(self, stage: str, payload: dict) -> None:
        self._capture_state(stage, payload)
        if stage == "scan_plan_complete":
            self._planned_files = int(payload.get("files_planned", 0) or 0)
            self._planned_chunks = int(payload.get("chunks_total", 0) or 0)
        line = self._format_log_line(stage, payload)
        if not line:
            return
        self._write_debug(line)
        self._render(stage, payload, line)

    def _capture_state(self, stage: str, payload: dict) -> None:
        if stage == "start":
            self._phase = "start"
            self._phase_started_at = time.monotonic()
            self._phase_current = 0
            self._phase_total = 0
            self._current_file = ""
            self._last_chunks_in_file = 0
            return
        if stage == "scan_file":
            self._switch_phase("scan")
            self._phase_current = int(payload.get("index", 0) or 0)
            self._phase_total = int(payload.get("total", 0) or 0)
            self._current_file = str(payload.get("path", "") or "")
            return
        if stage == "scan_plan_complete":
            self._switch_phase("scan")
            self._phase_current = int(payload.get("files_planned", 0) or 0)
            self._phase_total = int(payload.get("files_planned", 0) or 0)
            self._current_file = "scan complete"
            return
        if stage == "reset_collection":
            self._switch_phase("reset")
            self._phase_current = 1
            self._phase_total = 1
            self._current_file = str(payload.get("collection", "") or "")
            return
        if stage == "prepare_file":
            self._switch_phase("prepare")
            self._phase_current = int(payload.get("index", 0) or 0)
            self._phase_total = self._planned_files or int(payload.get("total", 0) or 0)
            self._current_file = str(payload.get("path", "") or "")
            self._last_chunks_in_file = int(payload.get("chunks_in_file", 0) or 0)
            return
        if stage == "upsert_batch":
            self._switch_phase("upsert")
            self._phase_current = int(payload.get("written", 0) or 0)
            self._phase_total = self._planned_chunks or int(payload.get("total", 0) or 0)
            return
        if stage == "delete_path":
            self._switch_phase("delete")
            self._phase_current = int(payload.get("index", 0) or 0)
            self._phase_total = int(payload.get("total", 0) or 0)
            self._current_file = str(payload.get("path", "") or "")
            return
        if stage == "done":
            self._switch_phase("done")
            self._phase_current = int(payload.get("written", 0) or 0)
            self._phase_total = self._planned_chunks or self._phase_current
            self._current_file = "done"

    def _switch_phase(self, phase: str) -> None:
        if self._phase != phase:
            self._phase = phase
            self._phase_started_at = time.monotonic()

    def _format_log_line(self, stage: str, payload: dict) -> str | None:
        if stage == "start":
            return (
                "[ingest] start"
                f" root={payload.get('root')}"
                f" collection={payload.get('collection')}"
            )
        if stage == "reset_collection":
            return f"[ingest] reset collection={payload.get('collection')}"
        if stage == "scan_file":
            return (
                f"[ingest] scan-file index={payload.get('index', 0)}"
                f" total={payload.get('total', 0)}"
                f" path={payload.get('path', '')}"
            )
        if stage == "scan_plan_complete":
            return (
                "[ingest] scan-done"
                f" files={payload.get('files_planned', 0)}"
                f" skipped={payload.get('files_skipped', 0)}"
                f" chunks={payload.get('chunks_total', 0)}"
            )
        if stage == "scan_complete":
            return (
                "[ingest] scan"
                f" files={payload.get('total_files', 0)}"
                f" new={payload.get('new_paths', 0)}"
                f" changed={payload.get('changed_paths', 0)}"
                f" deleted={payload.get('deleted_paths', 0)}"
                f" ratio={payload.get('changed_ratio', 0.0):.3f}"
            )
        if stage == "prepare_file":
            return (
                f"[ingest] prepare-file index={payload.get('index', 0)}"
                f" total={self._planned_files or payload.get('total', 0)}"
                f" path={payload.get('path', '')}"
                f" chunks_in_file={payload.get('chunks_in_file', 0)}"
            )
        if stage == "upsert_batch":
            return (
                f"[ingest] upsert-batch written={payload.get('written', 0)}"
                f" total={self._planned_chunks or payload.get('total', 0)}"
                f" batch_size={payload.get('batch_size', 0)}"
            )
        if stage == "delete_path":
            return f"[ingest] delete index={payload.get('index', 0)} total={payload.get('total', 0)} path={payload.get('path', '')}"
        if stage == "done":
            return (
                "[ingest] done"
                f" mode={payload.get('mode')}"
                f" written={payload.get('written', 0)}"
            )
        return None

    def _render(self, stage: str, payload: dict, line: str) -> None:
        if not self._tty:
            print(line, flush=True)
            return
        if stage in {"start", "reset_collection", "scan_plan_complete", "scan_complete", "done"}:
            self._render_status_block(payload)
            if stage in {"scan_plan_complete", "scan_complete", "done"}:
                print()
            return
        if stage in {"scan_file", "prepare_file", "upsert_batch", "delete_path"}:
            self._render_status_block(payload)

    def _render_status_block(self, payload: dict) -> None:
        elapsed = max(time.monotonic() - self._phase_started_at, 0.001)
        current = max(self._phase_current, 0)
        total = max(self._phase_total, 0)
        rate = (current / elapsed) if current > 0 else 0.0
        eta = ((total - current) / rate) if rate > 0 and total >= current and total > 0 else None
        percent = (current / total * 100.0) if total > 0 else 0.0
        bar = self._progress_bar(current, total)
        phase_label = self._phase.upper()
        file_line = f"[ingest] current {self._current_file}"
        if self._phase == "prepare" and self._last_chunks_in_file > 0:
            file_line += f" chunks={self._last_chunks_in_file}"
        status_line = (
            f"[ingest] {phase_label} {current}/{total} {percent:5.1f}% {bar} "
            f"rate={rate:.2f}/s eta={self._format_eta(eta)}"
        )
        lines = [file_line, status_line]
        if self._rendered_lines:
            sys.stdout.write(f"\x1b[{self._rendered_lines}F")
        for idx, text in enumerate(lines):
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(text)
            if idx != len(lines) - 1:
                sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._rendered_lines = len(lines)

    def _progress_bar(self, current: int, total: int, width: int = 28) -> str:
        if total <= 0:
            return "[" + ("." * width) + "]"
        filled = min(width, int(width * current / total))
        return "[" + ("#" * filled) + ("." * (width - filled)) + "]"

    def _format_eta(self, seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        total_seconds = max(0, int(seconds))
        minutes, sec = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _write_debug(self, line: str) -> None:
        ts = datetime.now(UTC).isoformat()
        self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._debug_log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{ts} {line}\n")


def main():
    ap = argparse.ArgumentParser(description="Ingest project code into RAG server (code-aware chunks)")
    ap.add_argument("--server", default=settings.rag_server)
    ap.add_argument("--root", default=settings.project_root or ".")
    ap.add_argument("--collection", default="code")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--replace-by-path", action="store_true", help="Delete prior chunks for each file path before upserting")
    ap.add_argument("--max-file-bytes", type=int, default=800_000, help="Skip files larger than this size")
    ap.add_argument("--exts", default=",".join(sorted(_get_default_exts())), help="Comma-separated extensions to include")
    ap.add_argument("--include-glob", default=None, help="Comma-separated glob patterns (relative to root) to include, e.g. '*chart*.*,*.py'")
    ap.add_argument("--limit-files", type=int, default=None, help="Optional limit on number of files to ingest")
    ap.add_argument("--full", action="store_true", help="Force full re-index even if manifest is up-to-date")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    exts = {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower() for e in args.exts.split(",") if e.strip()}
    includes = [s.strip() for s in (args.include_glob.split(",") if args.include_glob else []) if s.strip()]
    progress = IngestProgressReporter(DEBUG_LOG_PATH)

    progress("start", {"root": str(root), "collection": args.collection})
    plan_ingest_work(
        root=root,
        exts=exts,
        includes=includes or None,
        limit_files=args.limit_files,
        max_file_bytes=args.max_file_bytes,
        progress_cb=progress,
    )

    report = sync_project(
        root=root,
        collection=args.collection,
        exts=exts,
        includes=includes or None,
        limit_files=args.limit_files,
        max_file_bytes=args.max_file_bytes,
        force_full=args.full,
        progress_cb=progress,
    )
    progress("done", {"mode": report.get("mode"), "written": report.get("written", 0)})
    print(report)


if __name__ == "__main__":
    main()
