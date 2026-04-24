"""
Example: ingest a project directory via rag_server HTTP API.

Runs a background ingest job and polls for completion.

Usage:
    python examples/ingest.py /path/to/your/project
    python examples/ingest.py /path/to/your/project --force-full
    python examples/ingest.py /path/to/your/project --scan-only
"""

import argparse
import json
import sys
import time

import requests

RAG_URL = "http://127.0.0.1:8000"


def load_project_key(path: str = ".mcp-rag") -> str:
    try:
        with open(path) as f:
            data = json.load(f)
            return data.get("project_key", "")
    except FileNotFoundError:
        return ""


def scan(project_path: str, key: str) -> None:
    headers = {"X-Project-Key": key}
    resp = requests.post(
        f"{RAG_URL}/rag/scan",
        json={"root": project_path},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    d = resp.json()
    print(f"\nScan result for: {d['root']}")
    print(f"  Files found  : {d['files_found']}")
    print(f"  Total size   : {d['total_mb']} MB")
    print(f"  By extension : {d['by_extension']}")
    if d.get("sample_paths"):
        print(f"  Sample paths :")
        for p in d["sample_paths"][:10]:
            print(f"    {p}")


def ingest(project_path: str, key: str, force_full: bool = False) -> None:
    headers = {"X-Project-Key": key}
    resp = requests.post(
        f"{RAG_URL}/rag/sync",
        json={"root": project_path, "collection": "code", "force_full": force_full},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print(f"\nIngest job started: {job_id}")
    print(f"  root={job['root']}  collection={job['collection']}")

    # Poll until done
    while True:
        time.sleep(3)
        r = requests.get(f"{RAG_URL}/rag/jobs/{job_id}", headers=headers, timeout=10)
        r.raise_for_status()
        j = r.json()
        status = j["status"]
        progress = j.get("progress", {})
        summary = progress.get("summary", "")
        elapsed = progress.get("elapsed_human", "")
        percent = progress.get("percent", "")

        pct = f" {percent}%" if percent != "" else ""
        print(f"  [{status}]{pct}  {elapsed}  {summary}", end="\r")

        if status in ("completed", "error", "stopped"):
            print()
            break

    print(f"\nResult: {j['status']}")
    if j.get("result"):
        r = j["result"]
        print(f"  mode           : {r.get('mode')}")
        print(f"  files indexed  : {r.get('files_indexed', r.get('paths_requested', '?'))}")
        print(f"  chunks written : {r.get('written', '?')}")
    if j.get("error"):
        print(f"  ERROR: {j['error']['message']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project_path", help="Path to project directory to index")
    ap.add_argument("--force-full", action="store_true", help="Force full re-index")
    ap.add_argument("--scan-only", action="store_true", help="Preview files without indexing")
    ap.add_argument("--key", default="", help="Project key (auto-detected from .mcp-rag if not set)")
    ap.add_argument("--url", default=RAG_URL)
    args = ap.parse_args()

    RAG_URL = args.url
    key = args.key or load_project_key()
    if not key:
        print("ERROR: project key not found. Run mcp_server/gateway.py from your project dir first.")
        sys.exit(1)

    if args.scan_only:
        scan(args.project_path, key)
    else:
        scan(args.project_path, key)
        print()
        ingest(args.project_path, key, force_full=args.force_full)
