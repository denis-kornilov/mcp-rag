"""
One-shot script: reads all chunk paths from ChromaDB collection and writes a
manifest so that the next sync_project run uses incremental mode instead of
triggering a full re-index.

Run BEFORE rebooting (while ingest is still running or after killing it):
    cd mcp_rag_package
    python3 save_manifest_from_chroma.py

Safe to run multiple times — it always reflects what's actually in ChromaDB.
"""
import sys
import pathlib

# Make sure project imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rag_server.settings import settings
from rag_server.store import get_collection
from rag_server.ingest_ops import load_manifest, save_manifest, _manifest_path

ROOT = pathlib.Path(settings.project_root).resolve()
COLLECTION = "code"


def main():
    print(f"Connecting to ChromaDB at {settings.chroma_path} ...")
    col = get_collection(COLLECTION)

    print("Fetching all documents (this may take a moment) ...")
    # ChromaDB: get all with metadata only (no embeddings needed)
    result = col.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []

    # Collect unique relative paths
    paths = set()
    for meta in metadatas:
        p = (meta or {}).get("path")
        if p:
            paths.add(str(p))

    print(f"Found {len(metadatas)} chunks across {len(paths)} unique paths.")

    # Load existing manifest (may be empty)
    manifest = load_manifest(COLLECTION)
    updated = 0
    skipped = 0

    for rel_path in sorted(paths):
        fp = ROOT / rel_path
        if not fp.exists() or not fp.is_file():
            skipped += 1
            continue
        try:
            stat = fp.stat()
            manifest[rel_path] = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
            updated += 1
        except Exception as e:
            print(f"  WARN: could not stat {rel_path}: {e}")
            skipped += 1

    save_manifest(COLLECTION, manifest)
    manifest_file = _manifest_path(COLLECTION)
    print(f"\nManifest saved: {manifest_file}")
    print(f"  Entries written : {updated}")
    print(f"  Paths skipped   : {skipped}")
    print(f"  Total in manifest: {len(manifest)}")
    print("\nNext sync_project run will use incremental mode for these files.")


if __name__ == "__main__":
    main()
