# mcp-rag

Semantic search engine for codebases and documents, exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). Connects to any MCP-compatible AI agent (Claude Code, Claude Desktop, Cursor, Gemini, Codex, etc.) and provides contextual code search without blocking the agent's workflow.

## Features

- **Non-blocking ingest** — ingest runs in background threads, returns `job_id` immediately; search is always available
- **Dual ORT sessions** — search and ingest use separate ONNX Runtime sessions in true OS-level parallelism
- **Multi-project isolation** — each project gets a UUID4 key (stored in `.mcp-rag`); data isolated in `mcp_rag_projects/<key>/`
- **Hybrid search** — BM25 + vector embeddings fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranker** — optional `sentence-transformers` CrossEncoder for higher precision
- **CPU-first** — works out of the box on any machine; NVIDIA CUDA and AMD ROCm supported via `embed_server/install.sh`
- **MCP stdio + HTTP** — gateway works as a child process (stdio) or standalone HTTP server

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  AI Agent (Claude Code / Cursor / Gemini / Codex)           │
│                    │  MCP (stdio or HTTP)                    │
└────────────────────┼────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │    mcp_server           │  port 8002 (HTTP mode)
        │    gateway.py           │  - resolves project key from CWD/.mcp-rag
        └────────────┬────────────┘  - registers project with rag_server
                     │  HTTP  X-Project-Key
        ┌────────────▼────────────┐
        │    rag_server           │  port 8000
        │    FastAPI + ChromaDB   │  - multi-project isolation
        │    ingest / search /    │  - BM25 + vector hybrid search
        │    sync / jobs          │  - optional cross-encoder reranker
        └────────────┬────────────┘
                     │  HTTP  priority=0 (search) | 10 (ingest)
        ┌────────────▼────────────┐
        │    embed_server         │  port 8001
        │    ONNX Runtime         │  - BGE-M3 model (INT8 quantized)
        │    BGE-M3 embeddings    │  - dual ORT sessions: search (1 thread)
        │                         │    + ingest (4 threads), truly parallel
        └─────────────────────────┘
```

### Components

| Component | Conda env | Port | Description |
|---|---|---|---|
| `embed_server` | `mcp-embed` | 8001 | ONNX Runtime inference, BGE-M3 embeddings |
| `rag_server` | `mcp-rag` | 8000 | FastAPI, ChromaDB, ingest/search/sync API |
| `mcp_server` | `mcp-gateway` | 8002 | MCP gateway, RAG HTTP client |

## Quick Start

### 1. Install

```bash
git clone https://github.com/denis-kornilov/mcp-rag.git
cd mcp-rag

# Install all three subsystems (interactive, detects CPU/GPU)
bash install.sh

# Or install individually:
bash embed_server/install.sh   # choose CPU / CUDA / ROCm
bash rag_server/install.sh
bash mcp_server/install.sh
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — adjust paths if needed (defaults work for local single-machine setup)
```

Each subsystem also has its own `.env` for fine-grained overrides.
See [`examples/env/`](examples/env/) for annotated templates (local, remote GPU, multi-server).

### 3. Start servers

```bash
bash embed_server/start_embed.sh   # terminal 1
bash rag_server/start_rag.sh       # terminal 2
bash mcp_server/start_mcp.sh       # terminal 3 (HTTP mode)
```

Start scripts validate `.env` before launching and warn about misconfigurations.

### 4. Connect your IDE

**HTTP mode** — gateway runs as a standalone server:

```json
{
  "mcpServers": {
    "mcp-rag": {
      "type": "http",
      "url": "http://127.0.0.1:8002/mcp"
    }
  }
}
```

**stdio mode** — gateway as a child process, no separate server needed:

```json
{
  "mcpServers": {
    "mcp-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/mcp-rag/mcp_server/gateway.py"],
      "env": {"PYTHONPATH": "/path/to/mcp-rag"}
    }
  }
}
```

See [`examples/mcp_config.json`](examples/mcp_config.json) for all variants.

### 5. Index your project

The MCP gateway auto-generates a project key from CWD on startup. Ask your agent:

```
scan my project and ingest it
```

Or use the CLI directly:

```bash
python examples/ingest.py /path/to/your/project
python examples/search.py "how does authentication work"
```

## MCP Tools

| Tool | Description |
|---|---|
| `health` | Check all servers status |
| `scan_project` | Preview files that would be indexed (no embedding) |
| `ingest_project` | Scan + confirm before indexing |
| `confirm_ingest_project` | Start background ingest job |
| `ingest_paths` | Ingest specific file paths |
| `confirm_ingest_paths` | Start background ingest for specific paths |
| `get_job_status` | Poll background job progress |
| `stop_job` | Gracefully stop a running ingest job |
| `list_recent_jobs` | List recent ingest jobs |
| `search` | Semantic search (optional hybrid mode and reranking) |
| `fetch_chunks` | Fetch specific chunks by ID |
| `project_status` | Current project info and collection stats |
| `register_project` | Manually register project with rag_server |

## REST API

### embed_server `http://localhost:8001`

```
GET  /healthz     — status, model info, thread counts
POST /embed       — {"texts": [...], "priority": 0|10}
                    priority=0  → search session (SEARCH_THREADS cores)
                    priority=10 → ingest session (ONNX_NUM_THREADS cores)
```

### rag_server `http://localhost:8000`

All endpoints except `/healthz`, `/project/register`, `/project/list` require `X-Project-Key` header.

```
GET  /healthz
POST /project/register             — {"key": "...", "name": "...", "project_path": "..."}
GET  /project/list

GET  /query/?q=...&k=5&hybrid=true — semantic + optional BM25 hybrid search
POST /rag/scan                     — {"root": "/path"} — preview files
POST /rag/sync                     — {"root": "/path", "force_full": false} — start job
GET  /rag/jobs/{job_id}            — poll job status
POST /rag/jobs/{job_id}/stop       — stop job gracefully
```

See [`examples/api_curl.sh`](examples/api_curl.sh) for complete curl examples.

## Project Structure

```
mcp-rag/
├── install.sh                    # root installer: bash install.sh [all|embed|rag|mcp|gpu]
├── install_adds.sh               # GPU driver installer (CUDA / ROCm)
├── server.py                     # uvicorn launcher for rag_server
├── ingest_project.py             # CLI: manual project ingest with progress
├── save_manifest_from_chroma.py  # emergency: rebuild manifest from ChromaDB
├── requirements.txt              # unified pip requirements
├── environment.yml               # unified conda env (mcp-rag-all)
├── .env.example                  # copy to .env and edit
│
├── embed_server/
│   ├── model.py                  # dual ORT sessions (search + ingest)
│   ├── batcher.py                # per-session FIFO batching queues
│   ├── main.py                   # FastAPI app, routes by priority
│   ├── server.py                 # uvicorn entry point
│   ├── lifecycle.py              # ensure_running() health check
│   ├── install.sh                # interactive installer (CPU/CUDA/ROCm)
│   ├── start_embed.sh            # start with .env validation
│   ├── requirements.txt
│   ├── environment.yml           # conda env: mcp-embed
│   └── .env.example
│
├── rag_server/
│   ├── main.py                   # FastAPI app
│   ├── middleware.py             # X-Project-Key enforcement (401/403)
│   ├── project_manager.py        # project registry (JSON files)
│   ├── project_context.py        # per-request ContextVar isolation
│   ├── ingest_ops.py             # file scan, chunking, embedding, sync
│   ├── store.py                  # ChromaDB client factory
│   ├── chunker.py                # code-aware chunking (Python AST etc.)
│   ├── embeddings.py             # HTTP client to embed_server
│   ├── hybrid_search.py          # in-memory BM25 + RRF fusion
│   ├── reranker.py               # CrossEncoder reranker (optional)
│   ├── watcher.py                # FS watcher (disabled by default)
│   ├── router_{ingest,query,sync,project}.py
│   ├── settings.py               # pydantic-settings
│   ├── error_reporter.py         # structured error logging
│   ├── install.sh
│   ├── install_service.sh        # systemd service installer
│   ├── start_rag.sh              # start with .env validation
│   ├── requirements.txt
│   ├── environment.yml           # conda env: mcp-rag
│   └── .env.example
│
├── mcp_server/
│   ├── gateway.py                # MCP tools + project key resolution
│   ├── lifecycle.py              # embed_server health check helper
│   ├── install.sh
│   ├── start_mcp.sh
│   ├── requirements.txt
│   ├── environment.yml           # conda env: mcp-gateway
│   └── .env.example
│
├── examples/
│   ├── search.py                 # semantic search via HTTP API
│   ├── ingest.py                 # ingest + poll job status
│   ├── mcp_config.json           # MCP config for stdio/http/remote setups
│   ├── api_curl.sh               # curl examples for all endpoints
│   └── env/                      # annotated .env templates
│       ├── local.env.example     # all components on one machine
│       ├── remote.env.example    # GPU server + local gateway
│       └── README.md
│
└── tests/
    ├── test_onnx.py              # embed_server ONNX CPU inference test
    └── test_gpu.py               # embed_server GPU (CUDA/ROCm) test
```

## Requirements

| Component | Python | Key packages |
|---|---|---|
| embed_server | 3.12 | `onnxruntime` (or `-gpu`/`-rocm`), `transformers>=4.40`, `optimum>=1.21`, `fastapi` |
| rag_server | 3.12 | `chromadb==0.5.5`, `fastapi`, `pathspec`, `watchdog` |
| mcp_server | 3.12 | `mcp>=1.12.4`, `chromadb==0.5.5`, `requests` |

Optional: `sentence-transformers>=2.7.0` for cross-encoder reranking (installs PyTorch ~2 GB).

## GPU Support

`embed_server/install.sh` auto-detects hardware and installs the correct `onnxruntime` variant:

| Hardware | Package | `EMBED_BACKEND` |
|---|---|---|
| Any CPU | `onnxruntime` | `onnx-cpu` |
| NVIDIA GPU | `onnxruntime-gpu` | `onnx-cuda` |
| AMD GPU | `onnxruntime-rocm` | `onnx-rocm` |
| Auto-detect | — | `onnx-auto` |

For GPU driver installation: `bash install_adds.sh`

## Multi-Project Isolation

Each project directory gets its own namespace:

1. Gateway reads CWD, generates/reads UUID4 from `.mcp-rag` file
2. Registers key with rag_server on startup
3. All ChromaDB data in `mcp_rag_projects/<key>/chroma_db/`
4. Every request carries `X-Project-Key`; rag_server returns 401/403 without it

Multiple agents on different projects run simultaneously with full data isolation.

## Troubleshooting

**IDE hangs on startup**
- Gateway never kills embed_server on startup (cascade failure fix)
- Check embed_server is running: `curl http://localhost:8001/healthz`

**First start is slow (3–8 min)**
- BGE-M3 is exported to ONNX on first run; result cached in `HF_HOME/onnx_exports/bge-m3/`
- Subsequent starts load from cache in under 5 seconds

**Search returns no results**
- Verify project is indexed: `python examples/search.py "test"`
- Check project key: `GET http://localhost:8000/project/list`
- rag_server returns 401 if `X-Project-Key` header is missing

**inotify watch limit exceeded**
- `FS_WATCHER_ENABLED=false` is the default — watchdog is disabled
- If you enabled it: `sudo sysctl fs.inotify.max_user_watches=524288`

## License

MIT

---

## Connecting to AI Agents

### Claude Code

Official CLI: https://docs.anthropic.com/en/docs/claude-code/getting-started

```bash
# Installation
npm install -g @anthropic-ai/claude-code

# Add MCP server (HTTP mode — gateway runs separately)
claude mcp add mcp-rag --transport http http://127.0.0.1:8002/mcp

# Or stdio mode (gateway runs automatically from project folder)
claude mcp add mcp-rag python /path/to/mcp-rag/mcp_server/gateway.py \
  --env PYTHONPATH=/path/to/mcp-rag

# Verify
claude mcp list
```

Config is saved to `~/.claude.json`:

```json
{
  "mcpServers": {
    "mcp-rag": {
      "type": "http",
      "url": "http://127.0.0.1:8002/mcp"
    }
  }
}
```

### Gemini CLI

Official CLI: https://cloud.google.com/gemini/docs/codeassist/cli-getting-started

```bash
# Installation
pip install google-generativeai
# or via gcloud
gcloud components install gemini

# MCP config in ~/.gemini/settings.json
```

```json
{
  "mcpServers": {
    "mcp-rag": {
      "url": "http://127.0.0.1:8002/mcp",
      "transport": "http"
    }
  }
}
```

For stdio mode, specify `command` instead of `url` — format is similar to Claude Code.

### OpenAI Codex CLI

Official CLI: https://github.com/openai/codex

```bash
# Installation
npm install -g @openai/codex

# MCP config in ~/.codex/config.json or specify via flag
codex --mcp-server "http://127.0.0.1:8002/mcp" "search for authentication logic"
```

```json
{
  "mcpServers": {
    "mcp-rag": {
      "type": "http",
      "url": "http://127.0.0.1:8002/mcp"
    }
  }
}
```

---

## Deployment Topologies

### Topology 1 — All Local

All three components on a single developer machine. The simplest option.

```
[Developer Machine]
  IDE/Agent → mcp_server:8002 → rag_server:8000 → embed_server:8001
```

Config (copy from example):
```bash
cp examples/env/local.env.example .env
```

Key `.env` settings:
```env
EMBED_HOST=127.0.0.1
EMBED_SERVER_URL=http://127.0.0.1:8001
RAG_SERVER=http://127.0.0.1:8000
RAG_BACKEND=http
MCP_TRANSPORT=http
```

Startup:
```bash
bash embed_server/start_embed.sh
bash rag_server/start_rag.sh
bash mcp_server/start_mcp.sh
```

---

### Topology 2 — Everything remote, MCP connects remotely

Entire stack on a GPU server. IDE connects to the MCP gateway over the network.

```
[GPU Server: gpu-host]
  embed_server:8001
  rag_server:8000
  mcp_server:8002   ← IDE connects here via HTTP

[Developer Machine]
  IDE/Agent → http://gpu-host:8002/mcp
```

On server — `.env`:
```env
EMBED_HOST=0.0.0.0
EMBED_PORT=8001
EMBED_BACKEND=onnx-cuda      # or onnx-rocm for AMD

RAG_HOST=0.0.0.0
RAG_PORT=8000
EMBED_SERVER_URL=http://127.0.0.1:8001

MCP_HOST=0.0.0.0
MCP_PORT=8002
RAG_BACKEND=http
RAG_SERVER=http://127.0.0.1:8000
```

In IDE (locally):
```json
{
  "mcpServers": {
    "mcp-rag": {
      "type": "http",
      "url": "http://gpu-host:8002/mcp"
    }
  }
}
```

---

### Topology 3 — MCP local, RAG and embed remote

Lightweight gateway on the developer machine, heavy components on the server.

```
[GPU Server: gpu-host]
  embed_server:8001
  rag_server:8000

[Developer Machine]
  mcp_server:8002  → http://gpu-host:8000
  IDE/Agent → mcp_server:8002 (or stdio)
```

On server, run only embed and rag:
```bash
bash embed_server/start_embed.sh
bash rag_server/start_rag.sh
```

Locally — `mcp_server/.env`:
```env
RAG_BACKEND=http
RAG_SERVER=http://gpu-host:8000
MCP_TRANSPORT=http
MCP_HOST=127.0.0.1
MCP_PORT=8002
```

Locally — startup:
```bash
bash mcp_server/start_mcp.sh
```

Or without a separate server (stdio):
```json
{
  "mcpServers": {
    "mcp-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/mcp-rag/mcp_server/gateway.py"],
      "env": {
        "PYTHONPATH": "/path/to/mcp-rag",
        "RAG_BACKEND": "http",
        "RAG_SERVER": "http://gpu-host:8000"
      }
    }
  }
}
```

---

### Topology 4 — MCP + RAG local, multiple remote embed_servers

For large projects or parallel ingest: multiple GPU workers compute embeddings simultaneously. rag_server divides batches equally among them, executing in parallel.

```
[GPU Server 1: gpu1-host]
  embed_server:8001  ← processes half of the batch

[GPU Server 2: gpu2-host]
  embed_server:8001  ← processes the other half of the batch

[Local/Server: rag-host]
  rag_server:8000    ← divides the batch among workers

[Developer Machine]
  mcp_server → rag_server
```

In `rag_server/.env` — list all workers separated by commas:
```env
# If EMBED_SERVER_URLS is present, EMBED_SERVER_URL is ignored
EMBED_SERVER_URLS=http://gpu1-host:8001,http://gpu2-host:8001

# If workers are on different ports:
# EMBED_SERVER_URLS=http://gpu1-host:8001,http://gpu2-host:8001,http://gpu3-host:8002
```

How it works: rag_server divides the list of chunks equally among workers, sends requests in parallel via `ThreadPoolExecutor`, and collects results in the original order. Linear scaling of ingest speed with the number of workers.

Each embed_server is an independent process with its own `.env` and choice of GPU backend. You can mix NVIDIA and AMD workers.

---

## Compute Resource Management

All performance settings are configured via environment variables in `embed_server/.env`.

> **Note on GPU support**
>
> The author did not have access to dedicated GPU hardware during development, so the NVIDIA CUDA and AMD ROCm code paths have not been thoroughly tested. Apologies in advance if something does not work correctly or contains errors in that area. Any feedback, bug reports, or pull requests that improve GPU stability and quality are very much appreciated.
>
> — *Denis Igorovich Kornilov, Kyiv, Ukraine*

### CPU — core management

embed_server creates **two independent ORT sessions**: one for ingest, one for search. Both run in parallel at the OS level — the Python GIL does not block native C++ ORT inference.

| Variable | Default | Description |
|---|---|---|
| `ONNX_NUM_THREADS` | `4` | Ingest session threads (intra-op: parallelism within a single op) |
| `SEARCH_THREADS` | `1` | Search session threads (queries are short — one core is sufficient) |
| `ONNX_INTER_THREADS` | `1` | Threads between ORT graph ops (inter-op parallelism) |
| `ONNX_EXECUTION_MODE` | `sequential` | Graph execution mode: `sequential` or `parallel` |

**Thread count recommendations:**

```env
# Server with 8 physical cores — ingest + search running simultaneously
ONNX_NUM_THREADS=6
SEARCH_THREADS=2

# Laptop, 4 cores
ONNX_NUM_THREADS=3
SEARCH_THREADS=1

# parallel mode + inter_threads only makes sense on servers with 16+ cores
ONNX_EXECUTION_MODE=parallel
ONNX_INTER_THREADS=2
```

> `ONNX_NUM_THREADS + SEARCH_THREADS` = total cores allocated to inference.  
> Leave 1–2 cores free for FastAPI request handling.

---

### NVIDIA GPU — CUDA

**Supported devices:**

| Series | Examples | Requirements |
|---|---|---|
| GeForce GTX 9xx / 10xx / 16xx | GTX 970, GTX 1080 Ti, GTX 1660 | CUDA 11+, compute capability 5.2+ |
| GeForce RTX 20xx / 30xx / 40xx / 50xx | RTX 2080, RTX 3090, RTX 4090 | CUDA 11.8+ |
| Data Center / HPC | T4, V100, A100, H100, H200, L40S | CUDA 12+ for H100/H200 |
| Embedded / Jetson | Jetson Orin, AGX Xavier | JetPack 5+ |

**Libraries:**

- `onnxruntime-gpu>=1.19.0` — replaces `onnxruntime` (both packages cannot be installed at the same time)
- CUDA Toolkit 11.8 or 12.x + cuDNN 8.x / 9.x (matching versions)
- NVIDIA driver ≥ 525

**Installation:**

```bash
bash embed_server/install.sh
# → select "NVIDIA (CUDA)"
```

Or manually:
```bash
pip uninstall onnxruntime -y
pip install "onnxruntime-gpu>=1.19.0"
```

**Configuration:**

```env
EMBED_BACKEND=onnx-cuda

# Select GPU when multiple devices are present (nvidia-smi -L to list)
CUDA_DEVICE_ID=0
```

**Verify:**

```bash
nvidia-smi                             # list all GPUs
python -c "import onnxruntime as o; print(o.get_available_providers())"
# → [..., 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

### AMD GPU — ROCm

**Supported devices:**

| Series | Examples |
|---|---|
| RDNA 2 (RX 6xxx) | RX 6700 XT, RX 6800 XT, RX 6900 XT |
| RDNA 3 (RX 7xxx) | RX 7800 XT, RX 7900 XTX |
| Instinct (HPC) | MI100, MI200, MI250, MI300X |

> **Note:** Consumer GPUs (RX 6xxx/7xxx) may require `HSA_OVERRIDE_GFX_VERSION` to bypass ROCm restrictions on certain driver versions.

**Libraries:**

- ROCm 5.6 / 6.x — AMD GPU platform (Linux only)
- `onnxruntime-rocm` — built against a specific ROCm version

**Installation:**

```bash
bash embed_server/install.sh
# → select "AMD (ROCm)"
```

Or manually:
```bash
pip uninstall onnxruntime -y
pip install onnxruntime-rocm  # version depends on installed ROCm
```

**Configuration:**

```env
EMBED_BACKEND=onnx-rocm

# Select GPU when multiple devices are present (rocm-smi --showproductname to list)
ROCM_DEVICE_ID=0

# Consumer GPUs (RX 6xxx/7xxx) may need:
# export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RDNA2
# export HSA_OVERRIDE_GFX_VERSION=11.0.0   # RDNA3
```

**Verify:**

```bash
rocm-smi --showproductname
python -c "import onnxruntime as o; print(o.get_available_providers())"
# → [..., 'ROCMExecutionProvider', 'CPUExecutionProvider']
```

---

### Auto backend selection

```env
EMBED_BACKEND=onnx-auto
```

Priority order: CUDA → ROCm → CPU. Convenient when deploying across machines with different hardware.

---

### CUDA provider options passed to ORT

| ORT option | Value | Description |
|---|---|---|
| `device_id` | `CUDA_DEVICE_ID` | GPU index |
| `arena_extend_strategy` | `kNextPowerOfTwo` | GPU memory growth strategy |
| `cudnn_conv_algo_search` | `EXHAUSTIVE` | cuDNN algorithm search (slower on first run, faster on subsequent runs) |
| `do_copy_in_default_stream` | `True` | Synchronous H↔D copy (safer for parallel sessions) |

