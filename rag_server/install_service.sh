#!/bin/bash
# Install rag_server as a systemd service (auto-start on boot)
set -e

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="rag"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"
UVICORN="$CONDA_DIR/envs/$ENV_NAME/bin/uvicorn"
SERVICE_NAME="rag-server"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"
USER_NAME="$(whoami)"

echo "=== Installing systemd service: $SERVICE_NAME ==="

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=RAG Server (mcp-rag)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$UVICORN rag_server.main:app --host 0.0.0.0 --port 8000 --log-level info
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=$PROJECT_DIR

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo ""
echo "✓ Service installed and started"
echo ""
echo "Commands:"
echo "  sudo systemctl status $SERVICE_NAME"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo "  sudo systemctl restart $SERVICE_NAME"
