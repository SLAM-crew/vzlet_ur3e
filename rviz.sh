#!/usr/bin/env bash
set -euo pipefail

ensure_x11_access() {
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is not set. Visualization is required; export DISPLAY first (example: export DISPLAY=:0)." >&2
    exit 1
  fi
  if ! command -v xhost >/dev/null 2>&1; then
    echo "xhost is not installed. Install xhost to grant X11 access for Docker visualization." >&2
    exit 1
  fi

  # Allow local Docker clients to connect to the host X server.
  if ! xhost +local:docker >/dev/null 2>&1 && ! xhost +SI:localuser:root >/dev/null 2>&1; then
    echo "Failed to grant X11 access via xhost. Run xhost manually and retry." >&2
    exit 1
  fi
}

ensure_x11_access
sudo docker compose -f docker/rviz.compose.yaml up rviz