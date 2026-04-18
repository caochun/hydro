#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
SUBMODULE_DIR="$REPO_DIR/qgis-mcp"

if [ ! -d "$SUBMODULE_DIR" ]; then
  echo "qgis-mcp submodule is missing at: $SUBMODULE_DIR" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  cd "$SUBMODULE_DIR"
  exec uv run --no-sync src/qgis_mcp/server.py
fi

if [ -x "$SUBMODULE_DIR/.venv/bin/python" ]; then
  exec "$SUBMODULE_DIR/.venv/bin/python" "$SUBMODULE_DIR/src/qgis_mcp/server.py"
fi

cat >&2 <<EOF
QGIS MCP runtime is not ready.

Choose one setup path:
1. Install uv, then run:
   cd "$SUBMODULE_DIR" && uv sync
2. Or run the upstream installer:
   cd "$SUBMODULE_DIR" && python install.py

After that:
- restart QGIS
- enable the "QGIS MCP" plugin
- click "Start Server" inside QGIS
- restart your MCP client
EOF

exit 1
