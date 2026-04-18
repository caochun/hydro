# QGIS MCP Setup

This repository now vendors [qgis-mcp](./qgis-mcp) as a git submodule and exposes a project-local `qgis` MCP server entry.

## What was added

- Git submodule: `qgis-mcp`
- Claude Code style config: [`.mcp.json`](/Users/chun/Develop/hydro/.mcp.json)
- VS Code MCP config: [`.vscode/mcp.json`](/Users/chun/Develop/hydro/.vscode/mcp.json)
- Server launcher: [`scripts/run-qgis-mcp.sh`](/Users/chun/Develop/hydro/scripts/run-qgis-mcp.sh)

Both MCP configs launch the local wrapper script, which then starts the vendored `qgis-mcp` server from the submodule.

## One-time setup

### 1. Prepare the QGIS MCP runtime

Recommended:

```bash
cd qgis-mcp
python install.py
```

If you prefer `uv`:

```bash
cd qgis-mcp
uv sync
```

## 2. Install and enable the QGIS plugin

The upstream installer can symlink the plugin automatically. If you do it manually on macOS, link:

```bash
ln -s "/Users/chun/Develop/hydro/qgis-mcp/qgis_mcp_plugin" \
  "$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/qgis_mcp_plugin"
```

Then restart QGIS and enable `QGIS MCP` in `Plugins` -> `Manage and Install Plugins`.

## 3. Start the QGIS side server

Inside QGIS:

1. Open the `QGIS MCP` panel or toolbar button.
2. Click `Start Server`.

## 4. Restart your MCP client

Once your client reloads project MCP config, it should see a `qgis` server entry that launches from this repository.

## Notes

- The wrapper script prefers `uv`; if `uv` is not installed, it falls back to `qgis-mcp/.venv/bin/python`.
- The QGIS desktop plugin must be running, otherwise the MCP server starts but cannot talk to QGIS.
- The vendored submodule is currently pinned to commit `22e506a` (`v0.2.1`).
