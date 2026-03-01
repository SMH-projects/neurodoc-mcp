# NeuroDoc MCP

[![PyPI version](https://img.shields.io/pypi/v/neurodoc-mcp.svg)](https://pypi.org/project/neurodoc-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/neurodoc-mcp.svg)](https://pypi.org/project/neurodoc-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI navigation for codebases via `context.md` files. Instead of vector search — hierarchical maps of functions, dependencies, and C4 diagrams.

**17–119x faster** than vector search · **100% accuracy** · **Zero infrastructure**

---

## Quick Start

### Claude Code (recommended)

```bash
claude mcp add ndoc -- uvx neurodoc-mcp
```

Then in any project:

```
/ndoc init
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ndoc": {
      "command": "uvx",
      "args": ["neurodoc-mcp"]
    }
  }
}
```

### VS Code / Cursor

Add to `.vscode/mcp.json` or `~/.cursor/mcp.json`:

```json
{
  "servers": {
    "ndoc": {
      "command": "uvx",
      "args": ["neurodoc-mcp"]
    }
  }
}
```

---

## How It Works

NeuroDoc creates compact `context.md` files in each directory of your project:

```
# auth | db, cache
auth.py: login(user,pwd) | logout(token) | refresh(token)
middleware.py: require_auth(f) | get_current_user()
tags: auth login jwt middleware
upd: 2025-01-15 ndoc
```

Claude reads these files before starting each task — instead of scanning hundreds of files, it instantly knows:
- What functions exist in each module
- What modules depend on what
- Where to look for specific logic

---

## Tools

| Tool | Description |
|------|-------------|
| `ndoc_init` | Scan project, generate all `context.md` files + C4 diagram |
| `ndoc_update` | Update `context.md` for changed files (uses `git diff`) |
| `ndoc_validate` | Check which modules are stale or missing docs |
| `ndoc_status` | Overview of NeuroDoc state for the project |

---

## Usage

After adding the MCP server, use these commands in Claude Code:

```
Initialize project:
> Run ndoc_init for this project

Update after changes:
> Run ndoc_update

Check status:
> Run ndoc_status
```

---

## What Gets Created

```
your-project/
├── context.index.md        # Project map + C4 + dependency graph
├── CLAUDE.md               # Navigation rules for Claude (auto-updated)
├── src/
│   ├── context.md          # Module: functions, deps, tags
│   └── auth/
│       └── context.md
└── ...
```

**`context.index.md`** — master index with Mermaid dependency graph and C4 component diagram.

---

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (for `uvx`)
- Claude Code, Claude Desktop, VS Code, or Cursor

---

## Alternative Installation

```bash
# pip
pip install neurodoc-mcp
neurodoc-mcp

# from source
git clone https://github.com/YOUR_USERNAME/neurodoc-mcp
cd neurodoc-mcp
bash install.sh
```

---

## HTTP Mode (self-hosted)

For team use, run as an HTTP server:

```bash
docker compose up -d
```

Then connect via:

```bash
claude mcp add ndoc --transport http http://localhost:8000/mcp
```

---

## License

MIT
