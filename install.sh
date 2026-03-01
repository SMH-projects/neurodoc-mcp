#!/bin/bash
# NeuroDoc MCP — установка одной командой

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PATH="$SCRIPT_DIR/server.py"
VENV_PATH="$SCRIPT_DIR/.venv"
PYTHON="$VENV_PATH/bin/python"

echo ""
echo "🚀 NeuroDoc MCP — Установка"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Зависимости через uv
echo ""
echo "📦 Шаг 1/3 — Установка зависимостей..."
if command -v uv &>/dev/null; then
    uv venv "$VENV_PATH" --python 3.12 --quiet 2>/dev/null || uv venv "$VENV_PATH" --quiet
    uv pip install mcp --python "$PYTHON" --quiet
else
    python3 -m venv "$VENV_PATH"
    "$PYTHON" -m pip install mcp --quiet
fi
echo "   ✓ mcp установлен"

# 2. Регистрация в Claude Code
echo ""
echo "🔌 Шаг 2/3 — Регистрация MCP сервера в Claude Code..."
claude mcp add --transport stdio ndoc -- "$PYTHON" "$SERVER_PATH"
echo "   ✓ ndoc зарегистрирован"

# 3. Проверка
echo ""
echo "✅ Шаг 3/3 — Проверка..."
claude mcp list | grep ndoc && echo "   ✓ ndoc активен" || echo "   ⚠ Проверь вручную: claude mcp list"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ NeuroDoc установлен!"
echo ""
echo "Команды в Claude Code:"
echo "  ndoc_init(\"/путь/к/проекту\")      — инициализация"
echo "  ndoc_update(\"/путь/к/проекту\")    — обновление"
echo "  ndoc_validate(\"/путь/к/проекту\")  — проверка актуальности"
echo "  ndoc_status(\"/путь/к/проекту\")    — статус"
echo ""
