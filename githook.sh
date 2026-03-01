#!/bin/bash
# NeuroDoc git pre-commit hook — автообновление context.md
# Установка: cp githook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

CHANGED=$(git diff --cached --name-only --diff-filter=ACMR | grep -vE "context\.(md|index\.md)")
[ -z "$CHANGED" ] && exit 0

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
NEURODOC_SERVER="${NEURODOC_SERVER:-/Users/soliev/Desktop/SMH/neurodoc-mcp/server.py}"

# Собираем уникальные директории изменённых файлов
DIRS=$(echo "$CHANGED" | xargs -I{} dirname "$PROJECT_ROOT/{}" | sort -u)
FILES_LIST=$(echo "$CHANGED" | tr '\n' ',' | sed 's/,$//')

echo "🔄 NeuroDoc: обновление context.md..."
python3 -c "
import sys
sys.path.insert(0, '$(dirname $NEURODOC_SERVER)')
from server import ndoc_update
result = ndoc_update('$PROJECT_ROOT', '$FILES_LIST')
print(result)
"

# Добавляем обновлённые context.md в коммит
git add "$PROJECT_ROOT"/**/context.md "$PROJECT_ROOT"/context.index.md 2>/dev/null || true

exit 0
