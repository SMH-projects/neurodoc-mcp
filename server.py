#!/usr/bin/env python3
"""
NeuroDoc MCP Server
Commands: ndoc_init · ndoc_update · ndoc_validate · ndoc_status

Usage:
  python server.py                        # stdio (default, for local Claude Code)
  python server.py --transport http       # HTTP server on port 8000
  python server.py --transport http --port 9000 --host 0.0.0.0
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from datetime import datetime
import os
import re
import ast
import subprocess
import argparse
import sys

# ── Transport config ──────────────────────────────────────────────────────────
def _get_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--transport", default=os.getenv("NDOC_TRANSPORT", "stdio"),
                   choices=["stdio", "streamable-http", "sse"])
    p.add_argument("--host", default=os.getenv("NDOC_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("NDOC_PORT", "8000")))
    args, _ = p.parse_known_args()
    return args

_args = _get_args()

mcp = FastMCP(
    "ndoc",
    host=_args.host,
    port=_args.port,
    instructions="NeuroDoc — AI navigation for codebases via context.md files",
)

# WORKSPACE_ROOT: если задан, все пути резолвятся внутри него
# Пример: WORKSPACE_ROOT=/workspace → ndoc_init("my-app") → /workspace/my-app
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", "")

def resolve_path(user_path: str = "") -> Path:
    """Резолвит путь. Если пусто — берёт текущую директорию."""
    if not user_path or user_path.strip() == ".":
        base = Path(WORKSPACE_ROOT) if WORKSPACE_ROOT else Path.cwd()
        return base.resolve()
    p = Path(user_path).expanduser()
    if WORKSPACE_ROOT and not p.is_absolute():
        p = Path(WORKSPACE_ROOT) / p
    return p.resolve()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv',
    'dist', 'build', '.next', '.nuxt', 'vendor', '.idea',
    '.vscode', 'coverage', '.cache', 'tmp', 'logs',
}

CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go'}


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────

def parse_python(filepath: Path) -> dict:
    try:
        source = filepath.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(source)
    except Exception:
        return {'functions': [], 'imports': []}

    functions, imports = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args if a.arg != 'self'][:4]
            functions.append({'name': node.name, 'params': params})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return {'functions': functions, 'imports': imports}


def parse_js_ts(filepath: Path) -> dict:
    try:
        source = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}

    functions, imports = [], []
    patterns = [
        r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
        r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>',
        r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(([^)]*)\)',
    ]
    for pat in patterns:
        for m in re.finditer(pat, source):
            params = [p.strip().split(':')[0].split('=')[0].strip()
                      for p in m.group(2).split(',') if p.strip()][:4]
            functions.append({'name': m.group(1), 'params': params})

    for m in re.finditer(r"from\s+['\"]([^'\"]+)['\"]", source):
        imports.append(m.group(1))
    return {'functions': functions, 'imports': imports}


def parse_go(filepath: Path) -> dict:
    try:
        source = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}

    functions, imports = [], []
    for m in re.finditer(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)', source):
        params = [p.strip() for p in m.group(2).split(',') if p.strip()][:4]
        functions.append({'name': m.group(1), 'params': params})

    in_import_block = False
    for line in source.split('\n'):
        line = line.strip()
        if line.startswith('import ('):
            in_import_block = True
            continue
        if in_import_block:
            if line == ')':
                in_import_block = False
            else:
                m = re.search(r'"([^"]+)"', line)
                if m:
                    imports.append(m.group(1).split('/')[-1])
        elif line.startswith('import "'):
            m = re.search(r'"([^"]+)"', line)
            if m:
                imports.append(m.group(1).split('/')[-1])

    return {'functions': functions, 'imports': imports}


def parse_file(filepath: Path) -> dict:
    ext = filepath.suffix.lower()
    if ext == '.py':
        return parse_python(filepath)
    elif ext in ('.js', '.jsx', '.ts', '.tsx'):
        return parse_js_ts(filepath)
    elif ext == '.go':
        return parse_go(filepath)
    return {'functions': [], 'imports': []}


def scan_dir(dir_path: Path) -> dict:
    """Scan directory, return {filename: parsed_data} for files with functions."""
    result = {}
    for f in dir_path.iterdir():
        if f.is_file() and f.suffix.lower() in CODE_EXTENSIONS:
            parsed = parse_file(f)
            if parsed['functions']:
                result[f.name] = parsed
    return result


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def short_name(dir_path: Path, root: Path) -> str:
    if dir_path == root:
        return root.name
    return dir_path.name


def resolve_deps(imports: list, all_modules: list) -> list:
    deps = []
    for imp in imports:
        imp_lower = imp.lower().replace('-', '_').replace('/', '_')
        for mod in all_modules:
            mod_lower = mod.lower()
            if mod_lower in imp_lower or imp_lower == mod_lower:
                if mod not in deps:
                    deps.append(mod)
    return deps


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


# ─────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────

def make_context_md(dir_path: Path, root: Path, all_modules: list) -> str:
    files_data = scan_dir(dir_path)
    mod = short_name(dir_path, root)

    all_imports = []
    for fd in files_data.values():
        all_imports.extend(fd.get('imports', []))

    deps = resolve_deps(all_imports, [m for m in all_modules if m != mod])

    header = f"# {mod}"
    if deps:
        header += f" | {', '.join(deps[:5])}"

    lines = [header]
    for fname, fd in files_data.items():
        funcs = fd.get('functions', [])[:8]
        if funcs:
            parts = []
            for fn in funcs:
                p = ','.join(fn['params'][:3])
                parts.append(f"{fn['name']}({p})")
            lines.append(f"{fname}: {' | '.join(parts)}")

    tags = {mod}
    for d in deps:
        tags.add(d)
    for fname in files_data:
        base = re.sub(r'\.(py|go|ts|js|tsx|jsx)$', '', fname)
        tags.add(base)

    lines.append(f"tags: {' '.join(list(tags)[:12])}")
    lines.append(f"upd: {datetime.now().strftime('%Y-%m-%d')} ndoc")
    return '\n'.join(lines)


def make_index(modules: dict, project_name: str) -> str:
    now = datetime.now().strftime('%Y-%m-%d')
    lines = [
        f"# Project: {project_name} | modules: {len(modules)} | updated: {now}",
        "",
        "## Map",
    ]
    for mod, data in modules.items():
        deps = data.get('deps', [])
        kw = data.get('keywords', [])[:4]
        dep_str = ', '.join(deps) if deps else '(none)'
        kw_str = ', '.join(kw)
        lines.append(f"{mod} → {dep_str} | {kw_str}")

    lines += ["", "## Dependency Graph", "", "```mermaid", "graph TD"]
    for mod, data in modules.items():
        for dep in data.get('deps', []):
            if dep in modules:
                lines.append(f"    {mod} --> {dep}")
    lines.append("```")

    lines += ["", "## C4 Component Diagram", "", "```mermaid", "C4Component"]
    lines.append(f'    title Component diagram for {project_name}')
    for mod in modules:
        lines.append(f'    Component({mod}, "{mod}", "module")')
    for mod, data in modules.items():
        for dep in data.get('deps', []):
            if dep in modules:
                lines.append(f'    Rel({mod}, {dep}, "uses")')
    lines.append("```")

    return '\n'.join(lines)


CLAUDE_MD_RULES = """
## NeuroDoc — навигационные правила

### ПЕРЕД каждой задачей:
1. Прочитай `context.index.md` в корне проекта
2. Определи какие модули затрагивает задача
3. Прочитай `context.md` этих модулей
4. Прочитай `context.md` их зависимостей (→ links)
5. Только потом приступай

### ПОСЛЕ каждого изменения кода:
1. Обнови `context.md` в изменённой директории
2. Обнови `context.index.md` если добавился новый модуль или изменились связи
3. Формат: компактный (не более 120 токенов на файл)

### Формат context.md:
```
# module_name | dep1, dep2
file.ext: FuncA(params) | FuncB(params)
tags: keyword1 keyword2
upd: YYYY-MM-DD author
```
"""


# ─────────────────────────────────────────────
# MCP TOOLS
# ─────────────────────────────────────────────

@mcp.tool()
def ndoc_init(project_path: str = "") -> str:
    """
    Инициализирует NeuroDoc для проекта.
    Создаёт context.md в каждой папке с кодом, context.index.md с C4 диаграммой,
    и добавляет правила в CLAUDE.md.

    project_path: путь к проекту. Если не указан — использует текущую директорию.
    Примеры:
      ndoc_init()                          # текущая папка
      ndoc_init(".")                       # текущая папка
      ndoc_init("/workspace/my-project")   # конкретный путь
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"❌ Папка не найдена: {project_path}"

    out = []
    out.append(f"🚀 NeuroDoc Init — {root.name}")
    out.append("━" * 48)

    # ── Шаг 1: Сканирование ──────────────────────────
    out.append("\n📂 Шаг 1/5 — Сканирование проекта...")
    dirs_with_code = []
    for dp, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        dp = Path(dp)
        if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files):
            dirs_with_code.append(dp)

    out.append(f"   ✓ Найдено папок с кодом: {len(dirs_with_code)}")

    # ── Шаг 2: Парсинг ───────────────────────────────
    out.append("\n🔍 Шаг 2/5 — Парсинг файлов...")
    all_module_names = [short_name(d, root) for d in dirs_with_code]
    modules: dict = {}

    for dp in dirs_with_code:
        mod = short_name(dp, root)
        files_data = scan_dir(dp)
        all_imports, keywords = [], []
        for fd in files_data.values():
            all_imports.extend(fd.get('imports', []))
            keywords.extend([fn['name'] for fn in fd.get('functions', [])[:2]])

        deps = resolve_deps(all_imports, [m for m in all_module_names if m != mod])
        func_count = sum(len(fd.get('functions', [])) for fd in files_data.values())

        modules[mod] = {
            'dir_path': dp,
            'files': files_data,
            'deps': deps,
            'keywords': keywords[:4],
        }
        out.append(f"   ✓ {safe_rel(dp, root) or root.name}/ — "
                   f"{len(files_data)} файлов, {func_count} функций")

    # ── Шаг 3: context.md ────────────────────────────
    out.append(f"\n📝 Шаг 3/5 — Генерация context.md...")
    generated = 0
    for mod, data in modules.items():
        dp = data['dir_path']
        content = make_context_md(dp, root, all_module_names)
        (dp / 'context.md').write_text(content, encoding='utf-8')
        rel = safe_rel(dp, root) or root.name
        out.append(f"   ✓ {rel}/context.md")
        generated += 1

    out.append(f"   → Создано: {generated} файлов")

    # ── Шаг 4: context.index.md + C4 ─────────────────
    out.append(f"\n🗺️  Шаг 4/5 — context.index.md + C4 диаграмма...")
    index_content = make_index(modules, root.name)
    (root / 'context.index.md').write_text(index_content, encoding='utf-8')
    out.append(f"   ✓ context.index.md — {len(modules)} модулей")
    out.append(f"   ✓ Dependency Graph (Mermaid)")
    out.append(f"   ✓ C4 Component Diagram (Mermaid)")

    # ── Шаг 5: CLAUDE.md ─────────────────────────────
    out.append(f"\n⚙️  Шаг 5/5 — Обновление CLAUDE.md...")
    claude_path = root / 'CLAUDE.md'
    if claude_path.exists():
        existing = claude_path.read_text(encoding='utf-8')
        if 'NeuroDoc' not in existing:
            claude_path.write_text(existing + CLAUDE_MD_RULES, encoding='utf-8')
            out.append("   ✓ Правила NeuroDoc добавлены в существующий CLAUDE.md")
        else:
            out.append("   ℹ️  CLAUDE.md уже содержит правила NeuroDoc")
    else:
        claude_path.write_text(f"# {root.name}\n{CLAUDE_MD_RULES}", encoding='utf-8')
        out.append("   ✓ CLAUDE.md создан с правилами NeuroDoc")

    # ── Итог ─────────────────────────────────────────
    out.append("")
    out.append("━" * 48)
    out.append(f"✅ NeuroDoc инициализирован успешно!")
    out.append(f"")
    out.append(f"   📁 {generated} context.md файлов создано")
    out.append(f"   🗺️  context.index.md с C4 диаграммой")
    out.append(f"   ⚙️  CLAUDE.md обновлён")
    out.append(f"")
    out.append(f"💡 Теперь Claude будет читать context.md перед каждой задачей.")
    out.append(f"   Запусти ndoc_validate для проверки актуальности.")
    return '\n'.join(out)


@mcp.tool()
def ndoc_update(project_path: str = "", changed_files: str = "") -> str:
    """
    Обновляет context.md для изменённых файлов.
    changed_files — список файлов через запятую (относительные пути).
    Если пусто — берёт из git diff или обновляет всё.
    project_path: если не указан — текущая директория.
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"❌ Папка не найдена: {project_path}"

    out = []
    out.append("🔄 NeuroDoc Update")
    out.append("━" * 40)

    # Определяем папки для обновления
    if changed_files.strip():
        files = [root / f.strip() for f in changed_files.split(',')]
        dirs_to_update = list({f.parent for f in files if f.exists()})
        out.append(f"\n   Файлы: {changed_files}")
    else:
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD'],
                cwd=root, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                files = [root / f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                dirs_to_update = list({f.parent for f in files if f.parent.exists()})
                out.append(f"\n   Из git diff: {len(files)} изменённых файлов")
            else:
                raise ValueError("no diff")
        except Exception:
            dirs_to_update = []
            for dp, subdirs, files_in_dir in os.walk(root):
                subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
                dp = Path(dp)
                if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files_in_dir):
                    dirs_to_update.append(dp)
            out.append(f"\n   Git diff не найден — обновляем все {len(dirs_to_update)} модулей")

    if not dirs_to_update:
        return "ℹ️  Нечего обновлять"

    # Собираем все имена модулей
    all_module_names = []
    for dp, subdirs, _ in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        all_module_names.append(short_name(Path(dp), root))

    out.append(f"\n   Обновляем {len(dirs_to_update)} модуль(ей):\n")
    updated = 0
    for dp in dirs_to_update:
        content = make_context_md(dp, root, all_module_names)
        (dp / 'context.md').write_text(content, encoding='utf-8')
        rel = safe_rel(dp, root) or root.name
        out.append(f"   ✓ {rel}/context.md")
        updated += 1

    # Пересобираем индекс
    modules: dict = {}
    for dp, subdirs, files_in_dir in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        dp = Path(dp)
        if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files_in_dir):
            mod = short_name(dp, root)
            fd_map = scan_dir(dp)
            all_imp, kw = [], []
            for fd in fd_map.values():
                all_imp.extend(fd.get('imports', []))
                kw.extend([fn['name'] for fn in fd.get('functions', [])[:2]])
            deps = resolve_deps(all_imp, [m for m in all_module_names if m != mod])
            modules[mod] = {'dir_path': dp, 'files': fd_map, 'deps': deps, 'keywords': kw[:4]}

    index_content = make_index(modules, root.name)
    (root / 'context.index.md').write_text(index_content, encoding='utf-8')

    out.append(f"\n   ✓ context.index.md обновлён")
    out.append(f"\n✅ Обновлено {updated} context.md файлов")
    return '\n'.join(out)


@mcp.tool()
def ndoc_validate(project_path: str = "") -> str:
    """
    Проверяет актуальность context.md файлов.
    Показывает какие модули свежие, устаревшие или без документации.
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"❌ Папка не найдена: {project_path}"

    out = []
    out.append("🔍 NeuroDoc Validate")
    out.append("━" * 40)

    fresh, stale, missing = [], [], []

    for dp, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        dp = Path(dp)
        code_files = [f for f in files if Path(f).suffix.lower() in CODE_EXTENSIONS]
        if not code_files:
            continue

        ctx = dp / 'context.md'
        rel = safe_rel(dp, root) or root.name

        if not ctx.exists():
            missing.append(rel)
            continue

        ctx_mtime = ctx.stat().st_mtime
        code_mtimes = [(dp / f).stat().st_mtime for f in code_files if (dp / f).exists()]
        if code_mtimes and max(code_mtimes) > ctx_mtime:
            stale.append(rel)
        else:
            fresh.append(rel)

    total = len(fresh) + len(stale) + len(missing)
    out.append("")

    if fresh:
        out.append(f"✅ Актуальные ({len(fresh)}):")
        for m in fresh:
            out.append(f"   ✓ {m}/")

    if stale:
        out.append(f"\n⚠️  Устаревшие ({len(stale)}) — код изменён после context.md:")
        for m in stale:
            out.append(f"   ⚠ {m}/")

    if missing:
        out.append(f"\n❌ Нет context.md ({len(missing)}):")
        for m in missing:
            out.append(f"   ✗ {m}/")

    out.append("")
    if not stale and not missing:
        out.append("✅ Все context.md актуальны!")
    else:
        pct = int(len(fresh) / total * 100) if total > 0 else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        out.append(f"📊 [{bar}] {pct}% актуально")
        out.append(f"   {len(fresh)}/{total} свежих · {len(stale)} устаревших · {len(missing)} отсутствуют")
        out.append("")
        out.append("💡 Запусти ndoc_update для обновления")

    return '\n'.join(out)


@mcp.tool()
def ndoc_status(project_path: str = "") -> str:
    """
    Показывает обзор состояния NeuroDoc для проекта.
    Отображает карту модулей и граф зависимостей.
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"❌ Папка не найдена: {project_path}"

    out = []
    out.append(f"📊 NeuroDoc Status — {root.name}")
    out.append("━" * 40)

    index_path = root / 'context.index.md'
    ctx_count = sum(
        1 for dp, subdirs, files in os.walk(root)
        for _ in [subdirs.__setitem__(slice(None), [d for d in subdirs if d not in SKIP_DIRS])]
        if 'context.md' in files
    )

    out.append(f"\n   📁 context.md файлов: {ctx_count}")
    out.append(f"   📄 context.index.md: {'✓ найден' if index_path.exists() else '✗ отсутствует'}")
    claude_path = root / 'CLAUDE.md'
    has_neurodoc = claude_path.exists() and 'NeuroDoc' in claude_path.read_text(encoding='utf-8')
    out.append(f"   ⚙️  CLAUDE.md правила: {'✓ подключены' if has_neurodoc else '✗ не подключены'}")

    if index_path.exists():
        out.append("\n─── context.index.md ───────────────────")
        lines = index_path.read_text(encoding='utf-8').split('\n')
        # Show Map section (skip mermaid blocks for brevity)
        in_mermaid = False
        shown = 0
        for line in lines:
            if '```mermaid' in line:
                in_mermaid = True
                continue
            if in_mermaid and '```' in line:
                in_mermaid = False
                continue
            if not in_mermaid and shown < 25:
                out.append(f"  {line}")
                shown += 1
        if shown >= 25:
            out.append("  ...")
    else:
        out.append("\n💡 Запусти ndoc_init для инициализации")

    return '\n'.join(out)


if __name__ == "__main__":
    transport = _args.transport
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        print(f"🚀 NeuroDoc MCP server starting on http://{_args.host}:{_args.port}/mcp",
              file=sys.stderr)
        mcp.run(transport=transport)
