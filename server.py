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

# ── Transport config ───────────────────────────────────────────────────────────
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

WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", "")

def resolve_path(user_path: str = "") -> Path:
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
# BODY EXTRACTOR (brace counting)
# ─────────────────────────────────────────────

def extract_body(source: str, open_brace_pos: int) -> str:
    """Extract content between matching braces, handling strings."""
    depth = 1
    i = open_brace_pos + 1
    length = len(source)
    while i < length and depth > 0:
        c = source[i]
        if c in ('"', "'", '`'):
            i += 1
            while i < length and source[i] != c:
                if source[i] == '\\':
                    i += 1
                i += 1
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1
    return source[open_brace_pos:i]


def find_func_body(source: str, after_pos: int) -> str:
    """Find and extract function body (first { after position)."""
    brace_pos = source.find('{', after_pos)
    if brace_pos == -1:
        return ''
    return extract_body(source, brace_pos)


# ─────────────────────────────────────────────
# PYTHON PARSER (AST-based)
# ─────────────────────────────────────────────

def build_python_import_map(tree) -> dict:
    """alias → short_module_name"""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                short = alias.name.split('.')[-1]
                aliases[name] = short
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod_short = node.module.split('.')[-1]
            for alias in node.names:
                name = alias.asname or alias.name
                aliases[name] = mod_short
    return aliases


def get_python_func_calls(func_node, import_map: dict) -> list:
    """Extract external calls from a Python function node."""
    calls, seen = [], set()
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call):
            if (isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)):
                alias = child.func.value.id
                if alias in import_map and alias not in ('self', 'cls'):
                    key = f"{import_map[alias]}.{child.func.attr}"
                    if key not in seen:
                        seen.add(key)
                        calls.append(key)
            elif isinstance(child.func, ast.Name):
                name = child.func.id
                if name in import_map:
                    key = import_map[name]
                    if key not in seen:
                        seen.add(key)
                        calls.append(key)
    return calls[:3]


def parse_python(filepath: Path) -> dict:
    try:
        source = filepath.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(source)
    except Exception:
        return {'functions': [], 'imports': []}

    import_map = build_python_import_map(tree)
    functions, imports = [], []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args if a.arg != 'self'][:4]
            calls = get_python_func_calls(node, import_map)
            functions.append({'name': node.name, 'params': params, 'calls': calls})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# JS/TS PARSER
# ─────────────────────────────────────────────

def build_js_import_map(source: str) -> dict:
    """Build name → module_short map from JS/TS imports."""
    aliases = {}
    # import { A, B as C } from 'module'
    for m in re.finditer(r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]", source):
        mod = m.group(2).split('/')[-1]
        mod = re.sub(r'\.(ts|tsx|js|jsx)$', '', mod)
        for part in m.group(1).split(','):
            part = part.strip()
            if ' as ' in part:
                _, alias = part.split(' as ', 1)
                aliases[alias.strip()] = mod
            elif part:
                aliases[part] = mod
    # import Default from 'module'
    for m in re.finditer(r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]", source):
        mod = m.group(2).split('/')[-1]
        mod = re.sub(r'\.(ts|tsx|js|jsx)$', '', mod)
        aliases[m.group(1)] = mod
    return aliases


def get_js_calls_in_body(body: str, import_map: dict) -> list:
    """Find which imported names are used in a function body."""
    calls, seen = [], set()
    for name, mod in import_map.items():
        if re.search(rf'\b{re.escape(name)}\s*[\.(]', body):
            if mod not in seen:
                seen.add(mod)
                calls.append(mod)
    return calls[:3]


def parse_js_ts(filepath: Path) -> dict:
    try:
        source = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}

    import_map = build_js_import_map(source)
    functions, imports = [], []
    seen_names = set()

    patterns = [
        r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
        r'(?:export\s+)?const\s+(\w+)(?:\s*:[^=]+)?\s*=\s*(?:async\s+)?\(([^)]*)\)(?:\s*:[^=]*)?\s*=>',
        r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(([^)]*)\)',
    ]

    for pat in patterns:
        for m in re.finditer(pat, source):
            name = m.group(1)
            if name in seen_names:
                continue
            seen_names.add(name)
            params = [p.strip().split(':')[0].split('=')[0].strip()
                      for p in m.group(2).split(',') if p.strip()][:4]
            body = find_func_body(source, m.end())
            calls = get_js_calls_in_body(body, import_map) if body else []
            functions.append({'name': name, 'params': params, 'calls': calls})

    for m in re.finditer(r"from\s+['\"]([^'\"]+)['\"]", source):
        imports.append(m.group(1))

    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# GO PARSER
# ─────────────────────────────────────────────

def build_go_import_map(source: str) -> dict:
    """Build pkg_alias → short_name map from Go imports."""
    aliases = {}
    in_block = False
    for line in source.split('\n'):
        line = line.strip()
        if line.startswith('import ('):
            in_block = True
            continue
        if in_block:
            if line == ')':
                in_block = False
            else:
                m = re.search(r'"([^"]+)"', line)
                if m:
                    pkg = m.group(1).split('/')[-1]
                    alias_m = re.match(r'^(\w+)\s+"', line)
                    alias = alias_m.group(1) if alias_m else pkg
                    aliases[alias] = pkg
        elif re.match(r'^import\s+"', line):
            m = re.search(r'"([^"]+)"', line)
            if m:
                pkg = m.group(1).split('/')[-1]
                aliases[pkg] = pkg
    return aliases


def parse_go(filepath: Path) -> dict:
    try:
        source = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}

    import_map = build_go_import_map(source)
    functions = []

    for m in re.finditer(
        r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)', source
    ):
        name = m.group(1)
        raw_params = [p.strip() for p in m.group(2).split(',') if p.strip()]
        params = [p.split()[-1].lstrip('*').split('.')[0] for p in raw_params][:4]
        body = find_func_body(source, m.end())
        calls, seen = [], set()
        for pkg, short in import_map.items():
            if body and re.search(rf'\b{re.escape(pkg)}\.', body):
                if short not in seen:
                    seen.add(short)
                    calls.append(short)
        functions.append({'name': name, 'params': params, 'calls': calls[:3]})

    imports = list(import_map.values())
    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────

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
    result = {}
    try:
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in CODE_EXTENSIONS:
                parsed = parse_file(f)
                if parsed['functions']:
                    result[f.name] = parsed
    except PermissionError:
        pass
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
        imp_norm = imp.lower().replace('-', '_').replace('/', '_')
        for mod in all_modules:
            if mod.lower() in imp_norm or imp_norm == mod.lower():
                if mod not in deps:
                    deps.append(mod)
    return deps


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def child_subdirs_with_code(dir_path: Path) -> list:
    """Return immediate subdirs that contain code files."""
    result = []
    try:
        for item in sorted(dir_path.iterdir()):
            if item.is_dir() and item.name not in SKIP_DIRS:
                has_code = any(
                    f.suffix.lower() in CODE_EXTENSIONS
                    for f in item.rglob('*') if f.is_file()
                )
                if has_code:
                    result.append(item)
    except PermissionError:
        pass
    return result


# ─────────────────────────────────────────────
# CONTEXT.MD GENERATORS
# ─────────────────────────────────────────────

def make_context_md(dir_path: Path, root: Path, all_modules: list) -> str:
    files_data = scan_dir(dir_path)
    mod = short_name(dir_path, root)

    # Check for subdirs with code (always, not only when files_data is empty)
    children = child_subdirs_with_code(dir_path)

    # Pure parent directory: no direct code files
    if not files_data:
        if children:
            child_names = [c.name for c in children[:12]]
            lines = [
                f"# {mod}",
                f"submodules: {', '.join(child_names)}",
                f"tags: {mod} {' '.join(child_names[:8])}",
                f"upd: {datetime.now().strftime('%Y-%m-%d')} ndoc",
            ]
            return '\n'.join(lines)
        return f"# {mod}\ntags: {mod}\nupd: {datetime.now().strftime('%Y-%m-%d')} ndoc"

    # Collect imports → compute module deps
    all_imports = []
    for fd in files_data.values():
        all_imports.extend(fd.get('imports', []))
    deps = resolve_deps(all_imports, [m for m in all_modules if m != mod])

    # Header: # module → dep1, dep2
    header = f"# {mod}"
    if deps:
        header += f" → {', '.join(deps[:5])}"

    lines = [header]

    # File lines with function call tracking
    for fname, fd in files_data.items():
        funcs = fd.get('functions', [])[:8]
        if not funcs:
            continue
        parts = []
        for fn in funcs:
            p = ','.join(fn['params'][:3])
            entry = f"{fn['name']}({p})"
            calls = fn.get('calls', [])
            if calls:
                entry += f"→{','.join(calls[:2])}"
            parts.append(entry)
        lines.append(f"{fname}: {' | '.join(parts)}")

    # Add submodules line if has child dirs with code
    if children:
        child_names = [c.name for c in children[:10]]
        lines.append(f"submodules: {', '.join(child_names)}")

    # Tags
    tags = {mod}
    for d in deps:
        tags.add(d)
    for fname in files_data:
        base = re.sub(r'\.(py|go|ts|js|tsx|jsx)$', '', fname)
        tags.add(base)
    for fd in files_data.values():
        for fn in fd.get('functions', [])[:2]:
            tags.add(fn['name'].lower())

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
        line = f"{mod} → {dep_str}"
        if kw:
            line += f" | {', '.join(kw)}"
        lines.append(line)

    # Build graph chains
    dep_graph = {mod: data.get('deps', []) for mod, data in modules.items()}
    all_deps_flat = {d for deps in dep_graph.values() for d in deps}
    entry_points = [m for m in modules if m not in all_deps_flat]

    lines += ["", "## Graph"]
    seen_chains = set()
    for entry in (entry_points or list(modules.keys()))[:6]:
        chain = [entry]
        current = entry
        for _ in range(6):
            deps = dep_graph.get(current, [])
            if not deps:
                break
            nxt = next((d for d in deps if d in modules and d not in chain), None)
            if not nxt:
                break
            chain.append(nxt)
            current = nxt
        if len(chain) > 1:
            chain_str = ' → '.join(chain)
            if chain_str not in seen_chains:
                seen_chains.add(chain_str)
                lines.append(chain_str)

    # Mermaid graph
    lines += ["", "## Mermaid", "", "```mermaid", "graph TD"]
    for mod, data in modules.items():
        for dep in data.get('deps', []):
            if dep in modules:
                lines.append(f"    {mod} --> {dep}")
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
# module_name → dep1, dep2
file.ext: FuncA(params)→dep.call | FuncB(params)
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
    Создаёт context.md в каждой папке с кодом, context.index.md с картой зависимостей,
    и добавляет правила в CLAUDE.md.

    project_path: путь к проекту. Если не указан — использует текущую директорию.
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"❌ Папка не найдена: {project_path}"

    out = [f"🚀 NeuroDoc Init — {root.name}", "━" * 48]

    # Сканирование
    out.append("\n📂 Шаг 1/5 — Сканирование проекта...")
    dirs_with_code = []
    for dp, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        dp = Path(dp)
        if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files):
            dirs_with_code.append(dp)

    # Also include parent dirs that have subdirs with code
    all_dirs = set(dirs_with_code)
    for dp in dirs_with_code:
        parent = dp.parent
        while parent != root.parent and parent != root:
            all_dirs.add(parent)
            parent = parent.parent
    all_dirs.add(root)

    out.append(f"   ✓ Найдено директорий: {len(all_dirs)}")

    # Парсинг
    out.append("\n🔍 Шаг 2/5 — Парсинг файлов...")
    all_module_names = [short_name(d, root) for d in all_dirs]
    modules: dict = {}

    for dp in sorted(all_dirs):
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
        if files_data:
            out.append(f"   ✓ {safe_rel(dp, root) or root.name}/ — "
                       f"{len(files_data)} файлов, {func_count} функций")

    # context.md
    out.append(f"\n📝 Шаг 3/5 — Генерация context.md...")
    generated = 0
    for mod, data in modules.items():
        dp = data['dir_path']
        content = make_context_md(dp, root, all_module_names)
        (dp / 'context.md').write_text(content, encoding='utf-8')
        generated += 1

    out.append(f"   → Создано: {generated} файлов")

    # context.index.md
    out.append(f"\n🗺️  Шаг 4/5 — context.index.md...")
    index_content = make_index(modules, root.name)
    (root / 'context.index.md').write_text(index_content, encoding='utf-8')
    out.append(f"   ✓ context.index.md — {len(modules)} модулей")
    out.append(f"   ✓ Map + Graph + Mermaid диаграмма")

    # CLAUDE.md
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

    out += [
        "",
        "━" * 48,
        "✅ NeuroDoc инициализирован!",
        "",
        f"   📁 {generated} context.md файлов",
        f"   🗺️  context.index.md с Map + Graph",
        f"   ⚙️  CLAUDE.md обновлён",
        "",
        "💡 Запусти ndoc_validate для проверки актуальности.",
    ]
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

    out = ["🔄 NeuroDoc Update", "━" * 40]

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
            out.append(f"\n   Обновляем все {len(dirs_to_update)} модулей")

    if not dirs_to_update:
        return "ℹ️  Нечего обновлять"

    all_module_names = []
    for dp, subdirs, _ in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        all_module_names.append(short_name(Path(dp), root))

    out.append(f"\n   Обновляем {len(dirs_to_update)} модуль(ей):\n")
    updated = 0
    for dp in dirs_to_update:
        content = make_context_md(dp, root, all_module_names)
        (dp / 'context.md').write_text(content, encoding='utf-8')
        out.append(f"   ✓ {safe_rel(dp, root) or root.name}/context.md")
        updated += 1

    # Rebuild index
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

    (root / 'context.index.md').write_text(make_index(modules, root.name), encoding='utf-8')
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

    out = ["🔍 NeuroDoc Validate", "━" * 40, ""]
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

    if fresh:
        out.append(f"✅ Актуальные ({len(fresh)}):")
        for m in fresh:
            out.append(f"   ✓ {m}/")
    if stale:
        out.append(f"\n⚠️  Устаревшие ({len(stale)}):")
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
        out.append("\n💡 Запусти ndoc_update для обновления")

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

    out = [f"📊 NeuroDoc Status — {root.name}", "━" * 40]

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
        in_mermaid, shown = False, 0
        for line in lines:
            if '```mermaid' in line:
                in_mermaid = True
                continue
            if in_mermaid and '```' in line:
                in_mermaid = False
                continue
            if not in_mermaid and shown < 30:
                out.append(f"  {line}")
                shown += 1
        if shown >= 30:
            out.append("  ...")
    else:
        out.append("\n💡 Запусти ndoc_init для инициализации")

    return '\n'.join(out)


def main():
    transport = _args.transport
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        print(f"🚀 NeuroDoc MCP server starting on http://{_args.host}:{_args.port}/mcp",
              file=sys.stderr)
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
