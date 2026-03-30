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
# C4 DIAGRAM HELPERS
# ─────────────────────────────────────────────

def detect_tech(files_data: dict) -> str:
    ext_counts: dict = {}
    for fname in files_data:
        ext = Path(fname).suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    if not ext_counts:
        return "Code"
    primary = max(ext_counts, key=ext_counts.get)
    return {
        '.py': 'Python', '.go': 'Go',
        '.ts': 'TypeScript', '.tsx': 'TypeScript/React',
        '.js': 'JavaScript', '.jsx': 'JavaScript/React',
        '.rb': 'Ruby', '.java': 'Java', '.rs': 'Rust',
        '.cs': 'C#', '.cpp': 'C++', '.kt': 'Kotlin', '.swift': 'Swift',
    }.get(primary, primary.lstrip('.').upper() or 'Code')


def c4_alias(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_') or 'mod'


def c4_label(s: str) -> str:
    return s.replace('"', "'")[:60]


def module_container_type(mod_name: str) -> str:
    name = mod_name.lower().split('/')[-1]
    if re.search(r'\b(db|database|store|storage|repo|repository|dao|model|models|migration|schema|cache|redis|mongo)\b', name):
        return 'ContainerDb'
    if re.search(r'\b(queue|worker|broker|kafka|celery|task|tasks|job|jobs|consumer|producer|amqp)\b', name):
        return 'ContainerQueue'
    return 'Container'


_STDLIB_NOISE = {
    # Go stdlib
    'os', 'fmt', 'io', 'http', 'log', 'err', 'ctx', 'pkg',
    'time', 'math', 'net', 'url', 'strings', 'strconv',
    'bytes', 'errors', 'sync', 'context', 'runtime', 'reflect',
    'encoding', 'unicode', 'sort', 'bufio', 'regexp', 'json',
    'atomic', 'maps', 'slices', 'testing', 'main', 'flag',
    'filepath', 'crypto', 'hash', 'binary', 'hex', 'base64',
    # Python stdlib
    'sys', 'os', 'pathlib', 'datetime', 're', 'ast', 'subprocess',
    'argparse', 'typing', 'collections', 'itertools', 'functools',
    'abc', 'copy', 'dataclasses', 'enum', 'inspect', 'logging',
    'threading', 'multiprocessing', 'asyncio', 'socket', 'ssl',
    'json', 'csv', 'xml', 'html', 'urllib', 'http', 'email',
    'hashlib', 'hmac', 'secrets', 'random', 'math', 'statistics',
    'struct', 'io', 'tempfile', 'shutil', 'glob', 'fnmatch',
    'traceback', 'warnings', 'weakref', 'gc', 'platform', 'signal',
    'unittest', 'pytest', 'pprint', 'string', 'textwrap', 'difflib',
    # Node/JS built-ins
    'path', 'fs', 'url', 'http', 'https', 'crypto', 'stream',
    'events', 'util', 'buffer', 'child_process', 'cluster', 'net',
    'dns', 'tls', 'zlib', 'readline', 'repl', 'vm', 'module',
    'process', 'console', 'timers', 'perf_hooks',
}


def collect_external_deps(modules: dict, all_module_names: list) -> list:
    """Return top external deps sorted by usage frequency."""
    mod_set = {m.lower() for m in all_module_names}
    mod_leaves = {m.lower().split('/')[-1] for m in all_module_names}
    counts: dict = {}
    for mod, data in modules.items():
        seen = set()
        for fd in data.get('files', {}).values():
            for imp in fd.get('imports', []):
                if imp.startswith('.') or imp.startswith('/'):
                    continue
                root_pkg = re.sub(r'[^a-z0-9_-]', '', imp.split('/')[0].split('.')[0].lower())
                if not root_pkg or len(root_pkg) < 2:
                    continue
                if root_pkg in mod_set or root_pkg in mod_leaves:
                    continue
                if root_pkg in _STDLIB_NOISE:
                    continue
                if root_pkg not in seen:
                    seen.add(root_pkg)
                    counts[root_pkg] = counts.get(root_pkg, 0) + 1
    return sorted(counts, key=lambda k: -counts[k])[:12]


def make_c4_context(modules: dict, project_name: str, all_module_names: list) -> list:
    """Return lines for C4Context diagram."""
    ext_deps = collect_external_deps(modules, all_module_names)
    all_files: dict = {}
    for data in modules.values():
        all_files.update(data.get('files', {}))
    tech = detect_tech(all_files)
    proj = c4_alias(project_name)

    _DB = {'postgres', 'postgresql', 'mysql', 'mongodb', 'mongo', 'redis',
           'sqlite', 'elasticsearch', 'cassandra', 'dynamodb', 'mariadb', 'mssql'}
    _QUEUE = {'kafka', 'rabbitmq', 'celery', 'amqp', 'sqs', 'pubsub', 'nats', 'zeromq'}

    lines = [
        '```mermaid', 'C4Context',
        f'  title System Context — {project_name}', '',
        f'  System({proj}, "{project_name}", "{tech} application")',
    ]
    for dep in ext_deps:
        alias = c4_alias(dep)
        if dep in _DB:
            lines.append(f'  SystemDb_Ext({alias}, "{dep}", "External database")')
        elif dep in _QUEUE:
            lines.append(f'  SystemQueue_Ext({alias}, "{dep}", "Message broker")')
        else:
            lines.append(f'  System_Ext({alias}, "{dep}", "External dependency")')
    lines.append('')
    for dep in ext_deps:
        lines.append(f'  Rel({proj}, {c4_alias(dep)}, "uses")')
    lines.append('```')
    return lines


def make_c4_container(modules: dict, project_name: str, root: Path) -> list:
    """Return lines for C4Container diagram (top-level modules only)."""
    top: dict = {}
    for mod, data in modules.items():
        dp = data.get('dir_path')
        if dp:
            try:
                depth = len(dp.relative_to(root).parts)
            except ValueError:
                depth = 99
            if depth <= 2:
                top[mod] = data
    if not top:
        top = dict(list(modules.items())[:20])

    proj = c4_alias(project_name)
    lines = [
        '```mermaid', 'C4Container',
        f'  title Container diagram — {project_name}', '',
        f'  System_Boundary({proj}, "{project_name}") {{',
    ]
    for mod, data in top.items():
        alias = c4_alias(mod)
        label = mod.split('/')[-1]
        tech = detect_tech(data.get('files', {}))
        kw = data.get('keywords', [])[:3]
        descr = c4_label(', '.join(kw)) if kw else label
        ctype = module_container_type(mod)
        lines.append(f'    {ctype}({alias}, "{label}", "{tech}", "{descr}")')
    lines.append('  }')
    lines.append('')
    top_set = set(top)
    for mod, data in top.items():
        src = c4_alias(mod)
        for dep in data.get('deps', []):
            if dep in top_set:
                lines.append(f'  Rel({src}, {c4_alias(dep)}, "uses")')
    lines.append('```')
    return lines


def make_c4_component(mod: str, files_data: dict) -> list:
    """Return lines for C4Component diagram for a single module."""
    if not files_data:
        return []
    tech = detect_tech(files_data)
    mod_alias = c4_alias(mod)

    # Build file → alias map (only files with functions, max 15)
    file_aliases: dict = {}
    for fname, fd in list(files_data.items())[:15]:
        if fd.get('functions'):
            base = re.sub(r'\.(py|go|ts|js|tsx|jsx)$', '', fname)
            file_aliases[fname] = c4_alias(base)

    if not file_aliases:
        return []

    lines = [
        '```mermaid', 'C4Component',
        f'  title Component diagram — {mod}', '',
        f'  Container_Boundary({mod_alias}, "{mod}") {{',
    ]
    for fname, alias in file_aliases.items():
        funcs = files_data[fname].get('functions', [])
        func_names = [fn['name'] for fn in funcs[:3]]
        descr = c4_label(', '.join(func_names))
        lines.append(f'    Component({alias}, "{fname}", "{tech}", "{descr}")')
    lines.append('  }')
    lines.append('')

    added: set = set()
    for fname, alias in file_aliases.items():
        all_calls: set = set()
        for fn in files_data[fname].get('functions', []):
            all_calls.update(fn.get('calls', []))
        for other_fname, other_alias in file_aliases.items():
            if other_fname == fname:
                continue
            other_base = re.sub(r'\.(py|go|ts|js|tsx|jsx)$', '', other_fname)
            if any(other_base in call or call in other_base for call in all_calls):
                key = (alias, other_alias)
                if key not in added:
                    added.add(key)
                    lines.append(f'  Rel({alias}, {other_alias}, "calls")')
    lines.append('```')
    return lines


# ─────────────────────────────────────────────
# CONTEXT.MD GENERATORS
# ─────────────────────────────────────────────

def make_context_md(dir_path: Path, root: Path, all_modules: list, reverse_deps: dict = None) -> str:
    files_data = scan_dir(dir_path)
    mod = short_name(dir_path, root)
    now = datetime.now().strftime('%Y-%m-%d')

    # Check for subdirs with code (always, not only when files_data is empty)
    children = child_subdirs_with_code(dir_path)

    # Pure parent directory: no direct code files
    if not files_data:
        if children:
            child_names = [c.name for c in children]
            lines = [f"# {mod}", ""]
            lines.append(f"**submodules:** {', '.join(child_names)}")
            # Show what each child does if we can infer
            for child in children[:8]:
                child_data = scan_dir(child)
                if child_data:
                    child_funcs = []
                    for fd in child_data.values():
                        for fn in fd.get('functions', [])[:2]:
                            child_funcs.append(fn['name'])
                    if child_funcs:
                        lines.append(f"- **{child.name}**: {', '.join(child_funcs[:4])}")
                    else:
                        lines.append(f"- **{child.name}**")
            lines += [
                "",
                f"**tags:** {mod} {' '.join(child_names[:8])}",
                f"**updated:** {now} ndoc",
            ]
            return '\n'.join(lines)
        return f"# {mod}\n\n**tags:** {mod}\n**updated:** {now} ndoc"

    # Collect imports → compute module deps
    all_imports = []
    for fd in files_data.values():
        all_imports.extend(fd.get('imports', []))
    deps = resolve_deps(all_imports, [m for m in all_modules if m != mod])

    # Reverse deps: who uses this module
    used_by = []
    if reverse_deps:
        used_by = reverse_deps.get(mod, [])

    # Header: # module → dep1, dep2
    header = f"# {mod}"
    if deps:
        header += f" → {', '.join(deps)}"

    lines = [header, ""]

    if used_by:
        lines.append(f"**used by:** {', '.join(used_by)}")
        lines.append("")

    # Per-file detailed sections
    for fname, fd in files_data.items():
        funcs = fd.get('functions', [])
        if not funcs:
            continue

        lines.append(f"## {fname}")
        for fn in funcs:
            params = ', '.join(fn['params'])
            entry = f"`{fn['name']}({params})`"
            calls = fn.get('calls', [])
            if calls:
                entry += f" → {', '.join(calls)}"
            lines.append(f"- {entry}")
        lines.append("")

    # Submodules section
    if children:
        child_names = [c.name for c in children]
        lines.append(f"**submodules:** {', '.join(child_names)}")
        lines.append("")

    # Tags
    tags = {mod}
    for d in deps:
        tags.add(d)
    for fname in files_data:
        base = re.sub(r'\.(py|go|ts|js|tsx|jsx)$', '', fname)
        tags.add(base)
    for fd in files_data.values():
        for fn in fd.get('functions', [])[:3]:
            tags.add(fn['name'].lower())

    lines.append(f"**tags:** {' '.join(sorted(tags)[:16])}")
    lines.append(f"**updated:** {now} ndoc")

    # C4 Component diagram
    c4_lines = make_c4_component(mod, files_data)
    if c4_lines:
        lines += ['', '## C4 Component', ''] + c4_lines

    return '\n'.join(lines)


def make_index(modules: dict, project_name: str, root: Path = None) -> str:
    now = datetime.now().strftime('%Y-%m-%d')
    all_module_names = list(modules.keys())

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

    # Dependency chains (Graph)
    dep_graph = {mod: data.get('deps', []) for mod, data in modules.items()}
    all_deps_flat = {d for deps in dep_graph.values() for d in deps}
    entry_points = [m for m in modules if m not in all_deps_flat]

    lines += ["", "## Graph"]
    seen_chains: set = set()
    for entry in (entry_points or list(modules.keys()))[:6]:
        chain = [entry]
        current = entry
        for _ in range(8):
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

    # C4 Context
    lines += ["", "## C4 Context", ""]
    lines += make_c4_context(modules, project_name, all_module_names)

    # C4 Container
    lines += ["", "## C4 Container", ""]
    lines += make_c4_container(modules, project_name, root or Path('.'))

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
1. Обнови `context.md` в изменённой директории (запусти ndoc_update)
2. Обнови `context.index.md` если добавился новый модуль или изменились связи

### Формат context.md:
```markdown
# module_name → dep1, dep2

**used by:** parent_module, other_module

## filename.go
- `FuncA(param1, param2)` → dep1, dep2
- `FuncB()`

**submodules:** child1, child2

**tags:** keyword1 keyword2
**updated:** YYYY-MM-DD ndoc
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

    # Build reverse dependency map: mod → [list of modules that depend on it]
    reverse_deps: dict = {}
    for mod, data in modules.items():
        for dep in data.get('deps', []):
            reverse_deps.setdefault(dep, []).append(mod)

    # context.md
    out.append(f"\n📝 Шаг 3/5 — Генерация context.md...")
    generated = 0
    for mod, data in modules.items():
        dp = data['dir_path']
        content = make_context_md(dp, root, all_module_names, reverse_deps)
        (dp / 'context.md').write_text(content, encoding='utf-8')
        generated += 1

    out.append(f"   → Создано: {generated} файлов")

    # context.index.md
    out.append(f"\n🗺️  Шаг 4/5 — context.index.md...")
    index_content = make_index(modules, root.name, root)
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

    # Rebuild index and reverse deps
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

    reverse_deps: dict = {}
    for mod, data in modules.items():
        for dep in data.get('deps', []):
            reverse_deps.setdefault(dep, []).append(mod)

    # Re-generate context.md for updated dirs with reverse_deps
    for dp in dirs_to_update:
        content = make_context_md(dp, root, all_module_names, reverse_deps)
        (dp / 'context.md').write_text(content, encoding='utf-8')

    (root / 'context.index.md').write_text(make_index(modules, root.name, root), encoding='utf-8')
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


# ══════════════════════════════════════════════════════════════════════════════
# ndoc_c4 — Full C4 Architecture Model Generator
# Supports: Laravel, Django, FastAPI, Flask, Rails, Express, NestJS, Next.js,
#           Spring Boot, Gin, Echo, Fiber, Phoenix, and any other framework.
# ══════════════════════════════════════════════════════════════════════════════

def _read_safe(path):
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


# ── Framework detection ───────────────────────────────────────────────────────

_FRAMEWORK_SIGNATURES = [
    # (filename, content_regex_or_None, framework_name, language)
    ('composer.json',     r'"laravel/framework"',          'Laravel',     'PHP'),
    ('composer.json',     r'"symfony/symfony"',             'Symfony',     'PHP'),
    ('composer.json',     r'"slim/slim"',                   'Slim',        'PHP'),
    ('composer.json',     r'"cakephp/cakephp"',             'CakePHP',     'PHP'),
    ('artisan',           None,                             'Laravel',     'PHP'),
    ('requirements.txt',  r'(?i)\bdjango\b',                'Django',      'Python'),
    ('requirements.txt',  r'(?i)\bfastapi\b',               'FastAPI',     'Python'),
    ('requirements.txt',  r'(?i)\bflask\b',                 'Flask',       'Python'),
    ('requirements.txt',  r'(?i)\bsanic\b',                 'Sanic',       'Python'),
    ('pyproject.toml',    r'(?i)django',                    'Django',      'Python'),
    ('pyproject.toml',    r'(?i)fastapi',                   'FastAPI',     'Python'),
    ('pyproject.toml',    r'(?i)flask',                     'Flask',       'Python'),
    ('Gemfile',           r"gem ['\"]rails['\"]",           'Rails',       'Ruby'),
    ('Gemfile',           r"gem ['\"]sinatra['\"]",         'Sinatra',     'Ruby'),
    ('Gemfile',           r"gem ['\"]hanami['\"]",          'Hanami',      'Ruby'),
    ('package.json',      r'"@nestjs/core"',                'NestJS',      'TypeScript'),
    ('package.json',      r'"next"[^a-z]',                  'Next.js',     'TypeScript'),
    ('package.json',      r'"nuxt"[^a-z]',                  'Nuxt',        'TypeScript'),
    ('package.json',      r'"@remix-run',                   'Remix',       'TypeScript'),
    ('package.json',      r'"svelte"[^a-z]',                'SvelteKit',   'TypeScript'),
    ('package.json',      r'"express"[^a-z]',               'Express',     'Node.js'),
    ('package.json',      r'"fastify"[^a-z]',               'Fastify',     'Node.js'),
    ('package.json',      r'"hono"[^a-z]',                  'Hono',        'Node.js'),
    ('package.json',      r'"koa"[^a-z]',                   'Koa',         'Node.js'),
    ('go.mod',            r'gin-gonic/gin',                  'Gin',         'Go'),
    ('go.mod',            r'labstack/echo',                  'Echo',        'Go'),
    ('go.mod',            r'gofiber/fiber',                  'Fiber',       'Go'),
    ('go.mod',            r'go-chi/chi',                     'Chi',         'Go'),
    ('pom.xml',           r'spring-boot',                   'Spring Boot', 'Java'),
    ('build.gradle',      r'spring-boot',                   'Spring Boot', 'Java/Kotlin'),
    ('build.gradle.kts',  r'spring-boot',                   'Spring Boot', 'Kotlin'),
    ('mix.exs',           r'phoenix',                       'Phoenix',     'Elixir'),
    ('Cargo.toml',        r'actix-web|axum|warp|rocket',    'Rust Web',    'Rust'),
    ('*.csproj',          None,                             'ASP.NET',     'C#'),
]

_LANG_BY_EXT = {
    '.php': 'PHP', '.py': 'Python', '.rb': 'Ruby',
    '.java': 'Java', '.kt': 'Kotlin', '.cs': 'C#',
    '.go': 'Go', '.rs': 'Rust', '.ts': 'TypeScript',
    '.js': 'JavaScript', '.ex': 'Elixir', '.exs': 'Elixir',
    '.swift': 'Swift', '.scala': 'Scala',
}


def _detect_framework(root):
    result = {'name': 'Application', 'language': 'Unknown', 'version': '', 'pkg_manager': ''}
    for filename, pattern, fw_name, lang in _FRAMEWORK_SIGNATURES:
        if '*' in filename:
            # glob match
            matches = list(root.glob(filename))
            if not matches:
                continue
            content = _read_safe(matches[0])
        else:
            f = root / filename
            if not f.exists():
                continue
            content = _read_safe(f)
        if pattern is None or re.search(pattern, content):
            result.update({'name': fw_name, 'language': lang})
            if filename == 'composer.json':
                result['pkg_manager'] = 'Composer'
                m = re.search(r'"laravel/framework":\s*"([^"]+)"', content)
                if m:
                    result['version'] = m.group(1).lstrip('^~>=')
            elif 'requirements' in filename or 'pyproject' in filename:
                result['pkg_manager'] = 'pip'
            elif filename == 'package.json':
                result['pkg_manager'] = 'npm/yarn/pnpm'
            elif filename == 'go.mod':
                result['pkg_manager'] = 'Go modules'
            elif filename == 'Gemfile':
                result['pkg_manager'] = 'Bundler'
            elif filename in ('pom.xml', 'build.gradle', 'build.gradle.kts'):
                result['pkg_manager'] = 'Maven/Gradle'
            elif filename == 'Cargo.toml':
                result['pkg_manager'] = 'Cargo'
            break
    if result['language'] == 'Unknown':
        ext_count = {}
        for dp, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in _LANG_BY_EXT:
                    ext_count[ext] = ext_count.get(ext, 0) + 1
        if ext_count:
            primary = max(ext_count, key=ext_count.get)
            result['language'] = _LANG_BY_EXT[primary]
    return result


# ── Docker / infrastructure detection ────────────────────────────────────────

_IMAGE_TO_CONTAINER = {
    r'postgres|postgresql': ('Database',     'PostgreSQL',    '🐘', '#89b4fa'),
    r'mysql|mariadb':       ('Database',     'MySQL',         '🗄️', '#89b4fa'),
    r'mongo':               ('Database',     'MongoDB',       '🍃', '#89b4fa'),
    r'clickhouse':          ('Analytics DB', 'ClickHouse',    '📊', '#cba6f7'),
    r'elasticsearch':       ('Search',       'Elasticsearch', '🔍', '#f9e2af'),
    r'influxdb':            ('Time-Series',  'InfluxDB',      '📈', '#cba6f7'),
    r'redis':               ('Cache',        'Redis',         '⚡', '#f38ba8'),
    r'memcached':           ('Cache',        'Memcached',     '⚡', '#f38ba8'),
    r'rabbitmq':            ('Queue',        'RabbitMQ',      '🐇', '#fab387'),
    r'kafka':               ('Queue',        'Kafka',         '📨', '#fab387'),
    r'nats':                ('Queue',        'NATS',          '📨', '#fab387'),
    r'traefik':             ('Proxy',        'Traefik',       '🔀', '#89dceb'),
    r'nginx':               ('Proxy',        'nginx',         '🔀', '#89dceb'),
    r'caddy':               ('Proxy',        'Caddy',         '🔀', '#89dceb'),
    r'haproxy':             ('Proxy',        'HAProxy',       '🔀', '#89dceb'),
    r'minio':               ('Storage',      'MinIO',         '🪣', '#f9e2af'),
    r'prometheus':          ('Monitoring',   'Prometheus',    '📡', '#eba0ac'),
    r'grafana':             ('Monitoring',   'Grafana',       '📊', '#eba0ac'),
    r'meilisearch':         ('Search',       'Meilisearch',   '🔍', '#f9e2af'),
    r'typesense':           ('Search',       'Typesense',     '🔍', '#f9e2af'),
}


def _parse_docker_compose(root):
    for fname in ('docker-compose.yml', 'docker-compose.yaml',
                  'docker-compose.prod.yml', 'compose.yml', 'compose.yaml'):
        dc = root / fname
        if not dc.exists():
            dc = root / 'deploy' / fname
        if not dc.exists():
            continue
        content = _read_safe(dc)
        containers = []
        # Find service names (2-space indent)
        svc_names = re.findall(r'^  (\w[\w-]*):\s*$', content, re.M)
        for svc in svc_names:
            # Extract block for this service
            blk_m = re.search(
                rf'^  {re.escape(svc)}:\s*\n((?:    .*\n?)*)', content, re.M
            )
            blk = blk_m.group(1) if blk_m else ''
            img_m = re.search(r'image:\s*[\'"]?([^\s\'"#]+)', blk)
            image = img_m.group(1).lower() if img_m else ''
            has_build = bool(re.search(r'^\s*build:', blk, re.M))
            port_m = re.findall(r'["\']?(\d{2,5}):\d+["\']?', blk)
            port = port_m[0] if port_m else ''
            ctype, tech, icon, color = 'App', svc, '📦', '#89b4fa'
            for pat, (t, te, ic, co) in _IMAGE_TO_CONTAINER.items():
                if re.search(pat, image or svc.lower()):
                    ctype, tech, icon, color = t, te, ic, co
                    break
            containers.append({
                'name': svc, 'type': ctype, 'tech': tech,
                'icon': icon, 'color': color,
                'port': port, 'has_build': has_build,
            })
        if containers:
            return containers
    return []


# ── External integration detection ───────────────────────────────────────────

_ENV_PATTERNS = [
    # Databases
    (r'DATABASE_URL|DB_HOST|DB_CONNECTION',         'Database',    'Database',          '🗄️'),
    (r'POSTGRES|PGSQL',                             'Database',    'PostgreSQL',         '🐘'),
    (r'MYSQL|MARIADB',                              'Database',    'MySQL',              '🗄️'),
    (r'MONGODB|MONGO_URI',                          'Database',    'MongoDB',            '🍃'),
    (r'REDIS_URL|REDIS_HOST',                       'Cache',       'Redis',              '⚡'),
    (r'CLICKHOUSE',                                 'Analytics DB','ClickHouse',         '📊'),
    # Queues
    (r'RABBITMQ|AMQP_URL',                          'Queue',       'RabbitMQ',           '🐇'),
    (r'(?<!_)KAFKA',                                'Queue',       'Kafka',              '📨'),
    (r'(?<![A-Z])SQS|AWS_QUEUE',                    'Queue',       'AWS SQS',            '📨'),
    # Auth / OAuth
    (r'GOOGLE.*CLIENT|GOOGLE.*SECRET|GOOGLE_OAUTH', 'OAuth',       'Google OAuth',       '🔐'),
    (r'FACEBOOK.*CLIENT|FACEBOOK.*APP',             'OAuth',       'Facebook OAuth',     '🔐'),
    (r'GITHUB.*CLIENT|GITHUB.*SECRET',              'OAuth',       'GitHub OAuth',       '🔐'),
    (r'TWITTER.*KEY|TWITTER.*SECRET',               'OAuth',       'Twitter OAuth',      '🔐'),
    (r'TELEGRAM.*BOT|TELEGRAM.*TOKEN',              'OAuth',       'Telegram Login',     '🔐'),
    (r'VK.*CLIENT|VK_APP',                          'OAuth',       'VK OAuth',           '🔐'),
    (r'STEAM.*KEY|STEAM.*SECRET',                   'OAuth',       'Steam OAuth',        '🔐'),
    (r'APPLE.*CLIENT|SIGN_IN_WITH_APPLE',           'OAuth',       'Apple Sign-In',      '🔐'),
    (r'AUTH0_|OKTA_|COGNITO_',                      'Auth',        'OAuth2 Provider',    '🔐'),
    # Payments
    (r'STRIPE',                                     'Payments',    'Stripe',             '💳'),
    (r'PAYPAL',                                     'Payments',    'PayPal',             '💳'),
    (r'BRAINTREE',                                  'Payments',    'Braintree',          '💳'),
    (r'ADYEN',                                      'Payments',    'Adyen',              '💳'),
    (r'MOLLIE',                                     'Payments',    'Mollie',             '💳'),
    (r'PADDLE',                                     'Payments',    'Paddle',             '💳'),
    (r'LEMON_SQUEEZY',                              'Payments',    'Lemon Squeezy',      '💳'),
    # Storage & CDN
    (r'AWS_.*S3|S3_BUCKET|S3_KEY',                  'Storage',     'AWS S3',             '🪣'),
    (r'CLOUDFLARE_R2|R2_BUCKET',                    'Storage',     'Cloudflare R2',      '🪣'),
    (r'CLOUDFLARE',                                 'CDN/DNS',     'Cloudflare',         '☁️'),
    (r'GCS_|GOOGLE_CLOUD_STORAGE',                  'Storage',     'Google Cloud Storage','🪣'),
    (r'MINIO',                                      'Storage',     'MinIO',              '🪣'),
    (r'DIGITALOCEAN_SPACES',                        'Storage',     'DO Spaces',          '🪣'),
    # Email
    (r'MAIL_HOST|SMTP_HOST|MAIL_SERVER',            'Email',       'SMTP',               '📧'),
    (r'MAILGUN',                                    'Email',       'Mailgun',            '📧'),
    (r'SENDGRID',                                   'Email',       'SendGrid',           '📧'),
    (r'POSTMARK',                                   'Email',       'Postmark',           '📧'),
    (r'SES_|AWS.*MAIL|MAIL.*SES',                   'Email',       'AWS SES',            '📧'),
    (r'RESEND',                                     'Email',       'Resend',             '📧'),
    (r'SPARKPOST',                                  'Email',       'SparkPost',          '📧'),
    # Monitoring
    (r'SENTRY',                                     'Monitoring',  'Sentry',             '🔍'),
    (r'DATADOG',                                    'Monitoring',  'Datadog',            '🔍'),
    (r'NEWRELIC|NEW_RELIC',                         'Monitoring',  'New Relic',          '🔍'),
    (r'ROLLBAR',                                    'Monitoring',  'Rollbar',            '🔍'),
    (r'BUGSNAG',                                    'Monitoring',  'Bugsnag',            '🔍'),
    # Alerting
    (r'SLACK.*WEBHOOK|SLACK.*TOKEN',                'Alerting',    'Slack',              '💬'),
    (r'PAGERDUTY',                                  'Alerting',    'PagerDuty',          '🚨'),
    (r'DISCORD.*WEBHOOK',                           'Alerting',    'Discord',            '💬'),
    # Analytics & CRM
    (r'SEGMENT',                                    'Analytics',   'Segment',            '📈'),
    (r'MIXPANEL',                                   'Analytics',   'Mixpanel',           '📈'),
    (r'AMPLITUDE',                                  'Analytics',   'Amplitude',          '📈'),
    (r'GOOGLE_ANALYTICS|GA_TRACKING|GTM_',          'Analytics',   'Google Analytics',   '📈'),
    (r'CUSTOMER_?IO|CUSTOMERIO',                    'CRM',         'Customer.io',        '📊'),
    (r'HUBSPOT',                                    'CRM',         'HubSpot',            '📊'),
    (r'INTERCOM',                                   'CRM',         'Intercom',           '📊'),
    (r'SALESFORCE',                                 'CRM',         'Salesforce',         '📊'),
    (r'KLAVIYO',                                    'CRM',         'Klaviyo',            '📊'),
    # Realtime / Push
    (r'PUSHER',                                     'Realtime',    'Pusher',             '📡'),
    (r'ABLY',                                       'Realtime',    'Ably',               '📡'),
    (r'FCM_|FIREBASE.*SERVER',                      'Push',        'Firebase FCM',       '📡'),
    (r'ONESIGNAL',                                  'Push',        'OneSignal',          '📡'),
    # SMS
    (r'TWILIO',                                     'SMS',         'Twilio',             '📱'),
    (r'VONAGE|NEXMO',                               'SMS',         'Vonage',             '📱'),
    (r'AWS_SNS',                                    'SMS',         'AWS SNS',            '📱'),
    # KYC / Identity
    (r'KYC_|JUMIO|ONFIDO|SUMSUB|VERIFF|SUM_SUB',   'KYC',         'KYC Provider',       '🪪'),
    # AI / ML
    (r'OPENAI|GPT_',                                'AI',          'OpenAI',             '🤖'),
    (r'ANTHROPIC',                                  'AI',          'Anthropic Claude',   '🤖'),
    (r'GEMINI|GOOGLE_AI',                           'AI',          'Google AI',          '🤖'),
    # Maps
    (r'GOOGLE_MAPS|MAPS_API',                       'Maps',        'Google Maps',        '🗺️'),
    (r'MAPBOX',                                     'Maps',        'Mapbox',             '🗺️'),
    # Finance
    (r'OPENEXCHANGERATES|EXCHANGE_RATE|FIXER',      'Finance',     'Exchange Rate API',  '💱'),
    (r'AMBITO',                                     'Finance',     'Ambito Finance',     '💱'),
    # Internal microservices
    (r'FIN_SERVICE|FIN_URL|PAYMENT_SERVICE',        'Microservice','Payment Service',    '💳'),
    (r'SLOTS_SERVICE|GAMES_SERVICE|GAME_URL',       'Microservice','Games Service',      '🎮'),
    (r'AUTH_SERVICE|AUTH_URL|SSO_URL',              'Microservice','Auth Service',       '🔐'),
    (r'STAT_SERVICE|ANALYTICS_SERVICE',             'Microservice','Stats Service',      '📊'),
    (r'NOTIFICATION_SERVICE|NOTIFY_URL',            'Microservice','Notification Svc',  '📬'),
]


def _detect_integrations(root):
    env_content = ''
    for name in ('.env.example', '.env.sample', '.env.template', '.env.dist', '.env.test'):
        f = root / name
        if f.exists():
            env_content = _read_safe(f)
            break
    for cfg in ('config/services.php', 'config/database.php', 'config/queue.php',
                'config/settings.py', 'application.properties',
                'src/main/resources/application.yml'):
        f = root / cfg
        if f.exists():
            env_content += '\n' + _read_safe(f)
    if not env_content:
        return []
    found = {}
    for pattern, category, name, icon in _ENV_PATTERNS:
        if re.search(pattern, env_content, re.I):
            if name not in found:
                found[name] = {
                    'name': name, 'category': category,
                    'icon': icon, 'alias': c4_alias(name),
                }
    return list(found.values())


# ── Actor detection ───────────────────────────────────────────────────────────

_ACTOR_PATTERNS = [
    (r"Route::prefix\(['\"]api",                    'API Client / Mobile',  '📱'),
    (r"Route::prefix\(['\"]admin|/nova\b|->nova\(",  'Admin Operator',       '🛡️'),
    (r"Route::prefix\(['\"]partner|/partner/",       'Partner / Affiliate',  '🤝'),
    (r'middleware.*auth|auth.*middleware',           'Authenticated User',   '👤'),
    (r'webhook|postback|callback',                  'External Webhook',     '🔔'),
    (r'namespace :api',                             'API Client',           '📱'),
    (r'namespace :admin',                           'Admin',                '🛡️'),
    (r'path\([\'"]api/',                            'API Client',           '📱'),
    (r'path\([\'"]admin/',                          'Admin',                '🛡️'),
    (r'/v\d+/',                                     'API Consumer',         '📱'),
    (r'swagger|openapi|redoc',                      'Developer',            '👩‍💻'),
    (r'grpc|proto',                                 'gRPC Client',          '🔗'),
]

_ACTOR_COLORS = {
    '📱': '#89b4fa', '🛡️': '#f38ba8', '🤝': '#a6e3a1',
    '👤': '#89dceb', '🔔': '#fab387', '👩‍💻': '#cba6f7',
    '🔗': '#94e2d5',
}


def _detect_actors(root):
    route_content = ''
    for rp in ('routes/api.php', 'routes/web.php', 'routes/admin.php',
               'config/routes.rb', 'app/router.js', 'src/routes/index.ts',
               'src/router/index.ts', 'src/app.ts', 'src/app.js'):
        f = root / rp
        if f.exists():
            route_content += _read_safe(f) + '\n'
    for f in list(root.rglob('urls.py'))[:3]:
        try:
            if not any(p in SKIP_DIRS for p in f.relative_to(root).parts):
                route_content += _read_safe(f) + '\n'
        except ValueError:
            pass
    found = {}
    for pattern, name, icon in _ACTOR_PATTERNS:
        if re.search(pattern, route_content, re.I):
            if name not in found:
                color = _ACTOR_COLORS.get(icon, '#89dceb')
                found[name] = {'name': name, 'icon': icon, 'color': color, 'alias': c4_alias(name)}
    if not found:
        found['User'] = {'name': 'User', 'icon': '👤', 'color': '#89dceb', 'alias': 'User'}
        found['Admin'] = {'name': 'Admin', 'icon': '🛡️', 'color': '#f38ba8', 'alias': 'Admin'}
    return list(found.values())


# ── Module detection ──────────────────────────────────────────────────────────

def _detect_modules(root, framework):
    modules, fw, lang = [], framework['name'], framework['language']
    ext_map = {
        'PHP': '.php', 'Python': '.py', 'Ruby': '.rb', 'TypeScript': '.ts',
        'JavaScript': '.js', 'Go': '.go', 'Java': '.java', 'Kotlin': '.kt',
        'Rust': '.rs', 'C#': '.cs', 'Elixir': '.ex',
    }
    ext = ext_map.get(lang, '.php')

    def count_files(d):
        return sum(1 for _ in d.rglob(f'*{ext}') if _.is_file())

    # Laravel — nwidart modules
    md = root / 'Modules'
    if md.exists():
        for item in sorted(md.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                modules.append({'name': item.name, 'path': str(item.relative_to(root)),
                                'files': count_files(item), 'type': 'domain'})
    # Laravel — app/ subdirs
    if not modules and fw in ('Laravel', 'Symfony', 'Slim', 'CakePHP'):
        for sub in ('app/Domain', 'app/Modules', 'app/Services', 'app/Http/Controllers'):
            d = root / sub
            if d.exists():
                for item in sorted(d.iterdir()):
                    if item.is_dir() and count_files(item) > 0:
                        modules.append({'name': item.name, 'path': str(item.relative_to(root)),
                                        'files': count_files(item), 'type': 'service'})
                if modules:
                    break
    # Django — apps (has models.py or views.py)
    if fw in ('Django', 'Flask', 'FastAPI', 'Sanic'):
        for item in sorted(root.iterdir()):
            if item.is_dir() and item.name not in SKIP_DIRS:
                if (item / 'models.py').exists() or (item / 'views.py').exists() \
                        or (item / 'routes.py').exists() or (item / 'router.py').exists():
                    modules.append({'name': item.name, 'path': item.name,
                                    'files': count_files(item), 'type': 'app'})
    # Rails
    if fw in ('Rails', 'Sinatra', 'Hanami'):
        for sub in ('app/controllers', 'app/models', 'app/services'):
            d = root / sub
            if d.exists():
                for item in sorted(d.iterdir()):
                    if item.is_dir():
                        modules.append({'name': item.name.title(), 'path': str(item.relative_to(root)),
                                        'files': count_files(item), 'type': 'controller_group'})
                if modules:
                    break
    # NestJS / Express / Node
    if fw in ('NestJS', 'Express', 'Fastify', 'Hono', 'Koa', 'Next.js', 'Nuxt', 'Remix'):
        for src in ('src', 'app', 'lib', 'modules'):
            d = root / src
            if d.exists():
                for item in sorted(d.iterdir()):
                    if item.is_dir() and item.name not in SKIP_DIRS:
                        n = count_files(item)
                        if n > 0:
                            modules.append({'name': item.name, 'path': str(item.relative_to(root)),
                                            'files': n, 'type': 'module'})
                if modules:
                    break
    # Go — internal/pkg/cmd
    if lang == 'Go':
        for src in ('internal', 'pkg', 'cmd', 'service', 'services'):
            d = root / src
            if d.exists():
                for item in sorted(d.iterdir()):
                    if item.is_dir():
                        n = count_files(item)
                        if n > 0:
                            modules.append({'name': item.name, 'path': str(item.relative_to(root)),
                                            'files': n, 'type': 'package'})
    # Spring Boot — src/main/java packages
    if fw == 'Spring Boot':
        java_src = root / 'src' / 'main' / 'java'
        if java_src.exists():
            # Find the deepest common package
            all_dirs = [d for d in java_src.rglob('*') if d.is_dir()]
            for d in sorted(all_dirs):
                n = sum(1 for f in d.iterdir() if f.suffix == '.java')
                if n > 0:
                    modules.append({'name': d.name, 'path': str(d.relative_to(root)),
                                    'files': n, 'type': 'package'})
    # Phoenix — lib/app_name (contexts)
    if fw == 'Phoenix':
        lib_dir = root / 'lib'
        if lib_dir.exists():
            for item in sorted(lib_dir.iterdir()):
                if item.is_dir():
                    n = count_files(item)
                    if n > 0:
                        modules.append({'name': item.name, 'path': str(item.relative_to(root)),
                                        'files': n, 'type': 'context'})
    # Generic fallback
    if not modules:
        for src in ('src', 'lib', 'app', 'packages'):
            d = root / src
            if d.exists():
                for item in sorted(d.iterdir()):
                    if item.is_dir() and item.name not in SKIP_DIRS:
                        modules.append({'name': item.name, 'path': item.name,
                                        'files': 0, 'type': 'module'})
                if modules:
                    break
    return modules[:20]


# ── Mermaid dark theme ────────────────────────────────────────────────────────

_MERMAID_INIT = ('%%{init: {"theme": "base", "themeVariables": {'
                 '"primaryColor": "#1e1e2e", "primaryTextColor": "#cdd6f4", '
                 '"primaryBorderColor": "#89b4fa", "lineColor": "#89b4fa", '
                 '"secondaryColor": "#181825", "tertiaryColor": "#313244", '
                 '"background": "#1e1e2e", "mainBkg": "#1e1e2e", '
                 '"nodeBorder": "#89b4fa", "clusterBkg": "#1e1e2e", '
                 '"clusterBorder": "#585b70", "titleColor": "#cdd6f4", '
                 '"edgeLabelBackground": "#313244", '
                 '"fontFamily": "ui-monospace, monospace", "fontSize": "14px"'
                 '}}}%%')

_CLASSDEFS = [
    'classDef actor      fill:#89dceb,stroke:#74c7ec,color:#1e1e2e,font-weight:bold',
    'classDef system     fill:#89b4fa,stroke:#74c7ec,color:#1e1e2e,font-weight:bold',
    'classDef proxy      fill:#24a1c1,stroke:#74c7ec,color:#1e1e2e,font-weight:bold',
    'classDef app        fill:#1e3a5f,stroke:#89b4fa,color:#89b4fa,font-weight:bold',
    'classDef worker     fill:#1a2a3a,stroke:#74c7ec,color:#74c7ec',
    'classDef database   fill:#1a1a2e,stroke:#89b4fa,color:#89b4fa',
    'classDef cache      fill:#2e1a1a,stroke:#f38ba8,color:#f38ba8',
    'classDef queue      fill:#2e1e0a,stroke:#fab387,color:#fab387',
    'classDef analytdb   fill:#1e1a2e,stroke:#cba6f7,color:#cba6f7',
    'classDef storage    fill:#2a2a0a,stroke:#f9e2af,color:#f9e2af',
    'classDef oauth      fill:#1a2e1a,stroke:#a6e3a1,color:#a6e3a1',
    'classDef monitoring fill:#2e1a1e,stroke:#eba0ac,color:#eba0ac',
    'classDef analytics  fill:#1a1a2e,stroke:#b4befe,color:#b4befe',
    'classDef realtime   fill:#1a2a2e,stroke:#89dceb,color:#89dceb',
    'classDef email      fill:#1a2a2a,stroke:#94e2d5,color:#94e2d5',
    'classDef payments   fill:#1a2e1a,stroke:#a6e3a1,color:#a6e3a1',
    'classDef microsvc   fill:#0a2a1a,stroke:#a6e3a1,color:#a6e3a1',
    'classDef external   fill:#1e1e2e,stroke:#585b70,color:#6c7086',
    'classDef domain     fill:#1e3a5f,stroke:#89b4fa,color:#89b4fa',
    'classDef service    fill:#1e3a2f,stroke:#a6e3a1,color:#a6e3a1',
    'classDef entity     fill:#2e1a2e,stroke:#cba6f7,color:#cba6f7',
    'classDef repo       fill:#2a1e3a,stroke:#cba6f7,color:#cba6f7',
    'classDef job        fill:#2e1e0a,stroke:#fab387,color:#fab387',
    'classDef event      fill:#0a2a2a,stroke:#89dceb,color:#89dceb',
    'classDef resource   fill:#2e1a1a,stroke:#f38ba8,color:#f38ba8',
]

_CAT_CSS = {
    'Database': 'database', 'Cache': 'cache', 'Queue': 'queue',
    'Analytics DB': 'analytdb', 'OAuth': 'oauth', 'Auth': 'oauth',
    'Storage': 'storage', 'CDN/DNS': 'storage',
    'Monitoring': 'monitoring', 'Alerting': 'monitoring',
    'Analytics': 'analytics', 'CRM': 'analytics',
    'Realtime': 'realtime', 'Push': 'realtime',
    'Payments': 'payments', 'Email': 'email', 'SMS': 'email',
    'KYC': 'external', 'Microservice': 'microsvc',
    'AI': 'analytics', 'Maps': 'external', 'Finance': 'external',
    'Time-Series': 'analytdb', 'Search': 'database',
}

_REL_LABELS = {
    'Database': 'SQL/reads/writes', 'Cache': 'cache ops',
    'Queue': 'pub / sub', 'Analytics DB': 'analytics writes',
    'OAuth': 'OAuth2 flow', 'Auth': 'auth tokens',
    'Storage': 'file storage', 'CDN/DNS': 'CDN · DNS · API',
    'Monitoring': 'errors · traces', 'Alerting': 'webhooks',
    'Analytics': 'track events', 'CRM': 'user events',
    'Realtime': 'broadcast', 'Push': 'push notify',
    'Payments': 'payment API', 'Email': 'send email',
    'SMS': 'send SMS', 'KYC': 'identity verify',
    'Microservice': 'HTTP · gRPC', 'Finance': 'rate feed',
    'AI': 'inference API', 'Maps': 'map API', 'Search': 'search queries',
}


def _classdefs_block(indent='    '):
    return [f'{indent}{cd}' for cd in _CLASSDEFS]


# ── L1: System Context ────────────────────────────────────────────────────────

def _make_l1(project_name, framework, actors, integrations):
    fw, lang, ver = framework['name'], framework['language'], framework.get('version', '')
    stack = f"{fw} · {lang}" + (f" {ver}" if ver else '')
    sys_alias = c4_alias(project_name)

    lines = ['```mermaid', _MERMAID_INIT, 'graph TB']
    lines += _classdefs_block()
    lines.append('')

    # Central system node
    lines.append(f'    {sys_alias}["🏗️ **{project_name}**\\n─────────────────\\n{stack}"]:::system')
    lines.append('')

    # Actors subgraph
    lines.append('    subgraph ACTORS["👥 Users & Clients"]')
    for a in actors:
        lines.append(f'        {a["alias"]}["{a["icon"]} {a["name"]}"]:::actor')
    lines.append('    end')
    lines.append('')

    # Integration subgraphs per category
    cat_groups = {}
    for intg in integrations:
        cat_groups.setdefault(intg['category'], []).append(intg)

    for cat, items in cat_groups.items():
        grp = c4_alias(cat.replace('/', '_'))
        lines.append(f'    subgraph {grp}["{cat}"]')
        for item in items:
            css = _CAT_CSS.get(cat, 'external')
            lines.append(f'        {item["alias"]}["{item["icon"]} {item["name"]}"]:::{css}')
        lines.append('    end')
        lines.append('')

    # Arrows
    for a in actors:
        lines.append(f'    {a["alias"]} -->|"HTTP / HTTPS"| {sys_alias}')
    lines.append('')
    for intg in integrations:
        label = _REL_LABELS.get(intg['category'], 'uses')
        lines.append(f'    {sys_alias} -->|"{label}"| {intg["alias"]}')

    lines.append('```')
    return '\n'.join(lines)


# ── L2: Container Diagram ─────────────────────────────────────────────────────

_CONTAINER_CSS = {
    'App': 'app', 'Proxy': 'proxy', 'Database': 'database',
    'Cache': 'cache', 'Queue': 'queue', 'Analytics DB': 'analytdb',
    'Storage': 'storage', 'Search': 'database', 'Monitoring': 'monitoring',
    'Time-Series': 'analytdb',
}


def _infer_containers(framework, integrations):
    fw, lang = framework['name'], framework['language']
    icons = {'PHP': '🐘', 'Python': '🐍', 'Ruby': '💎', 'Go': '🐹',
             'TypeScript': '📘', 'JavaScript': '📗', 'Java': '☕',
             'Rust': '🦀', 'Elixir': '💜', 'C#': '🔷', 'Kotlin': '🟣'}
    icon = icons.get(lang, '📦')
    ctrs = [{'name': f'{fw} App', 'type': 'App', 'tech': f'{fw} · {lang}',
              'icon': icon, 'port': '8080', 'has_build': True}]
    if any(i['category'] == 'Queue' for i in integrations):
        ctrs.append({'name': 'Queue Worker', 'type': 'App', 'tech': f'{lang} Worker',
                     'icon': '⚙️', 'port': '', 'has_build': True})
    if fw in ('Laravel', 'Django', 'Rails', 'Spring Boot', 'Phoenix'):
        ctrs.append({'name': 'Scheduler', 'type': 'App', 'tech': 'Cron',
                     'icon': '⏰', 'port': '', 'has_build': True})
    return ctrs


def _make_l2(project_name, framework, containers, integrations):
    if not containers:
        containers = _infer_containers(framework, integrations)

    lines = ['```mermaid', _MERMAID_INIT, 'graph TD']
    lines += _classdefs_block()
    lines.append('')

    lines.append(f'    subgraph SYS["{project_name} — Infrastructure"]')
    for c in containers:
        alias = c4_alias(c['name'])
        port = f"\\n:{c['port']}" if c.get('port') else ''
        lbl = f"{c.get('icon','📦')} **{c['name']}**\\n{c.get('tech', c['name'])}{port}"
        css = _CONTAINER_CSS.get(c['type'], 'app')
        lines.append(f'        {alias}["{lbl}"]:::{css}')
    lines.append('    end')
    lines.append('')

    # External infra from integrations
    ext_cats = {'Database', 'Cache', 'Queue', 'Analytics DB', 'Storage', 'CDN/DNS', 'Search', 'Time-Series'}
    ext = [i for i in integrations if i['category'] in ext_cats]
    if ext:
        lines.append('    subgraph EXT["☁️ External Infrastructure"]')
        for i in ext:
            css = _CAT_CSS.get(i['category'], 'external')
            lines.append(f'        {i["alias"]}["{i["icon"]} {i["name"]}"]:::{css}')
        lines.append('    end')
        lines.append('')

    # Relationships
    alias_map = {c['name']: c4_alias(c['name']) for c in containers}
    proxy = next((c for c in containers if c['type'] == 'Proxy'), None)
    app = next((c for c in containers if c['type'] == 'App' and c.get('has_build')), None)
    worker = next((c for c in containers if c['type'] == 'App' and 'worker' in c['name'].lower()), None)
    db = next((c for c in containers if c['type'] == 'Database'), None)
    cache = next((c for c in containers if c['type'] == 'Cache'), None)
    queue = next((c for c in containers if c['type'] == 'Queue'), None)

    if proxy and app:
        lines.append(f'    {alias_map[proxy["name"]]} -->|"routes HTTP"| {alias_map[app["name"]]}')
    if app and db:
        lines.append(f'    {alias_map[app["name"]]} -->|"SQL queries"| {alias_map[db["name"]]}')
    if app and cache:
        lines.append(f'    {alias_map[app["name"]]} -->|"cache · sessions"| {alias_map[cache["name"]]}')
    if app and queue:
        lines.append(f'    {alias_map[app["name"]]} -->|"publish jobs"| {alias_map[queue["name"]]}')
    if worker and queue:
        lines.append(f'    {alias_map[queue["name"]]} -->|"consume"| {alias_map[worker["name"]]}')
    if worker and db:
        lines.append(f'    {alias_map[worker["name"]]} -->|"reads/writes"| {alias_map[db["name"]]}')

    lines.append('```')
    return '\n'.join(lines)


# ── L3: Component Diagram ─────────────────────────────────────────────────────

_TYPE_ICONS = {
    'domain': '🧩', 'app': '🔧', 'service': '⚙️',
    'module': '📦', 'package': '📁', 'controller_group': '🎮',
    'context': '🔷', 'package': '📁',
}

_COMMON_RELS = [
    ('notification', ['user', 'auth', 'order', 'payment', 'wallet']),
    ('auth',         ['user']),
    ('payment',      ['order', 'user', 'wallet']),
    ('wallet',       ['user', 'transaction']),
    ('bonus',        ['user', 'wallet']),
    ('loyalty',      ['user', 'bonus']),
    ('geo',          ['user']),
    ('mirror',       ['geo']),
    ('kyc',          ['user']),
    ('daily',        ['user', 'bonus']),
    ('order',        ['user', 'payment']),
    ('analytics',    ['user', 'order']),
]


def _make_l3(project_name, framework, modules):
    if not modules:
        return ''
    fw = framework['name']
    lines = ['```mermaid', _MERMAID_INIT, 'graph LR']
    lines += _classdefs_block()
    lines.append('')

    lines.append(f'    subgraph APP["{fw} Application"]')
    for mod in modules:
        alias = c4_alias(mod['name'])
        icon = _TYPE_ICONS.get(mod.get('type', 'module'), '📦')
        fcount = f"\\n{mod['files']} files" if mod.get('files', 0) > 0 else ''
        lines.append(f'        {alias}["{icon} {mod["name"]}{fcount}"]:::domain')
    lines.append('    end')
    lines.append('')

    # Infer relationships
    mod_lower = {m['name'].lower(): c4_alias(m['name']) for m in modules}
    added = set()
    for src_pat, tgt_pats in _COMMON_RELS:
        src_alias = next((a for n, a in mod_lower.items() if src_pat in n), None)
        if not src_alias:
            continue
        for tp in tgt_pats:
            tgt_alias = next((a for n, a in mod_lower.items() if tp in n and a != src_alias), None)
            if tgt_alias:
                key = (src_alias, tgt_alias)
                if key not in added:
                    added.add(key)
                    lines.append(f'    {src_alias} -.->|"uses"| {tgt_alias}')

    lines.append('```')
    return '\n'.join(lines)


# ── L4: Code Diagram ──────────────────────────────────────────────────────────

_FILE_CSS = {
    'model': 'entity', 'entity': 'entity',
    'service': 'service', 'manager': 'service', 'handler': 'service',
    'repo': 'repo', 'repository': 'repo', 'store': 'repo',
    'job': 'job', 'command': 'job', 'console': 'job',
    'event': 'event', 'listener': 'event', 'subscriber': 'event',
    'resource': 'resource', 'transformer': 'resource', 'nova': 'resource',
}

_FILE_ICONS = {
    'entity': '🧱', 'service': '⚙️', 'repo': '🗄️',
    'job': '⏳', 'event': '📡', 'resource': '📄', 'external': '📦',
}


def _guess_css(name, group=''):
    n = (name + group).lower()
    for key, css in _FILE_CSS.items():
        if key in n:
            return css
    return 'external'


def _make_l4(root, framework, modules):
    if not modules:
        return '', ''
    target = max(modules, key=lambda m: m.get('files', 0))
    tpath = root / target['path']
    if not tpath.exists():
        return target['name'], ''
    ext_map = {
        'PHP': '.php', 'Python': '.py', 'Ruby': '.rb',
        'TypeScript': '.ts', 'JavaScript': '.js', 'Go': '.go',
        'Java': '.java', 'Kotlin': '.kt', 'Rust': '.rs',
        'C#': '.cs', 'Elixir': '.ex',
    }
    ext = ext_map.get(framework['language'], '.php')
    groups = {}
    for f in tpath.rglob(f'*{ext}'):
        try:
            rel = f.relative_to(tpath)
            grp = str(rel.parent) if len(rel.parts) > 1 else 'root'
            groups.setdefault(grp, []).append(f.name)
        except ValueError:
            pass
    if not groups:
        return target['name'], ''

    lines = ['```mermaid', _MERMAID_INIT, 'graph TB']
    lines += _classdefs_block()
    lines.append('')

    for grp, files in sorted(groups.items()):
        glbl = target['name'] if grp == 'root' else grp.replace('/', ' › ')
        ga = c4_alias(f"{target['name']}_{grp}")
        lines.append(f'    subgraph {ga}["{glbl}"]')
        for fname in sorted(files[:12]):
            base = fname.replace(ext, '')
            fa = c4_alias(f"{target['name']}_{base}")
            css = _guess_css(fname, grp)
            icon = _FILE_ICONS.get(css, '📦')
            lines.append(f'        {fa}["{icon} {base}"]:::{css}')
        lines.append('    end')
        lines.append('')

    lines.append('```')
    return target['name'], '\n'.join(lines)


# ── MCP Tool ──────────────────────────────────────────────────────────────────

@mcp.tool()
def ndoc_c4(project_path: str = "", output: str = "docs/c4-architecture.md") -> str:
    """
    Генерирует полную C4-архитектурную модель (уровни 1–4) из кодовой базы.

    Анализирует конфигурационные файлы, маршруты, зависимости и инфраструктуру.
    Поддерживает Laravel, Symfony, Django, FastAPI, Flask, Rails, Express, NestJS,
    Next.js, Nuxt, Spring Boot, Gin, Echo, Fiber, Phoenix, Rust Web и другие.

    project_path: путь к проекту (по умолчанию — текущая директория)
    output:       путь к выходному файлу (по умолчанию docs/c4-architecture.md)
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"❌ Папка не найдена: {project_path}"

    log = [f"🏗️  NeuroDoc C4 — {root.name}", "━" * 48, ""]

    log.append("🔍 1/6 — Определяем стек...")
    fw = _detect_framework(root)
    ver_str = f" {fw['version']}" if fw.get('version') else ''
    log.append(f"   ✓ {fw['name']} · {fw['language']}{ver_str}")

    log.append("👥 2/6 — Анализируем акторов...")
    actors = _detect_actors(root)
    log.append(f"   ✓ {len(actors)} тип(а): {', '.join(a['name'] for a in actors)}")

    log.append("🔌 3/6 — Ищем интеграции...")
    integrations = _detect_integrations(root)
    log.append(f"   ✓ {len(integrations)} внешних сервисов")

    log.append("🐳 4/6 — Читаем инфраструктуру...")
    containers = _parse_docker_compose(root)
    if containers:
        log.append(f"   ✓ docker-compose: {len(containers)} контейнеров")
    else:
        log.append("   ℹ️  docker-compose не найден — инферируем из стека")

    log.append("🧩 5/6 — Определяем модули...")
    modules = _detect_modules(root, fw)
    log.append(f"   ✓ {len(modules)} модулей")

    log.append("✍️  6/6 — Генерируем диаграммы...")
    name = root.name
    today = datetime.now().strftime('%Y-%m-%d')
    stack_str = f"{fw['name']} · {fw['language']}{ver_str}"

    l1 = _make_l1(name, fw, actors, integrations)
    l2 = _make_l2(name, fw, containers, integrations)
    l3 = _make_l3(name, fw, modules)
    l4_mod, l4 = _make_l4(root, fw, modules)

    doc = [
        f"# {name} — C4 Architecture Model",
        "",
        f"> **System:** {name}  ",
        f"> **Stack:** {stack_str}  ",
        f"> **Updated:** {today}",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"**{name}** is built on **{fw['name']}** ({fw['language']}).",
        f"It integrates with **{len(integrations)} external services** and",
        f"is composed of **{len(modules)} domain modules**.",
        "",
        "---",
        "",
        "## Level 1 — System Context",
        "",
        "> Who uses the system and what external systems does it touch?",
        "> *Audience: executives, stakeholders.*",
        "",
        l1,
        "",
    ]

    if actors:
        doc += ["**Actors:**", ""]
        for a in actors:
            doc.append(f"- {a['icon']} **{a['name']}**")
        doc.append("")

    if integrations:
        doc += [
            "**External services:**",
            "",
            "| Service | Category |",
            "|---------|----------|",
        ]
        for i in integrations:
            doc.append(f"| {i['icon']} {i['name']} | {i['category']} |")
        doc.append("")

    doc += [
        "---",
        "",
        "## Level 2 — Container Diagram",
        "",
        "> Separately deployable units and their interactions.",
        "> *Audience: architects, senior developers.*",
        "",
        l2,
        "",
    ]

    eff_containers = containers or _infer_containers(fw, integrations)
    if eff_containers:
        doc += [
            "| Container | Type | Technology |",
            "|-----------|------|------------|",
        ]
        for c in eff_containers:
            doc.append(f"| {c.get('icon','📦')} {c['name']} | {c['type']} | {c.get('tech','—')} |")
        doc.append("")

    doc += [
        "---",
        "",
        "## Level 3 — Component Diagram",
        "",
        f"> Internal structure of the **{fw['name']}** application.",
        "> *Audience: developers.*",
        "",
        l3 if l3 else "_No modules detected — run ndoc_init first._",
        "",
    ]

    if modules:
        doc += [
            "| Module | Type | Files |",
            "|--------|------|-------|",
        ]
        for m in modules:
            doc.append(f"| {m['name']} | {m.get('type','module')} | {m.get('files',0)} |")
        doc.append("")

    if l4:
        doc += [
            "---",
            "",
            f"## Level 4 — Code Diagram: {l4_mod}",
            "",
            f"> Internal file/class structure of the **{l4_mod}** module.",
            "> *Audience: developers working in this module.*",
            "",
            l4,
            "",
        ]

    out_path = resolve_path(project_path) / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(doc), encoding='utf-8')

    log += [
        "",
        "━" * 48,
        "✅ C4 модель готова!",
        "",
        f"   📄 {output} ({len(doc)} строк)",
        f"   📐 L1 — {len(actors)} акторов, {len(integrations)} интеграций",
        f"   📐 L2 — {len(eff_containers)} контейнеров",
        f"   📐 L3 — {len(modules)} модулей",
        f"   📐 L4 — {l4_mod or 'не найден'}",
        "",
        f"💡 Файл: {out_path}",
    ]
    return '\n'.join(log)


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
