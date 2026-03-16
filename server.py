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
import warnings

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

CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go',
                   '.php', '.rb', '.java', '.cs', '.swift', '.kt', '.kts', '.rs'}

_MINIFIED_RE = re.compile(
    r'(\.min\.|[.\-]bundle\.|[.\-]bundle-|-bundle\b)',
    re.IGNORECASE,
)

def is_minified_file(filepath: Path) -> bool:
    """Detect minified/bundled files by name or line length."""
    if _MINIFIED_RE.search(filepath.name):
        return True
    try:
        with filepath.open(encoding='utf-8', errors='ignore') as fh:
            sample = [fh.readline() for _ in range(5)]
        non_empty = [l for l in sample if l.strip()]
        if non_empty and sum(len(l) for l in non_empty) / len(non_empty) > 250:
            return True
    except Exception:
        pass
    return False


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
    if is_minified_file(filepath):
        return {'functions': [], 'imports': []}
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
# PHP PARSER
# ─────────────────────────────────────────────

def parse_php(filepath: Path) -> dict:
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}
    functions = []
    imports = []
    seen = set()
    try:
        for m in re.finditer(r'(?:abstract\s+|final\s+)?class\s+(\w+)', content):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                functions.append({'name': name, 'params': [], 'calls': []})
        for m in re.finditer(r'(?:public|protected|private)(?:\s+static)?\s+function\s+(\w+)\s*\(([^)]*)\)', content):
            name = m.group(1)
            raw_params = m.group(2).strip()
            params = []
            for p in raw_params.split(','):
                p = p.strip()
                pm = re.search(r'\$(\w+)', p)
                if pm:
                    params.append('$' + pm.group(1))
            key = f"{name}({','.join(params)})"
            if key not in seen:
                seen.add(key)
                functions.append({'name': name, 'params': params, 'calls': []})
        for m in re.finditer(r'^use\s+([\w\\]+)(?:\s+as\s+\w+)?;', content, re.MULTILINE):
            parts = m.group(1).split('\\')
            imp = parts[-1]
            if imp not in imports:
                imports.append(imp)
    except Exception:
        pass
    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# RUBY PARSER
# ─────────────────────────────────────────────

def parse_ruby(filepath: Path) -> dict:
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}
    functions = []
    imports = []
    seen = set()
    try:
        for m in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                functions.append({'name': name, 'params': [], 'calls': []})
        for m in re.finditer(r'^\s*def\s+(\w+)\s*(?:\(([^)]*)\))?', content, re.MULTILINE):
            name = m.group(1)
            raw = (m.group(2) or '').strip()
            params = [p.strip() for p in raw.split(',')] if raw else []
            key = f"{name}({','.join(params)})"
            if key not in seen:
                seen.add(key)
                functions.append({'name': name, 'params': params, 'calls': []})
        for m in re.finditer(r'^require(?:_relative)?\s+[\'"]([^\'"]+)[\'"]', content, re.MULTILINE):
            imp = Path(m.group(1)).stem
            if imp not in imports:
                imports.append(imp)
    except Exception:
        pass
    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# JAVA PARSER
# ─────────────────────────────────────────────

def parse_java(filepath: Path) -> dict:
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}
    functions = []
    imports = []
    seen = set()
    try:
        for m in re.finditer(r'(?:public|private|protected)?\s*(?:abstract\s+)?class\s+(\w+)', content):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                functions.append({'name': name, 'params': [], 'calls': []})
        for m in re.finditer(r'(?:public|private|protected)(?:\s+static)?(?:\s+final)?\s+\w[\w<>,\s]*\s+(\w+)\s*\(([^)]*)\)', content):
            name = m.group(1)
            raw = m.group(2).strip()
            params = []
            for p in raw.split(','):
                p = p.strip()
                parts = p.split()
                if len(parts) >= 2:
                    params.append(parts[-2])
            key = f"{name}({','.join(params)})"
            if key not in seen:
                seen.add(key)
                functions.append({'name': name, 'params': params, 'calls': []})
        for m in re.finditer(r'^import\s+(?:static\s+)?([\w.]+);', content, re.MULTILINE):
            imp = m.group(1).split('.')[-1]
            if imp not in imports:
                imports.append(imp)
    except Exception:
        pass
    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# C# PARSER
# ─────────────────────────────────────────────

def parse_csharp(filepath: Path) -> dict:
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}
    functions = []
    imports = []
    seen = set()
    try:
        for m in re.finditer(r'(?:public|internal|private|protected)?\s*(?:abstract\s+|sealed\s+)?class\s+(\w+)', content):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                functions.append({'name': name, 'params': [], 'calls': []})
        for m in re.finditer(r'(?:public|private|protected|internal)(?:\s+static)?(?:\s+async)?\s+\w[\w<>?,\s]*\s+(\w+)\s*\(([^)]*)\)', content):
            name = m.group(1)
            raw = m.group(2).strip()
            params = []
            for p in raw.split(','):
                p = p.strip()
                parts = p.split()
                if len(parts) >= 2:
                    params.append(parts[0])
            key = f"{name}({','.join(params)})"
            if key not in seen:
                seen.add(key)
                functions.append({'name': name, 'params': params, 'calls': []})
        for m in re.finditer(r'^using\s+([\w.]+);', content, re.MULTILINE):
            imp = m.group(1).split('.')[-1]
            if imp not in imports:
                imports.append(imp)
    except Exception:
        pass
    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# SWIFT PARSER
# ─────────────────────────────────────────────

def parse_swift(filepath: Path) -> dict:
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}
    functions = []
    imports = []
    seen = set()
    try:
        for m in re.finditer(r'(?:class|struct|enum|protocol)\s+(\w+)', content):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                functions.append({'name': name, 'params': [], 'calls': []})
        for m in re.finditer(r'func\s+(\w+)\s*\(([^)]*)\)', content):
            name = m.group(1)
            raw = m.group(2).strip()
            params = [p.split(':')[0].strip() for p in raw.split(',') if ':' in p]
            key = f"{name}({','.join(params)})"
            if key not in seen:
                seen.add(key)
                functions.append({'name': name, 'params': params, 'calls': []})
        for m in re.finditer(r'^import\s+(\w+)', content, re.MULTILINE):
            imp = m.group(1)
            if imp not in imports:
                imports.append(imp)
    except Exception:
        pass
    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# KOTLIN PARSER
# ─────────────────────────────────────────────

def parse_kotlin(filepath: Path) -> dict:
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}
    functions = []
    imports = []
    seen = set()
    try:
        for m in re.finditer(r'(?:class|object|interface|data\s+class)\s+(\w+)', content):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                functions.append({'name': name, 'params': [], 'calls': []})
        for m in re.finditer(r'fun\s+(\w+)\s*\(([^)]*)\)', content):
            name = m.group(1)
            raw = m.group(2).strip()
            params = []
            for p in raw.split(','):
                p = p.strip()
                if ':' in p:
                    params.append(p.split(':')[0].strip())
            key = f"{name}({','.join(params)})"
            if key not in seen:
                seen.add(key)
                functions.append({'name': name, 'params': params, 'calls': []})
        for m in re.finditer(r'^import\s+([\w.]+)', content, re.MULTILINE):
            imp = m.group(1).split('.')[-1]
            if imp not in imports:
                imports.append(imp)
    except Exception:
        pass
    return {'functions': functions, 'imports': imports}


# ─────────────────────────────────────────────
# RUST PARSER
# ─────────────────────────────────────────────

def parse_rust(filepath: Path) -> dict:
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'functions': [], 'imports': []}
    functions = []
    imports = []
    seen = set()
    try:
        for m in re.finditer(r'(?:pub\s+)?struct\s+(\w+)', content):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                functions.append({'name': name, 'params': [], 'calls': []})
        for m in re.finditer(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)', content):
            name = m.group(1)
            raw = m.group(2).strip()
            params = []
            for p in raw.split(','):
                p = p.strip()
                if ':' in p:
                    params.append(re.sub(r'^[&\s]*(mut\s+)?', '', p.split(':')[0].strip()))
            key = f"{name}({','.join(params)})"
            if key not in seen:
                seen.add(key)
                functions.append({'name': name, 'params': params, 'calls': []})
        for m in re.finditer(r'^use\s+([\w:]+)', content, re.MULTILINE):
            parts = re.split(r'::', m.group(1))
            imp = parts[-1]
            if imp not in imports:
                imports.append(imp)
    except Exception:
        pass
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
    elif ext == '.php':
        return parse_php(filepath)
    elif ext == '.rb':
        return parse_ruby(filepath)
    elif ext == '.java':
        return parse_java(filepath)
    elif ext == '.cs':
        return parse_csharp(filepath)
    elif ext == '.swift':
        return parse_swift(filepath)
    elif ext in ('.kt', '.kts'):
        return parse_kotlin(filepath)
    elif ext == '.rs':
        return parse_rust(filepath)
    return {'functions': [], 'imports': []}


def scan_dir(dir_path: Path) -> dict:
    result = {}
    try:
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in CODE_EXTENSIONS:
                parsed = parse_file(f)
                if parsed['functions']:
                    result[f.name] = parsed
    except PermissionError as e:
        warnings.warn(f"Permission denied: {e}")
    return result


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def short_name(dir_path: Path, root: Path) -> str:
    if dir_path == root:
        return root.name
    try:
        return str(dir_path.relative_to(root))
    except ValueError:
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
    except PermissionError as e:
        warnings.warn(f"Permission denied: {e}")
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


_c4_alias_registry: dict = {}


def c4_alias(name: str) -> str:
    """Create a short, valid Mermaid node ID from a name.
    Extracts first meaningful word(s) to avoid 80-char monster IDs.
    'HTTP API — Chi router — REST endpoints' → 'HTTP_API'
    'mail-backend — image:latest, ports 8080' → 'mail_backend'
    Deduplicates aliases by appending _2, _3, etc. when collisions occur.
    """
    # Take only the part before first separator (—, -, ,) if name is long
    if len(name) > 25:
        for sep in (' — ', ' – ', ' - ', ', ', ' ('):
            idx = name.find(sep)
            if idx > 3:
                name = name[:idx]
                break
    # Strip non-alphanumeric chars
    alias = re.sub(r'[^a-zA-Z0-9]', '_', name.strip()).strip('_')
    # Collapse runs of underscores
    alias = re.sub(r'_+', '_', alias)
    # Truncate to 40 chars max
    base = alias[:40] or 'mod'
    if base not in _c4_alias_registry:
        _c4_alias_registry[base] = 1
        return base
    _c4_alias_registry[base] += 1
    return f"{base}_{_c4_alias_registry[base]}"


def c4_label(s: str) -> str:
    return s.replace('"', "'")[:120]


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

# Libraries/frameworks that should NOT appear as external systems in C4 Context.
# These are code dependencies, not architectural systems.
_LIBRARY_NOISE = {
    # Go frameworks & tooling
    'grpc', 'protobuf', 'proto', 'golang', 'google', 'go',
    'gorilla', 'gin', 'echo', 'fiber', 'chi', 'mux',
    'gorm', 'sqlx', 'pgx', 'ent', 'bun',
    'zap', 'logrus', 'zerolog', 'slog',
    'viper', 'cobra', 'pflag',
    'testify', 'gomock', 'mockery',
    'wire', 'fx', 'dig',
    'jwt', 'uuid', 'validator',
    'otel', 'opentelemetry', 'jaeger', 'zipkin',
    # Python frameworks
    'fastapi', 'flask', 'django', 'starlette', 'uvicorn', 'gunicorn',
    'sqlalchemy', 'alembic', 'tortoise', 'peewee',
    'pydantic', 'marshmallow', 'attrs',
    'celery', 'dramatiq', 'rq',
    'httpx', 'aiohttp', 'requests',
    'pytest', 'unittest', 'hypothesis',
    'boto3', 'botocore',
    # Node frameworks
    'axios', 'fetch', 'superagent', 'node-fetch', 'ky', 'undici',
    'express', 'fastify', 'koa', 'hapi', 'nestjs', 'nest',
    'prisma', 'typeorm', 'sequelize', 'mongoose', 'knex',
    'jest', 'mocha', 'vitest', 'chai',
    'webpack', 'vite', 'esbuild', 'rollup', 'babel',
    'eslint', 'prettier', 'typescript',
    'lodash', 'underscore', 'ramda', 'dayjs', 'moment',
    'zod', 'yup', 'joi',
    # Java frameworks
    'spring', 'springframework', 'hibernate', 'lombok',
    'junit', 'mockito', 'assertj',
    # PHP frameworks
    'laravel', 'symfony', 'slim', 'lumen',
    'eloquent', 'doctrine',
    # Ruby frameworks
    'rails', 'sinatra', 'hanami',
    'activerecord', 'rspec',
    # Rust frameworks
    'tokio', 'actix', 'axum', 'warp', 'rocket',
    'serde', 'reqwest', 'sqlx',
    # .NET frameworks
    'aspnet', 'efcore', 'xunit', 'nunit',
    # Generic
    'utils', 'helpers', 'common', 'shared', 'core', 'base',
    'config', 'internal', 'pkg', 'lib', 'src',
}

# Only these qualify as real external SYSTEMS in C4 Context
_REAL_EXTERNAL_SYSTEMS = {
    # Databases
    'postgres', 'postgresql', 'mysql', 'mariadb', 'mongodb', 'mongo',
    'sqlite', 'mssql', 'cockroachdb', 'dynamodb', 'cassandra', 'neo4j',
    'clickhouse', 'bigquery', 'snowflake', 'redshift',
    # Cache
    'redis', 'memcached', 'dragonflydb',
    # Message brokers / queues
    'kafka', 'rabbitmq', 'nats', 'pulsar', 'sqs', 'pubsub',
    'activemq', 'zeromq', 'celery',
    # Search
    'elasticsearch', 'opensearch', 'algolia', 'typesense', 'meilisearch',
    # Storage
    's3', 'gcs', 'minio', 'cloudinary', 'azure',
    # Email providers
    'mailgun', 'sendgrid', 'postmark', 'ses', 'smtp', 'mandrill', 'sparkpost',
    # Monitoring / Observability
    'prometheus', 'grafana', 'datadog', 'newrelic', 'sentry', 'bugsnag',
    'honeycomb', 'lightstep', 'dynatrace',
    # Payment
    'stripe', 'paypal', 'braintree', 'square', 'adyen',
    # Auth providers
    'auth0', 'okta', 'keycloak', 'cognito', 'firebase',
    # CDN / Infra
    'cloudflare', 'nginx', 'traefik', 'envoy',
    # Specific external APIs (user-named)
    'twilio', 'plivo', 'vonage', 'pusher', 'ably',
    'github', 'gitlab', 'jira', 'slack', 'discord',
    'google', 'facebook', 'twitter', 'apple',
}


def is_real_external_system(name: str) -> bool:
    """Return True only if this dep is an actual external system (not a library)."""
    n = name.lower().replace('-', '').replace('_', '').replace('.', '')
    # Check exact match in real systems
    if n in _REAL_EXTERNAL_SYSTEMS or name.lower() in _REAL_EXTERNAL_SYSTEMS:
        return True
    # Check if it's known library noise
    if n in _LIBRARY_NOISE or name.lower() in _LIBRARY_NOISE:
        return False
    # Check stdlib
    if name.lower() in _STDLIB_NOISE:
        return False
    # Heuristic: Go module paths like "google.golang.org/grpc" → library
    if re.search(r'\.(go|org|com|io|dev|net)\b', name.lower()):
        return False
    return True


# ─────────────────────────────────────────────
# RELATIONSHIP LABELS
# ─────────────────────────────────────────────

_REL_LABELS: dict = {
    # Databases
    'postgres': 'reads/writes',
    'postgresql': 'reads/writes',
    'mysql': 'reads/writes',
    'mariadb': 'reads/writes',
    'mongodb': 'reads/writes',
    'mongo': 'reads/writes',
    'sqlite': 'reads/writes',
    'mssql': 'reads/writes',
    'dynamodb': 'reads/writes',
    'cassandra': 'reads/writes',
    'clickhouse': 'OLAP events',
    # Cache
    'redis': 'cache/sessions',
    'memcached': 'cache',
    'valkey': 'cache',
    # HTTP / API
    'axios': 'HTTP requests',
    'httpx': 'HTTP requests',
    'requests': 'HTTP requests',
    'got': 'HTTP requests',
    'superagent': 'HTTP requests',
    'guzzle': 'HTTP requests',
    # Auth
    'jwt': 'auth',
    'passport': 'auth',
    'sanctum': 'auth',
    'oauth': 'auth',
    'auth': 'auth',
    # Queues / workers
    'kafka': 'event stream',
    'rabbitmq': 'async jobs',
    'celery': 'async tasks',
    'sqs': 'async jobs',
    'bull': 'async jobs',
    'beanstalkd': 'async jobs',
    'horizon': 'queue management',
    # Search
    'elasticsearch': 'full-text search',
    'algolia': 'search',
    'typesense': 'search',
    'meilisearch': 'search',
    'scout': 'search',
    # Storage / CDN
    's3': 'file storage',
    'cloudinary': 'media storage',
    'minio': 'file storage',
    'flysystem': 'file storage',
    # Monitoring / Logging
    'sentry': 'error reporting',
    'datadog': 'metrics',
    'prometheus': 'metrics',
    'grafana': 'dashboards',
    'bugsnag': 'error reporting',
    # Mail
    'mailgun': 'sends email',
    'sendgrid': 'sends email',
    'smtp': 'sends email',
    'mailer': 'sends email',
    # Payment
    'stripe': 'payments',
    'paypal': 'payments',
    'braintree': 'payments',
    # Real-time
    'pusher': 'WebSocket events',
    'aws_s3': 'file storage',
    'aws s3': 'file storage',
}


def get_rel_label(dep: str) -> str:
    return _REL_LABELS.get(dep.lower(), 'uses')


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
    _c4_alias_registry.clear()
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
        label = get_rel_label(dep)
        lines.append(f'  Rel({proj}, {c4_alias(dep)}, "{label}")')
    lines.append('```')
    return lines


def detect_layer(mod: str, dp: Path, root: Path) -> str:
    """Detect architectural layer: backend | frontend | module | admin | public | test."""
    try:
        parts = [p.lower() for p in dp.relative_to(root).parts]
    except ValueError:
        parts = [mod.lower()]
    if any(p in ('nova-components', 'nova_components') for p in parts):
        return 'admin'
    if any(p == 'modules' for p in parts):
        return 'module'
    if any(p in ('resources', 'js', 'vue', 'react', 'assets', 'frontend', 'client') for p in parts):
        return 'frontend'
    if any(p in ('public', 'static', 'dist', 'build') for p in parts):
        return 'public'
    if any(p in ('test', 'tests', 'spec', 'specs', '__tests__', 'testing') for p in parts):
        return 'test'
    return 'backend'


def make_c4_container(modules: dict, project_name: str, root: Path) -> list:
    """Return lines for C4Container diagram grouped by architectural layer."""
    # Collect modules at depth 0-2
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

    # Group by architectural layer
    layers: dict = {
        'backend': [], 'frontend': [], 'admin': [],
        'module': [], 'public': [], 'test': [],
    }
    for mod, data in top.items():
        dp = data.get('dir_path', root)
        layer = detect_layer(mod, dp, root)
        layers.setdefault(layer, []).append((mod, data))

    # Skip layers with no content
    active_layers = {k: v for k, v in layers.items() if v}

    proj = c4_alias(project_name)
    lines = ['```mermaid', 'C4Container', f'  title Container diagram — {project_name}', '']

    # User persona
    lines.append(f'  Person(user, "User", "End-user of {project_name}")')
    lines.append('')

    # System boundary
    lines.append(f'  System_Boundary({proj}, "{project_name}") {{')

    alias_map: dict = {}  # mod → alias
    layer_reps: dict = {}  # layer → first alias (for cross-layer rels)

    layer_labels = {
        'backend': 'Backend', 'frontend': 'Frontend',
        'admin': 'Admin Panel', 'module': 'Modules',
        'public': 'Public', 'test': 'Tests',
    }

    for layer, mods in active_layers.items():
        if layer in ('test', 'public'):
            continue
        for mod, data in mods:
            alias = c4_alias(mod)
            alias_map[mod] = alias
            if layer not in layer_reps:
                layer_reps[layer] = alias
            label = mod.split('/')[-1]
            tech = detect_tech(data.get('files', {}))
            kw = data.get('keywords', [])[:3]
            descr = c4_label(', '.join(kw)) if kw else layer_labels.get(layer, label)
            if layer == 'admin':
                tech = f'{tech}/Nova'
            ctype = module_container_type(mod)
            lines.append(f'    {ctype}({alias}, "{label}", "{tech}", "{descr}")')

    lines.append('  }')
    lines.append('')

    # --- Relationships ---

    # User → frontend or backend entry point
    fe_rep = layer_reps.get('frontend')
    be_rep = layer_reps.get('backend')
    adm_rep = layer_reps.get('admin')

    if fe_rep:
        lines.append(f'  Rel(user, {fe_rep}, "accesses via browser")')
    elif be_rep:
        lines.append(f'  Rel(user, {be_rep}, "accesses via browser")')
    if adm_rep and adm_rep != fe_rep:
        lines.append(f'  Rel(user, {adm_rep}, "manages via admin")')

    # Frontend → Backend (API)
    if fe_rep and be_rep:
        lines.append(f'  Rel({fe_rep}, {be_rep}, "API requests [HTTP/JSON]")')

    # Admin → Backend
    if adm_rep and be_rep:
        lines.append(f'  Rel({adm_rep}, {be_rep}, "extends [Laravel Nova]")')

    # Backend → Modules
    mod_reps = [alias for mod, _ in active_layers.get('module', []) if mod in alias_map
                for alias in [alias_map[mod]]]
    if be_rep and mod_reps:
        for m_alias in mod_reps[:5]:
            lines.append(f'  Rel({be_rep}, {m_alias}, "delegates to")')

    # Dep-based relationships (same-layer)
    added_rels: set = set()
    for mod, data in top.items():
        if mod not in alias_map:
            continue
        src = alias_map[mod]
        for dep in data.get('deps', []):
            if dep in alias_map:
                dst = alias_map[dep]
                key = (src, dst)
                if key not in added_rels and src != dst:
                    added_rels.add(key)
                    lines.append(f'  Rel({src}, {dst}, "uses")')

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
    # Build base-name → alias lookup for import matching
    base_to_alias: dict = {}
    for fname, alias in file_aliases.items():
        base = re.sub(r'\.(py|go|ts|js|tsx|jsx)$', '', fname).lower()
        base_to_alias[base] = alias
        base_to_alias[base.split('/')[-1]] = alias

    for fname, alias in file_aliases.items():
        fd = files_data[fname]

        # 1. Import-based relationships (most reliable)
        for imp in fd.get('imports', []):
            imp_base = re.sub(r'\.(ts|tsx|js|jsx|py|go)$', '', imp.split('/')[-1]).lower()
            if imp_base in base_to_alias:
                other_alias = base_to_alias[imp_base]
                if other_alias != alias:
                    key = (alias, other_alias)
                    if key not in added:
                        added.add(key)
                        lines.append(f'  Rel({alias}, {other_alias}, "imports")')

        # 2. Call-based relationships (fallback)
        all_calls: set = set()
        for fn in fd.get('functions', []):
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

def _infer_purpose(dir_path: Path, root: Path) -> str:
    """Infer a one-line purpose description from the directory name/path."""
    parts = dir_path.parts
    name = dir_path.name

    # Check path parts for known patterns
    path_str = str(dir_path)

    if 'nova-components' in path_str or 'nova_components' in path_str:
        return f"Custom Laravel Nova admin panel component: {name}"
    if 'app/Http/Controllers' in path_str or 'Http/Controllers' in path_str:
        return f"HTTP API controllers grouped by domain: {name}"
    if 'app/Services' in path_str or '/Services' in path_str:
        return f"Application service layer: {name}"
    if 'Modules/' in path_str:
        # Extract module name
        for part in parts:
            if part == 'Modules':
                idx = list(parts).index(part)
                if idx + 1 < len(parts):
                    domain = parts[idx + 1]
                    return f"Bounded context module: {domain} domain"
        return f"Bounded context module: {name}"

    # Check directory name suffixes/keywords
    name_lower = name.lower()
    if 'controller' in name_lower:
        return f"HTTP request handlers for {name} domain"
    if 'service' in name_lower:
        return f"Business logic layer for {name}"
    if 'repository' in name_lower or 'repo' in name_lower:
        return f"Data access layer — database queries for {name}"
    if 'middleware' in name_lower:
        return f"Request/response middleware for {name}"
    if 'model' in name_lower:
        return f"Eloquent models for {name} data"
    if 'provider' in name_lower:
        return f"Laravel service provider for {name}"
    if 'resource' in name_lower:
        return f"API resource transformers for {name}"
    if 'migration' in name_lower:
        return f"Database migrations for {name}"
    if 'event' in name_lower:
        return f"Domain events for {name}"
    if 'listener' in name_lower:
        return f"Event listeners for {name}"
    if 'job' in name_lower:
        return f"Async jobs/queue workers for {name}"
    if 'observer' in name_lower:
        return f"Eloquent model observers for {name}"
    if 'policy' in name_lower:
        return f"Authorization policies for {name}"
    if 'request' in name_lower:
        return f"Form request validators for {name}"
    if 'command' in name_lower:
        return f"Artisan console commands for {name}"
    if 'test' in name_lower or 'spec' in name_lower:
        return f"Test suite for {name}"
    if 'helper' in name_lower or 'util' in name_lower:
        return f"Utility helpers for {name}"
    if 'config' in name_lower:
        return f"Configuration files for {name}"

    return f"Code module: {name}"


def make_context_md(dir_path: Path, root: Path, all_modules: list, reverse_deps: dict = None) -> str:
    files_data = scan_dir(dir_path)
    mod = short_name(dir_path, root)
    now = datetime.now().strftime('%Y-%m-%d')

    # Check for ALL immediate subdirs (not just those with code)
    children = child_subdirs_with_code(dir_path)
    # Also get truly all subdirs for parent-only directories
    all_children = []
    try:
        all_children = sorted(
            [item for item in dir_path.iterdir() if item.is_dir() and item.name not in SKIP_DIRS],
            key=lambda p: p.name,
        )
    except PermissionError:
        pass

    # Collect imports → compute module deps
    all_imports = []
    for fd in files_data.values():
        all_imports.extend(fd.get('imports', []))
    deps = resolve_deps(all_imports, [m for m in all_modules if m != mod])

    # Reverse deps: who uses this module
    used_by = []
    if reverse_deps:
        used_by = reverse_deps.get(mod, [])

    # ── Header ──────────────────────────────────────────────────────────
    header = f"# {mod}"
    if deps:
        header += f" → {', '.join(deps)}"
    lines = [header, ""]

    if used_by:
        lines.append(f"**used by:** {', '.join(used_by)}")
        lines.append("")

    # ── Purpose ─────────────────────────────────────────────────────────
    purpose = _infer_purpose(dir_path, root)
    lines.append("## Назначение (Purpose)")
    lines.append("")
    lines.append(purpose)
    lines.append("")

    # ── Files & Functions ────────────────────────────────────────────────
    lines.append("## Файлы и функции")
    lines.append("")

    if files_data:
        for fname, fd in sorted(files_data.items()):
            funcs = fd.get('functions', [])
            lines.append(f"### {fname}")
            if not funcs:
                lines.append("- (no public API)")
            else:
                for fn in funcs:
                    params = ', '.join(fn['params'])
                    calls = fn.get('calls', [])
                    entry = f"- `{fn['name']}({params})`"
                    if calls:
                        entry += f" → {', '.join(calls)}"
                    lines.append(entry)
            lines.append("")
    elif all_children:
        # Parent-only dir: list subdirectories with a brief summary
        for child in all_children[:12]:
            child_data = scan_dir(child)
            child_funcs = []
            for fd in child_data.values():
                for fn in fd.get('functions', [])[:2]:
                    child_funcs.append(fn['name'])
            if child_funcs:
                lines.append(f"- **{child.name}/**: {', '.join(child_funcs[:4])}")
            else:
                has_any = any(True for _ in child.rglob('*') if _.is_file())
                lines.append(f"- **{child.name}/**" + (" — (no code files)" if not has_any else ""))
        lines.append("")
    else:
        lines.append("_No code files in this directory._")
        lines.append("")

    # ── Dependencies ─────────────────────────────────────────────────────
    lines.append("## Зависимости (Dependencies)")
    lines.append("")
    if deps:
        for dep in deps:
            lines.append(f"- `../{dep}/` — {_infer_purpose(root / dep, root)}")
    else:
        lines.append("_No detected dependencies._")
    lines.append("")

    # ── Submodules ───────────────────────────────────────────────────────
    if children:
        child_names = [c.name for c in children]
        lines.append(f"**submodules:** {', '.join(child_names)}")
        lines.append("")

    # ── Tags ─────────────────────────────────────────────────────────────
    tags: set = {mod}
    for d in deps:
        tags.add(d)
    # Add name parts (split by CamelCase and underscores)
    for part in re.split(r'[_\-/]', mod):
        if len(part) >= 3:
            tags.add(part.lower())
    for fname in files_data:
        base = re.sub(r'\.[a-z]+$', '', fname, flags=re.IGNORECASE)
        for part in re.split(r'[_\-]', base):
            if len(part) >= 3:
                tags.add(part.lower())
    for fd in files_data.values():
        for fn in fd.get('functions', [])[:4]:
            name = fn['name'].lower()
            # Skip obfuscated/minified names
            if len(name) >= 3 and not re.match(r'^[a-z]{1,2}\d*$', name):
                tags.add(name)
    for child in all_children[:6]:
        if len(child.name) >= 3:
            tags.add(child.name.lower())

    meaningful_tags = sorted(t for t in tags if len(t) >= 3)[:16]
    lines.append(f"## Теги")
    lines.append("")
    lines.append(', '.join(meaningful_tags))
    lines.append("")
    lines.append(f"**updated:** {now} ndoc")

    # ── C4 Component diagram (kept from original) ─────────────────────────
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
# NeuroDoc Navigation Rules

## Before Every Task
1. Read `context.index.md` in the project root — get architecture overview
2. Identify which modules your task touches
3. Read `context.md` in those module directories
4. Read `context.md` of their dependencies (listed after `→` in the header)
5. Only then start implementation

## After Every Code Change
1. Run `ndoc` (leave changed_files empty for git auto-detection)
2. If you added a new module or changed cross-module dependencies, also update `context.index.md`

## Context File Format
```markdown
# module_name → dep1, dep2

**used by:** parent_module

## filename.ext
- `FunctionName(param1 Type, param2 Type)` → dep1
- `HelperName()`

**submodules:** child1, child2
**tags:** keyword1 keyword2
**updated:** YYYY-MM-DD ndoc
```

## Rules
- Read context.md BEFORE reading source files — it is faster and preserves context window
- When context.md is stale (updated date old), run ndoc to find and update all stale modules
- Dependencies listed in `→` header are the modules this module imports from
- Modules listed in `**used by:**` are modules that import this one
- `**tags:**` are searchable keywords for this module's domain
"""


def _ensure_claude_md(root: Path) -> None:
    """Append NeuroDoc rules to CLAUDE.md if not already present."""
    claude_path = root / 'CLAUDE.md'
    if claude_path.exists():
        existing = claude_path.read_text(encoding='utf-8')
        if 'NeuroDoc' not in existing:
            claude_path.write_text(existing + CLAUDE_MD_RULES, encoding='utf-8')
    else:
        claude_path.write_text(f"# {root.name}\n{CLAUDE_MD_RULES}", encoding='utf-8')


# ─────────────────────────────────────────────
# MCP TOOL
# ─────────────────────────────────────────────

@mcp.tool()
def ndoc(project_path: str = "", changed_files: str = "") -> str:
    """
    Analyze, document, and keep a codebase navigable — one command does everything.

    Run this tool:
    - At the start of any session to understand project structure
    - After making code changes to update documentation
    - When you need architecture overview or dependency maps

    What it does automatically:
    1. Detects project state (fresh install vs. existing docs vs. code changes)
    2. If no context.md files exist → runs full deep architectural analysis and generates C4 diagrams (Context/Container/Component levels)
    3. If context.md exists and code changed → updates only the affected modules
    4. If everything is fresh → returns current architecture summary instantly
    5. Always returns: architecture overview, dependency map, stale module list

    Parameters:
    - project_path: Path to project root. Leave empty to use WORKSPACE_ROOT env var or current directory.
    - changed_files: Comma-separated changed file paths for targeted update (e.g. "app/service.py,app/models.py"). Leave empty for automatic git detection.

    Returns:
    - Architecture summary from context.index.md
    - List of modules updated (if any)
    - List of stale modules needing attention (if any)
    - C4 diagram summary (on first run)

    Use this as your primary tool for any codebase navigation task. No need to call multiple tools separately.
    """
    import warnings as _warnings

    root = resolve_path(project_path)
    if not root.exists():
        return f"Error: folder not found: {project_path!r}"

    results = []

    # --- PHASE 1: DETECT STATE ---
    context_index = root / "context.index.md"
    context_files = list(root.rglob("context.md"))
    context_files = [f for f in context_files if not any(
        skip in f.parts for skip in SKIP_DIRS
    )]

    has_docs = len(context_files) > 0
    has_index = context_index.exists()

    # --- PHASE 2: DETECT CHANGES ---
    changed = []
    if changed_files.strip():
        changed = [root / p.strip() for p in changed_files.split(",") if p.strip()]
    else:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=root, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                changed = [root / p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
        except Exception:
            pass

    # --- PHASE 3: ACT ---
    module_map = None  # will be populated in whichever branch runs

    if not has_docs:
        # FULL ANALYSIS - first time
        results.append("No documentation found — running full project analysis...")

        dirs_with_code = []
        for dp, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
            dp = Path(dp)
            if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files):
                dirs_with_code.append(dp)

        # Include ALL subdirectories (not just those containing code),
        # so every directory gets a context.md file.
        all_dirs = set(dirs_with_code)
        for dp_str, subdirs, _ in os.walk(root, followlinks=False):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
            all_dirs.add(Path(dp_str))
        all_dirs.add(root)

        all_module_names = [short_name(d, root) for d in all_dirs]
        module_map = {}
        for dp in sorted(all_dirs):
            mod = short_name(dp, root)
            files_data = scan_dir(dp)
            all_imports, keywords = [], []
            for fd in files_data.values():
                all_imports.extend(fd.get('imports', []))
                keywords.extend([fn['name'] for fn in fd.get('functions', [])[:2]])
            deps = resolve_deps(all_imports, [m for m in all_module_names if m != mod])
            module_map[mod] = {'dir_path': dp, 'files': files_data, 'deps': deps, 'keywords': keywords[:4]}

        reverse_deps: dict = {}
        for mod, info in module_map.items():
            for dep in info.get('deps', []):
                reverse_deps.setdefault(dep, []).append(mod)

        written = 0
        for mod, info in module_map.items():
            dp = info['dir_path']
            content = make_context_md(dp, root, all_module_names, reverse_deps)
            ctx_file = dp / "context.md"
            try:
                ctx_file.write_text(content, encoding="utf-8")
                written += 1
            except PermissionError as e:
                _warnings.warn(f"Permission denied writing {ctx_file}: {e}")

        index_content = make_index(module_map, root.name, root)
        try:
            context_index.write_text(index_content, encoding="utf-8")
        except PermissionError as e:
            _warnings.warn(f"Permission denied writing context.index.md: {e}")

        _ensure_claude_md(root)

        results.append(f"Generated {written} context.md files")
        results.append(f"Generated context.index.md with architecture overview")

    elif changed:
        # INCREMENTAL UPDATE
        results.append(f"Detected {len(changed)} changed files — updating affected modules...")

        affected_dirs = set()
        for f in changed:
            if f.parent.exists():
                affected_dirs.add(f.parent)

        all_module_names = []
        for dp, subdirs, _ in os.walk(root, followlinks=False):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
            all_module_names.append(short_name(Path(dp), root))

        module_map = {}
        for dp, subdirs, files_in_dir in os.walk(root, followlinks=False):
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
                module_map[mod] = {'dir_path': dp, 'files': fd_map, 'deps': deps, 'keywords': kw[:4]}

        reverse_deps = {}
        for mod, info in module_map.items():
            for dep in info.get('deps', []):
                reverse_deps.setdefault(dep, []).append(mod)

        updated = []
        for d in affected_dirs:
            mod = short_name(d, root)
            if mod in module_map:
                content = make_context_md(d, root, all_module_names, reverse_deps)
                ctx_file = d / "context.md"
                try:
                    ctx_file.write_text(content, encoding="utf-8")
                    updated.append(str(d.relative_to(root)))
                except PermissionError as e:
                    _warnings.warn(f"Permission denied: {e}")

        index_content = make_index(module_map, root.name, root)
        try:
            context_index.write_text(index_content, encoding="utf-8")
        except PermissionError as e:
            _warnings.warn(f"Permission denied writing context.index.md: {e}")

        if updated:
            results.append(f"Updated: {', '.join(updated)}")
        else:
            results.append("Changed files not in documented modules — no updates needed")

    else:
        results.append("Documentation is current — no changes detected")

    # --- PHASE 4: VALIDATE (always) ---
    stale = []
    missing = []

    if module_map is None:
        # Build module map for validation only
        val_module_names = []
        for dp, subdirs, _ in os.walk(root, followlinks=False):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
            val_module_names.append(short_name(Path(dp), root))
        module_map = {}
        for dp, subdirs, files_in_dir in os.walk(root, followlinks=False):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
            dp = Path(dp)
            if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files_in_dir):
                mod = short_name(dp, root)
                module_map[mod] = {'dir_path': dp}

    for mod, info in module_map.items():
        dir_path = info.get('dir_path', root / mod)
        ctx = dir_path / "context.md"
        if not ctx.exists():
            try:
                missing.append(str(dir_path.relative_to(root)))
            except ValueError:
                missing.append(mod)
        else:
            ctx_mtime = ctx.stat().st_mtime
            try:
                code_mtimes = [
                    f.stat().st_mtime
                    for f in dir_path.iterdir()
                    if f.suffix.lower() in CODE_EXTENSIONS and f.is_file()
                ]
                if code_mtimes and max(code_mtimes) > ctx_mtime:
                    try:
                        stale.append(str(dir_path.relative_to(root)))
                    except ValueError:
                        stale.append(mod)
            except (PermissionError, OSError) as e:
                _warnings.warn(f"Cannot check freshness of {ctx}: {e}")

    if stale:
        results.append(f"Stale modules ({len(stale)}): {', '.join(stale[:10])}")
    if missing:
        results.append(f"Undocumented modules ({len(missing)}): {', '.join(missing[:10])}")

    # --- PHASE 5: SUMMARY ---
    if context_index.exists():
        try:
            idx_text = context_index.read_text(encoding="utf-8", errors="ignore")
            lines = idx_text.splitlines()
            preview = []
            in_mermaid = False
            for line in lines:
                if '```mermaid' in line:
                    in_mermaid = True
                    continue
                if in_mermaid and line.strip() == '```':
                    in_mermaid = False
                    continue
                if not in_mermaid:
                    preview.append(line)
                if len(preview) >= 40:
                    break
            results.append("\n--- Architecture Summary ---")
            results.append("\n".join(preview))
        except Exception:
            pass

    return "\n".join(results)


import json as _json


# ─────────────────────────────────────────────
# AGENT FINDINGS → C4 GENERATORS
# ─────────────────────────────────────────────

_SYSTEM_SYNONYMS: dict = {
    'postgres': 'postgresql',
    'pg': 'postgresql',
    'mongo': 'mongodb',
    'mssql': 'sqlserver',
    'maria': 'mariadb',
    'rabbit': 'rabbitmq',
    'rmq': 'rabbitmq',
    'elastic': 'elasticsearch',
    'es': 'elasticsearch',
    'clickhouse': 'clickhouse',
}


def _normalize_system_name(name: str) -> str:
    """Normalize for dedup: 'PostgreSQL 15' → 'postgresql', 'postgres' → 'postgresql'."""
    # Strip trailing version numbers
    n = re.sub(r'[\s:_-]+\d[\d.x]*$', '', name.strip(), flags=re.IGNORECASE)
    n = n.lower().replace(' ', '').replace('-', '').replace('_', '')
    return _SYSTEM_SYNONYMS.get(n, n)


_CANONICAL_DISPLAY: dict = {
    'postgresql': 'PostgreSQL',
    'mysql': 'MySQL',
    'mariadb': 'MariaDB',
    'mongodb': 'MongoDB',
    'redis': 'Redis',
    'valkey': 'Valkey',
    'memcached': 'Memcached',
    'kafka': 'Kafka',
    'rabbitmq': 'RabbitMQ',
    'elasticsearch': 'Elasticsearch',
    'clickhouse': 'ClickHouse',
    'cassandra': 'Cassandra',
    'dynamodb': 'DynamoDB',
    'sqlserver': 'SQL Server',
    'minio': 'MinIO',
    'mailgun': 'Mailgun',
    'sendgrid': 'SendGrid',
    'stripe': 'Stripe',
    'paypal': 'PayPal',
    'prometheus': 'Prometheus',
    'grafana': 'Grafana',
    'sentry': 'Sentry',
    'datadog': 'Datadog',
    'nats': 'NATS',
}


def _collect_ext_deps(layers: list) -> dict:
    """Collect real external systems from all layers, deduplicated by normalized name."""
    seen: dict = {}  # normalized_key → {display, type, label}
    for layer in layers:
        for dep in layer.get('external_deps', []):
            name = dep.get('name', '').strip()
            if not name or len(name) < 2:
                continue
            if not is_real_external_system(name):
                continue
            key = _normalize_system_name(name)
            # Prefer canonical capitalized display name
            display = _CANONICAL_DISPLAY.get(key, name)
            if key not in seen:
                seen[key] = {
                    'display': display,
                    'type': dep.get('type', 'other'),
                    'label': dep.get('label') or get_rel_label(key),
                }
    return seen


def make_c4_context_from_findings(findings: dict, project_name: str) -> list:
    """Generate C4Context — real external systems only, deduplicated."""
    _c4_alias_registry.clear()
    proj = c4_alias(project_name)
    layers = findings.get('layers', [])
    techs = [l.get('tech', '') for l in layers if l.get('tech')]
    main_tech = techs[0] if techs else 'Application'

    ext_deps = _collect_ext_deps(layers)

    _DB = {'database', 'db'}
    _QUEUE = {'queue', 'broker', 'messaging'}
    _CACHE = {'cache'}

    # Build alias map FIRST so Rel() reuses the same IDs (fixes ID mismatch bug)
    alias_map: dict = {}
    for key in ext_deps:
        alias_map[key] = c4_alias(key)

    lines = [
        '```mermaid', 'C4Context',
        f'  title System Context — {project_name}', '',
        f'  Person(user, "User", "End-user of {project_name}")',
        f'  System({proj}, "{project_name}", "{main_tech} application")',
        '',
    ]
    for key, info in list(ext_deps.items())[:12]:
        alias = alias_map[key]
        display = info['display']
        dep_type = info['type'].lower()
        if dep_type in _DB:
            lines.append(f'  SystemDb_Ext({alias}, "{display}", "Database")')
        elif dep_type in _CACHE:
            lines.append(f'  SystemDb_Ext({alias}, "{display}", "Cache")')
        elif dep_type in _QUEUE:
            lines.append(f'  SystemQueue_Ext({alias}, "{display}", "Message broker")')
        elif dep_type == 'mail':
            lines.append(f'  System_Ext({alias}, "{display}", "Email provider")')
        elif dep_type == 'monitoring':
            lines.append(f'  System_Ext({alias}, "{display}", "Monitoring")')
        elif dep_type == 'payment':
            lines.append(f'  System_Ext({alias}, "{display}", "Payment gateway")')
        else:
            lines.append(f'  System_Ext({alias}, "{display}", "External system")')
    lines.append('')
    lines.append(f'  Rel(user, {proj}, "uses")')
    for key, info in list(ext_deps.items())[:12]:
        lines.append(f'  Rel({proj}, {alias_map[key]}, "{info["label"]}")')
    lines.append('```')
    return lines


def _extract_docker_services(layers: list) -> dict:
    """
    Parse Docker/Infrastructure layer key_components into real services.
    Returns {normalized_name: {display, type, label}} for infra-defined systems.
    Docker/infra layer itself is NOT a C4 container — it's a deployment descriptor.
    """
    services: dict = {}
    _INFRA_KEYS = ('infra', 'docker', 'infrastructure')
    for layer in layers:
        if not any(k in layer.get('name', '').lower() for k in _INFRA_KEYS):
            continue
        for comp in layer.get('key_components', []):
            # key_components can be string or {name, image, purpose}
            if isinstance(comp, dict):
                name = comp.get('name', '') or comp.get('image', '')
                purpose = comp.get('purpose', '')
            else:
                raw = str(comp).strip()
                # Agent ignored instructions and returned "service-name — image, ports, ..."
                # Extract just the short service name before the first separator
                for sep in (' — ', ' – ', ' - ', ': ', ', '):
                    idx = raw.find(sep)
                    if 2 < idx < 40:
                        name = raw[:idx].strip()
                        purpose = raw[idx + len(sep):][:80]
                        break
                else:
                    name = raw
                    purpose = ''
            name = name.strip()
            if not name or len(name) < 2:
                continue
            # Determine type from name/purpose
            lower = (name + ' ' + purpose).lower()
            if any(k in lower for k in ('postgres', 'mysql', 'mariadb', 'mongo', 'sqlite', 'clickhouse', 'mssql')):
                stype = 'database'
            elif any(k in lower for k in ('redis', 'memcach', 'valkey')):
                stype = 'cache'
            elif any(k in lower for k in ('kafka', 'rabbitmq', 'nats', 'activemq', 'sqs', 'pubsub')):
                stype = 'queue'
            elif any(k in lower for k in ('nginx', 'traefik', 'haproxy', 'caddy', 'envoy')):
                stype = 'proxy'
            elif any(k in lower for k in ('prometheus', 'grafana', 'jaeger', 'tempo', 'loki', 'datadog')):
                stype = 'monitoring'
            else:
                stype = 'service'
            key = _normalize_system_name(name)
            if key not in services:
                # Use canonical capitalized display name when available
                display = _CANONICAL_DISPLAY.get(key, name)
                services[key] = {
                    'display': display,
                    'type': stype,
                    'label': get_rel_label(key) if stype != 'proxy' else 'routes traffic',
                }
    return services


def make_c4_container_from_findings(findings: dict, project_name: str) -> list:
    """
    Generate C4Container — only DEPLOYABLE UNITS.
    - Docker/Infrastructure layer is skipped as a container (it's a deployment descriptor)
    - Its key_components (postgres, redis, ...) become real C4 containers inside the boundary
    - External systems from agent findings go outside the boundary
    """
    proj = c4_alias(project_name)
    layers = findings.get('layers', [])
    cross_rels = findings.get('cross_layer_relations', [])

    _SKIP_LAYERS = ('proto', 'protobuf', 'generated', 'gen', 'infra', 'docker', 'infrastructure')
    _INFRA_KEYS = ('infra', 'docker', 'infrastructure')
    _DB_TYPES = {'database', 'db'}
    _QUEUE_TYPES = {'queue', 'broker', 'messaging'}

    # External systems from all agent findings (deduped)
    ext_deps = _collect_ext_deps(layers)

    # Services defined in Docker compose / infra layer
    docker_services = _extract_docker_services(layers)

    lines = ['```mermaid', 'C4Container', f'  title Container diagram — {project_name}', '']
    lines.append(f'  Person(user, "User", "End-user of {project_name}")')
    lines.append('')

    alias_map: dict = {}

    lines.append(f'  System_Boundary({proj}, "{project_name}") {{')

    # 1. Real application layers (skip infra/proto/gen)
    app_layers = [l for l in layers if not any(k in l.get('name', '').lower() for k in _SKIP_LAYERS)]
    for layer in app_layers:
        name = layer.get('name', 'Unknown')
        alias = c4_alias(name)
        alias_map[name] = alias
        tech = layer.get('tech', 'Code')
        desc = c4_label(layer.get('description', name))
        name_lower = name.lower()
        if any(k in name_lower for k in ('db', 'database', 'store')):
            ctype = 'ContainerDb'
        elif any(k in name_lower for k in ('queue', 'worker', 'broker', 'consumer')):
            ctype = 'ContainerQueue'
        else:
            ctype = 'Container'
        lines.append(f'    {ctype}({alias}, "{name}", "{tech}", "{desc}")')

    # 2. Docker compose services that are deployable within the system (proxy, app services)
    #    DBs/caches from docker go OUTSIDE as SystemDb_Ext if not already in ext_deps
    docker_internal: dict = {}  # services that live inside the system boundary
    for key, info in docker_services.items():
        stype = info['type']
        norm = _normalize_system_name(info['display'])
        # If this service is already covered by ext_deps, skip (avoid duplicate)
        if norm in ext_deps:
            continue
        if stype in ('proxy', 'service'):
            docker_internal[key] = info
        else:
            # database/cache/queue from docker → treat as external deployable
            ext_deps[norm] = info

    for key, info in docker_internal.items():
        alias = c4_alias(key)
        alias_map[info['display']] = alias
        lines.append(f'    Container({alias}, "{info["display"]}", "Docker", "{info["label"]}")')

    lines.append('  }')
    lines.append('')

    # External systems outside boundary
    for key, info in list(ext_deps.items())[:10]:
        alias = c4_alias(key)
        alias_map[info['display']] = alias
        stype = info['type'].lower()
        if stype in _DB_TYPES:
            lines.append(f'  SystemDb_Ext({alias}, "{info["display"]}", "Database")')
        elif stype == 'cache':
            lines.append(f'  SystemDb_Ext({alias}, "{info["display"]}", "Cache")')
        elif stype in _QUEUE_TYPES:
            lines.append(f'  SystemQueue_Ext({alias}, "{info["display"]}", "Message broker")')
        elif stype == 'monitoring':
            lines.append(f'  System_Ext({alias}, "{info["display"]}", "Monitoring")')
        elif stype == 'mail':
            lines.append(f'  System_Ext({alias}, "{info["display"]}", "Email provider")')
        elif stype == 'payment':
            lines.append(f'  System_Ext({alias}, "{info["display"]}", "Payment gateway")')
        else:
            lines.append(f'  System_Ext({alias}, "{info["display"]}", "External system")')
    lines.append('')

    # Relationships
    fe = next((l for l in app_layers if any(k in l.get('name', '').lower() for k in ('front', 'vue', 'react', 'ui', 'browser', 'client'))), None)
    be = next((l for l in app_layers if any(k in l.get('name', '').lower() for k in ('backend', 'api', 'server', 'php', 'laravel', 'django', 'rails', 'go'))), None)
    adm = next((l for l in app_layers if any(k in l.get('name', '').lower() for k in ('admin', 'nova', 'panel'))), None)

    if fe and fe['name'] in alias_map:
        lines.append(f'  Rel(user, {alias_map[fe["name"]]}, "accesses via browser")')
    elif be and be['name'] in alias_map:
        lines.append(f'  Rel(user, {alias_map[be["name"]]}, "accesses via HTTP/gRPC")')
    if adm and adm != fe and adm['name'] in alias_map:
        lines.append(f'  Rel(user, {alias_map[adm["name"]]}, "manages via admin panel")')
    if fe and be and fe['name'] in alias_map and be['name'] in alias_map:
        lines.append(f'  Rel({alias_map[fe["name"]]}, {alias_map[be["name"]]}, "API requests [HTTP/JSON]")')
    if adm and be and adm != fe and adm['name'] in alias_map and be['name'] in alias_map:
        lines.append(f'  Rel({alias_map[adm["name"]]}, {alias_map[be["name"]]}, "extends [Nova]")')

    # Explicit cross-layer relations
    added: set = set()
    for rel in cross_rels:
        src = rel.get('from', '')
        dst = rel.get('to', '')
        label = rel.get('label', 'uses')
        src_alias = alias_map.get(src)
        dst_alias = alias_map.get(dst)
        if src_alias and dst_alias:
            key = (src_alias, dst_alias)
            if key not in added:
                added.add(key)
                lines.append(f'  Rel({src_alias}, {dst_alias}, "{label}")')

    # Main layer → external systems
    main_layer = be or (app_layers[0] if app_layers else None)
    if main_layer and main_layer['name'] in alias_map:
        main_alias = alias_map[main_layer['name']]
        for key, info in list(ext_deps.items())[:10]:
            ext_alias = c4_alias(key)
            label = info.get('label') or get_rel_label(info['display'])
            rel_key = (main_alias, ext_alias)
            if rel_key not in added:
                added.add(rel_key)
                lines.append(f'  Rel({main_alias}, {ext_alias}, "{label}")')

    lines.append('```')
    return lines


def make_c4_component_from_findings(layer: dict) -> list:
    """
    Generate C4Component for a single architectural layer from agent findings.
    Uses key_components[] and component_relationships[] returned by agents.
    """
    name = layer.get('name', 'Unknown')
    tech = layer.get('tech', 'Code')
    components = layer.get('key_components', [])
    relationships = layer.get('component_relationships', [])

    if not components:
        return []

    mod_alias = c4_alias(name)
    comp_aliases: dict = {}  # component name → alias

    lines = [
        '```mermaid', 'C4Component',
        f'  title Component diagram — {name}', '',
        f'  Container_Boundary({mod_alias}, "{name}") {{',
    ]

    for comp in components[:15]:
        if isinstance(comp, dict):
            cname = comp.get('name', '')
            cpurpose = c4_label(comp.get('purpose', cname))
        else:
            cname = str(comp)
            cpurpose = c4_label(cname)
        if not cname:
            continue
        # If agent ignored instructions and returned a long description as name,
        # extract the first word/token as the short name for the alias
        display_name = cname
        if len(cname) > 30:
            # take up to first separator
            short = re.split(r'[\s—\-,:(]', cname)[0].strip()
            display_name = short if len(short) >= 3 else cname[:25]
        alias = c4_alias(display_name)
        comp_aliases[cname] = alias
        comp_aliases[display_name] = alias
        # Normalize variations: "AuthHandler" → also reachable as "auth_handler", "auth"
        comp_aliases[cname.lower()] = alias
        comp_aliases[display_name.lower()] = alias
        # Use display_name as the label so it's short and readable
        label_name = display_name if len(display_name) <= 30 else display_name[:30]
        lines.append(f'    Component({alias}, "{label_name}", "{tech}", "{cpurpose}")')

    lines.append('  }')
    lines.append('')

    # Build relationships from agent-provided data
    added: set = set()
    for rel in relationships:
        src_name = rel.get('from', '')
        dst_name = rel.get('to', '')
        label = rel.get('label', 'calls')
        src_alias = comp_aliases.get(src_name) or comp_aliases.get(src_name.lower())
        dst_alias = comp_aliases.get(dst_name) or comp_aliases.get(dst_name.lower())
        if src_alias and dst_alias and src_alias != dst_alias:
            key = (src_alias, dst_alias)
            if key not in added:
                added.add(key)
                lines.append(f'  Rel({src_alias}, {dst_alias}, "{label}")')

    if not added:
        # Fallback: infer relationships from naming patterns (handler→service, service→repo)
        _LAYERS = [
            (['handler', 'controller', 'router', 'endpoint'], ['service', 'usecase', 'manager'], 'calls'),
            (['service', 'usecase', 'manager'], ['repo', 'repository', 'store', 'dao'], 'queries'),
            (['service', 'usecase', 'manager'], ['client', 'provider', 'gateway', 'sender'], 'uses'),
        ]
        for comp in components[:15]:
            cname = str(comp.get('name', comp) if isinstance(comp, dict) else comp).lower()
            src_alias = comp_aliases.get(cname)
            if not src_alias:
                continue
            for src_patterns, dst_patterns, label in _LAYERS:
                if any(p in cname for p in src_patterns):
                    for other in components[:15]:
                        oname = str(other.get('name', other) if isinstance(other, dict) else other).lower()
                        dst_alias = comp_aliases.get(oname)
                        if dst_alias and dst_alias != src_alias and any(p in oname for p in dst_patterns):
                            key = (src_alias, dst_alias)
                            if key not in added:
                                added.add(key)
                                lines.append(f'  Rel({src_alias}, {dst_alias}, "{label}")')

    lines.append('```')
    return lines


def make_sequence_from_findings(findings: dict, project_name: str) -> list:
    """Generate Mermaid sequence diagrams for key flows from agent findings."""
    flows = findings.get('key_flows', [])
    if not flows:
        return []

    all_lines = []
    for flow in flows[:3]:  # max 3 sequence diagrams
        title = flow.get('title', 'Main Flow')
        steps = flow.get('steps', [])
        if not steps:
            continue
        lines = [
            '```mermaid', 'sequenceDiagram',
            f'  title {title}',
        ]
        # Collect actors
        actors: list = []
        for step in steps:
            src = step.get('from', '')
            dst = step.get('to', '')
            for a in [src, dst]:
                if a and a not in actors:
                    actors.append(a)
        for actor in actors:
            lines.append(f'  participant {c4_alias(actor)} as {actor}')
        lines.append('')
        for step in steps:
            src = c4_alias(step.get('from', 'Client'))
            dst = c4_alias(step.get('to', 'Server'))
            msg = step.get('message', 'calls')
            arrow = '->>' if step.get('async') else '->>'
            lines.append(f'  {src}->>{dst}: {msg}')
            if step.get('response'):
                lines.append(f'  {dst}-->>{src}: {step["response"]}')
        lines.append('```')
        all_lines += ['', f'### Sequence: {title}', ''] + lines

    return all_lines


def generate_c4_context(findings: dict, project_name: str, c4_alias_fn=None) -> str:
    """L1: C4 System Context — system as black box with user personas and real external systems.

    Node limit: max 10 total (2 personas + system + up to 7 external).
    External: only database/queue/cache types, top 5 preferred systems.
    Rel() labels: max 25 characters.
    """
    _alias = c4_alias_fn if c4_alias_fn is not None else c4_alias

    lines = ["```mermaid", "C4Context", f'  title System Context — {project_name}', ""]

    # 2 user personas
    lines.append(f'  Person(player, "Player", "Registers, deposits, plays")')
    lines.append(f'  Person(admin, "Admin", "Manages via admin panel")')
    lines.append("")

    # System as a black box
    description = c4_label(findings.get("description", f"Platform: {project_name}"))[:60]
    lines.append(f'  System(main_system, "{project_name}", "{description}")')
    lines.append("")

    # Collect external deps — ONLY database, queue, cache types
    _ALLOWED_TYPES = {"database", "db", "cache", "queue", "message_queue", "broker", "messaging"}
    # Preferred order for top-5 selection
    _PRIORITY_KEYS = ["mysql", "postgres", "postgresql", "redis", "rabbitmq", "clickhouse", "elasticsearch", "mongo", "mongodb"]

    ext_seen: set = set()
    ext_candidates = []  # (key, alias, display, dep_type)

    for layer in findings.get("layers", []):
        for dep in layer.get("external_deps", []):
            name = dep.get("name", "")
            if not name or not is_real_external_system(name):
                continue
            dep_type = dep.get("type", "other")
            if dep_type not in _ALLOWED_TYPES:
                continue  # skip http, auth, mail, other, monitoring, payment
            key = _normalize_system_name(name)
            if key not in ext_seen:
                ext_seen.add(key)
                display = _CANONICAL_DISPLAY.get(key, name)
                raw_alias = re.sub(r'[^a-zA-Z0-9]', '_', key).strip('_')
                raw_alias = re.sub(r'_+', '_', raw_alias) or 'ext_sys'
                ext_candidates.append((key, raw_alias, display, dep_type))

    # Pick top 5: preferred systems first, then others
    def _ext_priority(item):
        key = item[0]
        try:
            return _PRIORITY_KEYS.index(key)
        except ValueError:
            return len(_PRIORITY_KEYS)

    ext_candidates.sort(key=_ext_priority)
    ext_list = ext_candidates[:5]  # max 5 external nodes

    # Short Rel labels by type (max 20 chars)
    _TYPE_LABEL = {
        "database": "reads/writes",
        "db": "reads/writes",
        "cache": "cache/sessions",
        "queue": "async jobs",
        "message_queue": "async jobs",
        "broker": "async jobs",
        "messaging": "async jobs",
    }

    # Emit external system nodes
    for key, alias, display, dep_type in ext_list:
        if dep_type in ("database", "db"):
            lines.append(f'  SystemDb_Ext({alias}, "{display}", "Database")')
        elif dep_type == "cache":
            lines.append(f'  SystemDb_Ext({alias}, "{display}", "Cache")')
        elif dep_type in ("queue", "message_queue", "broker", "messaging"):
            lines.append(f'  SystemQueue_Ext({alias}, "{display}", "Message broker")')

    lines.append("")
    lines.append("  %% Relations")
    lines.append(f'  Rel(player, main_system, "plays & deposits", "HTTPS")')
    lines.append(f'  Rel(admin, main_system, "manages platform", "HTTPS")')
    for key, alias, display, dep_type in ext_list:
        label = _TYPE_LABEL.get(dep_type, "uses")[:25]
        lines.append(f'  Rel(main_system, {alias}, "{label}")')

    lines.append("```")
    return "\n".join(lines)


def generate_c4_container(findings: dict, project_name: str, c4_alias_fn=None) -> str:
    """L2: C4 Container — runtime containers inside system boundary (NOT architectural layers).

    For lafa.main-style projects these are: nginx, octane (Laravel app server),
    nova_admin (admin panel), queue_workers, mysql, redis, rabbitmq, clickhouse, etc.
    """
    _alias = c4_alias_fn if c4_alias_fn is not None else c4_alias

    layers = findings.get("layers", [])
    cross_rels = findings.get("cross_layer_relations", [])

    # Build external deps (databases, caches, queues outside the boundary)
    ext_seen: set = set()
    ext_list = []  # (alias, display, dep_type, label)
    ext_db_aliases: list = []   # aliases of database-type external deps
    ext_queue_aliases: list = []  # aliases of queue-type external deps

    for layer in layers:
        for dep in layer.get("external_deps", []):
            name = dep.get("name", "")
            if not name or not is_real_external_system(name):
                continue
            key = _normalize_system_name(name)
            if key not in ext_seen:
                ext_seen.add(key)
                dep_type = dep.get("type", "other")
                label = get_rel_label(key)  # always use short canonical label
                display = _CANONICAL_DISPLAY.get(key, name)
                raw_alias = re.sub(r'[^a-zA-Z0-9]', '_', key).strip('_')
                raw_alias = re.sub(r'_+', '_', raw_alias) or 'ext_sys'
                ext_list.append((raw_alias, display, dep_type, label))
                if dep_type in ("database", "db", "cache"):
                    ext_db_aliases.append(raw_alias)
                elif dep_type in ("queue", "message_queue", "broker", "messaging"):
                    ext_queue_aliases.append(raw_alias)

    lines = ["```mermaid", "C4Container", f'  title Container diagram — {project_name}', ""]
    lines.append(f'  Person(player, "Игрок / Player", "End user")')
    lines.append(f'  Person(admin, "Администратор / Admin", "Admin panel user")')
    lines.append("")

    # External systems outside the boundary (real infra: DBs, caches, queues)
    for alias, display, dep_type, label in ext_list:
        if dep_type in ("database", "db"):
            lines.append(f'  SystemDb_Ext({alias}, "{display}", "Database")')
        elif dep_type == "cache":
            lines.append(f'  SystemDb_Ext({alias}, "{display}", "Cache")')
        elif dep_type in ("queue", "message_queue", "broker", "messaging"):
            lines.append(f'  SystemQueue_Ext({alias}, "{display}", "Message broker")')
        elif dep_type == "mail":
            lines.append(f'  System_Ext({alias}, "{display}", "Email provider")')
        elif dep_type == "payment":
            lines.append(f'  System_Ext({alias}, "{display}", "Payment gateway")')
        else:
            lines.append(f'  System_Ext({alias}, "{display}", "External system")')
    lines.append("")

    # ── Determine runtime containers to place inside System_Boundary ──
    # Look for an infrastructure layer with explicit key_components first.
    infra_layer = None
    for layer in layers:
        lname_low = layer.get("name", "").lower()
        if any(k in lname_low for k in ('infra', 'infrastructure', 'docker', 'deploy')):
            infra_layer = layer
            break

    # Detect special layers: admin panel, modules/business logic
    has_nova_layer = any(
        any(k in l.get('name', '').lower() for k in ('admin', 'nova', 'panel'))
        for l in layers
    )
    has_modules_layer = any(
        any(k in l.get('name', '').lower() for k in ('module', 'modules', 'business'))
        for l in layers
    )

    # Core runtime containers — always present for Laravel/PHP projects
    # If infra layer has explicit components use them, otherwise use defaults
    if infra_layer and infra_layer.get("key_components"):
        infra_components = infra_layer["key_components"]
    else:
        # Default runtime containers for a typical lafa.main-style project
        infra_components = [
            {"name": "nginx", "purpose": "Reverse proxy / TLS termination / static files"},
            {"name": "octane", "purpose": "Laravel Octane PHP app server — business logic, REST API"},
            {"name": "queue_workers", "purpose": "Background job processors consuming RabbitMQ queues"},
        ]
        if has_nova_layer:
            infra_components.append({"name": "nova_admin", "purpose": "Laravel Nova admin panel — backoffice management"})
        if has_modules_layer:
            infra_components.append({"name": "modules", "purpose": "Domain business modules — Geo, Loyalty, Bonus, Mirrors"})

    # Build alias map for containers
    container_aliases: dict = {}  # component name -> alias

    lines.append(f'  System_Boundary(sys, "{project_name}") {{')
    for comp in infra_components:
        if isinstance(comp, dict):
            cname = comp.get("name", "container")
            cpurpose = c4_label(comp.get("purpose", cname))
        else:
            cname = str(comp)
            cpurpose = c4_label(cname)
        cname_low = cname.lower()
        raw_alias = re.sub(r'[^a-zA-Z0-9]', '_', cname).strip('_').lower()
        raw_alias = re.sub(r'_+', '_', raw_alias) or 'container'
        container_aliases[cname] = raw_alias
        container_aliases[cname_low] = raw_alias

        if any(k in cname_low for k in ('db', 'database', 'mysql', 'postgres', 'mongo', 'clickhouse')):
            ctype = 'ContainerDb'
        elif any(k in cname_low for k in ('queue', 'worker', 'rabbit', 'kafka', 'broker')):
            ctype = 'ContainerQueue'
        elif any(k in cname_low for k in ('redis', 'cache', 'memcache')):
            ctype = 'ContainerDb'
        else:
            ctype = 'Container'

        # Pick a human-readable tech label
        if 'nginx' in cname_low:
            tech = 'nginx'
        elif 'octane' in cname_low or 'laravel' in cname_low:
            tech = 'PHP/Laravel Octane'
        elif 'nova' in cname_low:
            tech = 'Laravel Nova'
        elif 'queue' in cname_low or 'worker' in cname_low:
            tech = 'PHP/RabbitMQ'
        elif 'module' in cname_low:
            tech = 'PHP/Laravel'
        else:
            tech = 'Code'

        lines.append(f'    {ctype}({raw_alias}, "{cname}", "{tech}", "{cpurpose}")')
    lines.append("  }")
    lines.append("")

    # ── Relationships ──
    nginx_alias = container_aliases.get('nginx')
    octane_alias = container_aliases.get('octane')
    nova_alias = container_aliases.get('nova_admin')
    workers_alias = container_aliases.get('queue_workers')
    modules_alias = container_aliases.get('modules')

    added: set = set()

    def _add_rel(src, dst, label, protocol=""):
        if src and dst and (src, dst) not in added:
            added.add((src, dst))
            if protocol:
                lines.append(f'  Rel({src}, {dst}, "{label}", "{protocol}")')
            else:
                lines.append(f'  Rel({src}, {dst}, "{label}")')

    # Player and Admin → nginx (entry point)
    if nginx_alias:
        _add_rel("player", nginx_alias, "HTTPS")
        _add_rel("admin", nginx_alias, "HTTPS")
    elif octane_alias:
        _add_rel("player", octane_alias, "HTTPS")
    if nova_alias and not nginx_alias:
        _add_rel("admin", nova_alias, "HTTPS")

    # nginx → backend containers
    if nginx_alias and octane_alias:
        _add_rel(nginx_alias, octane_alias, "proxy", "HTTP")
    if nginx_alias and nova_alias:
        _add_rel(nginx_alias, nova_alias, "proxy admin", "HTTP")

    # octane → modules (if present)
    if octane_alias and modules_alias:
        _add_rel(octane_alias, modules_alias, "event dispatch")

    # octane → queue workers
    if octane_alias and workers_alias:
        _add_rel(octane_alias, workers_alias, "dispatch jobs", "AMQP")

    # octane → external databases and caches
    if octane_alias:
        for db_alias in ext_db_aliases:
            _add_rel(octane_alias, db_alias, get_rel_label(db_alias))

    # octane → queues
    if octane_alias:
        for q_alias in ext_queue_aliases:
            _add_rel(octane_alias, q_alias, "async jobs", "AMQP")

    # queue workers → databases (read/write for job processing)
    if workers_alias:
        for db_alias in ext_db_aliases:
            _add_rel(workers_alias, db_alias, "reads/writes")

    # Apply any explicit cross-layer relations from findings
    for rel in cross_rels:
        src_name = rel.get('from', '')
        dst_name = rel.get('to', '')
        raw_label = rel.get('label', rel.get('description', 'calls'))
        # Keep labels short — strip protocol details like [JSON/HTTPS]
        label = re.sub(r'\[.*?\]', '', raw_label).strip()
        label = re.split(r'\s*[+&]\s*', label)[0].strip()  # take first part if joined
        label = label[:25]
        src_a = container_aliases.get(src_name) or container_aliases.get(src_name.lower())
        dst_a = container_aliases.get(dst_name) or container_aliases.get(dst_name.lower())
        if src_a and dst_a:
            _add_rel(src_a, dst_a, label)

    lines.append("```")
    return "\n".join(lines)


def generate_c4_component(findings: dict, project_name: str, c4_alias_fn=None) -> str:
    """L3: C4 Component — zoom into the octane/backend container showing controllers, services, repos.

    Traces the real user→controller→service→repository→DB data flow.
    """
    _alias = c4_alias_fn if c4_alias_fn is not None else c4_alias

    # Find the main backend layer — prefer one with component_relationships
    layers = findings.get("layers", [])
    backend_layer = None
    # First pass: layer with component_relationships (richest data)
    for layer in layers:
        if layer.get("component_relationships"):
            backend_layer = layer
            break
    # Second pass: layer named 'backend' / 'api' / etc.
    if not backend_layer:
        for layer in layers:
            lname_low = layer.get("name", "").lower()
            if any(k in lname_low for k in ('backend', 'api', 'server', 'php', 'laravel', 'octane')):
                backend_layer = layer
                break
    # Fallback: first layer
    if not backend_layer:
        backend_layer = layers[0] if layers else {}

    if not backend_layer:
        return ""

    lname = backend_layer.get("name", "Backend")
    tech = backend_layer.get("tech", "PHP/Laravel")
    components = backend_layer.get("key_components", [])
    rels = backend_layer.get("component_relationships", [])
    ext_deps = backend_layer.get("external_deps", [])

    if not components:
        return ""

    lines = ["```mermaid", "C4Component", f'  title Component diagram — {lname} (octane)', ""]

    # Player persona — traces user entry into the system
    lines.append(f'  Person(player, "Игрок / Player", "End user")')
    lines.append("")

    # External system nodes (databases only — what repos talk to)
    ext_aliases: dict = {}  # original name -> alias
    db_aliases: list = []   # database/cache aliases
    for dep in ext_deps:
        name = dep.get("name", "")
        if not name or not is_real_external_system(name):
            continue
        dep_type = dep.get("type", "other")
        key = _normalize_system_name(name)
        label = get_rel_label(key)  # always use short canonical label
        display = _CANONICAL_DISPLAY.get(key, name)
        raw_alias = re.sub(r'[^a-zA-Z0-9]', '_', key).strip('_')
        raw_alias = re.sub(r'_+', '_', raw_alias) or 'ext_sys'
        ext_aliases[name] = raw_alias
        ext_aliases[name.lower()] = raw_alias
        if dep_type in ("database", "db"):
            lines.append(f'  SystemDb_Ext({raw_alias}, "{display}", "Database")')
            db_aliases.append(raw_alias)
        elif dep_type == "cache":
            lines.append(f'  SystemDb_Ext({raw_alias}, "{display}", "Cache")')
            db_aliases.append(raw_alias)
        elif dep_type in ("queue", "message_queue", "broker", "messaging"):
            lines.append(f'  SystemQueue_Ext({raw_alias}, "{display}", "Message broker")')
        else:
            lines.append(f'  System_Ext({raw_alias}, "{display}", "External system")')

    lines.append("")

    # Container_Boundary representing the octane process (API server)
    boundary_alias = re.sub(r'[^a-zA-Z0-9]', '_', lname).strip('_').lower()
    boundary_alias = re.sub(r'_+', '_', boundary_alias) or 'octane'
    lines.append(f'  Container_Boundary({boundary_alias}, "Laravel Octane / API Server") {{')

    # Build component aliases — deduplicated, stable
    comp_aliases: dict = {}   # original name (and lower) -> alias
    alias_counts: dict = {}
    first_controller_alias: str = ""
    last_repo_alias: str = ""

    for comp in components[:20]:
        if isinstance(comp, dict):
            cname = comp.get('name', str(comp))
            cpurpose = c4_label(comp.get('purpose', cname))
        else:
            cname = str(comp)
            cpurpose = c4_label(cname)
        if not cname:
            continue
        display_name = cname
        if len(cname) > 30:
            short = re.split(r'[\s\u2014\-,:(]', cname)[0].strip()
            display_name = short if len(short) >= 3 else cname[:25]
        base_alias = re.sub(r'[^a-zA-Z0-9]', '_', display_name).strip('_').lower()
        base_alias = re.sub(r'_+', '_', base_alias) or 'comp'
        # Deduplicate aliases
        if base_alias in alias_counts:
            alias_counts[base_alias] += 1
            comp_alias = f"{base_alias}_{alias_counts[base_alias]}"
        else:
            alias_counts[base_alias] = 1
            comp_alias = base_alias
        comp_aliases[cname] = comp_alias
        comp_aliases[cname.lower()] = comp_alias
        comp_aliases[display_name] = comp_alias
        comp_aliases[display_name.lower()] = comp_alias

        # Track first controller and last repo for entry/exit relationships
        cname_low = cname.lower()
        if not first_controller_alias and any(k in cname_low for k in ('controller', 'handler', 'endpoint')):
            first_controller_alias = comp_alias
        if any(k in cname_low for k in ('repository', 'repo', 'store', 'dao')):
            last_repo_alias = comp_alias

        label_name = display_name if len(display_name) <= 30 else display_name[:30]
        lines.append(f'    Component({comp_alias}, "{label_name}", "Laravel Component", "{cpurpose}")')

    lines.append("  }")
    lines.append("")

    # ── Relationships ──
    added: set = set()

    def _add_rel(src, dst, label):
        if src and dst and src != dst and (src, dst) not in added:
            added.add((src, dst))
            lines.append(f'  Rel({src}, {dst}, "{label}")')

    # Player → first controller (entry point of the flow)
    if first_controller_alias:
        _add_rel("player", first_controller_alias, "POST /api/v1/... action", )
    elif comp_aliases:
        first_comp = next(iter(comp_aliases.values()))
        _add_rel("player", first_comp, "API request", )

    # Explicit component relationships from agent findings
    for rel in rels:
        src_name = rel.get('from', '')
        dst_name = rel.get('to', '')
        label = c4_label(rel.get('label', 'calls'))
        src_a = comp_aliases.get(src_name) or comp_aliases.get(src_name.lower())
        dst_a = comp_aliases.get(dst_name) or comp_aliases.get(dst_name.lower())
        if src_a and dst_a:
            _add_rel(src_a, dst_a, label)

    # Fallback: infer from naming patterns when no explicit rels provided
    if len(added) <= 1:
        _INFERENCE = [
            (['handler', 'controller', 'router', 'endpoint'], ['service', 'usecase', 'manager'], 'calls service'),
            (['service', 'usecase', 'manager'], ['repo', 'repository', 'store', 'dao'], 'queries data'),
            (['service', 'usecase', 'manager'], ['client', 'provider', 'gateway', 'sender'], 'uses'),
        ]
        for comp in components[:20]:
            cname_low = str(comp.get('name', comp) if isinstance(comp, dict) else comp).lower()
            src_a = comp_aliases.get(cname_low)
            if not src_a:
                continue
            for src_patterns, dst_patterns, lbl in _INFERENCE:
                if any(p in cname_low for p in src_patterns):
                    for other in components[:20]:
                        oname_low = str(other.get('name', other) if isinstance(other, dict) else other).lower()
                        dst_a = comp_aliases.get(oname_low)
                        if dst_a and dst_a != src_a and any(p in oname_low for p in dst_patterns):
                            _add_rel(src_a, dst_a, lbl)

    # Last repository → database external systems
    if last_repo_alias and db_aliases:
        _add_rel(last_repo_alias, db_aliases[0], "SQL queries")
    elif db_aliases and comp_aliases:
        # Connect any service/repo component to db
        for cname_low, comp_a in comp_aliases.items():
            if any(k in cname_low for k in ('repo', 'repository', 'service')):
                _add_rel(comp_a, db_aliases[0], "SQL queries")
                break

    lines.append("```")
    return "\n".join(lines)


def generate_c4_dynamic(findings: dict, project_name: str, c4_alias_fn=None) -> str:
    """L4: C4 Dynamic — numbered sequence tracing one key user flow through all runtime containers/components."""
    _alias_fn = c4_alias_fn if c4_alias_fn is not None else c4_alias

    # Collect key_flows: prefer top-level, then scan layers
    flows = findings.get('key_flows', [])
    if not flows:
        for layer in findings.get("layers", []):
            layer_flows = layer.get('key_flows', [])
            if layer_flows:
                flows = layer_flows
                break
    if not flows:
        return ""

    # Use the first key flow (most important user journey)
    flow = flows[0]
    title = flow.get("title", "Key Flow")
    steps = flow.get("steps", [])
    if not steps:
        return ""

    lines = ["```mermaid", "C4Dynamic", f'  title Dynamic — {title}', ""]

    # Collect unique participants in order of first appearance
    # Determine node type per participant: Person vs Container
    _PERSON_KEYWORDS = ('user', 'player', 'admin', 'client', 'игрок', 'пользователь')
    _DB_KEYWORDS = ('mysql', 'postgres', 'redis', 'clickhouse', 'mongo', 'database', 'db', 'cache')
    _QUEUE_KEYWORDS = ('rabbit', 'kafka', 'queue', 'worker', 'broker')

    participants: dict = {}        # display name -> alias
    participant_types: dict = {}   # alias -> 'Person' | 'Container' | 'ContainerDb' | 'ContainerQueue'
    aliases_set: set = set()

    for step in steps:
        for role in ("from", "to"):
            name = step.get(role, "")
            if not name or name in participants:
                continue
            raw_alias = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_').lower()
            raw_alias = re.sub(r'_+', '_', raw_alias) or 'actor'
            # Deduplicate alias
            base = raw_alias
            counter = 2
            while raw_alias in aliases_set:
                raw_alias = f"{base}_{counter}"
                counter += 1
            aliases_set.add(raw_alias)
            participants[name] = raw_alias

            # Classify node type
            name_low = name.lower()
            if any(k in name_low for k in _PERSON_KEYWORDS):
                ptype = 'Person'
            elif any(k in name_low for k in _DB_KEYWORDS):
                ptype = 'ContainerDb'
            elif any(k in name_low for k in _QUEUE_KEYWORDS):
                ptype = 'ContainerQueue'
            else:
                ptype = 'Container'
            participant_types[raw_alias] = ptype

    # Emit node declarations
    for name, palias in participants.items():
        ptype = participant_types.get(palias, 'Container')
        if ptype == 'Person':
            lines.append(f'  Person({palias}, "{name}", "")')
        elif ptype == 'ContainerDb':
            lines.append(f'  ContainerDb({palias}, "{name}", "", "")')
        elif ptype == 'ContainerQueue':
            lines.append(f'  ContainerQueue({palias}, "{name}", "", "")')
        else:
            lines.append(f'  Container({palias}, "{name}", "", "")')

    lines.append("")

    # Numbered RelIndex for each step
    idx = 1
    for step in steps:
        from_alias = participants.get(step.get("from", ""), "unknown")
        to_alias = participants.get(step.get("to", ""), "unknown")
        msg = c4_label(step.get("message", "calls"))
        lines.append(f'  RelIndex({idx}, {from_alias}, {to_alias}, "{msg}")')
        idx += 1
        # Optional synchronous response gets its own numbered step
        resp = step.get("response", "")
        if resp:
            lines.append(f'  RelIndex({idx}, {to_alias}, {from_alias}, "{c4_label(resp)}")')
            idx += 1

    lines.append("```")
    return "\n".join(lines)


def _search(root: Path, names: list, max_depth: int = 2) -> Path | None:
    """Find first matching file/dir within max_depth levels."""
    for name in names:
        if (root / name).exists():
            return root / name
    if max_depth > 1:
        for child in root.iterdir():
            if child.is_dir() and child.name not in SKIP_DIRS:
                for name in names:
                    if (child / name).exists():
                        return child / name
    return None


def _subdirs(path: Path) -> list:
    try:
        return [d.name for d in path.iterdir() if d.is_dir() and d.name not in SKIP_DIRS]
    except Exception:
        return []


def detect_project_layers(root: Path) -> list:
    """Detect architectural layers for any tech stack."""
    layers = []
    seen_paths: set = set()

    def add_layer(name, path, tech, prompt):
        p = str(path)
        if p not in seen_paths:
            seen_paths.add(p)
            layers.append({'name': name, 'path': p, 'tech': tech, 'agent_prompt': prompt})

    # ── Go ──
    go_mod = _search(root, ['go.mod'])
    if go_mod:
        go_root = go_mod.parent
        try:
            mod_line = next((l for l in go_mod.read_text(encoding='utf-8').splitlines() if l.startswith('module ')), '')
            mod_name = mod_line.replace('module ', '').strip()
        except Exception:
            mod_name = go_root.name
        subs = _subdirs(go_root)
        has_grpc = any(d in subs for d in ('proto', 'gen', 'grpc', 'pb'))
        tech = 'Go/gRPC' if has_grpc else 'Go'
        add_layer('Backend', go_root, tech,
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Backend layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the Backend layer at {go_root}.\n"
            f"Module: {mod_name} | Dirs: {', '.join(subs[:15])}\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (Go version, key frameworks from go.mod)\n"
            f"2. All major components (handlers, services, repositories, middleware)\n"
            f"3. Key external dependencies (database driver, cache, queue — read actual imports)\n"
            f"4. How this layer exposes its API (HTTP routes or gRPC service definitions)\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read go.mod — list ALL dependencies with their purpose\n"
            f"2. Explore cmd/ — what services/binaries are defined\n"
            f"3. Explore internal/, pkg/ — map packages: handlers, services, repositories, models\n"
            f"4. Find API layer: read actual route files for HTTP or .proto files for gRPC\n"
            f"5. Find DB driver: read actual imports in db/ or repository files\n"
            f"6. Find cache, queue, and external HTTP clients from config/env files\n"
            f"7. Map cross-package data flow (which package calls which)\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Backend",\n'
            f'  "tech": "Go version, Key Libraries",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "component_relationships": [{{"from": "HandlerA", "to": "ServiceB", "label": "calls"}}],\n'
            f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
            f'  "api_endpoints": [],\n'
            f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Handler", "message": "request", "response": "result"}}]}}]\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact class/file names found in the code (max 25 chars each)\n"
            f"- tech: include version numbers visible in go.mod\n"
            f"- description: one sentence maximum\n"
            f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual imports"
        )
        for proto_dir in ('proto', 'gen', 'pb'):
            if (go_root / proto_dir).exists():
                add_layer('gRPC / Protobuf', go_root / proto_dir, 'gRPC/Protobuf',
                    f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the gRPC/Protobuf layer of this project.\n\n"
                    f"WHY this matters: Your findings will be used to generate C4 architecture diagrams showing service contracts and inter-service communication.\n\n"
                    f"Your task: Analyze the gRPC/Protobuf layer at {go_root / proto_dir}.\n\n"
                    f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
                    f"1. All .proto service definitions with their RPC methods\n"
                    f"2. Request/response message types for each RPC\n"
                    f"3. Which generated clients/servers exist in the codebase\n"
                    f"4. Cross-service dependencies\n\n"
                    f"ANALYSIS STEPS:\n"
                    f"1. List all .proto files — services and messages defined\n"
                    f"2. For each service: list RPC methods with request/response types\n"
                    f"3. Find generated code — what clients/servers are generated\n"
                    f"4. Identify cross-service dependencies\n\n"
                    f"OUTPUT FORMAT:\n"
                    f"<findings>\n"
                    f"{{{{\n"
                    f'  "layer": "gRPC / Protobuf",\n'
                    f'  "tech": "gRPC/Protobuf, version info",\n'
                    f'  "description": "One sentence: what contracts this layer defines",\n'
                    f'  "key_components": ["ServiceName1", "ServiceName2"],\n'
                    f'  "external_deps": []\n'
                    f"}}}}\n"
                    f"</findings>\n\n"
                    f"CONSTRAINTS:\n"
                    f"- key_components: use exact service names from .proto files\n"
                    f"- description: one sentence maximum"
                )
                break

    # ── PHP / Laravel / Symfony ──
    composer = _search(root, ['composer.json'])
    if composer:
        php_root = composer.parent
        try:
            data = _json.loads(composer.read_text(encoding='utf-8'))
            reqs = data.get('require', {})
            framework = 'Laravel' if any('laravel/framework' in k for k in reqs) \
                else 'Symfony' if any('symfony' in k for k in reqs) \
                else 'PHP'
        except Exception:
            framework = 'PHP'
        subs = _subdirs(php_root)
        add_layer('Backend', php_root, f'PHP/{framework}',
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Backend layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the {framework} Backend layer at {php_root}.\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (framework version, PHP version from composer.json)\n"
            f"2. All major components (Controllers, Services, Models, Repositories, Providers)\n"
            f"3. Key external dependencies (database driver, cache, queue — read actual config)\n"
            f"4. How this layer exposes its API (routes grouped by domain)\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read composer.json — list ALL dependencies with their purpose\n"
            f"2. Explore app/ — Controllers, Services, Models, Repositories, Providers\n"
            f"3. Read routes/ — list API/web endpoints grouped by domain\n"
            f"4. Find DB type: read .env or config/database.php — check actual driver (mysql/pgsql/sqlite)\n"
            f"5. Map layer interactions (which controller calls which service, which service calls which repo)\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Backend",\n'
            f'  "tech": "PHP/Framework version, Key Libraries",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "component_relationships": [{{"from": "ControllerA", "to": "ServiceB", "label": "calls"}}],\n'
            f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
            f'  "api_endpoints": [],\n'
            f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Controller", "message": "request", "response": "result"}}]}}]\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact class names found in the code (max 25 chars each)\n"
            f"- tech: include version numbers visible in composer.json\n"
            f"- description: one sentence maximum\n"
            f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual config files"
        )
        # Laravel Nova
        nova_path = php_root / 'nova-components'
        if nova_path.exists():
            components = [d.name for d in nova_path.iterdir() if d.is_dir()]
            add_layer('Admin Panel', nova_path, 'Laravel Nova/Vue.js',
                f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Admin Panel layer of this project.\n\n"
                f"WHY this matters: Your findings will be used to generate C4 architecture diagrams showing the admin interface and its backend model connections.\n\n"
                f"Your task: Analyze the Admin Panel layer at {nova_path}.\n"
                f"Components: {', '.join(components[:15])}\n\n"
                f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
                f"1. The technology stack (Nova version, Vue.js version)\n"
                f"2. All Nova components with their purpose\n"
                f"3. Cross-component dependencies (imports)\n"
                f"4. Which backend models each component interacts with\n\n"
                f"ANALYSIS STEPS:\n"
                f"1. For each component — read resources/js/ files, understand its purpose\n"
                f"2. Find props, events, key functions\n"
                f"3. Find cross-component dependencies (imports)\n"
                f"4. Identify what backend models each component interacts with\n\n"
                f"OUTPUT FORMAT:\n"
                f"<findings>\n"
                f"{{{{\n"
                f'  "layer": "Admin Panel",\n'
                f'  "tech": "Laravel Nova/Vue.js, version info",\n'
                f'  "description": "One sentence: what this admin panel manages",\n'
                f'  "key_components": [{{"name": "ComponentName", "purpose": "what it does", "dependencies": []}}],\n'
                f'  "external_deps": []\n'
                f"}}}}\n"
                f"</findings>\n\n"
                f"CONSTRAINTS:\n"
                f"- key_components: use exact component directory names\n"
                f"- description: one sentence maximum"
            )
        # Laravel Modules
        modules_path = php_root / 'Modules'
        if modules_path.exists():
            mod_names = [d.name for d in modules_path.iterdir() if d.is_dir()]
            add_layer('Modules', modules_path, 'PHP/Laravel Modules',
                f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Modules (bounded contexts) layer of this project.\n\n"
                f"WHY this matters: Your findings will be used to generate C4 architecture diagrams showing bounded contexts and their inter-module communication.\n\n"
                f"Your task: Analyze the Laravel modules at {modules_path}.\n"
                f"Modules: {', '.join(mod_names)}\n\n"
                f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
                f"1. The domain each module serves (e.g., Auth, Billing, Notifications)\n"
                f"2. Cross-module dependencies (which module imports from which)\n"
                f"3. Events/jobs produced or consumed by each module\n"
                f"4. The public API each module exposes to others\n\n"
                f"ANALYSIS STEPS:\n"
                f"1. For each module — examine Routes/, Controllers/, Models/ — what domain it serves\n"
                f"2. Find cross-module dependencies (use statements, service provider registrations)\n"
                f"3. Identify events/jobs produced or consumed\n"
                f"4. Find public API exposed to other modules\n\n"
                f"OUTPUT FORMAT:\n"
                f"<findings>\n"
                f"{{{{\n"
                f'  "layer": "Modules",\n'
                f'  "tech": "PHP/Laravel Modules",\n'
                f'  "description": "One sentence: what domain these modules collectively cover",\n'
                f'  "modules": [{{"name": "ModuleName", "description": "domain purpose", "domain": "domain", "depends_on": [], "provides": []}}]\n'
                f"}}}}\n"
                f"</findings>\n\n"
                f"CONSTRAINTS:\n"
                f"- modules: list all modules found in the directory\n"
                f"- description: one sentence maximum per module"
            )

    # ── Python ──
    py_marker = _search(root, ['pyproject.toml', 'requirements.txt', 'setup.py', 'Pipfile', 'poetry.lock'])
    if py_marker:
        py_root = py_marker.parent
        subs = _subdirs(py_root)
        framework = 'FastAPI' if any(s in subs for s in ('routers', 'api')) \
            else 'Django' if any(s in subs for s in ('urls.py',)) or (py_root / 'manage.py').exists() \
            else 'Flask' if any(s in subs for s in ('blueprints',)) \
            else 'Python'
        add_layer('Backend', py_root, f'Python/{framework}',
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Backend layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the Python/{framework} Backend layer at {py_root}.\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (framework version, Python version from pyproject.toml)\n"
            f"2. All major components (routers/views, services, models, schemas)\n"
            f"3. Key external dependencies (database, cache, queue — read actual connection strings)\n"
            f"4. How this layer exposes its API (routes/endpoints grouped by domain)\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read pyproject.toml or requirements.txt — list ALL dependencies with their purpose\n"
            f"2. Explore source directories — map modules: routers/views, services, models, schemas\n"
            f"3. Find API endpoints (FastAPI routes, Django urls, Flask blueprints)\n"
            f"4. Find DB: read actual connection string in settings.py or .env\n"
            f"5. Map data flow between layers (which router calls which service, which service calls which model)\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Backend",\n'
            f'  "tech": "Python/Framework version, Key Libraries",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "component_relationships": [{{"from": "RouterA", "to": "ServiceB", "label": "calls"}}],\n'
            f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
            f'  "api_endpoints": [],\n'
            f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Router", "message": "request", "response": "result"}}]}}]\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact class/file names found in the code (max 25 chars each)\n"
            f"- tech: include version numbers visible in pyproject.toml or requirements.txt\n"
            f"- description: one sentence maximum\n"
            f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual config files"
        )

    # ── Node.js / TypeScript ──
    pkg_json = _search(root, ['package.json'])
    if pkg_json and not composer:  # skip if already handled as frontend of PHP project
        pkg_root = pkg_json.parent
        try:
            data = _json.loads(pkg_json.read_text(encoding='utf-8'))
            deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
            framework = 'Next.js' if 'next' in deps \
                else 'NestJS' if '@nestjs/core' in deps \
                else 'Express' if 'express' in deps \
                else 'React' if 'react' in deps \
                else 'Vue' if 'vue' in deps \
                else 'Node.js'
        except Exception:
            framework = 'Node.js'
            deps = {}
        subs = _subdirs(pkg_root)
        has_src = (pkg_root / 'src').exists()
        is_frontend = framework in ('React', 'Vue', 'Next.js') and not any(
            d in subs for d in ('controllers', 'services', 'handlers', 'routes'))
        layer_name = 'Frontend' if is_frontend else 'Backend'
        add_layer(layer_name, pkg_root, f'TypeScript/{framework}' if (pkg_root / 'tsconfig.json').exists() else framework,
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the {layer_name} layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the {framework} {layer_name.lower()} layer at {pkg_root}.\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (framework version, language version from package.json)\n"
            f"2. All major components ({'components, pages, hooks, stores' if is_frontend else 'controllers, services, repositories, modules'})\n"
            f"3. Key external dependencies ({'API endpoints called' if is_frontend else 'DB driver, cache, queue, external APIs'})\n"
            f"4. How this layer {'communicates with the backend' if is_frontend else 'exposes its API'}\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read package.json — list ALL dependencies with their purpose\n"
            f"2. Explore src/ — map modules: {'components, pages, hooks, stores' if is_frontend else 'controllers, services, repositories, modules'}\n"
            f"3. Find {'API calls (axios/fetch) — what endpoints are called' if is_frontend else 'API routes/endpoints grouped by domain'}\n"
            f"4. Find {'state management (Redux/Zustand/Pinia)' if is_frontend else 'DB driver, cache, queue, external APIs'}\n"
            f"5. Map cross-module dependencies\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "{layer_name}",\n'
            f'  "tech": "TypeScript/Framework version, Key Libraries",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "external_deps": [{{"name": "dep", "type": "other", "label": "uses"}}],\n'
            + ('  "api_calls": []\n' if is_frontend else '  "api_endpoints": []\n') +
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact component/file names found in the code (max 25 chars each)\n"
            f"- tech: include version numbers visible in package.json\n"
            f"- description: one sentence maximum"
        )

    # ── Java / Kotlin / Spring ──
    java_marker = _search(root, ['pom.xml', 'build.gradle', 'build.gradle.kts'])
    if java_marker:
        java_root = java_marker.parent
        is_kotlin = java_marker.name.endswith('.kts') or any((java_root / 'src').rglob('*.kt'))
        tech = 'Kotlin/Spring' if is_kotlin else 'Java/Spring'
        subs = _subdirs(java_root)
        add_layer('Backend', java_root, tech,
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Backend layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the {tech} Backend layer at {java_root}.\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (Spring version, Java/Kotlin version from pom.xml/build.gradle)\n"
            f"2. All major components (controllers, services, repositories, models, config)\n"
            f"3. Key external dependencies (DB driver, cache, queue — read actual configuration)\n"
            f"4. How this layer exposes its API (REST endpoints grouped by domain)\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read pom.xml or build.gradle — list ALL dependencies with their purpose\n"
            f"2. Explore src/main/ — map packages: controllers, services, repositories, models, config\n"
            f"3. Find REST endpoints (@RestController, @GetMapping etc)\n"
            f"4. Find DB (JPA/Hibernate/JDBC), cache (Redis?), queue (Kafka/RabbitMQ?), external clients\n"
            f"5. Map Spring beans and their relationships\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Backend",\n'
            f'  "tech": "{tech} version, Key Libraries",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "component_relationships": [{{"from": "ControllerA", "to": "ServiceB", "label": "calls"}}],\n'
            f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
            f'  "api_endpoints": [],\n'
            f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Controller", "message": "request", "response": "result"}}]}}]\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact class names found in the code (max 25 chars each)\n"
            f"- tech: include version numbers visible in pom.xml or build.gradle\n"
            f"- description: one sentence maximum\n"
            f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual configuration"
        )

    # ── Rust ──
    cargo = _search(root, ['Cargo.toml'])
    if cargo:
        rust_root = cargo.parent
        subs = _subdirs(rust_root)
        add_layer('Backend', rust_root, 'Rust',
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Backend layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the Rust Backend layer at {rust_root}.\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (Rust edition, web framework, key crates from Cargo.toml)\n"
            f"2. All major components (handlers, services, models, db, config)\n"
            f"3. Key external dependencies (DB crate, cache, queue — read actual Cargo.toml)\n"
            f"4. How this layer exposes its API (Axum/Actix/Warp routes)\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read Cargo.toml — list ALL dependencies with their purpose\n"
            f"2. Explore src/ — map modules: handlers, services, models, db, config\n"
            f"3. Find API layer (Axum/Actix/Warp routes)\n"
            f"4. Find DB (sqlx/diesel/sea-orm), cache, queue, external HTTP clients\n"
            f"5. Map module dependencies\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Backend",\n'
            f'  "tech": "Rust edition, Framework, Key Crates",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "component_relationships": [{{"from": "HandlerA", "to": "ServiceB", "label": "calls"}}],\n'
            f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
            f'  "api_endpoints": [],\n'
            f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Handler", "message": "request", "response": "result"}}]}}]\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact module/struct names found in the code (max 25 chars each)\n"
            f"- tech: include version info visible in Cargo.toml\n"
            f"- description: one sentence maximum\n"
            f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual Cargo.toml"
        )

    # ── Ruby / Rails ──
    gemfile = _search(root, ['Gemfile'])
    if gemfile:
        ruby_root = gemfile.parent
        subs = _subdirs(ruby_root)
        framework = 'Rails' if (ruby_root / 'config' / 'routes.rb').exists() else 'Ruby'
        add_layer('Backend', ruby_root, f'Ruby/{framework}',
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Backend layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the Ruby/{framework} Backend layer at {ruby_root}.\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (Rails/Ruby version from Gemfile)\n"
            f"2. All major components (controllers, models, services, jobs, mailers)\n"
            f"3. Key external dependencies (DB adapter, cache, queue — read actual config/database.yml)\n"
            f"4. How this layer exposes its API (routes from config/routes.rb)\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read Gemfile — list ALL dependencies with their purpose\n"
            f"2. Explore app/ — controllers, models, services, jobs, mailers\n"
            f"3. Read config/routes.rb — list API endpoints\n"
            f"4. Find DB (ActiveRecord adapter), cache (Redis?), queue (Sidekiq?), external APIs\n"
            f"5. Map layer interactions\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Backend",\n'
            f'  "tech": "Ruby/Framework version, Key Gems",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "component_relationships": [{{"from": "ControllerA", "to": "ServiceB", "label": "calls"}}],\n'
            f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
            f'  "api_endpoints": [],\n'
            f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Controller", "message": "request", "response": "result"}}]}}]\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact class names found in the code (max 25 chars each)\n"
            f"- tech: include version numbers visible in Gemfile\n"
            f"- description: one sentence maximum\n"
            f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual config files"
        )

    # ── .NET / C# ──
    csproj = _search(root, ['*.sln', '*.csproj'])
    if not csproj:
        # search manually
        for f in root.rglob('*.csproj'):
            csproj = f
            break
    if csproj:
        dotnet_root = csproj.parent if csproj.suffix == '.csproj' else csproj.parent
        subs = _subdirs(dotnet_root)
        add_layer('Backend', dotnet_root, 'C#/.NET',
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Backend layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
            f"Your task: Analyze the C#/.NET Backend layer at {dotnet_root}.\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. The primary technology stack (.NET version, ASP.NET version from .csproj)\n"
            f"2. All major components (Controllers, Services, Repositories, Models, DTOs)\n"
            f"3. Key external dependencies (DB provider, cache, queue — read actual appsettings.json)\n"
            f"4. How this layer exposes its API (ASP.NET controllers or minimal API endpoints)\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read .csproj — list ALL NuGet dependencies with their purpose\n"
            f"2. Explore project structure — Controllers, Services, Repositories, Models, DTOs\n"
            f"3. Find API endpoints (ASP.NET controllers, minimal APIs)\n"
            f"4. Find DB (EF Core/Dapper), cache, queue (MassTransit/RabbitMQ?), external clients\n"
            f"5. Map dependency injection registrations\n\n"
            f"OUTPUT FORMAT — return exactly this JSON structure:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Backend",\n'
            f'  "tech": "C#/.NET version, Key Packages",\n'
            f'  "description": "One sentence: what this layer does and its role in the system",\n'
            f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
            f'  "component_relationships": [{{"from": "ControllerA", "to": "ServiceB", "label": "calls"}}],\n'
            f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
            f'  "api_endpoints": [],\n'
            f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Controller", "message": "request", "response": "result"}}]}}]\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components: list 10-15 items, use exact class names found in the code (max 25 chars each)\n"
            f"- tech: include version numbers visible in .csproj\n"
            f"- description: one sentence maximum\n"
            f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual appsettings.json"
        )

    # ── Frontend (standalone Vue/React/Angular) ──
    if not layers or all(l['name'] != 'Frontend' for l in layers):
        for fe_path in [root / 'resources' / 'js', root / 'frontend', root / 'client', root / 'web']:
            if fe_path.exists():
                pkg = fe_path / 'package.json'
                if not pkg.exists():
                    pkg = root / 'package.json'
                try:
                    data = _json.loads(pkg.read_text(encoding='utf-8')) if pkg.exists() else {}
                    deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                    fw = 'Vue' if 'vue' in deps else 'React' if 'react' in deps \
                        else 'Angular' if '@angular/core' in deps else 'JavaScript'
                except Exception:
                    fw = 'JavaScript'
                add_layer('Frontend', fe_path, fw,
                    f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Frontend layer of this project.\n\n"
                    f"WHY this matters: Your findings will be used to generate C4 architecture diagrams showing the frontend layer and its communication with the backend.\n\n"
                    f"Your task: Analyze the {fw} Frontend layer at {fe_path}.\n\n"
                    f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
                    f"1. The primary technology stack (framework version from package.json)\n"
                    f"2. All major components/pages grouped by feature domain\n"
                    f"3. Which API endpoints this frontend calls (from axios/fetch usage)\n"
                    f"4. How this layer manages state and routing\n\n"
                    f"ANALYSIS STEPS:\n"
                    f"1. Read package.json — list key dependencies\n"
                    f"2. Map components/pages by feature domain\n"
                    f"3. Find API calls (axios/fetch) — what endpoints are called\n"
                    f"4. Find state management (Vuex/Pinia/Redux/Zustand)\n"
                    f"5. Find routing config\n\n"
                    f"OUTPUT FORMAT:\n"
                    f"<findings>\n"
                    f"{{{{\n"
                    f'  "layer": "Frontend",\n'
                    f'  "tech": "{fw}, version info",\n'
                    f'  "description": "One sentence: what this frontend does and who uses it",\n'
                    f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
                    f'  "external_deps": [],\n'
                    f'  "api_calls": []\n'
                    f"}}}}\n"
                    f"</findings>\n\n"
                    f"CONSTRAINTS:\n"
                    f"- key_components: list 10-15 items, use exact component/page names (max 25 chars each)\n"
                    f"- tech: include version numbers visible in package.json\n"
                    f"- description: one sentence maximum"
                )
                break

    # ── Docker / Infrastructure ──
    docker_compose = _search(root, ['docker-compose.yml', 'docker-compose.yaml'])
    if docker_compose:
        add_layer('Infrastructure', docker_compose.parent, 'Docker/Compose',
            f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Infrastructure layer of this project.\n\n"
            f"WHY this matters: Your findings will be used to generate C4 architecture diagrams showing all deployable services and their relationships. Each service you identify becomes a node in the container diagram.\n\n"
            f"Your task: Analyze the Infrastructure layer at {docker_compose.parent}.\n\n"
            f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
            f"1. All services defined in docker-compose.yml with their roles\n"
            f"2. The role of each service (database/cache/queue/proxy/app)\n"
            f"3. Service dependencies (depends_on, network connections)\n"
            f"4. Image names and versions for each service\n\n"
            f"ANALYSIS STEPS:\n"
            f"1. Read docker-compose.yml — list ALL services with their roles\n"
            f"2. For each service identify: its role (database/cache/queue/proxy/app), image name\n"
            f"3. Map service dependencies (depends_on, networks)\n\n"
            f"OUTPUT FORMAT:\n"
            f"<findings>\n"
            f"{{{{\n"
            f'  "layer": "Infrastructure",\n'
            f'  "tech": "Docker/Compose",\n'
            f'  "description": "One sentence: what infrastructure this compose file defines",\n'
            f'  "key_components": [{{"name": "postgres", "image": "postgres:15", "purpose": "primary database"}}, {{"name": "redis", "image": "redis:7", "purpose": "session cache"}}],\n'
            f'  "external_deps": []\n'
            f"}}}}\n"
            f"</findings>\n\n"
            f"CONSTRAINTS:\n"
            f"- key_components[].name: ONLY the short service name from docker-compose (e.g. 'postgres', 'redis', 'nginx')\n"
            f"- Do NOT include image versions, ports, or descriptions in the name field — those go in 'image' and 'purpose'\n"
            f"- description: one sentence maximum"
        )

    # ── Fallback: generic scan ──
    if not layers:
        subs = _subdirs(root)
        layers.append({
            'name': 'Project',
            'path': str(root),
            'tech': 'Unknown',
            'agent_prompt': (
                f"You are a code architecture analyst. Your job is to produce a precise JSON findings object for the Project layer of this project.\n\n"
                f"WHY this matters: Your findings will be used to generate C4 architecture diagrams and navigation context files that help AI assistants navigate this codebase efficiently. Incomplete or vague findings produce unusable diagrams.\n\n"
                f"Your task: Analyze the project '{root.name}' at {root}.\n"
                f"Directories: {', '.join(subs[:20])}\n\n"
                f"SUCCESS CRITERIA — your output is complete when you have identified:\n"
                f"1. The technology stack and primary framework\n"
                f"2. The main architectural layers and their purposes\n"
                f"3. Key components with their roles\n"
                f"4. External dependencies (databases, APIs, services)\n\n"
                f"ANALYSIS STEPS:\n"
                f"1. Identify the technology stack and framework from config files\n"
                f"2. Map the main architectural layers\n"
                f"3. List key components with their purpose\n"
                f"4. Find external dependencies (databases, APIs, services)\n"
                f"5. Map component relationships\n\n"
                f"OUTPUT FORMAT — return exactly this JSON structure:\n"
                f"<findings>\n"
                f"{{{{\n"
                f'  "layer": "Project",\n'
                f'  "tech": "Framework, Language, Key Libraries",\n'
                f'  "description": "One sentence: what this project does and its main purpose",\n'
                f'  "key_components": ["ComponentName1", "ComponentName2"],\n'
                f'  "component_relationships": [{{"from": "ComponentA", "to": "ComponentB", "label": "calls"}}],\n'
                f'  "external_deps": [{{"name": "PostgreSQL", "type": "database", "label": "reads/writes data"}}],\n'
                f'  "api_endpoints": [],\n'
                f'  "key_flows": [{{"title": "Flow name", "steps": [{{"from": "Client", "to": "Handler", "message": "request", "response": "result"}}]}}]\n'
                f"}}}}\n"
                f"</findings>\n\n"
                f"CONSTRAINTS:\n"
                f"- key_components: list 10-15 items, use exact class/file names found in the code (max 25 chars each)\n"
                f"- tech: include version numbers where visible in config files\n"
                f"- description: one sentence maximum\n"
                f"- external_deps: only real DBs/queues/APIs/services — verify by reading actual config files"
            ),
        })

    return layers


def ndoc_explore(project_path: str = "") -> str:
    """
    Detect architectural layers in a project and return structured prompts for spawning parallel research agents that will analyze each layer deeply.

    Use this tool as the FIRST STEP of a two-step workflow when ndoc_init produces low-quality context (shallow function lists). It returns agent instructions — you must then spawn agents using those instructions, collect their JSON findings, and pass them to ndoc_generate.

    Do NOT use if ndoc_init already produced good context.md files with full function signatures and dependency maps.
    Do NOT skip the agent spawning step — ndoc_explore only detects layers, ndoc_generate does the writing.

    Workflow:
    1. Call ndoc_explore → get layer detection + agent prompts
    2. Spawn one agent per layer (in parallel if possible)
    3. Each agent returns a JSON findings object
    4. Call ndoc_generate(findings=<combined JSON>) to write everything

    Parameters:
    - project_path: Path to project root. Leave empty ("") to use WORKSPACE_ROOT or current directory.

    Returns:
    - Detected architectural layers (Backend/Frontend/Infrastructure/etc.)
    - One agent prompt per layer with instructions for deep analysis
    - Expected JSON output format for agent findings

    Caveats: Layer detection is heuristic (checks for framework-specific files). May misclassify hybrid stacks. Works best on standard Laravel/Django/React/Go/Next.js projects.
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"Error: folder not found: {project_path}"

    layers = detect_project_layers(root)
    if not layers:
        return (
            f"Warning: no architectural layers detected in {root}.\n"
            f"Use ndoc_init for basic initialization."
        )

    n = len(layers)
    out = [
        f"NeuroDoc Explorer — {root.name}",
        "=" * 52,
        "",
        f"Detected {n} architectural layers:",
    ]
    for layer in layers:
        out.append(f"  • {layer['name']} ({layer['tech']}): {layer['path']}")

    out += [
        "",
        "=" * 52,
        "",
        "INSTRUCTION: Create an Agent Team to deeply analyze this project.",
        "",
        f"Spawn {n} teammates in parallel — one per architectural layer.",
        "Each teammate independently studies its layer and returns a JSON report.",
        "",
    ]

    for i, layer in enumerate(layers, 1):
        out.append("-" * 44)
        out.append(f"Teammate {i} — {layer['name']} ({layer['tech']})")
        out.append("")
        out.append(layer['agent_prompt'])
        out.append("")

    out += [
        "=" * 52,
        "",
        "After ALL teammates finish — collect their JSON reports and call:",
        "",
        f'  ndoc_generate(',
        f'    project_path="{root}",',
        f'    findings=<combined JSON string from all agents>',
        f'  )',
        "",
        "Expected findings JSON structure:",
        "```json",
        _json.dumps({
            "project": root.name,
            "architecture_type": "e.g. Laravel+Vue+Nova | Django+React | NestJS+Next.js",
            "layers": [
                {
                    "name": "layer name",
                    "path": "path/to/layer",
                    "tech": "technology stack",
                    "description": "what this layer does",
                    "key_components": ["ComponentA", "ServiceB"],
                    "external_deps": [
                        {
                            "name": "library-name",
                            "type": "database|cache|queue|http|auth|mail|payment|monitoring|other",
                            "label": "relationship description, e.g. reads/writes data"
                        }
                    ],
                }
            ],
            "modules": [
                {
                    "name": "ModuleName",
                    "description": "business purpose",
                    "domain": "domain name",
                    "depends_on": ["OtherModule"]
                }
            ],
            "cross_layer_relations": [
                {
                    "from": "Frontend",
                    "to": "Backend",
                    "label": "API requests [HTTP/JSON]"
                }
            ],
        }, ensure_ascii=False, indent=2),
        "```",
    ]

    return '\n'.join(out)


def generate_c4_overview_svg(findings: dict, project_name: str) -> str:
    """Generate a C4 model overview SVG showing all 4 levels on a single wide canvas."""

    # ── helpers ──────────────────────────────────────────────────────────────
    def esc(s: str) -> str:
        return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

    def rect(x, y, w, h, fill, stroke, rx=8, opacity=1.0, stroke_width=2):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
                f'opacity="{opacity}"/>')

    def text(x, y, s, size=13, anchor='middle', weight='normal', fill='#222', dy=0):
        dy_attr = f' dy="{dy}"' if dy else ''
        return (f'<text x="{x}" y="{y}"{dy_attr} text-anchor="{anchor}" '
                f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
                f'font-family="Arial,Helvetica,sans-serif">{esc(s)}</text>')

    def line(x1, y1, x2, y2, stroke, sw=2, dash=''):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

    def arrow_marker(mid, color):
        return (f'<marker id="arr_{mid}" markerWidth="10" markerHeight="7" '
                f'refX="9" refY="3.5" orient="auto">'
                f'<polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>'
                f'</marker>')

    def arrow(x1, y1, x2, y2, color, mid, sw=2, dash=''):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="{sw}"{d} '
                f'marker-end="url(#arr_{mid})"/>')

    def person_icon(cx, cy, color='#1168bd', size=28):
        r = size // 3
        head_y = cy - size // 2
        body_y = head_y + r * 2
        return (f'<circle cx="{cx}" cy="{head_y}" r="{r}" fill="{color}"/>'
                f'<line x1="{cx}" y1="{body_y}" x2="{cx}" y2="{body_y + size // 3}" '
                f'stroke="{color}" stroke-width="3"/>'
                f'<line x1="{cx - r}" y1="{body_y + 4}" x2="{cx + r}" y2="{body_y + 4}" '
                f'stroke="{color}" stroke-width="3"/>'
                f'<line x1="{cx}" y1="{body_y + size // 3}" x2="{cx - r}" y2="{cy + size // 2}" '
                f'stroke="{color}" stroke-width="3"/>'
                f'<line x1="{cx}" y1="{body_y + size // 3}" x2="{cx + r}" y2="{cy + size // 2}" '
                f'stroke="{color}" stroke-width="3"/>')

    def wrap_lines(s: str, max_len: int = 22) -> list:
        words = s.split()
        lines_out, cur = [], ''
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_len:
                cur += ' ' + w
            else:
                lines_out.append(cur)
                cur = w
        if cur:
            lines_out.append(cur)
        return lines_out or ['']

    def multiline_text(x, cy, s, size=11, anchor='middle', fill='#222', max_len=22):
        parts = wrap_lines(s, max_len)
        total = len(parts)
        line_h = size + 3
        start_y = cy - (total - 1) * line_h / 2
        out = []
        for i, part in enumerate(parts):
            out.append(text(x, start_y + i * line_h, part, size=size, anchor=anchor, fill=fill))
        return ''.join(out)

    # ── extract data from findings ────────────────────────────────────────────
    layers = findings.get('layers', [])
    modules = findings.get('modules', [])

    # L1: person, system, external deps
    external_systems_all: list = []
    for layer in layers:
        for dep in layer.get('external_deps', []):
            name = dep if isinstance(dep, str) else dep.get('name', str(dep))
            if name and name not in external_systems_all:
                external_systems_all.append(name)
    ext_l1 = external_systems_all[:3]

    # L2: containers from infra/backend layers
    containers: list = []
    for layer in layers:
        lname = layer.get('name', '')
        kc = layer.get('key_components', [])
        for item in kc[:4]:
            cname = item if isinstance(item, str) else item.get('name', str(item))
            if cname:
                containers.append({'name': cname, 'layer': lname})
    if not containers:
        containers = [{'name': 'API Server', 'layer': 'Backend'},
                      {'name': 'Database', 'layer': 'Infra'},
                      {'name': 'Cache', 'layer': 'Infra'}]
    containers = containers[:5]

    # L3: components from first non-infra layer with relationships
    comp_layer = None
    for layer in layers:
        if layer.get('key_components') and layer.get('component_relationships'):
            comp_layer = layer
            break
    if comp_layer is None:
        for layer in layers:
            if layer.get('key_components'):
                comp_layer = layer
                break
    comp_components: list = []
    comp_rels: list = []
    if comp_layer:
        for item in comp_layer.get('key_components', [])[:5]:
            cname = item if isinstance(item, str) else item.get('name', str(item))
            if cname:
                comp_components.append(cname)
        comp_rels = comp_layer.get('component_relationships', [])[:4]

    # L4: class names from key_flows steps
    l4_classes: list = []
    for layer in layers:
        for flow in layer.get('key_flows', [])[:1]:
            for step in flow.get('steps', [])[:4]:
                s = step if isinstance(step, str) else step.get('component', str(step))
                if s and s not in l4_classes:
                    l4_classes.append(s)
    if not l4_classes:
        for flow in findings.get('key_flows', [])[:1]:
            for step in flow.get('steps', [])[:4]:
                s = step if isinstance(step, str) else step.get('component', str(step))
                if s and s not in l4_classes:
                    l4_classes.append(s)
    if not l4_classes:
        l4_classes = ['Controller', 'Service', 'Repository', 'Model']
    l4_classes = l4_classes[:4]

    # ── canvas & panel layout ─────────────────────────────────────────────────
    W, H = 3300, 1380

    # Panel positions (x, y, w, h)
    P1 = (40,   60,  680, 560)   # L1 Context
    P2 = (780,  200, 780, 720)   # L2 Containers
    P3 = (1640, 340, 820, 760)   # L3 Components
    P4 = (2540, 480, 720, 700)   # L4 Code

    # Panel tints
    TINT1 = '#dbeafe'   # blue
    TINT2 = '#dcfce7'   # green
    TINT3 = '#fff3cd'   # orange/yellow
    TINT4 = '#fde8e8'   # red
    BORDER1 = '#3b82f6'
    BORDER2 = '#22c55e'
    BORDER3 = '#f59e0b'
    BORDER4 = '#ef4444'

    # Zoom arrow colors
    Z_COLOR1 = '#22c55e'  # green  L1→L2
    Z_COLOR2 = '#f59e0b'  # orange L2→L3
    Z_COLOR3 = '#ef4444'  # red    L3→L4

    svg_parts = []

    # ── defs: arrow markers ───────────────────────────────────────────────────
    svg_parts.append('<defs>')
    svg_parts.append(arrow_marker('green', Z_COLOR1))
    svg_parts.append(arrow_marker('orange', Z_COLOR2))
    svg_parts.append(arrow_marker('red', Z_COLOR3))
    svg_parts.append(arrow_marker('blue', '#1168bd'))
    svg_parts.append(arrow_marker('gray', '#888'))
    svg_parts.append('</defs>')

    # ── background ───────────────────────────────────────────────────────────
    svg_parts.append(rect(0, 0, W, H, '#f8f9fa', 'none', rx=0))

    # ── title ────────────────────────────────────────────────────────────────
    svg_parts.append(text(W // 2, 42, f'C4 Architecture Overview — {esc(project_name)}',
                          size=28, weight='bold', fill='#1a1a2e'))

    # ══ PANEL 1: L1 Context ══════════════════════════════════════════════════
    x1, y1, w1, h1 = P1
    svg_parts.append(rect(x1, y1, w1, h1, TINT1, BORDER1, rx=12, stroke_width=3))
    svg_parts.append(text(x1 + w1 // 2, y1 + 32, 'L1 — Context', size=18, weight='bold', fill=BORDER1))

    # Person
    pcx, pcy = x1 + 120, y1 + 160
    svg_parts.append(person_icon(pcx, pcy, color='#1168bd', size=40))
    svg_parts.append(text(pcx, pcy + 40, 'User', size=12, fill='#1168bd', weight='bold'))

    # System box
    sx, sy, sw, sh = x1 + 230, y1 + 120, 200, 80
    svg_parts.append(rect(sx, sy, sw, sh, '#1168bd', '#0d4f8c', rx=6))
    svg_parts.append(text(sx + sw // 2, sy + 30, project_name[:18], size=13, weight='bold', fill='white'))
    svg_parts.append(text(sx + sw // 2, sy + 50, '[Software System]', size=10, fill='#cce0ff'))

    # Arrow: person → system
    svg_parts.append(arrow(pcx + 25, pcy, sx, sy + sh // 2, '#1168bd', 'blue'))

    # External systems
    ext_colors = ['#999', '#777', '#aaa']
    for idx, ext in enumerate(ext_l1):
        ex = x1 + 60 + idx * 200
        ey = y1 + 360
        ew, eh = 140, 60
        svg_parts.append(rect(ex, ey, ew, eh, '#e5e7eb', '#6b7280', rx=6))
        svg_parts.append(multiline_text(ex + ew // 2, ey + eh // 2, ext[:20], size=10, fill='#333'))
        # Arrow from system to external
        svg_parts.append(arrow(sx + sw // 2, sy + sh, ex + ew // 2, ey, '#888', 'gray', dash='6,4'))

    if not ext_l1:
        ex, ey, ew, eh = x1 + 80, y1 + 360, 140, 60
        svg_parts.append(rect(ex, ey, ew, eh, '#e5e7eb', '#6b7280', rx=6))
        svg_parts.append(text(ex + ew // 2, ey + eh // 2, 'External System', size=10, fill='#555'))

    # ══ PANEL 2: L2 Containers ═══════════════════════════════════════════════
    x2, y2, w2, h2 = P2
    svg_parts.append(rect(x2, y2, w2, h2, TINT2, BORDER2, rx=12, stroke_width=3))
    svg_parts.append(text(x2 + w2 // 2, y2 + 32, 'L2 — Containers', size=18, weight='bold', fill='#166534'))

    # System boundary inner box
    bx, by, bw, bh = x2 + 30, y2 + 55, w2 - 60, h2 - 120
    svg_parts.append(rect(bx, by, bw, bh, 'none', BORDER2, rx=8, stroke_width=2))
    svg_parts.append(text(bx + 8, by + 16, f'[{project_name[:20]}]', size=10, fill=BORDER2, anchor='start'))

    # Container boxes
    cols, rows = 2, 3
    cw, ch = 140, 60
    cx_start = bx + (bw - cols * cw - (cols - 1) * 20) // 2
    cy_start = by + 35
    for idx, cont in enumerate(containers[:cols * rows]):
        col = idx % cols
        row = idx // cols
        cx_ = cx_start + col * (cw + 20)
        cy_ = cy_start + row * (ch + 18)
        cname = cont['name'] if isinstance(cont, dict) else str(cont)
        clayer = cont.get('layer', '') if isinstance(cont, dict) else ''
        c_fill = '#166534' if 'infra' in clayer.lower() or 'db' in cname.lower() else '#15803d'
        svg_parts.append(rect(cx_, cy_, cw, ch, c_fill, '#14532d', rx=5))
        svg_parts.append(multiline_text(cx_ + cw // 2, cy_ + ch // 2, cname[:22], size=10, fill='white'))

    # External DBs below boundary
    db_x, db_y = x2 + 50, y2 + h2 - 55
    svg_parts.append(rect(db_x, db_y, 130, 40, '#d1fae5', '#059669', rx=4))
    svg_parts.append(text(db_x + 65, db_y + 24, 'Database', size=10, fill='#065f46'))

    # ══ PANEL 3: L3 Components ═══════════════════════════════════════════════
    x3, y3, w3, h3 = P3
    svg_parts.append(rect(x3, y3, w3, h3, TINT3, BORDER3, rx=12, stroke_width=3))
    svg_parts.append(text(x3 + w3 // 2, y3 + 32, 'L3 — Components', size=18, weight='bold', fill='#92400e'))

    # Container boundary
    cbx, cby, cbw, cbh = x3 + 30, y3 + 55, w3 - 60, h3 - 120
    svg_parts.append(rect(cbx, cby, cbw, cbh, 'none', BORDER3, rx=8, stroke_width=2))
    layer_label = comp_layer.get('name', 'Container') if comp_layer else 'Container'
    svg_parts.append(text(cbx + 8, cby + 16, f'[{layer_label[:24]}]', size=10, fill=BORDER3, anchor='start'))

    comp_boxes = {}
    cw3, ch3 = 150, 52
    n_comp = len(comp_components) or 1
    comp_per_col = min(3, n_comp)
    comp_cols = (n_comp + comp_per_col - 1) // comp_per_col
    cx3_start = cbx + (cbw - comp_cols * cw3 - (comp_cols - 1) * 20) // 2
    for idx, comp in enumerate(comp_components):
        col = idx % comp_cols if comp_cols > 1 else 0
        row = idx // comp_cols if comp_cols > 1 else idx
        cx_ = cx3_start + col * (cw3 + 20)
        cy_ = cby + 35 + row * (ch3 + 14)
        svg_parts.append(rect(cx_, cy_, cw3, ch3, '#d97706', '#b45309', rx=5))
        svg_parts.append(multiline_text(cx_ + cw3 // 2, cy_ + ch3 // 2, comp[:24], size=10, fill='white'))
        comp_boxes[comp] = (cx_ + cw3 // 2, cy_ + ch3 // 2, cx_, cy_, cw3, ch3)

    # Relationship arrows
    for rel in comp_rels[:3]:
        frm = rel.get('from', '') if isinstance(rel, dict) else ''
        to_ = rel.get('to', '') if isinstance(rel, dict) else ''
        if frm in comp_boxes and to_ in comp_boxes:
            fx, fy = comp_boxes[frm][0], comp_boxes[frm][1] + comp_boxes[frm][5] // 2
            tx, ty = comp_boxes[to_][0], comp_boxes[to_][1] - comp_boxes[to_][5] // 2
            svg_parts.append(arrow(fx, fy, tx, ty, '#b45309', 'gray', sw=1, dash='4,3'))

    # ══ PANEL 4: L4 Code ═════════════════════════════════════════════════════
    x4, y4, w4, h4 = P4
    svg_parts.append(rect(x4, y4, w4, h4, TINT4, BORDER4, rx=12, stroke_width=3))
    svg_parts.append(text(x4 + w4 // 2, y4 + 32, 'L4 — Code', size=18, weight='bold', fill='#991b1b'))

    # Class boxes
    cw4, ch4 = 180, 90
    n4 = len(l4_classes)
    col_gap = 20
    grid_w = min(2, n4) * (cw4 + col_gap) - col_gap
    cx4_start = x4 + (w4 - grid_w) // 2
    for idx, cls in enumerate(l4_classes):
        col = idx % 2
        row = idx // 2
        cx_ = cx4_start + col * (cw4 + col_gap)
        cy_ = y4 + 60 + row * (ch4 + 18)
        svg_parts.append(rect(cx_, cy_, cw4, ch4, '#fee2e2', BORDER4, rx=5))
        # Class header
        svg_parts.append(rect(cx_, cy_, cw4, 26, BORDER4, BORDER4, rx=5))
        svg_parts.append(text(cx_ + cw4 // 2, cy_ + 17, cls[:22], size=11, weight='bold', fill='white'))
        # Methods placeholder
        svg_parts.append(text(cx_ + 8, cy_ + 42, '+ method()', size=9, anchor='start', fill='#555'))
        svg_parts.append(text(cx_ + 8, cy_ + 57, '+ property', size=9, anchor='start', fill='#555'))

    # Arrows between class boxes (chain)
    for i in range(min(n4 - 1, 3)):
        col_a, row_a = i % 2, i // 2
        col_b, row_b = (i + 1) % 2, (i + 1) // 2
        ax_ = cx4_start + col_a * (cw4 + col_gap) + cw4 // 2
        ay_ = y4 + 60 + row_a * (ch4 + 18) + ch4
        bx_ = cx4_start + col_b * (cw4 + col_gap) + cw4 // 2
        by__ = y4 + 60 + row_b * (ch4 + 18)
        svg_parts.append(arrow(ax_, ay_, bx_, by__, '#ef4444', 'red', sw=1))

    # ══ Zoom-in diagonal arrows between panels ════════════════════════════════
    # L1 → L2
    ax_l1 = P1[0] + P1[2]
    ay_l1 = P1[1] + P1[3] // 2
    bx_l2 = P2[0]
    by_l2 = P2[1] + P2[3] // 2
    mid_x12 = (ax_l1 + bx_l2) // 2
    mid_y12 = (ay_l1 + by_l2) // 2
    svg_parts.append(f'<path d="M {ax_l1} {ay_l1} Q {mid_x12} {mid_y12 - 60} {bx_l2} {by_l2}" '
                     f'fill="none" stroke="{Z_COLOR1}" stroke-width="3" '
                     f'marker-end="url(#arr_green)"/>')
    svg_parts.append(text(mid_x12, mid_y12 - 70, 'Zoom in', size=13, fill=Z_COLOR1, weight='bold'))

    # L2 → L3
    ax_l2 = P2[0] + P2[2]
    ay_l2 = P2[1] + P2[3] // 2
    bx_l3 = P3[0]
    by_l3 = P3[1] + P3[3] // 2
    mid_x23 = (ax_l2 + bx_l3) // 2
    mid_y23 = (ay_l2 + by_l3) // 2
    svg_parts.append(f'<path d="M {ax_l2} {ay_l2} Q {mid_x23} {mid_y23 - 60} {bx_l3} {by_l3}" '
                     f'fill="none" stroke="{Z_COLOR2}" stroke-width="3" '
                     f'marker-end="url(#arr_orange)"/>')
    svg_parts.append(text(mid_x23, mid_y23 - 70, 'Zoom in', size=13, fill=Z_COLOR2, weight='bold'))

    # L3 → L4
    ax_l3 = P3[0] + P3[2]
    ay_l3 = P3[1] + P3[3] // 2
    bx_l4 = P4[0]
    by_l4 = P4[1] + P4[3] // 2
    mid_x34 = (ax_l3 + bx_l4) // 2
    mid_y34 = (ay_l3 + by_l4) // 2
    svg_parts.append(f'<path d="M {ax_l3} {ay_l3} Q {mid_x34} {mid_y34 - 60} {bx_l4} {by_l4}" '
                     f'fill="none" stroke="{Z_COLOR3}" stroke-width="3" '
                     f'marker-end="url(#arr_red)"/>')
    svg_parts.append(text(mid_x34, mid_y34 - 70, 'Zoom in', size=13, fill=Z_COLOR3, weight='bold'))

    # ══ Level labels at bottom ════════════════════════════════════════════════
    label_y = H - 30
    labels = [
        (P1[0] + P1[2] // 2, 'Level 1 / Context', BORDER1),
        (P2[0] + P2[2] // 2, 'Level 2 / Containers', BORDER2),
        (P3[0] + P3[2] // 2, 'Level 3 / Components', BORDER3),
        (P4[0] + P4[2] // 2, 'Level 4 / Code', BORDER4),
    ]
    for lx, lbl, lcolor in labels:
        svg_parts.append(text(lx, label_y, lbl, size=15, fill=lcolor, weight='bold'))

    # ══ Assemble SVG ══════════════════════════════════════════════════════════
    header = (f'<svg xmlns="http://www.w3.org/2000/svg" '
              f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    return header + '\n'.join(svg_parts) + '</svg>'


def ndoc_generate(project_path: str = "", findings: str = "") -> str:
    """
    Generate enriched context.md files and context.index.md with full C4 architecture diagrams (Context, Container, Component levels) from agent research findings.

    Use this tool as the SECOND STEP after ndoc_explore — after parallel agents have analyzed each architectural layer and returned JSON findings.

    Do NOT call without findings JSON — it will fall back to basic ndoc_init behavior.
    Do NOT pass findings from a different project.

    Parameters:
    - project_path: Path to project root. Leave empty ("") to use WORKSPACE_ROOT or current directory.
    - findings: JSON string containing agent research results. Expected structure:
      {
        "architecture_type": "monolith|microservices|fullstack",
        "layers": [{"name": "Backend", "tech": "Laravel 9, PHP 8.1", "description": "...", "key_components": ["LoginController", "UserService"]}],
        "modules": [{"name": "Auth", "layer": "Backend", "dependencies": ["Database", "Redis"]}],
        "cross_layer_relations": [{"from": "Frontend", "to": "Backend", "protocol": "HTTP/JSON", "description": "API calls"}]
      }

    Returns:
    - Confirmation of files written: context.md in all code directories, context.index.md with C4 diagrams.
    - Warning messages for detected conflicts (e.g., multiple databases claimed by different agents).
    - Summary of C4 diagram levels generated (Context/Container/Component).

    Caveats: Database conflict detection is heuristic — review the warning section manually if multiple DB systems are flagged. C4 diagrams are generated from findings + static code scan; accuracy depends on agent research quality.
    """
    root = resolve_path(project_path)
    if not root.exists():
        return f"Error: folder not found: {project_path}"

    if not findings.strip():
        return (
            "Warning: findings is empty. First run ndoc_explore and spawn the Agent Team.\n"
            "Or use ndoc_init for basic initialization without agents."
        )

    # Parse findings
    try:
        data = _json.loads(findings)
    except _json.JSONDecodeError as e:
        return f"Error parsing findings JSON: {e}"

    # ── Normalize & validate findings ──

    # 1. Deduplicate layers by name (keep first occurrence)
    seen_layer_names: set = set()
    deduped_layers = []
    for layer in data.get('layers', []):
        lname = layer.get('name', '').strip()
        if lname and lname not in seen_layer_names:
            seen_layer_names.add(lname)
            deduped_layers.append(layer)
    data['layers'] = deduped_layers

    # 2. Normalize key_components — coerce dicts/objects, deduplicate, strip empty
    for layer in data.get('layers', []):
        kc = layer.get('key_components', [])
        normalized = []
        seen_kc: set = set()
        for item in kc:
            if isinstance(item, dict):
                s = item.get('name', str(item))
            elif isinstance(item, str):
                s = item
            else:
                s = str(item)
            s = s.strip()
            if s and s not in seen_kc:
                seen_kc.add(s)
                normalized.append(s if isinstance(item, str) else item)
        layer['key_components'] = normalized

    for mod in data.get('modules', []):
        kc = mod.get('key_components', [])
        mod['key_components'] = [
            item if isinstance(item, str)
            else item.get('name', str(item)) if isinstance(item, dict)
            else str(item)
            for item in kc
        ]

    # 3. Deduplicate modules by name
    seen_mod_names: set = set()
    deduped_mods = []
    for mod in data.get('modules', []):
        mname = mod.get('name', '').strip()
        if mname and mname not in seen_mod_names:
            seen_mod_names.add(mname)
            deduped_mods.append(mod)
    data['modules'] = deduped_mods

    # 4. Detect cross-layer DB conflicts (warn but don't block)
    db_claims: dict = {}  # normalized_key → [source, ...]

    # Collect from external_deps
    for layer in data.get('layers', []):
        lname = layer.get('name', '?')
        for dep in layer.get('external_deps', []):
            dep_name = dep.get('name', '').strip()
            dep_type = dep.get('type', '')
            if dep_type in ('database', 'db') and is_real_external_system(dep_name):
                key = _normalize_system_name(dep_name)
                db_claims.setdefault(key, []).append(lname)

    # Also collect from docker/infra key_components
    for layer in data.get('layers', []):
        if not any(k in layer.get('name', '').lower() for k in ('infra', 'docker', 'infrastructure')):
            continue
        for comp in layer.get('key_components', []):
            cname = comp.get('name', str(comp)) if isinstance(comp, dict) else str(comp)
            cname = cname.strip()
            lower = cname.lower()
            if any(k in lower for k in ('postgres', 'mysql', 'mariadb', 'mongo', 'clickhouse', 'mssql', 'sqlite')):
                key = _normalize_system_name(cname)
                if not any(k in lower for k in ('redis', 'kafka', 'rabbit', 'nginx')):
                    db_claims.setdefault(key, []).append('docker-compose')

    conflicts = []
    all_db_keys = list(db_claims.keys())
    if len(all_db_keys) > 1:
        # Flag if different DB families (e.g. postgresql vs mysql)
        unique_families = set(re.sub(r'\d', '', k) for k in all_db_keys)
        if len(unique_families) > 1:
            sources = {k: ', '.join(set(db_claims[k])) for k in all_db_keys}
            detail = '; '.join(f"{k} (from {v})" for k, v in sources.items())
            conflicts.append(
                f"Multiple DB families detected: {detail} — verify which is actually used"
            )
    data['_validation_warnings'] = conflicts

    project_name = data.get('project', root.name)
    out = [f"NeuroDoc Generate — {project_name}", "=" * 52, ""]

    # Show validation warnings upfront
    warnings = data.get('_validation_warnings', [])
    if warnings:
        out.append("⚠ Validation warnings (verify manually):")
        for w in warnings:
            out.append(f"  ! {w}")
        out.append("")

    # ── Step 1: Static scan (for per-directory context.md) ──
    out.append("Step 1/4 — Static scan...")
    dirs_with_code = []
    all_dirs = set()
    for dp, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        dp = Path(dp)
        all_dirs.add(dp)  # every directory gets a context.md
        if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files):
            dirs_with_code.append(dp)
    all_dirs.add(root)

    all_module_names = [short_name(d, root) for d in all_dirs]
    modules: dict = {}
    for dp in sorted(all_dirs):
        mod = short_name(dp, root)
        fd_map = scan_dir(dp)
        all_imp, kw = [], []
        for fd in fd_map.values():
            all_imp.extend(fd.get('imports', []))
            kw.extend([fn['name'] for fn in fd.get('functions', [])[:2]])
        deps = resolve_deps(all_imp, [m for m in all_module_names if m != mod])
        modules[mod] = {'dir_path': dp, 'files': fd_map, 'deps': deps, 'keywords': kw[:4]}

    out.append(f"   ok: {len(all_dirs)} directories, {len(dirs_with_code)} contain code")

    # ── Step 2: Generate per-directory context.md ──
    out.append("\nStep 2/4 — Generating context.md files...")
    reverse_deps: dict = {}
    for mod, mdata in modules.items():
        for dep in mdata.get('deps', []):
            reverse_deps.setdefault(dep, []).append(mod)

    generated = 0
    for mod, mdata in modules.items():
        dp = mdata['dir_path']
        content = make_context_md(dp, root, all_module_names, reverse_deps)
        (dp / 'context.md').write_text(content, encoding='utf-8')
        generated += 1
    out.append(f"   ok: {generated} context.md files written")

    # ── Step 3: Build enriched index with agent findings ──
    out.append("\nStep 3/4 — Building context.index.md with C4 diagrams...")
    now = datetime.now().strftime('%Y-%m-%d')
    arch_type = data.get('architecture_type', '')
    layers_info = data.get('layers', [])
    cross_rels = data.get('cross_layer_relations', [])

    index_lines = [
        f"# Project: {project_name} | modules: {len(modules)} | updated: {now}",
        "",
    ]
    if arch_type:
        index_lines.append(f"**architecture:** {arch_type}")
        index_lines.append("")

    # Show validation warnings in the index too
    if warnings:
        index_lines.append("> ⚠ **Verify manually:** " + "; ".join(warnings))
        index_lines.append("")

    # Architecture overview from agent findings
    if layers_info:
        index_lines.append("## Architecture Overview")
        for layer in layers_info:
            desc = layer.get('description', '')
            tech = layer.get('tech', '')
            name = layer.get('name', '')
            kc = layer.get('key_components', [])[:4]
            line = f"**{name}** ({tech})"
            if desc:
                line += f": {desc}"
            if kc:
                kc_names = [c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in kc]
                line += f" | {', '.join(kc_names)}"
            index_lines.append(line)
        index_lines.append("")

    # Module descriptions from agent findings
    found_modules = data.get('modules', [])
    if found_modules:
        index_lines.append("## Modules")
        for mod in found_modules:
            name = mod.get('name', '')
            desc = mod.get('description', '')
            domain = mod.get('domain', '')
            deps = mod.get('depends_on', [])
            line = f"**{name}**"
            if domain:
                line += f" [{domain}]"
            if desc:
                line += f": {desc}"
            if deps:
                line += f" → {', '.join(deps)}"
            index_lines.append(line)
        index_lines.append("")

    # Standard map
    index_lines.append("## Map")
    for mod, mdata in modules.items():
        deps = mdata.get('deps', [])
        kw = mdata.get('keywords', [])[:4]
        dep_str = ', '.join(deps) if deps else '(none)'
        line = f"{mod} → {dep_str}"
        if kw:
            line += f" | {', '.join(kw)}"
        index_lines.append(line)
    index_lines.append("")

    # ── 4-level C4 diagrams from agent findings ──
    c4_context_str = generate_c4_context(data, project_name) if layers_info else \
        "\n".join(make_c4_context(modules, project_name, all_module_names))
    c4_container_str = generate_c4_container(data, project_name) if layers_info else \
        "\n".join(make_c4_container(modules, project_name, root))
    c4_component_str = generate_c4_component(data, project_name)
    c4_dynamic_str = generate_c4_dynamic(data, project_name)

    # C4 Context (Level 1)
    index_lines.append("## C4 Context (Level 1)")
    index_lines.append("")
    index_lines.append(c4_context_str)
    index_lines.append("")

    # C4 Container (Level 2)
    index_lines.append("## C4 Container (Level 2)")
    index_lines.append("")
    index_lines.append(c4_container_str)
    index_lines.append("")

    # C4 Component — Backend (Level 3)
    if c4_component_str:
        index_lines.append("## C4 Component — Backend (Level 3)")
        index_lines.append("")
        index_lines.append(c4_component_str)
        index_lines.append("")

    # C4 Dynamic — Key Flow (Level 4)
    if c4_dynamic_str:
        index_lines.append("## C4 Dynamic — Key Flow (Level 4)")
        index_lines.append("")
        index_lines.append(c4_dynamic_str)
        index_lines.append("")

    # Also generate per-layer Component diagrams for non-backend layers
    if layers_info:
        _SKIP = ('proto', 'protobuf', 'generated', 'gen', 'infra', 'docker', 'infrastructure', 'backend')
        for layer in layers_info:
            lname = layer.get('name', '')
            if any(k in lname.lower() for k in _SKIP):
                continue
            comp_lines = make_c4_component_from_findings(layer)
            if comp_lines:
                index_lines.append(f"## C4 Component — {lname}")
                index_lines.append("")
                index_lines += comp_lines
                index_lines.append("")

    # Sequence diagrams (from agent findings)
    # Collect key_flows: top-level and per-layer
    all_flows = list(data.get('key_flows', []))
    for layer in layers_info:
        for flow in layer.get('key_flows', []):
            all_flows.append(flow)
    if all_flows:
        data_with_flows = dict(data)
        data_with_flows['key_flows'] = all_flows
        index_lines.append("## Key Flows")
        index_lines += make_sequence_from_findings(data_with_flows, project_name)
        index_lines.append("")

    # Append C4 Overview SVG link to context.index.md
    index_lines += ["", "## C4 Overview", "![C4 Architecture Overview](c4-overview.svg)", ""]
    (root / 'context.index.md').write_text('\n'.join(index_lines), encoding='utf-8')
    out.append("   ok: context.index.md written with C4 Context + Container + Component + Dynamic (4 levels)")

    # Generate C4 overview SVG
    try:
        svg_content = generate_c4_overview_svg(data, project_name)
        svg_path = root / 'c4-overview.svg'
        svg_path.write_text(svg_content, encoding='utf-8')
        out.append("   ok: c4-overview.svg generated (4-level C4 overview)")
    except Exception as _svg_err:
        out.append(f"   warn: c4-overview.svg generation failed: {_svg_err}")

    # ── Step 4: Update CLAUDE.md ──
    out.append("\nStep 4/4 — Updating CLAUDE.md...")
    claude_path = root / 'CLAUDE.md'
    if claude_path.exists():
        existing = claude_path.read_text(encoding='utf-8')
        if 'NeuroDoc' not in existing:
            claude_path.write_text(existing + CLAUDE_MD_RULES, encoding='utf-8')
            out.append("   ok: NeuroDoc rules added to existing CLAUDE.md")
        else:
            out.append("   info: CLAUDE.md already contains NeuroDoc rules")
    else:
        claude_path.write_text(f"# {project_name}\n{CLAUDE_MD_RULES}", encoding='utf-8')
        out.append("   ok: CLAUDE.md created")

    out += [
        "",
        "=" * 52,
        "NeuroDoc Generate complete!",
        "",
        f"   {generated} context.md files",
        f"   context.index.md with C4 Context + Container + Component + Dynamic (4 levels)",
        f"   C4 diagrams built from Agent Team findings (proper relationships, no ID mismatches)",
        f"   CLAUDE.md updated",
    ]

    if cross_rels:
        out.append(f"   {len(cross_rels)} cross-layer relationships from agent findings")

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
