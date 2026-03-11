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
    """Create a short, valid Mermaid node ID from a name.
    Extracts first meaningful word(s) to avoid 80-char monster IDs.
    'HTTP API — Chi router — REST endpoints' → 'HTTP_API'
    'mail-backend — image:latest, ports 8080' → 'mail_backend'
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
    return alias[:40] or 'mod'


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
    'postgres': 'reads/writes data',
    'postgresql': 'reads/writes data',
    'mysql': 'reads/writes data',
    'mariadb': 'reads/writes data',
    'mongodb': 'reads/writes data',
    'mongo': 'reads/writes data',
    'sqlite': 'reads/writes data',
    'mssql': 'reads/writes data',
    'dynamodb': 'reads/writes data',
    'cassandra': 'reads/writes data',
    # Cache
    'redis': 'caches via',
    'memcached': 'caches via',
    'valkey': 'caches via',
    # HTTP / API
    'axios': 'makes HTTP requests via',
    'httpx': 'makes HTTP requests via',
    'requests': 'makes HTTP requests via',
    'got': 'makes HTTP requests via',
    'superagent': 'makes HTTP requests via',
    'guzzle': 'makes HTTP requests via',
    # Auth
    'jwt': 'authenticates via',
    'passport': 'authenticates via',
    'sanctum': 'authenticates via',
    'oauth': 'authenticates via',
    'auth': 'authenticates via',
    # Queues / workers
    'kafka': 'publishes messages to',
    'rabbitmq': 'publishes messages to',
    'celery': 'enqueues tasks via',
    'sqs': 'publishes messages to',
    'bull': 'enqueues jobs via',
    'beanstalkd': 'enqueues jobs via',
    'horizon': 'manages queues via',
    # Search
    'elasticsearch': 'searches via',
    'algolia': 'searches via',
    'typesense': 'searches via',
    'meilisearch': 'searches via',
    'scout': 'searches via',
    # Storage / CDN
    's3': 'stores files in',
    'cloudinary': 'stores media in',
    'minio': 'stores files in',
    'flysystem': 'stores files via',
    # Monitoring / Logging
    'sentry': 'reports errors to',
    'datadog': 'sends metrics to',
    'prometheus': 'exposes metrics for',
    'grafana': 'visualizes in',
    'bugsnag': 'reports errors to',
    # Mail
    'mailgun': 'sends email via',
    'sendgrid': 'sends email via',
    'smtp': 'sends email via',
    'mailer': 'sends email via',
    # Payment
    'stripe': 'processes payments via',
    'paypal': 'processes payments via',
    'braintree': 'processes payments via',
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
            name = fn['name'].lower()
            # Skip obfuscated/minified names (1-2 chars, or pure digits, or single-letter+digit)
            if len(name) >= 3 and not re.match(r'^[a-z]{1,2}\d*$', name):
                tags.add(name)

    meaningful_tags = sorted(t for t in tags if len(t) >= 3)[:16]
    lines.append(f"**tags:** {' '.join(meaningful_tags)}")
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
    proj = c4_alias(project_name)
    layers = findings.get('layers', [])
    techs = [l.get('tech', '') for l in layers if l.get('tech')]
    main_tech = techs[0] if techs else 'Application'

    ext_deps = _collect_ext_deps(layers)

    _DB = {'database', 'db'}
    _QUEUE = {'queue', 'broker', 'messaging'}
    _CACHE = {'cache'}

    lines = [
        '```mermaid', 'C4Context',
        f'  title System Context — {project_name}', '',
        f'  Person(user, "User", "End-user of {project_name}")',
        f'  System({proj}, "{project_name}", "{main_tech} application")',
        '',
    ]
    for key, info in list(ext_deps.items())[:12]:
        alias = c4_alias(key)
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
        lines.append(f'  Rel({proj}, {c4_alias(key)}, "{info["label"]}")')
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
                services[key] = {
                    'display': name,
                    'type': stype,
                    'label': get_rel_label(name) if stype != 'proxy' else 'routes traffic',
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
            f"You are studying the Go backend of '{root.name}' at: {go_root}\n"
            f"Module: {mod_name} | Dirs: {', '.join(subs[:15])}\n\n"
            f"Tasks:\n"
            f"1. Read go.mod — list ALL dependencies with their purpose\n"
            f"2. Explore cmd/ — what services/binaries are defined\n"
            f"3. Explore internal/, pkg/ — map packages: handlers, services, repositories, models\n"
            f"4. Find API layer: HTTP routes or gRPC services\n"
            f"5. Find: DB driver, cache, queue, external HTTP clients\n"
            f"6. Map cross-package data flow\n\n"
            f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
        )
        for proto_dir in ('proto', 'gen', 'pb'):
            if (go_root / proto_dir).exists():
                add_layer('gRPC / Protobuf', go_root / proto_dir, 'gRPC/Protobuf',
                    f"Study the gRPC/Protobuf layer of '{root.name}' at: {go_root / proto_dir}\n\n"
                    f"Tasks:\n"
                    f"1. List all .proto files — services and messages defined\n"
                    f"2. For each service: list RPC methods with request/response types\n"
                    f"3. Find generated code — what clients/servers are generated\n"
                    f"4. Identify cross-service dependencies\n\n"
                    f"Return JSON: tech, description, key_components[], external_deps[{{name,type,label}}]"
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
            f"Study the {framework} backend of '{root.name}' at: {php_root}\n\n"
            f"Tasks:\n"
            f"1. Read composer.json — list ALL dependencies with their purpose\n"
            f"2. Explore app/ — Controllers, Services, Models, Repositories, Providers\n"
            f"3. Read routes/ — list API/web endpoints grouped by domain\n"
            f"4. Find: DB type, cache (Redis?), queue driver, external APIs\n"
            f"5. Map layer interactions\n\n"
            f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
        )
        # Laravel Nova
        nova_path = php_root / 'nova-components'
        if nova_path.exists():
            components = [d.name for d in nova_path.iterdir() if d.is_dir()]
            add_layer('Admin Panel', nova_path, 'Laravel Nova/Vue.js',
                f"Study the Nova admin panel of '{root.name}' at: {nova_path}\n"
                f"Components: {', '.join(components[:15])}\n\n"
                f"Tasks:\n"
                f"1. For each component — read resources/js/ files, understand its purpose\n"
                f"2. Find props, events, key functions\n"
                f"3. Find cross-component dependencies (imports)\n"
                f"4. Identify what backend models each component interacts with\n\n"
                f"Return JSON: tech, description, key_components[{{name,purpose,dependencies[]}}], external_deps[{{name,type,label}}]"
            )
        # Laravel Modules
        modules_path = php_root / 'Modules'
        if modules_path.exists():
            mod_names = [d.name for d in modules_path.iterdir() if d.is_dir()]
            add_layer('Modules', modules_path, 'PHP/Laravel Modules',
                f"Study the Laravel modules (bounded contexts) of '{root.name}' at: {modules_path}\n"
                f"Modules: {', '.join(mod_names)}\n\n"
                f"Tasks:\n"
                f"1. For each module — Routes/, Controllers/, Models/ — what domain does it serve\n"
                f"2. Find cross-module dependencies\n"
                f"3. Identify events/jobs produced or consumed\n"
                f"4. Find public API exposed to other modules\n\n"
                f"Return JSON: modules[{{name, description, domain, depends_on[], provides[]}}]"
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
            f"Study the Python/{framework} backend of '{root.name}' at: {py_root}\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"Tasks:\n"
            f"1. Read pyproject.toml or requirements.txt — list ALL dependencies with their purpose\n"
            f"2. Explore source directories — map modules: routers/views, services, models, schemas\n"
            f"3. Find API endpoints (FastAPI routes, Django urls, Flask blueprints)\n"
            f"4. Find: DB (SQLAlchemy/Django ORM/etc), cache, queue (Celery?), external HTTP clients\n"
            f"5. Map data flow between layers\n\n"
            f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
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
            f"Study the {framework} {'frontend' if is_frontend else 'backend'} of '{root.name}' at: {pkg_root}\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"Tasks:\n"
            f"1. Read package.json — list ALL dependencies with their purpose\n"
            f"2. Explore src/ — map modules: {'components, pages, hooks, stores' if is_frontend else 'controllers, services, repositories, modules'}\n"
            f"3. Find {'API calls (axios/fetch) — what endpoints are called' if is_frontend else 'API routes/endpoints grouped by domain'}\n"
            f"4. Find: {'state management (Redux/Zustand/Pinia)' if is_frontend else 'DB driver, cache, queue, external APIs'}\n"
            f"5. Map cross-module dependencies\n\n"
            f"Return JSON: tech, description, key_components[], external_deps[{{name,type,label}}], {'api_calls[]' if is_frontend else 'api_endpoints[]'}"
        )

    # ── Java / Kotlin / Spring ──
    java_marker = _search(root, ['pom.xml', 'build.gradle', 'build.gradle.kts'])
    if java_marker:
        java_root = java_marker.parent
        is_kotlin = java_marker.name.endswith('.kts') or any((java_root / 'src').rglob('*.kt'))
        tech = 'Kotlin/Spring' if is_kotlin else 'Java/Spring'
        subs = _subdirs(java_root)
        add_layer('Backend', java_root, tech,
            f"Study the {tech} backend of '{root.name}' at: {java_root}\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"Tasks:\n"
            f"1. Read pom.xml or build.gradle — list ALL dependencies with their purpose\n"
            f"2. Explore src/main/ — map packages: controllers, services, repositories, models, config\n"
            f"3. Find REST endpoints (@RestController, @GetMapping etc)\n"
            f"4. Find: DB (JPA/Hibernate/JDBC), cache (Redis?), queue (Kafka/RabbitMQ?), external clients\n"
            f"5. Map Spring beans and their relationships\n\n"
            f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
        )

    # ── Rust ──
    cargo = _search(root, ['Cargo.toml'])
    if cargo:
        rust_root = cargo.parent
        subs = _subdirs(rust_root)
        add_layer('Backend', rust_root, 'Rust',
            f"Study the Rust project '{root.name}' at: {rust_root}\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"Tasks:\n"
            f"1. Read Cargo.toml — list ALL dependencies with their purpose\n"
            f"2. Explore src/ — map modules: handlers, services, models, db, config\n"
            f"3. Find API layer (Axum/Actix/Warp routes)\n"
            f"4. Find: DB (sqlx/diesel/sea-orm), cache, queue, external HTTP clients\n"
            f"5. Map module dependencies\n\n"
            f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
        )

    # ── Ruby / Rails ──
    gemfile = _search(root, ['Gemfile'])
    if gemfile:
        ruby_root = gemfile.parent
        subs = _subdirs(ruby_root)
        framework = 'Rails' if (ruby_root / 'config' / 'routes.rb').exists() else 'Ruby'
        add_layer('Backend', ruby_root, f'Ruby/{framework}',
            f"Study the Ruby/{framework} backend of '{root.name}' at: {ruby_root}\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"Tasks:\n"
            f"1. Read Gemfile — list ALL dependencies with their purpose\n"
            f"2. Explore app/ — controllers, models, services, jobs, mailers\n"
            f"3. Read config/routes.rb — list API endpoints\n"
            f"4. Find: DB (ActiveRecord adapter), cache (Redis?), queue (Sidekiq?), external APIs\n"
            f"5. Map layer interactions\n\n"
            f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
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
            f"Study the .NET/C# project '{root.name}' at: {dotnet_root}\n"
            f"Dirs: {', '.join(subs[:15])}\n\n"
            f"Tasks:\n"
            f"1. Read .csproj — list ALL NuGet dependencies with their purpose\n"
            f"2. Explore project structure — Controllers, Services, Repositories, Models, DTOs\n"
            f"3. Find API endpoints (ASP.NET controllers, minimal APIs)\n"
            f"4. Find: DB (EF Core/Dapper), cache, queue (MassTransit/RabbitMQ?), external clients\n"
            f"5. Map dependency injection registrations\n\n"
            f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
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
                    f"Study the {fw} frontend of '{root.name}' at: {fe_path}\n\n"
                    f"Tasks:\n"
                    f"1. Read package.json — list key dependencies\n"
                    f"2. Map components/pages by feature domain\n"
                    f"3. Find API calls (axios/fetch) — what endpoints are called\n"
                    f"4. Find state management (Vuex/Pinia/Redux/Zustand)\n"
                    f"5. Find routing config\n\n"
                    f"Return JSON: tech, description, key_components[], external_deps[{{name,type,label}}], api_calls[]"
                )
                break

    # ── Docker / Infrastructure ──
    docker_compose = _search(root, ['docker-compose.yml', 'docker-compose.yaml'])
    if docker_compose:
        add_layer('Infrastructure', docker_compose.parent, 'Docker/Compose',
            f"Study the infrastructure of '{root.name}' at: {docker_compose.parent}\n\n"
            f"Tasks:\n"
            f"1. Read docker-compose.yml — list ALL services with their roles\n"
            f"2. For each service identify: its role (database/cache/queue/proxy/app), image name\n"
            f"3. Map service dependencies (depends_on, networks)\n\n"
            f"IMPORTANT for key_components: return ONLY objects with short 'name' (service name from docker-compose, e.g. 'postgres', 'redis', 'nginx', 'backend'). "
            f"Do NOT include image versions, ports, or descriptions in the name field — those go in 'purpose'.\n\n"
            f"Return JSON: tech, description, "
            f"key_components[{{\"name\": \"<short-service-name>\", \"image\": \"<image>\", \"purpose\": \"<one sentence>\"}}], "
            f"external_deps[{{name,type,label}}]\n\n"
            f"Example key_components: [{{\"name\": \"postgres\", \"image\": \"postgres:15\", \"purpose\": \"primary database\"}}, "
            f"{{\"name\": \"redis\", \"image\": \"redis:7\", \"purpose\": \"session cache\"}}]"
        )

    # ── Fallback: generic scan ──
    if not layers:
        subs = _subdirs(root)
        layers.append({
            'name': 'Project',
            'path': str(root),
            'tech': 'Unknown',
            'agent_prompt': (
                f"Study the project '{root.name}' at: {root}\n"
                f"Directories: {', '.join(subs[:20])}\n\n"
                f"Tasks:\n"
                f"1. Identify the technology stack and framework\n"
                f"2. Map the main architectural layers\n"
                f"3. List key components with their purpose\n"
                f"4. Find external dependencies (databases, APIs, services)\n"
                f"5. Map component relationships\n\n"
                f"Return JSON: tech, description, key_components[] (SHORT names ONLY — e.g. AuthHandler, UserService, MailRepo, HttpHandler. Max 25 chars each. NO descriptions or tech details in the name), component_relationships[{{from,to,label}}] (real call graph: which component calls which), external_deps[{{name,type,label}}] (only real DBs/queues/APIs/services), api_endpoints[], key_flows[{{title, steps[{{from,to,message,response}}]}}] (top 2 key flows)"
            ),
        })

    return layers


# ─────────────────────────────────────────────
# NEW MCP TOOLS
# ─────────────────────────────────────────────

@mcp.tool()
def ndoc_explore(project_path: str = "") -> str:
    """
    Scans project structure and returns instructions to launch an Agent Team.
    Each agent studies its own architectural layer in parallel.
    After all agents finish — call ndoc_generate with their combined findings.
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


@mcp.tool()
def ndoc_generate(project_path: str = "", findings: str = "") -> str:
    """
    Generates context.md files and C4 diagrams based on Agent Team findings.
    findings: JSON string with architectural description from agents (output of ndoc_explore flow).
    If findings is empty — use ndoc_init for basic initialization instead.
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

    # Normalize key_components — coerce dicts/objects to strings to prevent crash
    for layer in data.get('layers', []):
        kc = layer.get('key_components', [])
        layer['key_components'] = [
            item if isinstance(item, str)
            else item.get('name', str(item)) if isinstance(item, dict)
            else str(item)
            for item in kc
        ]
    for mod in data.get('modules', []):
        kc = mod.get('key_components', [])
        mod['key_components'] = [
            item if isinstance(item, str)
            else item.get('name', str(item)) if isinstance(item, dict)
            else str(item)
            for item in kc
        ]

    project_name = data.get('project', root.name)
    out = [f"NeuroDoc Generate — {project_name}", "=" * 52, ""]

    # ── Step 1: Static scan (for per-directory context.md) ──
    out.append("Step 1/4 — Static scan...")
    dirs_with_code = []
    all_dirs = set()
    for dp, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
        dp = Path(dp)
        if any(Path(f).suffix.lower() in CODE_EXTENSIONS for f in files):
            dirs_with_code.append(dp)
            parent = dp.parent
            while parent != root.parent:
                all_dirs.add(parent)
                parent = parent.parent
    all_dirs.update(dirs_with_code)
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
                line += f" | {', '.join(kc)}"
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

    # C4 Context (from agent findings)
    index_lines.append("## C4 Context")
    index_lines.append("")
    if layers_info:
        index_lines += make_c4_context_from_findings(data, project_name)
    else:
        index_lines += make_c4_context(modules, project_name, all_module_names)
    index_lines.append("")

    # C4 Container (from agent findings)
    index_lines.append("## C4 Container")
    index_lines.append("")
    if layers_info:
        index_lines += make_c4_container_from_findings(data, project_name)
    else:
        index_lines += make_c4_container(modules, project_name, root)

    # C4 Component — one diagram per non-infra layer that has key_components
    if layers_info:
        comp_sections = []
        _SKIP = ('proto', 'protobuf', 'generated', 'gen', 'infra', 'docker', 'infrastructure')
        for layer in layers_info:
            if any(k in layer.get('name', '').lower() for k in _SKIP):
                continue
            comp_lines = make_c4_component_from_findings(layer)
            if comp_lines:
                comp_sections.append(('## C4 Component — ' + layer['name'], comp_lines))
        if comp_sections:
            for section_title, comp_lines in comp_sections:
                index_lines.append("")
                index_lines.append(section_title)
                index_lines.append("")
                index_lines += comp_lines

    # Sequence diagrams (from agent findings)
    if data.get('key_flows'):
        index_lines.append("")
        index_lines.append("## Key Flows")
        index_lines += make_sequence_from_findings(data, project_name)

    # Also collect key_flows from individual layers
    layer_flows = []
    for layer in layers_info:
        for flow in layer.get('key_flows', []):
            layer_flows.append(flow)
    if layer_flows and not data.get('key_flows'):
        data_with_flows = dict(data)
        data_with_flows['key_flows'] = layer_flows
        index_lines.append("")
        index_lines.append("## Key Flows")
        index_lines += make_sequence_from_findings(data_with_flows, project_name)

    (root / 'context.index.md').write_text('\n'.join(index_lines), encoding='utf-8')
    out.append("   ok: context.index.md written with C4 Context + C4 Container + C4 Component from agent findings")

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
        f"   context.index.md with C4 Context + C4 Container",
        f"   C4 diagrams built from Agent Team findings",
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
