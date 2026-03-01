#!/bin/bash
# NeuroDoc — Деплой на Linux VPS (Ubuntu/Debian)
# Запускать от root: bash setup-server.sh your-domain.com
#
# Что делает скрипт:
#  1. Устанавливает Docker + Docker Compose
#  2. Клонирует репозиторий
#  3. Получает SSL сертификат (Let's Encrypt)
#  4. Запускает сервисы

set -e

DOMAIN="${1:-}"
REPO_URL="${2:-https://github.com/YOUR_ORG/neurodoc-mcp.git}"
WORKSPACE="/workspace"

if [ -z "$DOMAIN" ]; then
    echo "❌ Укажи домен: bash setup-server.sh ndoc.mycompany.com"
    exit 1
fi

echo ""
echo "🚀 NeuroDoc Server Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Домен:    $DOMAIN"
echo "   Воркспейс: $WORKSPACE"
echo ""

# ── 1. Docker ──────────────────────────────────────────────────────────────
echo "📦 Шаг 1/5 — Установка Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | bash
    systemctl enable docker
    systemctl start docker
    echo "   ✓ Docker установлен"
else
    echo "   ✓ Docker уже есть ($(docker --version | cut -d' ' -f3))"
fi

if ! command -v docker compose &>/dev/null; then
    apt-get install -y docker-compose-plugin 2>/dev/null || \
    pip3 install docker-compose 2>/dev/null || true
fi

# ── 2. Клонируем репо ──────────────────────────────────────────────────────
echo ""
echo "📁 Шаг 2/5 — Получение кода..."
if [ ! -d "/opt/neurodoc-mcp" ]; then
    git clone "$REPO_URL" /opt/neurodoc-mcp
    echo "   ✓ Клонировано в /opt/neurodoc-mcp"
else
    cd /opt/neurodoc-mcp && git pull
    echo "   ✓ Обновлено (git pull)"
fi

# ── 3. Рабочая директория для проектов ────────────────────────────────────
echo ""
echo "📂 Шаг 3/5 — Настройка workspace..."
mkdir -p "$WORKSPACE"
echo "   ✓ $WORKSPACE создан"
echo "   ℹ️  Клонируй сюда проекты команды:"
echo "      cd $WORKSPACE && git clone https://github.com/your-org/your-project"

# ── 4. Заменяем YOUR_DOMAIN в nginx.conf ──────────────────────────────────
echo ""
echo "⚙️  Шаг 4/5 — Конфигурация nginx для $DOMAIN..."
cp /opt/neurodoc-mcp/deploy/nginx.conf /opt/neurodoc-mcp/deploy/nginx.conf.bak
sed -i "s/YOUR_DOMAIN/$DOMAIN/g" /opt/neurodoc-mcp/deploy/nginx.conf
echo "   ✓ nginx.conf настроен"

# ── 5. SSL сертификат ─────────────────────────────────────────────────────
echo ""
echo "🔒 Шаг 5/5 — SSL сертификат (Let's Encrypt)..."

# Временно запускаем nginx на 80 для certbot challenge
mkdir -p /var/www/certbot

# Получаем сертификат
if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    docker run --rm \
        -v /etc/letsencrypt:/etc/letsencrypt \
        -v /var/www/certbot:/var/www/certbot \
        -p 80:80 \
        certbot/certbot certonly \
        --standalone \
        --non-interactive \
        --agree-tos \
        --email "admin@$DOMAIN" \
        -d "$DOMAIN"
    echo "   ✓ SSL сертификат получен"
else
    echo "   ✓ SSL сертификат уже есть"
fi

# ── Запуск ─────────────────────────────────────────────────────────────────
echo ""
echo "🐳 Собираем и запускаем контейнеры..."
cd /opt/neurodoc-mcp

# Собираем образ
docker build -t neurodoc-mcp:latest .

# Запускаем
WORKSPACE_ROOT="$WORKSPACE" docker compose -f deploy/docker-compose.prod.yml up -d

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ NeuroDoc задеплоен!"
echo ""
echo "   🌐 Эндпоинт: https://$DOMAIN/mcp"
echo "   📁 Проекты:  $WORKSPACE/"
echo ""
echo "Как добавить в Claude Code (разработчикам):"
echo "   claude mcp add --transport http ndoc https://$DOMAIN/mcp"
echo ""
echo "Как работать с проектом:"
echo "   1. cd $WORKSPACE && git clone https://github.com/ваш/проект"
echo "   2. В Claude Code: ndoc_init(\"/workspace/проект\")"
echo ""
