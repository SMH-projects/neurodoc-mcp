FROM python:3.12-slim

LABEL org.opencontainers.image.title="NeuroDoc MCP"
LABEL org.opencontainers.image.description="AI navigation for codebases via context.md + C4 diagrams"
LABEL org.opencontainers.image.source="https://github.com/your-org/neurodoc-mcp"

WORKDIR /app

# Зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir mcp uvicorn

# Код сервера
COPY server.py .

# Рабочая директория для монтирования проекта
RUN mkdir -p /workspace

EXPOSE 8000

ENV NDOC_TRANSPORT=streamable-http
ENV NDOC_HOST=0.0.0.0
ENV NDOC_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/mcp')" || exit 1

CMD ["python", "server.py"]
