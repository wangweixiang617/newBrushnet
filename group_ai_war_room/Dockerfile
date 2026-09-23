FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
RUN python scripts/init_db.py
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s --start-period=8s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/api/health',timeout=3)" || exit 1
CMD ["python","server.py","--host","0.0.0.0","--port","8050"]
