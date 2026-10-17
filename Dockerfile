FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ADE_ROOT=/app
WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 && pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 ade && mkdir -p runtime reviews && chown -R ade:ade /app
COPY web/ web/
COPY reports/ reports/
USER ade
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["uvicorn", "clinical_ade.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
