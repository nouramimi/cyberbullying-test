FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run_pipeline.py .
COPY data/raw.csv ./data/raw.csv

# Entraîne les modèles et génère rapports/figures au build (sans MongoDB)
RUN python run_pipeline.py --skip-mongo

RUN useradd -m appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
